import base64
import io
import os
from functools import lru_cache
from pathlib import Path

import dash
from dash import Input, Output, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from .forecast_failure import build_pipeline, extract_days_from_terms, load_data
from .paths import ASSETS_DIR, DATA_DIR, REPO_ROOT


def resolve_existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    return None


DEFAULT_SCORED_PATH = resolve_existing_path(
    Path(os.getenv("FORECAST_SCORED_PATH", "")) if os.getenv("FORECAST_SCORED_PATH") else None,
    REPO_ROOT / "forecast_failure_scored.csv",
    Path.cwd() / "forecast_failure_scored.csv",
)

DEFAULT_TRAIN_PATH = resolve_existing_path(
    Path(os.getenv("FORECAST_TRAIN_PATH", "")) if os.getenv("FORECAST_TRAIN_PATH") else None,
    DATA_DIR / "forecast_data.csv",
    Path.cwd() / "data" / "forecast_data.csv",
)

DEFAULT_SUPPLIER_PATH = resolve_existing_path(
    Path(os.getenv("FORECAST_SUPPLIER_PATH", "")) if os.getenv("FORECAST_SUPPLIER_PATH") else None,
    DATA_DIR / "supplier_attributes.csv",
    Path.cwd() / "data" / "supplier_attributes.csv",
)

TARGET = "Forecast_Failed_Flag"
EXCLUDED = {
    TARGET,
    "Actual_Spend",
    "Variance",
    "Absolute_Error",
    "Actual_Minus_Committed",
    "Forecast_Period_End_Date",
    "event_id",
}

CATEGORICAL_FEATURES = [
    "Programme_ID",
    "Commodity",
    "Supplier_ID",
    "Forecast_Period",
    "Forecast_Change_Direction",
    "Confidence_Band",
    "Contract_Type",
    "Supplier_Profile",
    "Region",
    "Payment_Terms",
]

NUMERIC_FEATURES = [
    "Forecast_Spend",
    "Revision_Number",
    "Previous_Forecast_Spend",
    "Forecast_Change",
    "Forecast_Stability_Score",
    "Days_Before_Period",
    "Committed_Spend",
    "Commitment_Ratio",
    "PO_Count",
    "Programme_Change_Count",
    "Programme_Scope_Churn_Index",
    "Programme_Change_Impact_Index",
    "forecast_version_month",
    "forecast_version_quarter",
    "forecast_period_month",
    "forecast_change_pct",
    "forecast_to_committed_ratio",
    "OTIF_Pct",
    "Avg_Lead_Time_Days",
    "Quality_Incidents_YTD",
    "Payment_Terms_Days",
    "Strategic_Flag_Num",
    "New_Supplier_Flag_Num",
]


def decode_uploaded_csv(contents: str) -> pd.DataFrame:
    content_type, content_string = contents.split(",", 1)
    _ = content_type
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))


def is_scored_input(df: pd.DataFrame) -> bool:
    if "predicted_failure_probability" not in df.columns:
        return False
    probs = pd.to_numeric(df["predicted_failure_probability"], errors="coerce")
    return probs.notna().any()


def enrich_raw_input(df: pd.DataFrame, supplier_path: Path = DEFAULT_SUPPLIER_PATH) -> pd.DataFrame:
    enriched = df.copy()

    if "Forecast_Version_Date" in enriched.columns:
        enriched["Forecast_Version_Date"] = pd.to_datetime(enriched["Forecast_Version_Date"], errors="coerce")
    if "Forecast_Period_End_Date" in enriched.columns:
        enriched["Forecast_Period_End_Date"] = pd.to_datetime(enriched["Forecast_Period_End_Date"], errors="coerce")

    id_cols = ["Programme_ID", "Supplier_ID", "Commodity", "Forecast_Period"]
    if "event_id" not in enriched.columns and all(c in enriched.columns for c in id_cols):
        enriched["event_id"] = (
            enriched["Programme_ID"].astype(str)
            + "|"
            + enriched["Supplier_ID"].astype(str)
            + "|"
            + enriched["Commodity"].astype(str)
            + "|"
            + enriched["Forecast_Period"].astype(str)
        )

    if "Forecast_Version_Date" in enriched.columns:
        enriched["forecast_version_month"] = enriched["Forecast_Version_Date"].dt.month
        enriched["forecast_version_quarter"] = enriched["Forecast_Version_Date"].dt.quarter

    if "Forecast_Period" in enriched.columns:
        enriched["forecast_period_month"] = pd.to_datetime(
            enriched["Forecast_Period"].astype(str) + "-01", errors="coerce"
        ).dt.month

    if "Previous_Forecast_Spend" in enriched.columns and "Forecast_Change" in enriched.columns:
        enriched["forecast_change_pct"] = np.where(
            enriched["Previous_Forecast_Spend"].fillna(0) != 0,
            enriched["Forecast_Change"] / enriched["Previous_Forecast_Spend"],
            0.0,
        )

    if "Committed_Spend" in enriched.columns and "Forecast_Spend" in enriched.columns:
        enriched["forecast_to_committed_ratio"] = np.where(
            enriched["Committed_Spend"].fillna(0) != 0,
            enriched["Forecast_Spend"] / enriched["Committed_Spend"],
            np.nan,
        )

    if supplier_path is not None and supplier_path.exists() and "Supplier_ID" in enriched.columns:
        supplier_df = pd.read_csv(supplier_path)
        supplier_df["Supplier_ID"] = supplier_df["Supplier_ID"].astype(str).str.strip()
        supplier_df = supplier_df.drop_duplicates(subset=["Supplier_ID"], keep="first")

        supplier_df["Payment_Terms_Days"] = supplier_df["Payment_Terms"].apply(extract_days_from_terms)
        supplier_df["Strategic_Flag_Num"] = supplier_df["Strategic_Flag"].map({"Yes": 1, "No": 0})
        supplier_df["New_Supplier_Flag_Num"] = supplier_df["New_Supplier_Flag"].map({"Yes": 1, "No": 0})

        enriched = enriched.merge(supplier_df, on="Supplier_ID", how="left")

    return enriched


@lru_cache(maxsize=1)
def get_trained_model() -> tuple:
    if DEFAULT_TRAIN_PATH is None:
        raise ValueError(
            "Training data not found. Looked for data/forecast_data.csv relative to app and working directory."
        )

    supplier_path = str(DEFAULT_SUPPLIER_PATH) if DEFAULT_SUPPLIER_PATH is not None else "data/supplier_attributes.csv"
    train_df = load_data(str(DEFAULT_TRAIN_PATH), supplier_path)
    candidate_features = [c for c in train_df.columns if c not in EXCLUDED]

    cat_features = [c for c in CATEGORICAL_FEATURES if c in candidate_features]
    num_features = [c for c in NUMERIC_FEATURES if c in candidate_features]

    if not (cat_features or num_features):
        raise ValueError("No trainable features available in historical data.")

    model = build_pipeline(cat_features, num_features)
    model.fit(train_df[cat_features + num_features], train_df[TARGET])
    return model, tuple(cat_features), tuple(num_features)


def score_raw_df(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    enriched = enrich_raw_input(df)
    model, cat_features, num_features = get_trained_model()

    all_features = list(cat_features) + list(num_features)
    for col in all_features:
        if col not in enriched.columns:
            enriched[col] = np.nan

    scored = enriched.copy()
    scored["predicted_failure_probability"] = model.predict_proba(scored[all_features])[:, 1]
    scored["derived_predicted_failure_flag"] = (
        scored["predicted_failure_probability"] >= threshold
    ).astype(int)

    if TARGET in scored.columns:
        scored[TARGET] = pd.to_numeric(scored[TARGET], errors="coerce")

    return scored


def prepare_scored_df(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if "predicted_failure_probability" not in df.columns:
        raise ValueError("Missing required column: predicted_failure_probability")

    df = df.copy()
    df["predicted_failure_probability"] = pd.to_numeric(
        df["predicted_failure_probability"], errors="coerce"
    )
    df = df[df["predicted_failure_probability"].notna()].copy()
    if df.empty:
        raise ValueError(
            "No valid predicted_failure_probability values were found. "
            "Upload a scored file or upload raw forecast data to auto-score."
        )

    df["derived_predicted_failure_flag"] = (
        df["predicted_failure_probability"] >= threshold
    ).astype(int)

    if "Forecast_Failed_Flag" in df.columns:
        df["Forecast_Failed_Flag"] = pd.to_numeric(df["Forecast_Failed_Flag"], errors="coerce")

    return df


def figure_probability_histogram(df: pd.DataFrame) -> go.Figure:
    color_col = "Forecast_Failed_Flag" if "Forecast_Failed_Flag" in df.columns else None
    fig = px.histogram(
        df,
        x="predicted_failure_probability",
        color=color_col,
        nbins=30,
        barmode="overlay",
        opacity=0.6,
        title="Predicted Failure Probability Distribution",
    )
    fig.update_layout(xaxis_title="Predicted Failure Probability", yaxis_title="Count")
    return fig


def figure_top_risk_rows(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    work = df.copy().sort_values("predicted_failure_probability", ascending=False).head(top_n)
    if "Programme_ID" in work.columns and "Supplier_ID" in work.columns and "Forecast_Period" in work.columns:
        work["forecast_key"] = (
            work["Programme_ID"].astype(str)
            + " | "
            + work["Supplier_ID"].astype(str)
            + " | "
            + work["Forecast_Period"].astype(str)
        )
    else:
        work["forecast_key"] = work.index.astype(str)

    fig = px.bar(
        work,
        x="predicted_failure_probability",
        y="forecast_key",
        orientation="h",
        title=f"Top {top_n} Highest-Risk Forecasts",
    )
    fig.update_layout(
        xaxis_title="Predicted Failure Probability",
        yaxis_title="Forecast Key",
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


def figure_group_risk(df: pd.DataFrame, group_col: str, top_n: int = 20) -> go.Figure:
    agg = (
        df.groupby(group_col, dropna=False)["predicted_failure_probability"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_failure_probability", "count": "forecast_count"})
        .sort_values("avg_failure_probability", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        agg,
        x="avg_failure_probability",
        y=group_col,
        orientation="h",
        color="forecast_count",
        color_continuous_scale="Blues",
        title=f"Average Risk by {group_col} (Top {top_n})",
    )
    fig.update_layout(
        xaxis_title="Average Predicted Failure Probability",
        yaxis_title=group_col,
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


def figure_period_trend(df: pd.DataFrame) -> go.Figure:
    trend = df.copy()
    trend["period_dt"] = pd.to_datetime(trend["Forecast_Period"] + "-01", errors="coerce")
    trend = trend[trend["period_dt"].notna()]

    agg = (
        trend.groupby("period_dt")["predicted_failure_probability"]
        .mean()
        .reset_index(name="avg_failure_probability")
        .sort_values("period_dt")
    )

    fig = px.line(
        agg,
        x="period_dt",
        y="avg_failure_probability",
        markers=True,
        title="Average Predicted Risk Over Forecast Period",
    )
    fig.update_layout(xaxis_title="Forecast Period", yaxis_title="Average Predicted Failure Probability")
    return fig


def figure_confusion_matrix(df: pd.DataFrame) -> go.Figure:
    matrix = pd.crosstab(
        df["Forecast_Failed_Flag"].fillna(-1).astype(int),
        df["derived_predicted_failure_flag"],
        rownames=["Actual"],
        colnames=["Predicted"],
    )

    fig = px.imshow(
        matrix.values,
        text_auto=True,
        x=[str(c) for c in matrix.columns],
        y=[str(i) for i in matrix.index],
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    fig.update_layout(xaxis_title="Predicted Flag", yaxis_title="Actual Flag")
    return fig


def figure_roc_pr(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    actual = df["Forecast_Failed_Flag"].dropna().astype(int)
    proba = df.loc[actual.index, "predicted_failure_probability"]

    fpr, tpr, _ = roc_curve(actual, proba)
    precision, recall, _ = precision_recall_curve(actual, proba)

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC = {roc_auc:.3f}"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash"}, name="Baseline"))
    roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

    pr_fig = go.Figure()
    pr_fig.add_trace(
        go.Scatter(x=recall, y=precision, mode="lines", name=f"PR AUC = {pr_auc:.3f}")
    )
    pr_fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")

    return roc_fig, pr_fig


def build_graphs(df: pd.DataFrame) -> list:
    graphs = [
        dcc.Graph(figure=figure_probability_histogram(df)),
        dcc.Graph(figure=figure_top_risk_rows(df)),
    ]

    for col in ["Supplier_ID", "Programme_ID", "Commodity"]:
        if col in df.columns:
            graphs.append(dcc.Graph(figure=figure_group_risk(df, col)))

    if "Forecast_Period" in df.columns:
        graphs.append(dcc.Graph(figure=figure_period_trend(df)))

    if "Forecast_Failed_Flag" in df.columns:
        perf_df = df[df["Forecast_Failed_Flag"].isin([0, 1])].copy()
        if not perf_df.empty:
            graphs.append(dcc.Graph(figure=figure_confusion_matrix(perf_df)))
            roc_fig, pr_fig = figure_roc_pr(perf_df)
            graphs.append(dcc.Graph(figure=roc_fig))
            graphs.append(dcc.Graph(figure=pr_fig))

    return graphs


app = dash.Dash(__name__, assets_folder=str(ASSETS_DIR))
app.title = "Forecast Failure Risk Dashboard"

app.layout = html.Div(
    [
        html.H2("Forecast Failure Risk Dashboard (Plotly)"),
        html.P(
            "Upload either a scored CSV (with predicted_failure_probability) "
            "or a raw forecast CSV to auto-score."
        ),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and drop or ", html.A("select a CSV file")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "marginBottom": "16px",
            },
            multiple=False,
        ),
        html.Div(
            [
                html.Label("Failure threshold for predicted flag"),
                dcc.Slider(
                    id="threshold-slider",
                    min=0.05,
                    max=0.95,
                    step=0.01,
                    value=0.5,
                    marks={0.1: "0.1", 0.3: "0.3", 0.5: "0.5", 0.7: "0.7", 0.9: "0.9"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        html.Div(id="status", style={"marginBottom": "10px", "fontWeight": "bold"}),
        html.Div(id="graphs"),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
)


@app.callback(
    Output("status", "children"),
    Output("graphs", "children"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename"),
    Input("threshold-slider", "value"),
)
def render_dashboard(contents, filename, threshold):
    try:
        if contents is None:
            if DEFAULT_SCORED_PATH is None:
                return (
                    "Upload a CSV to start. Default file forecast_failure_scored.csv not found.",
                    [],
                )
            df = pd.read_csv(DEFAULT_SCORED_PATH)
            source_name = str(DEFAULT_SCORED_PATH)
        else:
            df = decode_uploaded_csv(contents)
            source_name = filename or "uploaded file"

        if is_scored_input(df):
            scored_df = prepare_scored_df(df, float(threshold))
            source_type = "scored"
        else:
            scored_df = score_raw_df(df, float(threshold))
            source_type = "raw->scored"

        graphs = build_graphs(scored_df)

        status = (
            f"Loaded {source_name} ({source_type}): {len(scored_df):,} rows | "
            f"Threshold={threshold:.2f} | "
            f"Avg risk={scored_df['predicted_failure_probability'].mean():.3f}"
        )
        return status, graphs
    except Exception as exc:
        return f"Error: {exc}", []


def main() -> None:
    app.run(debug=True)


if __name__ == "__main__":
    main()


