import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pca_analysis import clean_data, get_numeric_features


DEFAULT_TARGET = "Forecast_Failed_Flag"


def ensure_event_id(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    id_cols = ["Programme_ID", "Supplier_ID", "Commodity", "Forecast_Period"]
    if "event_id" not in result.columns and all(col in result.columns for col in id_cols):
        result["event_id"] = (
            result["Programme_ID"].astype(str)
            + "|"
            + result["Supplier_ID"].astype(str)
            + "|"
            + result["Commodity"].astype(str)
            + "|"
            + result["Forecast_Period"].astype(str)
        )
    return result


def split_by_event_time(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "event_id" not in df.columns:
        raise ValueError("event_id is required for event-based split.")
    if "Forecast_Period_End_Date" not in df.columns:
        raise ValueError("Forecast_Period_End_Date is required for event-based split.")

    event_dates = df.groupby("event_id")["Forecast_Period_End_Date"].max().sort_values()
    if event_dates.empty:
        raise ValueError("No event dates available for splitting.")

    cutoff = max(1, int(len(event_dates) * train_frac))
    train_events = set(event_dates.iloc[:cutoff].index)
    test_events = set(event_dates.iloc[cutoff:].index)

    if not test_events:
        # Ensure there is always a test set when data is tiny.
        last_event = event_dates.index[-1]
        train_events.discard(last_event)
        test_events.add(last_event)

    train_df = df[df["event_id"].isin(train_events)].copy()
    test_df = df[df["event_id"].isin(test_events)].copy()
    return train_df, test_df


def build_pca_pipeline(n_components: float | int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )


def find_influential_factor(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_components: float | int,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    pca_pipeline = build_pca_pipeline(n_components=n_components)
    scores = pca_pipeline.fit_transform(df[feature_cols])
    pca_model = pca_pipeline.named_steps["pca"]

    component_names = [f"PC{i + 1}" for i in range(scores.shape[1])]

    explained_df = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": pca_model.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca_model.explained_variance_ratio_),
        }
    )

    loadings_df = pd.DataFrame(pca_model.components_.T, index=feature_cols, columns=component_names)
    scores_df = pd.DataFrame(scores, columns=component_names, index=df.index)

    chosen_component = component_names[0]
    component_corr = np.nan

    if target_col in df.columns:
        y = pd.Series(pd.to_numeric(df[target_col], errors="coerce"), index=df.index)
        scores_df[target_col] = y

        corrs: dict[str, float] = {}
        valid = y.notna()
        for component in component_names:
            comp_values = pd.Series(scores_df.loc[valid, component], index=y.loc[valid].index)
            if comp_values.nunique(dropna=True) < 2:
                corrs[component] = np.nan
            else:
                corrs[component] = float(comp_values.corr(y.loc[valid]))

        corr_series = pd.Series(corrs)
        if corr_series.notna().any():
            chosen_component = corr_series.abs().idxmax()
            component_corr = float(corr_series.loc[chosen_component])

    abs_loadings = loadings_df[chosen_component].abs().sort_values(ascending=False)
    selected_factor = abs_loadings.index[0]
    selected_loading = float(loadings_df.loc[selected_factor, chosen_component])

    selected_info = {
        "target_column": target_col,
        "selected_component": chosen_component,
        "selected_factor": selected_factor,
        "selected_factor_loading": selected_loading,
        "component_target_correlation": component_corr,
    }

    return explained_df, loadings_df, scores_df, selected_info


def train_test_linear(
    df: pd.DataFrame,
    factor: str,
    target_col: str,
    train_frac: float,
    threshold: float,
) -> tuple[pd.DataFrame, dict]:
    data = df.copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data[factor] = pd.to_numeric(data[factor], errors="coerce")
    data = data[data[target_col].notna() & data[factor].notna()].copy()

    train_df, test_df = split_by_event_time(data, train_frac=train_frac)
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty set. Adjust --train-frac or input data.")

    model = LinearRegression()
    model.fit(train_df[[factor]], train_df[target_col])

    raw_pred = model.predict(test_df[[factor]])
    pred_likelihood = np.clip(raw_pred, 0.0, 1.0)
    pred_flag = (pred_likelihood >= threshold).astype(int)

    scored = test_df.copy()
    scored["predicted_failure_likelihood_linear"] = pred_likelihood
    scored["derived_predicted_failure_flag"] = pred_flag
    scored["prediction_residual"] = scored[target_col] - scored["predicted_failure_likelihood_linear"]

    unique_target = sorted(pd.Series(scored[target_col].dropna().unique()).tolist())
    is_binary_target = set(unique_target).issubset({0, 1}) and len(unique_target) <= 2

    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "target_column": target_col,
        "feature_used": factor,
        "linear_intercept": float(model.intercept_),
        "linear_coefficient": float(model.coef_[0]),
        "mae": float(mean_absolute_error(scored[target_col], pred_likelihood)),
        "rmse": float(np.sqrt(mean_squared_error(scored[target_col], pred_likelihood))),
        "r2": float(r2_score(scored[target_col], pred_likelihood)),
        "threshold": float(threshold),
        "predicted_positive_rate": float(pred_flag.mean()),
        "is_binary_target": bool(is_binary_target),
    }

    if is_binary_target and scored[target_col].nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(scored[target_col], pred_likelihood))
    else:
        metrics["roc_auc"] = np.nan

    return scored, metrics


def build_plotly_report(
    predictions_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    output_html: Path,
    target_col: str,
) -> None:
    component = selected_df.loc[0, "selected_component"]
    factor = selected_df.loc[0, "selected_factor"]

    top_loadings = (
        loadings_df[[component]]
        .rename(columns={component: "loading"})
        .assign(abs_loading=lambda x: x["loading"].abs())
        .sort_values("abs_loading", ascending=False)
        .head(12)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    fig1 = px.bar(
        top_loadings,
        x="abs_loading",
        y="feature",
        orientation="h",
        title=f"Top PCA Loadings for {component}",
    )
    fig1.update_layout(yaxis={"categoryorder": "total ascending"})

    fig2 = px.scatter(
        predictions_df,
        x=factor,
        y="predicted_failure_likelihood_linear",
        color=target_col if target_col in predictions_df.columns else None,
        title=f"Linear Predicted Likelihood vs {factor}",
        hover_data=["event_id"] if "event_id" in predictions_df.columns else None,
    )

    fig3 = px.histogram(
        predictions_df,
        x="predicted_failure_likelihood_linear",
        color=target_col if target_col in predictions_df.columns else None,
        nbins=30,
        barmode="overlay",
        opacity=0.6,
        title="Predicted Failure Likelihood Distribution (Linear Model)",
    )

    bin_df = predictions_df.copy()
    bin_df["risk_bucket"] = pd.qcut(
        bin_df["predicted_failure_likelihood_linear"],
        q=10,
        labels=False,
        duplicates="drop",
    )
    can_calibrate = target_col in bin_df.columns
    if can_calibrate:
        unique_target = sorted(pd.Series(bin_df[target_col].dropna().unique()).tolist())
        can_calibrate = set(unique_target).issubset({0, 1}) and len(unique_target) <= 2

    if can_calibrate:
        calibration = (
            bin_df.groupby("risk_bucket", dropna=False)
            .agg(
                avg_predicted=("predicted_failure_likelihood_linear", "mean"),
                actual_failure_rate=(target_col, "mean"),
                count=(target_col, "size"),
            )
            .reset_index()
            .dropna(subset=["risk_bucket"])
        )

        fig4 = px.line(
            calibration,
            x="avg_predicted",
            y="actual_failure_rate",
            markers=True,
            title=f"Calibration: Actual {target_col} Rate vs Predicted Likelihood",
            hover_data=["count"],
        )
        fig4.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"dash": "dash"},
        )
    else:
        fig4 = px.scatter(
            predictions_df,
            x="predicted_failure_likelihood_linear",
            y="prediction_residual",
            title="Residuals vs Predicted Likelihood",
        )

    report_html = "\n".join(
        [
            "<h2>PCA + Linear Regression Workflow Report</h2>",
            pio.to_html(fig1, include_plotlyjs="cdn", full_html=False),
            pio.to_html(fig2, include_plotlyjs=False, full_html=False),
            pio.to_html(fig3, include_plotlyjs=False, full_html=False),
            pio.to_html(fig4, include_plotlyjs=False, full_html=False),
        ]
    )

    output_html.write_text(
        "<html><head><meta charset='utf-8'><title>PCA Linear Workflow</title></head><body>"
        + report_html
        + "</body></html>",
        encoding="utf-8",
    )


def parse_n_components(value: str) -> float | int:
    try:
        return int(value)
    except ValueError:
        return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PCA to find the most influential factor, then train/test a linear model and save Plotly-ready outputs."
    )
    parser.add_argument("--input", default="data/forecast_data.csv", help="Input forecast CSV")
    parser.add_argument("--n-components", default="0.95", help="PCA components (int) or variance target (float)")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train split by event chronology")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for derived flag")
    parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET,
        help="Column to predict with linear regression (default: Forecast_Failed_Flag)",
    )
    parser.add_argument("--output-dir", default="outputs", help="Folder for workflow artifacts")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = clean_data(args.input)
    df = ensure_event_id(df)

    if args.target_column not in df.columns:
        raise ValueError(f"Required target column missing: {args.target_column}")

    feature_cols = get_numeric_features(df)
    if not feature_cols:
        raise ValueError("No numeric features available for PCA.")

    explained_df, loadings_df, scores_df, selected_info = find_influential_factor(
        df=df,
        feature_cols=feature_cols,
        n_components=parse_n_components(args.n_components),
        target_col=args.target_column,
    )

    selected_df = pd.DataFrame([selected_info])

    predictions_df, metrics = train_test_linear(
        df=df,
        factor=selected_info["selected_factor"],
        target_col=args.target_column,
        train_frac=float(args.train_frac),
        threshold=float(args.threshold),
    )

    variance_path = output_dir / "workflow_pca_variance.csv"
    loadings_path = output_dir / "workflow_pca_loadings.csv"
    scores_path = output_dir / "workflow_pca_scores.csv"
    selected_path = output_dir / "workflow_selected_factor.csv"
    metrics_path = output_dir / "workflow_linear_metrics.csv"
    predictions_path = output_dir / "workflow_linear_predictions.csv"
    metadata_path = output_dir / "workflow_run_metadata.json"
    report_path = output_dir / "workflow_plotly_report.html"

    explained_df.to_csv(variance_path, index=False)
    loadings_df.to_csv(loadings_path)
    scores_df.to_csv(scores_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    metadata = {
        "input": args.input,
        "output_dir": str(output_dir),
        "target_column": args.target_column,
        "selected_factor": selected_info["selected_factor"],
        "selected_component": selected_info["selected_component"],
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    build_plotly_report(
        predictions_df=predictions_df,
        selected_df=selected_df,
        loadings_df=loadings_df,
        output_html=report_path,
        target_col=args.target_column,
    )

    print("\nWorkflow completed")
    print("------------------")
    print(f"Rows in source data: {len(df):,}")
    print(f"Target column:      {args.target_column}")
    print(f"Selected component: {selected_info['selected_component']}")
    print(f"Selected factor:    {selected_info['selected_factor']}")
    print(f"Linear coefficient: {metrics['linear_coefficient']:.6f}")
    print(f"ROC AUC (test):     {metrics['roc_auc']:.4f}" if not np.isnan(metrics["roc_auc"]) else "ROC AUC (test): n/a")
    print(f"Output folder:      {output_dir.resolve()}")
    print("Artifacts:")
    print(f"- {variance_path}")
    print(f"- {loadings_path}")
    print(f"- {scores_path}")
    print(f"- {selected_path}")
    print(f"- {metrics_path}")
    print(f"- {predictions_path}")
    print(f"- {metadata_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()

