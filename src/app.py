from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from model.dashboard_service import DashboardRepository, SERIES_KEYS


TIME_BUCKET_ORDER = ["150d+", "121-150d", "91-120d", "61-90d", "31-60d", "0-30d"]
CONFIDENCE_ORDER = ["High", "Medium", "Low"]
GRAPH_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["resetScale2d"],
}


@dataclass(frozen=True)
class DashboardData:
    forecast: pd.DataFrame
    supplier_profiles: list[str]
    programmes: list[str]


def currency(value: float) -> str:
    return f"GBP {value:,.0f}"


def percent(value: float) -> str:
    return f"{value:.1%}"


def load_dashboard_data() -> DashboardData:
    forecast = DashboardRepository.from_disk().prepare_forecast_frame()
    supplier_profiles = sorted(value for value in forecast["Supplier_Profile"].dropna().unique())
    programmes = sorted(value for value in forecast["Programme_ID"].dropna().unique())
    return DashboardData(
        forecast=forecast,
        supplier_profiles=supplier_profiles,
        programmes=programmes,
    )


DATA = load_dashboard_data()

app = Dash(
    __name__,
    title="Supply Chain Forecast Confidence Dashboard",
    assets_folder="assets",
)
server = app.server


def make_kpi_card(label: str, value: str) -> html.Div:
    return html.Div(
        className="kpi-card",
        children=[
            html.Span(label, className="kpi-label"),
            html.Strong(value, className="kpi-value"),
        ],
    )


def filter_forecast(programme_id: str | None, supplier_profile: str | None) -> pd.DataFrame:
    frame = DATA.forecast.copy()
    if programme_id and programme_id != "ALL":
        frame = frame.loc[frame["Programme_ID"] == programme_id]
    if supplier_profile and supplier_profile != "ALL":
        frame = frame.loc[frame["Supplier_Profile"] == supplier_profile]
    return frame


def summarise_series(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(SERIES_KEYS + ["Revision_Number", "Forecast_Version_Date"]).copy()
    summary = (
        ordered.groupby(SERIES_KEYS, as_index=False)
        .agg(
            Programme_Phase=("Programme_Phase", "first"),
            Delivery_Risk=("Delivery_Risk", "first"),
            Actual_Spend=("Actual_Spend", "first"),
            Initial_Forecast=("Forecast_Spend", "first"),
            Latest_Forecast=("Forecast_Spend", "last"),
            Latest_APE=("Absolute_Percentage_Error", "last"),
            Average_APE=("Absolute_Percentage_Error", "mean"),
            Average_Confidence_Band=("Confidence_Band", "last"),
            Low_Confidence_Share=("Low_Confidence_Flag", "mean"),
            Forecast_Fade_Pct=("Forecast_Fade_Pct", "last"),
            Average_Commitment_Ratio=("Commitment_Ratio", "mean"),
            Revision_Count=("Revision_Number", "max"),
        )
    )
    summary["Measured_Confidence"] = (1 - summary["Latest_APE"]).clip(lower=0, upper=1)
    summary["Risk_Score"] = (
        summary["Latest_APE"].fillna(0) * 55
        + (-summary["Forecast_Fade_Pct"].clip(upper=0).abs()).fillna(0) * 20
        + summary["Low_Confidence_Share"].fillna(0) * 15
        + (1 - summary["Average_Commitment_Ratio"].fillna(0)) * 10
    )
    return summary


def build_convergence_chart(frame: pd.DataFrame) -> go.Figure:
    bucket_view = (
        frame.dropna(subset=["Time_Bucket"])
        .groupby("Time_Bucket", as_index=False, observed=False)
        .agg(
            median_signed_error=("Signed_Error_Pct", "median"),
            lower_bound=("Signed_Error_Pct", lambda s: s.quantile(0.25)),
            upper_bound=("Signed_Error_Pct", lambda s: s.quantile(0.75)),
            median_ape=("Absolute_Percentage_Error", "median"),
        )
    )
    bucket_view["Time_Bucket"] = pd.Categorical(
        bucket_view["Time_Bucket"], categories=TIME_BUCKET_ORDER, ordered=True
    )
    bucket_view = bucket_view.sort_values("Time_Bucket")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bucket_view["Time_Bucket"],
            y=bucket_view["upper_bound"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bucket_view["Time_Bucket"],
            y=bucket_view["lower_bound"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(13, 107, 104, 0.12)",
            line=dict(width=0),
            name="Interquartile range",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bucket_view["Time_Bucket"],
            y=bucket_view["median_signed_error"],
            mode="lines+markers",
            line=dict(color="#0d6b68", width=4),
            marker=dict(size=9),
            name="Median signed error",
            customdata=bucket_view["median_ape"],
            hovertemplate=(
                "Bucket: %{x}<br>"
                "Median signed error: %{y:.1%}<br>"
                "Median APE: %{customdata:.1%}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_color="#a34f2d", line_dash="dash")
    fig.update_layout(
        title="Convergence Curve",
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        yaxis_title="Forecast bias vs actual",
        xaxis_title="Time before spend period",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def build_confidence_chart(frame: pd.DataFrame) -> go.Figure:
    confidence_view = (
        frame.dropna(subset=["Time_Bucket"])
        .groupby(["Time_Bucket", "Confidence_Band"], as_index=False, observed=False)
        .agg(
            mean_ape=("Absolute_Percentage_Error", "mean"),
            row_count=("Confidence_Band", "size"),
        )
    )
    confidence_view = confidence_view.loc[confidence_view["Confidence_Band"].isin(CONFIDENCE_ORDER)].copy()
    confidence_view["Measured_Confidence"] = (1 - confidence_view["mean_ape"]).clip(lower=0, upper=1)
    confidence_view["Time_Bucket"] = pd.Categorical(
        confidence_view["Time_Bucket"], categories=TIME_BUCKET_ORDER, ordered=True
    )
    confidence_view["Confidence_Band"] = pd.Categorical(
        confidence_view["Confidence_Band"], categories=CONFIDENCE_ORDER, ordered=True
    )
    confidence_view = confidence_view.sort_values(["Time_Bucket", "Confidence_Band"])
    if confidence_view.empty:
        return px.imshow(title="Confidence Heatmap")

    confidence_view["Display_Value"] = confidence_view.apply(
        lambda row: row["Measured_Confidence"] if row["row_count"] >= 20 else None,
        axis=1,
    )
    heatmap = confidence_view.pivot(
        index="Confidence_Band",
        columns="Time_Bucket",
        values="Display_Value",
    ).reindex(index=CONFIDENCE_ORDER, columns=TIME_BUCKET_ORDER)
    counts = confidence_view.pivot(
        index="Confidence_Band",
        columns="Time_Bucket",
        values="row_count",
    ).reindex(index=CONFIDENCE_ORDER, columns=TIME_BUCKET_ORDER)
    mean_ape = confidence_view.pivot(
        index="Confidence_Band",
        columns="Time_Bucket",
        values="mean_ape",
    ).reindex(index=CONFIDENCE_ORDER, columns=TIME_BUCKET_ORDER)

    fig = go.Figure(
        data=
        go.Heatmap(
            z=heatmap.values,
            x=list(heatmap.columns),
            y=list(heatmap.index),
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "#a34f2d"],
                [0.5, "#e3c269"],
                [1.0, "#0d6b68"],
            ],
            colorbar=dict(title="Measured confidence", tickformat=".0%"),
            customdata=list(zip(counts.values.tolist(), mean_ape.values.tolist())),
            hovertemplate=(
                "Declared confidence: %{y}<br>"
                "Bucket: %{x}<br>"
                "Measured confidence: %{z:.1%}<br>"
                "Rows: %{customdata[0]}<br>"
                "Mean APE: %{customdata[1]:.1%}<extra></extra>"
            ),
            text=counts.values,
            texttemplate="%{text}",
            textfont={"color": "#1f1b16", "size": 12},
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="Confidence Heatmap",
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Time before spend period",
        yaxis_title="Declared confidence",
    )
    fig.add_annotation(
        x=0.5,
        y=-0.22,
        xref="paper",
        yref="paper",
        text="Cells with fewer than 20 observations are left blank; numbers show row counts.",
        showarrow=False,
        font={"size": 12, "color": "#65594a"},
    )
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")
    return fig


def build_programme_risk_table(frame: pd.DataFrame) -> pd.DataFrame:
    series = summarise_series(frame)
    programme_risk = (
        series.groupby(["Programme_ID", "Programme_Phase", "Delivery_Risk"], as_index=False, dropna=False)
        .agg(
            Series_Count=("Programme_ID", "size"),
            Weighted_APE=("Latest_APE", "mean"),
            Average_Fade=("Forecast_Fade_Pct", "mean"),
            Low_Confidence_Share=("Low_Confidence_Share", "mean"),
            Average_Commitment_Ratio=("Average_Commitment_Ratio", "mean"),
            Average_Measured_Confidence=("Measured_Confidence", "mean"),
            Risk_Score=("Risk_Score", "mean"),
        )
        .sort_values("Risk_Score", ascending=False)
    )
    programme_risk["Weighted_APE"] = programme_risk["Weighted_APE"].map(percent)
    programme_risk["Average_Fade"] = programme_risk["Average_Fade"].map(percent)
    programme_risk["Low_Confidence_Share"] = programme_risk["Low_Confidence_Share"].map(percent)
    programme_risk["Average_Commitment_Ratio"] = programme_risk["Average_Commitment_Ratio"].map(percent)
    programme_risk["Average_Measured_Confidence"] = programme_risk["Average_Measured_Confidence"].map(percent)
    programme_risk["Risk_Score"] = programme_risk["Risk_Score"].map(lambda value: f"{value:.1f}")
    return programme_risk.head(10)


def build_layout() -> html.Main:
    return html.Main(
        className="page-shell",
        children=[
            html.Section(
                className="hero",
                children=[
                    html.P("Supply Chain Forecasting", className="eyebrow"),
                    html.H1("Forecast Convergence And Confidence Dashboard"),
                    html.P(
                        "This view focuses on prediction readiness: how forecasts converge, how trust changes as delivery approaches, and which programmes carry the most forecasting risk.",
                        className="hero-copy",
                    ),
                ],
            ),
            html.Section(
                className="control-strip",
                children=[
                    html.Div(
                        className="control-group",
                        children=[
                            html.Label("Programme"),
                            dcc.Dropdown(
                                id="programme-filter",
                                options=[{"label": "All programmes", "value": "ALL"}]
                                + [{"label": item, "value": item} for item in DATA.programmes],
                                value="ALL",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-group",
                        children=[
                            html.Label("Supplier profile"),
                            dcc.Dropdown(
                                id="profile-filter",
                                options=[{"label": "All profiles", "value": "ALL"}]
                                + [{"label": item, "value": item} for item in DATA.supplier_profiles],
                                value="ALL",
                                clearable=False,
                            ),
                        ],
                    ),
                ],
            ),
            html.Section(id="kpi-grid", className="kpi-grid"),
            html.Section(
                className="panel-grid",
                children=[
                    html.Article(
                        className="panel panel-wide",
                        children=[dcc.Graph(id="convergence-chart", config=GRAPH_CONFIG)],
                    ),
                    html.Article(
                        className="panel",
                        children=[dcc.Graph(id="confidence-chart", config=GRAPH_CONFIG)],
                    ),
                ],
            ),
            html.Section(
                className="panel-grid",
                children=[
                    html.Article(
                        className="panel panel-wide",
                        children=[
                            html.H2("Top Risky Programmes"),
                            dash_table.DataTable(
                                id="programme-risk-table",
                                page_size=10,
                                sort_action="native",
                                style_table={"overflowX": "auto"},
                                style_header={"backgroundColor": "#f5f1e9", "fontWeight": "700"},
                                style_cell={"padding": "10px", "textAlign": "left", "border": "none"},
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


app.layout = build_layout()


@app.callback(
    Output("kpi-grid", "children"),
    Output("convergence-chart", "figure"),
    Output("confidence-chart", "figure"),
    Output("programme-risk-table", "data"),
    Output("programme-risk-table", "columns"),
    Input("programme-filter", "value"),
    Input("profile-filter", "value"),
)
def update_dashboard(programme_id: str, supplier_profile: str):
    frame = filter_forecast(programme_id, supplier_profile)
    series = summarise_series(frame)

    average_ape = float(series["Latest_APE"].mean()) if not series.empty else 0.0
    average_fade = float(series["Forecast_Fade_Pct"].mean()) if not series.empty else 0.0
    measured_confidence = float(series["Measured_Confidence"].mean()) if not series.empty else 0.0
    risky_programmes = int(
        series.groupby("Programme_ID")["Risk_Score"].mean().sort_values(ascending=False).head(3).shape[0]
    )

    kpis = [
        make_kpi_card("Forecast series", f"{len(series):,}"),
        make_kpi_card("Average latest APE", percent(average_ape)),
        make_kpi_card("Average fade", percent(average_fade)),
        make_kpi_card("Measured confidence", percent(measured_confidence)),
        make_kpi_card("Programmes in scope", str(frame["Programme_ID"].nunique())),
        make_kpi_card("Top risky programmes shown", str(risky_programmes)),
    ]

    programme_risk = build_programme_risk_table(frame)

    return (
        kpis,
        build_convergence_chart(frame),
        build_confidence_chart(frame),
        programme_risk.to_dict("records"),
        [{"name": column.replace("_", " "), "id": column} for column in programme_risk.columns],
    )
