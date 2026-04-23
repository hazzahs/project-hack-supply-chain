import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

TARGET = "Forecast_Failed_Flag"


def build_visuals(output_dir: Path, html_name: str) -> Path:
    predictions_path = output_dir / "workflow_linear_predictions.csv"
    selected_path = output_dir / "workflow_selected_factor.csv"
    loadings_path = output_dir / "workflow_pca_loadings.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected-factor file: {selected_path}")
    if not loadings_path.exists():
        raise FileNotFoundError(f"Missing PCA loadings file: {loadings_path}")

    predictions = pd.read_csv(predictions_path)
    selected = pd.read_csv(selected_path)
    loadings = pd.read_csv(loadings_path, index_col=0)

    metadata_path = output_dir / "workflow_run_metadata.json"
    target_col = TARGET
    if "target_column" in selected.columns and pd.notna(selected.loc[0, "target_column"]):
        target_col = str(selected.loc[0, "target_column"])
    elif metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target_col = str(metadata.get("target_column", TARGET))

    component = selected.loc[0, "selected_component"]
    factor = selected.loc[0, "selected_factor"]

    top_loadings = (
        loadings[[component]]
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
        predictions,
        x=factor,
        y="predicted_failure_likelihood_linear",
        color=target_col if target_col in predictions.columns else None,
        title=f"Predicted Likelihood vs {factor}",
        hover_data=["event_id"] if "event_id" in predictions.columns else None,
    )

    fig3 = px.box(
        predictions,
        x=target_col if target_col in predictions.columns else None,
        y="predicted_failure_likelihood_linear",
        title=f"Predicted Likelihood by Actual {target_col}",
    )

    fig4 = px.histogram(
        predictions,
        x="prediction_residual",
        nbins=30,
        title="Residual Distribution (Actual - Predicted)",
    )

    html = "\n".join(
        [
            "<h2>Plotly Visuals: PCA + Linear Workflow Outputs</h2>",
            pio.to_html(fig1, include_plotlyjs="cdn", full_html=False),
            pio.to_html(fig2, include_plotlyjs=False, full_html=False),
            pio.to_html(fig3, include_plotlyjs=False, full_html=False),
            pio.to_html(fig4, include_plotlyjs=False, full_html=False),
        ]
    )

    report_path = output_dir / html_name
    report_path.write_text(
        "<html><head><meta charset='utf-8'><title>PCA Linear Visuals</title></head><body>"
        + html
        + "</body></html>",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Plotly visuals from saved PCA + linear workflow outputs.")
    parser.add_argument("--output-dir", default="outputs", help="Folder containing workflow CSV outputs")
    parser.add_argument("--html-name", default="workflow_plotly_visuals.html", help="Name of generated HTML report")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_path = build_visuals(output_dir=output_dir, html_name=args.html_name)
    print(f"Plotly report generated: {report_path.resolve()}")


if __name__ == "__main__":
    main()

