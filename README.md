# project-hack-supply-chain

## Project Overview

## Forecast Failure Logistic Model (with Supplier Features)

Train a logistic regression model for `Forecast_Failed_Flag` using forecast-time-safe fields plus supplier attributes from `data/supplier_attributes.csv`.

```powershell
python .\forecast_failure_model.py
python .\forecast_failure_model.py --train-frac 0.8 --threshold 0.45 --output scored_with_supplier_features.csv
python .\forecast_failure_model.py --feature-mode stability-only --output scored_stability_only.csv
```

## PCA Analysis

Run PCA on forecast features (using forecast-time-safe feature engineering) and inspect components linked to `Forecast_Failed_Flag`.

```powershell
python .\pca_analysis.py
python .\pca_analysis.py --n-components 5 --top-n-loadings 15 --output-prefix forecast_pca
```

The script writes these files:
- `*_variance.csv`
- `*_loadings.csv`
- `*_scores.csv`
- `*_top_loadings.csv`

## PCA -> Linear Regression Workflow (with Plotly Outputs)

This workflow:
1. runs PCA on forecast-time numeric features,
2. selects the most influential factor (largest loading on the PCA component most correlated with `Forecast_Failed_Flag`),
3. trains/tests a **linear regression** model using only that factor,
4. saves scored results and metrics, and
5. generates Plotly HTML visual reports.

```powershell
python .\pca_linear_workflow.py --input data\forecast_data.csv --output-dir outputs
python .\pca_linear_workflow.py --input data\forecast_data.csv --target-column Variance --output-dir outputs
python .\plotly_pca_linear_visuals.py --output-dir outputs
```

Use `--target-column` to choose the column to predict. If the target is binary (0/1), the workflow also reports ROC AUC and calibration-style plots.

Primary artifacts saved under `outputs/`:
- `workflow_pca_variance.csv`
- `workflow_pca_loadings.csv`
- `workflow_pca_scores.csv`
- `workflow_selected_factor.csv`
- `workflow_linear_metrics.csv`
- `workflow_linear_predictions.csv`
- `workflow_run_metadata.json`
- `workflow_plotly_report.html`
- `workflow_plotly_visuals.html`

## Plotly Upload Dashboard

Upload either:
- a scored CSV (must include `predicted_failure_probability`), or
- a raw forecast CSV (the app auto-scores it before plotting).

```powershell
python .\plotly_upload_dashboard.py
```

By default, if no file is uploaded and `forecast_failure_scored.csv` exists in the project root, it is loaded automatically.

## PCA Upload Front End (choose target column)

Use this Dash app when you want end users to upload their own CSV and select which column to predict.

```powershell
python .\plotly_pca_linear_upload_dashboard.py
```

**Features:**
- Upload CSV or use default `data/forecast_data.csv`
- Select any numeric column as target
- View multi-perspective dashboards:
  - **Executive View**: KPI summary, top 10 PCA factors, AI-generated recommendations
  - **Drill-Down View**: Model metrics, predictions vs factors, residuals
  - **Operations View**: Supplier/region heatmap, risk alerts
  - **PCA Technical**: Variance ratios, detailed loadings
- Download scored predictions

**AI-Generated Recommendations:**
If LLM utilities are configured (see below), the dashboard automatically generates contextual recommendations based on:
- Top 10 most influential factors from PCA
- Prediction likelihood and averagerisk
- Model metrics and performance

If LLM is unavailable, heuristic recommendations are generated.

**Business-friendly mode updates:**
- Main dashboard avoids technical model jargon for senior stakeholders.
- Technical diagnostics (including row counts and detailed metrics) are hidden behind **View detailed technical analysis** popup.
- Executive recommendations are shown as a concise table with a maximum of 3 improvements:
  - effort to implement,
  - expected improvement,
  - implementation steps.

**Executive landing updates (Phase 1):**
- `PCA n_components` is now fixed internally (not user-editable).
- Users choose a `Programme` (`Programme_ID`) as a business filter.
- Landing KPI cards now show:
  - forecast for next month,
  - probability the forecast will fail,
  - red/amber/green risk status.

Risk colour rules:
- Green: 0-25% failure probability
- Amber: 25-50% failure probability
- Red: 50-100% failure probability

**LLM Configuration (Optional):**
To enable AI-powered recommendations, configure these files:
- `C:\my_files\source_code\gen-ai\common\llm_utils.py` - LLM call utilities
- `C:\my_files\source_code\gen-ai\common\m2m_access_token.py` - Access token provider

Add your `.env` file with:
```
CLIENT_ID=<your_client_id>
CLIENT_SECRET=<your_client_secret>
API_KEY=<your_api_key>
```

You can provide credentials either via `.env` **or** via YAML fallback file:
- `C:\my_files\source_code\gen-ai\copilot_ignore\gaia_api_key.yaml`

Expected YAML keys:
```yaml
client_id: <CLIENT_ID>
client_secret: <CLIENT_SECRET>
```

`.env` values take priority if both are present.

For wireframe image previews inside the dashboard, place these files in `assets/`:
- `assets/wireframe_executive.png`
- `assets/wireframe_drilldown.png`
- `assets/wireframe_operations.png`
