# project-hack-supply-chain

Forecast risk analytics project with:
- logistic and PCA-based modeling,
- workflow artifact generation,
- interactive Plotly/Dash dashboards,
- a unified terminal entrypoint via `main.py`.

## Contents
- [1) Prerequisites](#1-prerequisites)
- [2) Install on Your Device (Windows / macOS / Linux)](#2-install-on-your-device-windows--macos--linux)
- [3) Terminal Help and Command Discovery](#3-terminal-help-and-command-discovery)
- [4) Quickstart Commands](#4-quickstart-commands)
- [5) Optional AI Recommendations](#5-optional-ai-recommendations)
- [6) Output Artifacts](#6-output-artifacts)
- [7) Contributing and Thanks](#7-contributing-and-thanks)

## 1) Prerequisites

- Python `>=3.14` (from `pyproject.toml`)
- `pip` (usually included with Python)
- Git (to clone this repository)

## 2) Install on Your Device (Windows / macOS / Linux)

### Step A: Clone

```bash
git clone <your-repo-url>
cd project_hack_27
```

### Step B: Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step C: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -e .
```

If you prefer not to install as an editable package, you can also run:

```bash
pip install pandas numpy scikit-learn plotly dash
```

## 3) Terminal Help and Command Discovery

All project workflows are routed through `main.py`.

Show top-level help:

```bash
python main.py --help
```

Show help for a specific command:

```bash
python main.py forecast-model --help
python main.py pca-analysis --help
python main.py pca-linear-workflow --help
python main.py pca-visuals --help
```

## 4) Quickstart Commands

### Forecast failure logistic model

```bash
python main.py forecast-model
python main.py forecast-model --train-frac 0.8 --threshold 0.45 --output scored_with_supplier_features.csv
python main.py forecast-model --feature-mode stability-only --output scored_stability_only.csv
```

### PCA analysis

```bash
python main.py pca-analysis
python main.py pca-analysis --n-components 5 --top-n-loadings 15 --output-prefix forecast_pca
```

### PCA -> linear workflow + visuals

```bash
python main.py pca-linear-workflow --input data/forecast_data.csv --output-dir outputs
python main.py pca-linear-workflow --input data/forecast_data.csv --target-column Variance --output-dir outputs
python main.py pca-visuals --output-dir outputs
```

### Dashboards

Scored/raw upload dashboard:

```bash
python main.py dashboard-upload --debug
```

PCA upload dashboard:

```bash
python main.py dashboard-pca --debug
```

## 5) Optional AI Recommendations

The PCA dashboard can generate recommendation text using an optional AI provider integration.

- If an AI provider and credentials are configured in your environment, recommendations are generated from model context.
- If no AI provider is configured, the dashboard falls back to heuristic recommendations.

Recommended setup approach:
- Store credentials in environment variables (or a secure secret manager).
- Keep provider-specific helper modules outside source control when possible.
- Never commit secrets to this repository.

## 6) Output Artifacts

Common generated files include:
- `workflow_pca_variance.csv`
- `workflow_pca_loadings.csv`
- `workflow_pca_scores.csv`
- `workflow_selected_factor.csv`
- `workflow_linear_metrics.csv`
- `workflow_linear_predictions.csv`
- `workflow_run_metadata.json`
- `workflow_plotly_report.html`
- `workflow_plotly_visuals.html`

## 7) Contributing and Thanks

Contributions are welcome via issues and pull requests.

Thank you to everyone who has contributed ideas, code, testing, and feedback to improve this project.

