# project-hack-supply-chain

Forecast-risk analytics project with PCA-driven feature selection, logistic and linear modelling workflows, and a multi-persona Plotly/Dash dashboard.

## Structure

Key files and folders in the current repository layout:

- `main.py` - unified command-line entrypoint for workflows and dashboards.
- `forecast_failure_model.py` - logistic regression workflow for forecast failure classification.
- `pca_analysis.py` - PCA analysis workflow and export utilities.
- `pca_linear_workflow.py` - PCA-to-linear workflow that selects one influential factor and scores risk.
- `plotly_pca_linear_upload_dashboard.py` - main multi-persona dashboard.
- `plotly_upload_dashboard.py` - scored/raw upload dashboard.
- `plotly_pca_linear_visuals.py` - standalone Plotly report builder for saved workflow outputs.
- `data/` - source CSV data files.
- `assets/` - dashboard styling and optional static assets.
- `outputs/` and `outputs_variance/` - generated workflow artefacts.
- `prompts/` - system prompt files used for AI-generated summaries and recommendations.

Persona prompt files currently stored under `prompts/`:

- `system_prompt_persona_programme_director.txt`
- `system_prompt_persona_commercial_manager.txt`
- `system_prompt_persona_cfo.txt`
- `system_prompt_persona_project_controls.txt`
- `system_prompt_Forecast_Failed_Flag.txt` (recommendation prompt generated for target-specific recommendations)

## Setup

### Docker

To run the application directly please ensure you have the following installed:

- Docker/Podman
- Docker/Podman compose plugin

After cloning the project to your local machine run

```bash
docker compose up --build

# OR

podman compose up --build
```

You will then be able to view the dashboard at `http:127.0.0.1:8050`

### Locally

Using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
uv run project-hack-supply-chain
```

Or with `pip` in an active virtual environment:

```bash
python -m pip install --upgrade pip
pip install -e .
```

## Terminal Help

Show all available commands:

```bash
python main.py --help
```

Show help for a specific workflow:

```bash
python main.py forecast-model --help
python main.py pca-analysis --help
python main.py pca-linear-workflow --help
python main.py pca-visuals --help
```

## Run the Dash Apps

PCA + linear upload dashboard:

```bash
python main.py dashboard-pca --debug
```

Scored/raw upload dashboard:

```bash
python main.py dashboard-upload --debug
```

## Run Offline Workflows

```bash
python main.py forecast-model
python main.py pca-analysis
python main.py pca-linear-workflow --input data/forecast_data.csv --output-dir outputs
python main.py pca-visuals --output-dir outputs
```

## Current Dashboard Highlights

- **Programme Director** view shows KPI cards, top drivers, and recommendations.
- **Commercial Manager** view includes contract/profile/commodity comparisons, a **region vs contract type heatmap**, and a supplier watchlist showing the **top 5 highest-risk** and **top 5 lowest-risk** suppliers.
- **CFO** view includes spend exposure visuals and a **spend versus failed proposal probability** chart with a best-fit trendline and narrative annotation.
- **Project Controls Lead** view focuses on trend diagnostics, stability, lead-time, and key risk-driver explanations.

Note: the older supplier-region risk-pair chart is no longer part of the active dashboard because it did not add enough additional insight.

## Optional AI Summaries

The dashboard can generate persona-specific narrative summaries and recommendation tables when the AI utilities are available.

- Persona summaries load their instructions from the external files in `prompts/`.
- Recommendation prompts are also stored under `prompts/`.
- If AI utilities are unavailable, the dashboard falls back to heuristic summaries and recommendations.
