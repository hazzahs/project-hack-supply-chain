# project-hack-supply-chain

Forecast-risk analytics project with offline modelling workflows and a multi-persona Dash dashboard.

## Quick Start

### Local

Using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
uv run project-hack-supply-chain
```

Open the app at `http://127.0.0.1:8050`.

### Docker

```bash
docker compose up --build
```

Open the app at `http://127.0.0.1:8050`.

## Project Layout

- `main.py` - unified local CLI entrypoint.
- `src/project_hack_supply_chain/` - real application and workflow code.
- `assets/` - dashboard CSS and static assets.
- `prompts/` - persona and recommendation prompt files for AI summaries.
- `data/` - source CSV inputs.
- `outputs/` and `outputs_variance/` - generated workflow artefacts when created locally.
- `scripts/` - legacy wrapper scripts kept for compatibility with older commands.

Main package modules:

- `src/project_hack_supply_chain/main.py` - app/CLI entrypoint used by `uv run project-hack-supply-chain`
- `src/project_hack_supply_chain/dashboard.py` - main multi-persona dashboard
- `src/project_hack_supply_chain/upload_dashboard.py` - scored/raw upload dashboard
- `src/project_hack_supply_chain/forecast_failure.py` - forecast-failure classification workflow
- `src/project_hack_supply_chain/pca.py` - PCA analysis workflow
- `src/project_hack_supply_chain/workflow.py` - PCA-to-linear workflow
- `src/project_hack_supply_chain/visuals.py` - standalone Plotly report builder
- `src/project_hack_supply_chain/llm.py` - LLM provider integration for AI summaries

## Running The App

Primary app startup:

```bash
uv run project-hack-supply-chain
```

Equivalent local CLI form:

```bash
python main.py
```

If you want debug mode:

```bash
python main.py dashboard-pca --debug
```

## CLI Help

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

## Offline Workflows

Run the main workflows directly through the unified CLI:

```bash
python main.py forecast-model
python main.py pca-analysis
python main.py pca-linear-workflow --input data/forecast_data.csv --output-dir outputs
python main.py pca-visuals --output-dir outputs
```

## Dashboard Views

- **Programme Director** - KPI cards, top drivers, recommendations, and executive summary
- **Commercial Manager** - contract/profile comparisons, heatmap, and supplier watchlist
- **CFO** - spend exposure, portfolio risk, and spend-vs-failure visuals
- **Project Controls Lead** - trend diagnostics, stability, lead-time, and risk-driver explanations

## AI Summaries

The dashboard can generate persona-specific narrative summaries and recommendation tables.

- Persona prompts are stored in `prompts/`
- Recommendation prompts are also stored in `prompts/`
- If no usable LLM configuration is available, the dashboard falls back to heuristic summaries and recommendations and logs a warning

### Supported Providers

Set one provider:

```bash
export LLM_PROVIDER=claude
```

Supported values:

- `claude`
- `gemini`
- `openai`
- `openrouter`

API key resolution:

- `claude`: `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`, then `API_KEY`
- `gemini`: `GEMINI_API_KEY` or `GOOGLE_API_KEY`, then `API_KEY`
- `openai`: `OPENAI_API_KEY`, then `API_KEY`
- `openrouter`: `OPENROUTER_API_KEY`, then `API_KEY`

Optional model overrides:

- `LLM_MODEL`
- `CLAUDE_MODEL`
- `GEMINI_MODEL`
- `OPENAI_MODEL`
- `OPENROUTER_MODEL`

Example:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key_here
uv run project-hack-supply-chain
```

## Environment Loading

If you use `direnv`, the repo `.envrc` can load `.envrc.local` automatically when present.

If you are not using `direnv`, you can still load it manually:

```bash
source .envrc
```

## Notes

- `main.py` is the intended local startup and CLI entrypoint
- `scripts/` contains legacy wrappers for older script names
- Docker uses `uv` and installs the project into the container environment rather than relying on a pre-baked Python image layout
