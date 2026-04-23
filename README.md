# project-hack-supply-chain

## Structure

- `main.py` is the repo-root startup for the main dashboard app.
- `src/project_hack_supply_chain/` contains the real application and workflow code.
- `scripts/` contains compatibility wrappers for script-style execution.
- `data/`, `assets/`, and `prompts/` contain project resources.

## Setup

```bash
uv sync
```

## Run The App

```bash
uv run project-hack-supply-chain
```

or:

```bash
python main.py
```

## Run Offline Workflows

```bash
uv run forecast-failure-model
uv run pca-analysis
uv run pca-linear-workflow --input data/forecast_data.csv --output-dir outputs
uv run plotly-pca-linear-visuals --output-dir outputs
```

If you prefer script-style commands while cleaning up the repo, use the wrappers under `scripts/`.

## LLM Provider Configuration

The dashboard recommendation summary can call multiple providers directly via API key environment variables.

Set:

```bash
export LLM_PROVIDER=claude
```

Supported `LLM_PROVIDER` values:
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
