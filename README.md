# project-hack-supply-chain

## Project Overview

This project is a Python-first Plotly Dash app for a supply-chain analytics dashboard.

The repository is structured around three responsibilities:

- `data/`: source CSV files used by the dashboard
- `model/`: ingestion and processing code that reads and shapes the data
- `src/`: the Dash application itself

The app is intentionally simple to run for non-technical judges: install Python packages, run one Python command, and open the local dashboard in a browser.

## What The App Does

- loads raw CSV data from `data/`
- uses Python code in `model/` to ingest and shape the data
- runs a local Plotly Dash web app
- shows headline KPIs for forecast prediction readiness
- provides simple filters for programme and supplier profile
- displays:
  - a convergence curve showing how forecasts fade or settle over time
  - a confidence-vs-time chart showing whether trust improves closer to delivery
  - a risk table highlighting the riskiest programmes

## Run Locally

Install dependencies into your virtual environment, then start the dashboard:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open:

```text
http://127.0.0.1:8050
```

## Why Dash

Dash is still a web app, but it is a Python-driven one. That means:

- Python loads and processes the data
- Python defines the page layout and charts
- the browser displays the interactive dashboard
- there is no separate JavaScript build step to run the dashboard locally

This keeps the development and demo workflow much simpler than a split backend/frontend architecture.

## Main Files

- `main.py`
  Starts the Dash development server.
- `model/ingestion.py`
  Loads the CSV files from `data/`.
- `model/dashboard_service.py`
  Holds shared Python-side data access logic.
- `src/app.py`
  Defines the Dash layout, charts, tables, and interactive callbacks.
- `src/assets/theme.css`
  Styles the dashboard.

## Current Scope

The current dashboard is still an initial prediction-readiness skeleton. It reads directly from the CSVs, derives forecast-history features in Python, and visualises:

- convergence behaviour
- measured confidence over time buckets
- programme-level forecast risk

It does not yet include a trained predictive model, scenario engine, or persisted processed datasets.
