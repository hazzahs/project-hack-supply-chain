from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.lstrip("\ufeff") for column in frame.columns]
    return frame


def load_raw_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "commitments_summary": read_csv(data_dir / "commitments_summary.csv"),
        "forecast_data": read_csv(data_dir / "forecast_data.csv"),
        "forecast_data_stripped": read_csv(data_dir / "forecast_data_stripped.csv"),
        "portfolio_funding_envelope": read_csv(data_dir / "portfolio_funding_envelope.csv"),
        "programme_attributes": read_csv(data_dir / "programme_attributes.csv"),
        "programme_budget": read_csv(data_dir / "programme_budget.csv"),
        "programme_month_change": read_csv(data_dir / "programme_month_change.csv"),
        "supplier_attributes": read_csv(data_dir / "supplier_attributes.csv"),
    }
