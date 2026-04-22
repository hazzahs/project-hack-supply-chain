from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from model.ingestion import load_raw_tables
from model.settings import get_settings


SERIES_KEYS = ["Programme_ID", "Supplier_ID", "Commodity", "Forecast_Period"]


@dataclass
class DashboardRepository:
    tables: dict[str, pd.DataFrame]

    @classmethod
    def from_disk(cls) -> "DashboardRepository":
        settings = get_settings()
        return cls(tables=load_raw_tables(settings.data_dir))

    def prepare_forecast_frame(self) -> pd.DataFrame:
        forecast = self.tables["forecast_data"].copy()
        supplier = self.tables["supplier_attributes"].copy()
        programme = self.tables["programme_attributes"].copy()

        for column in [
            "Forecast_Spend",
            "Actual_Spend",
            "Absolute_Error",
            "Committed_Spend",
            "Programme_Scope_Churn_Index",
            "Programme_Change_Impact_Index",
            "Days_Before_Period",
            "Revision_Number",
            "Commitment_Ratio",
        ]:
            forecast[column] = pd.to_numeric(forecast[column], errors="coerce")

        forecast["Forecast_Version_Date"] = pd.to_datetime(
            forecast["Forecast_Version_Date"], errors="coerce"
        )
        forecast = forecast.merge(
            supplier[["Supplier_ID", "Contract_Type", "Supplier_Profile", "Region"]],
            on="Supplier_ID",
            how="left",
        ).merge(
            programme[["Programme_ID", "Programme_Phase", "Delivery_Risk"]],
            on="Programme_ID",
            how="left",
        )

        forecast = forecast.dropna(subset=["Forecast_Spend", "Actual_Spend"]).copy()
        forecast["Absolute_Percentage_Error"] = (
            (forecast["Forecast_Spend"] - forecast["Actual_Spend"]).abs()
            / forecast["Actual_Spend"].replace(0, pd.NA)
        )
        forecast["Signed_Error_Pct"] = (
            (forecast["Forecast_Spend"] - forecast["Actual_Spend"])
            / forecast["Actual_Spend"].replace(0, pd.NA)
        )
        forecast["Forecast_to_Actual_Ratio"] = (
            forecast["Forecast_Spend"] / forecast["Actual_Spend"].replace(0, pd.NA)
        )

        ordered = forecast.sort_values(SERIES_KEYS + ["Revision_Number", "Forecast_Version_Date"]).copy()
        ordered["Initial_Forecast_Spend"] = ordered.groupby(SERIES_KEYS)["Forecast_Spend"].transform("first")
        ordered["Latest_Forecast_Spend"] = ordered.groupby(SERIES_KEYS)["Forecast_Spend"].transform("last")
        ordered["Forecast_Fade_Pct"] = (
            (ordered["Latest_Forecast_Spend"] - ordered["Initial_Forecast_Spend"])
            / ordered["Initial_Forecast_Spend"].replace(0, pd.NA)
        )
        ordered["Low_Confidence_Flag"] = (ordered["Confidence_Band"] == "Low").astype(int)

        bins = [-1, 30, 60, 90, 120, 150, 10_000]
        labels = ["0-30d", "31-60d", "61-90d", "91-120d", "121-150d", "150d+"]
        ordered["Time_Bucket"] = pd.cut(
            ordered["Days_Before_Period"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )
        return ordered

    def overview(self) -> dict[str, object]:
        forecast = self.prepare_forecast_frame()
        supplier = self.tables["supplier_attributes"].copy()
        programme = self.tables["programme_attributes"].copy()

        spend_rows = forecast.dropna(subset=["Forecast_Spend", "Actual_Spend"]).copy()

        supplier_error = (
            spend_rows.groupby("Supplier_ID", as_index=False)
            .agg(
                avg_ape=("Absolute_Percentage_Error", "mean"),
                forecast_rows=("Supplier_ID", "size"),
                total_forecast=("Forecast_Spend", "sum"),
            )
            .merge(
                supplier[["Supplier_ID", "Contract_Type", "Supplier_Profile", "Region"]],
                on="Supplier_ID",
                how="left",
            )
            .sort_values(["avg_ape", "forecast_rows"], ascending=[False, False])
            .head(10)
        )

        programme_spend = (
            spend_rows.groupby("Programme_ID", as_index=False)
            .agg(
                total_forecast=("Forecast_Spend", "sum"),
                total_actual=("Actual_Spend", "sum"),
                avg_ape=("Absolute_Percentage_Error", "mean"),
            )
            .merge(programme, on="Programme_ID", how="left")
            .sort_values("total_forecast", ascending=False)
        )

        kpis = {
            "programmes": int(forecast["Programme_ID"].nunique()),
            "suppliers": int(forecast["Supplier_ID"].nunique()),
            "forecast_rows": int(len(forecast)),
            "total_forecast_spend": float(spend_rows["Forecast_Spend"].sum()),
            "total_actual_spend": float(spend_rows["Actual_Spend"].sum()),
            "average_ape": float(spend_rows["Absolute_Percentage_Error"].mean()),
        }

        return {
            "kpis": kpis,
            "top_supplier_risk": supplier_error.fillna("").to_dict(orient="records"),
            "programme_spend": programme_spend.fillna("").to_dict(orient="records"),
            "available_tables": sorted(self.tables),
        }
