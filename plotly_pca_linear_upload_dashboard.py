import base64
import contextlib
import io
import json
import os
import sys
import threading
from pathlib import Path

import dash
from dash import Input, Output, State, dcc, html
from dash.development.base_component import Component
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from forecast_failure_model import extract_days_from_terms
from pca_analysis import get_numeric_features
from pca_linear_workflow import (
    DEFAULT_TARGET,
    ensure_event_id,
    find_influential_factor,
    parse_n_components,
    train_test_linear,
)

GEN_AI_ROOT = Path(r"C:\my_files\source_code\gen-ai")
sys.path.insert(0, str(GEN_AI_ROOT))
original_cwd = Path.cwd()
_GENAI_CWD_LOCK = threading.Lock()
try:
    os.chdir(GEN_AI_ROOT)
    from common.llm_utils import make_LLM_call  # type: ignore[import-not-found]
    from common.m2m_access_token import get_access_token  # type: ignore[import-not-found]
    LLM_AVAILABLE = True
except Exception as exc:
    LLM_AVAILABLE = False
    print(f"Warning: LLM utilities not available ({exc}). Recommendations will be generated from heuristics.")
finally:
    try:
        os.chdir(original_cwd)
    except Exception:
        pass


@contextlib.contextmanager
def with_genai_cwd():
    """Temporarily switch cwd so external gen-ai utilities can resolve relative paths."""
    with _GENAI_CWD_LOCK:
        current = Path.cwd()
        try:
            os.chdir(GEN_AI_ROOT)
            yield
        finally:
            os.chdir(current)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "forecast_data.csv"
DEFAULT_SUPPLIER_PATH = BASE_DIR / "data" / "supplier_attributes.csv"
PROMPTS_DIR = BASE_DIR / "prompts"
GAIA_CRED_PATH = Path(r"C:\my_files\source_code\gen-ai\copilot_ignore\gaia_api_key.yaml")
PROMPTS_DIR.mkdir(exist_ok=True)
FIXED_N_COMPONENTS = "0.95"
FILTER_DEFAULT_MIN_RISK = 0.0
FILTER_DEFAULT_SUPPLIER_TOPN = 25


def decode_uploaded_csv(contents: str) -> pd.DataFrame:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))


def enrich_with_supplier_attributes(
    df: pd.DataFrame,
    supplier_path: Path = DEFAULT_SUPPLIER_PATH,
) -> pd.DataFrame:
    enriched = df.copy()
    if "Supplier_ID" not in enriched.columns or supplier_path is None or not supplier_path.exists():
        return enriched

    supplier_df = pd.read_csv(supplier_path)
    if "Supplier_ID" not in supplier_df.columns:
        return enriched

    enriched["Supplier_ID"] = enriched["Supplier_ID"].astype(str).str.strip()
    supplier_df["Supplier_ID"] = supplier_df["Supplier_ID"].astype(str).str.strip()
    supplier_df = supplier_df.drop_duplicates(subset=["Supplier_ID"], keep="first")

    if "Payment_Terms" in supplier_df.columns and "Payment_Terms_Days" not in supplier_df.columns:
        supplier_df["Payment_Terms_Days"] = supplier_df["Payment_Terms"].apply(extract_days_from_terms)
    if "Strategic_Flag" in supplier_df.columns and "Strategic_Flag_Num" not in supplier_df.columns:
        supplier_df["Strategic_Flag_Num"] = supplier_df["Strategic_Flag"].map({"Yes": 1, "No": 0})
    if "New_Supplier_Flag" in supplier_df.columns and "New_Supplier_Flag_Num" not in supplier_df.columns:
        supplier_df["New_Supplier_Flag_Num"] = supplier_df["New_Supplier_Flag"].map({"Yes": 1, "No": 0})

    merge_columns = [col for col in supplier_df.columns if col == "Supplier_ID" or col not in enriched.columns]
    if merge_columns == ["Supplier_ID"]:
        return enriched

    return enriched.merge(supplier_df[merge_columns], on="Supplier_ID", how="left")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "Forecast_Version_Date" in work.columns:
        version_date = pd.Series(pd.to_datetime(work["Forecast_Version_Date"], errors="coerce"), index=work.index)
        work["Forecast_Version_Date"] = version_date
    if "Forecast_Period_End_Date" in work.columns:
        work["Forecast_Period_End_Date"] = pd.to_datetime(work["Forecast_Period_End_Date"], errors="coerce")

    if "Forecast_Version_Date" in work.columns:
        version_date = pd.Series(pd.to_datetime(work["Forecast_Version_Date"], errors="coerce"), index=work.index)
        work["forecast_version_month"] = version_date.apply(lambda x: x.month if pd.notna(x) else np.nan)
        work["forecast_version_quarter"] = version_date.apply(lambda x: ((x.month - 1) // 3) + 1 if pd.notna(x) else np.nan)

    if "Forecast_Period" in work.columns:
        period_series = pd.Series(pd.to_datetime(
            work["Forecast_Period"].astype(str) + "-01", errors="coerce"
        ), index=work.index)
        work["forecast_period_month"] = period_series.apply(lambda x: x.month if pd.notna(x) else np.nan)

    if "Previous_Forecast_Spend" in work.columns and "Forecast_Change" in work.columns:
        work["forecast_change_pct"] = np.where(
            work["Previous_Forecast_Spend"].fillna(0) != 0,
            work["Forecast_Change"] / work["Previous_Forecast_Spend"],
            0.0,
        )

    if "Committed_Spend" in work.columns and "Forecast_Spend" in work.columns:
        work["forecast_to_committed_ratio"] = np.where(
            work["Committed_Spend"].fillna(0) != 0,
            work["Forecast_Spend"] / work["Committed_Spend"],
            np.nan,
        )

    work = enrich_with_supplier_attributes(work)
    work = ensure_event_id(work)
    return work


def target_options_from_df(df: pd.DataFrame) -> list[dict]:
    options = []
    for col in df.columns:
        numeric = pd.Series(pd.to_numeric(df[col], errors="coerce"), index=df.index)
        if numeric.notna().sum() > 0:
            options.append({"label": col, "value": col})
    return options


def programme_options_from_df(df: pd.DataFrame) -> list[dict]:
    if "Programme_ID" not in df.columns:
        return []
    programmes = sorted(df["Programme_ID"].dropna().astype(str).unique().tolist())
    return [{"label": p, "value": p} for p in programmes]


def category_options_from_df(df: pd.DataFrame, column: str) -> list[dict]:
    if column not in df.columns:
        return []
    values = sorted(df[column].dropna().astype(str).unique().tolist())
    return [{"label": value, "value": value} for value in values]


def get_driver_description(driver_name: str) -> tuple[str, str]:
    """Return (reason, impact_explanation) for a driver."""
    name_lower = driver_name.lower()
    
    # Comprehensive descriptions for all features used in the model
    descriptions = {
        # Spend-related drivers
        "forecast_spend": (
            "Current forecast spend amount",
            "Higher or more volatile forecast amounts increase the likelihood of misalignment with actuals."
        ),
        "previous_forecast_spend": (
            "Historical forecast from prior period",
            "Large deviations from historical forecasts indicate planning uncertainty and higher failure risk."
        ),
        "committed_spend": (
            "Contractually committed spend",
            "Gaps between forecast and committed spend reflect poor planning alignment."
        ),
        "forecast_change": (
            "Magnitude of forecast revision",
            "Larger changes between forecast versions signal instability and lower confidence."
        ),
        "forecast_change_pct": (
            "Percentage change in forecast",
            "High percentage volatility relative to prior forecast indicates weak planning discipline."
        ),
        "forecast_to_committed_ratio": (
            "Forecast-to-committed alignment ratio",
            "Poor ratios indicate misalignment between operational forecasts and business commitments."
        ),
        
        # Supplier-related drivers
        "supplier": (
            "Supplier capacity and reliability",
            "Suppliers with lower reliability increase the chance of forecast misses due to delays or quality issues."
        ),
        "supplier_id": (
            "Specific supplier identity",
            "Certain suppliers have demonstrated patterns of delivery and quality issues affecting forecast accuracy."
        ),
        "supplier_profile": (
            "Supplier operational profile classification",
            "Different supplier profiles have varying risk levels; some categories are inherently less predictable."
        ),
        "strategic_flag_num": (
            "Strategic supplier designation",
            "Strategic suppliers may have different service level volatility compared to transactional suppliers."
        ),
        "new_supplier_flag_num": (
            "Supplier newness indicator",
            "New suppliers lack historical data for accurate forecasting and have unpredictable performance patterns."
        ),
        "payment_terms": (
            "Payment terms and conditions",
            "Different payment structures correlate with different fulfillment patterns and forecast volatility."
        ),
        "payment_terms_days": (
            "Payment terms in days",
            "Longer payment terms correlate with working capital flexibility, affecting forecast volatility."
        ),
        
        # Forecast stability and revision drivers
        "forecast_stability_score": (
            "Forecast stability metric",
            "Lower stability scores directly predict higher likelihood of forecast failure."
        ),
        "revision_number": (
            "Cumulative forecast revisions",
            "Multiple revisions indicate uncertainty and instability in planning."
        ),
        
        # Timing and seasonality drivers
        "days_before_period": (
            "Forecast lead time",
            "Forecasts created too far in advance are more prone to disruption and change."
        ),
        "forecast_version_month": (
            "Month of forecast creation",
            "Certain months have seasonal patterns affecting forecast accuracy."
        ),
        "forecast_version_quarter": (
            "Quarter of forecast creation",
            "Fiscal period impacts forecast accuracy due to budget and planning cycles."
        ),
        "forecast_period_month": (
            "Month being forecasted",
            "Target period seasonality affects forecast volatility and accuracy."
        ),
        
        # Commitment and logistics drivers
        "commitment_ratio": (
            "Commitment coverage percentage",
            "Low commitment ratios indicate poor order visibility and higher forecast risk."
        ),
        "po_count": (
            "Number of purchase orders",
            "More purchase orders indicate operational complexity and higher coordination risk."
        ),
        
        # Programme change drivers
        "programme": (
            "Programme scope and complexity",
            "Larger or more dynamic programmes experience greater forecast volatility."
        ),
        "programme_id": (
            "Specific programme identity",
            "Different programmes have different operational maturity levels affecting forecast accuracy."
        ),
        "programme_change_count": (
            "Count of scope changes",
            "Multiple changes to programme scope directly destabilize forecasts."
        ),
        "programme_scope_churn_index": (
            "Programme scope volatility index",
            "High scope churn creates cascading changes affecting all downstream forecasts."
        ),
        "programme_change_impact_index": (
            "Impact index of programme changes",
            "Large impact changes propagate through the supply chain, destabilizing forecasts."
        ),
        
        # Commodity and contract drivers
        "commodity": (
            "Type of material or service procured",
            "Complex or scarce commodities are harder to forecast accurately."
        ),
        "contract": (
            "Contract structure and terms",
            "Flexible contract terms can lead to unexpected changes in spend forecasts."
        ),
        "contract_type": (
            "Classification of contract type",
            "Different contract structures have different change management procedures and volatility."
        ),
        
        # Regional and logistics drivers
        "region": (
            "Geographic location and logistics complexity",
            "Certain regions have more volatile supply chains, leading to higher forecast variance."
        ),
        "otif_pct": (
            "On-Time In-Full delivery percentage",
            "Lower OTIF scores indicate supplier reliability issues that increase forecast risk."
        ),
        "avg_lead_time_days": (
            "Average supply chain lead time",
            "Longer lead times increase forecast window uncertainty and risk of disruption."
        ),
        
        # Quality drivers
        "quality_incidents_ytd": (
            "Year-to-date quality issues",
            "More quality incidents indicate supplier capability concerns affecting forecast reliability."
        ),
        
        # Direction and confidence drivers
        "forecast_change_direction": (
            "Direction of forecast revision",
            "Consistent directional changes indicate systematic forecasting bias."
        ),
        "confidence_band": (
            "Forecast confidence classification",
            "Lower confidence bands correlate with higher forecast failure rates."
        ),
    }
    
    # First try exact match
    if name_lower in descriptions:
        return descriptions[name_lower]
    
    # Then try partial match
    for key, (reason, impact) in descriptions.items():
        if key in name_lower:
            return reason, impact
    
    # Default fallback
    return f"{driver_name} characteristics", "This driver influences forecast volatility and failure risk."


def calculate_risk_alerts(predictions_df: pd.DataFrame, predictions_col: str = "predicted_failure_likelihood_linear") -> dict:
    """
    Calculate key risk metrics: Supplier Delay Risk, Cost Volatility, and Demand Spike.
    Returns a dictionary with risk scores and status.
    """
    alerts = {
        "supplier_delay_risk": {"score": 0.0, "status": "Low", "indicators": []},
        "cost_volatility": {"score": 0.0, "status": "Low", "indicators": []},
        "demand_spike": {"score": 0.0, "status": "Low", "indicators": []},
    }
    
    work = predictions_df.copy()
    
    # Supplier Delay Risk: Based on OTIF, Lead Time, and Payment Terms
    supplier_indicators = []
    if "OTIF_Pct" in work.columns:
        otif_mean = pd.to_numeric(work["OTIF_Pct"], errors="coerce").mean()
        if pd.notna(otif_mean) and otif_mean < 85:  # OTIF below 85% is a red flag
            supplier_indicators.append(f"Low OTIF: {otif_mean:.1f}%")
    
    if "Avg_Lead_Time_Days" in work.columns:
        lead_time_mean = pd.to_numeric(work["Avg_Lead_Time_Days"], errors="coerce").mean()
        if pd.notna(lead_time_mean) and lead_time_mean > 30:  # Lead time > 30 days
            supplier_indicators.append(f"High Lead Time: {lead_time_mean:.0f} days")
    
    if "Quality_Incidents_YTD" in work.columns:
        quality_mean = pd.to_numeric(work["Quality_Incidents_YTD"], errors="coerce").mean()
        if pd.notna(quality_mean) and quality_mean > 2:  # More than 2 incidents
            supplier_indicators.append(f"Quality Issues: {quality_mean:.1f} incidents/supplier")
    
    supplier_risk_score = len(supplier_indicators) / 3.0  # Normalize to 0-1
    alerts["supplier_delay_risk"]["score"] = supplier_risk_score
    alerts["supplier_delay_risk"]["indicators"] = supplier_indicators
    alerts["supplier_delay_risk"]["status"] = (
        "Red" if supplier_risk_score >= 0.66 else ("Amber" if supplier_risk_score >= 0.33 else "Green")
    )
    
    # Cost Volatility: Based on Forecast Change, Stability Score, and forecast_change_pct
    cost_indicators = []
    if "Forecast_Stability_Score" in work.columns:
        stability_mean = pd.to_numeric(work["Forecast_Stability_Score"], errors="coerce").mean()
        if pd.notna(stability_mean) and stability_mean < 0.5:  # Low stability
            cost_indicators.append(f"Low Stability Score: {stability_mean:.2f}")
    
    if "forecast_change_pct" in work.columns:
        change_pct = float(np.nanmean(np.abs(pd.to_numeric(work["forecast_change_pct"], errors="coerce"))))
        if pd.notna(change_pct) and change_pct > 0.3:  # >30% average change
            cost_indicators.append(f"High Change Rate: {change_pct:.1%}")
    
    if "Forecast_Change_Direction" in work.columns:
        ups = (work["Forecast_Change_Direction"] == "UP").sum()
        downs = (work["Forecast_Change_Direction"] == "DOWN").sum()
        total = ups + downs
        if total > 0:
            directional_volatility = max(ups, downs) / total if total > 0 else 0
            if directional_volatility < 0.6:  # Less than 60% consistent direction
                cost_indicators.append(f"Directional Inconsistency: {directional_volatility:.0%}")
    
    cost_risk_score = len(cost_indicators) / 3.0
    alerts["cost_volatility"]["score"] = cost_risk_score
    alerts["cost_volatility"]["indicators"] = cost_indicators
    alerts["cost_volatility"]["status"] = (
        "Red" if cost_risk_score >= 0.66 else ("Amber" if cost_risk_score >= 0.33 else "Green")
    )
    
    # Demand Spike Risk: Based on Programme Changes and Scope Churn
    demand_indicators = []
    if "Programme_Change_Count" in work.columns:
        change_count = pd.to_numeric(work["Programme_Change_Count"], errors="coerce").mean()
        if pd.notna(change_count) and change_count > 5:  # >5 changes on average
            demand_indicators.append(f"High Change Count: {change_count:.1f} changes/record")
    
    if "Programme_Scope_Churn_Index" in work.columns:
        churn = pd.to_numeric(work["Programme_Scope_Churn_Index"], errors="coerce").mean()
        if pd.notna(churn) and churn > 1.5:  # High churn index
            demand_indicators.append(f"Scope Churn: {churn:.2f}")
    
    if "Programme_Change_Impact_Index" in work.columns:
        impact = pd.to_numeric(work["Programme_Change_Impact_Index"], errors="coerce").mean()
        if pd.notna(impact) and impact > 1.5:  # High impact index
            demand_indicators.append(f"Change Impact: {impact:.2f}")
    
    demand_risk_score = len(demand_indicators) / 3.0
    alerts["demand_spike"]["score"] = demand_risk_score
    alerts["demand_spike"]["indicators"] = demand_indicators
    alerts["demand_spike"]["status"] = (
        "Red" if demand_risk_score >= 0.66 else ("Amber" if demand_risk_score >= 0.33 else "Green")
    )
    
    return alerts


def build_programme_director_summary(
    programme_view: pd.DataFrame,
    target_col: str,
    metrics: dict,
    next_month_forecast: float,
    next_month_failure_prob: float,
    risk_label: str,
) -> str:
    if programme_view.empty:
        return "No records are available for the current filter combination, so no reliable forecast narrative can be generated yet."

    def _mean_numeric(column: str) -> float:
        if column not in programme_view.columns:
            return float("nan")
        series = pd.Series(pd.to_numeric(programme_view[column], errors="coerce"), index=programme_view.index)
        return float(series.mean()) if series.notna().any() else float("nan")

    def _sum_numeric(column: str) -> float:
        if column not in programme_view.columns:
            return float("nan")
        series = pd.Series(pd.to_numeric(programme_view[column], errors="coerce"), index=programme_view.index)
        return float(series.sum()) if series.notna().any() else float("nan")

    actual_rate = _mean_numeric(target_col)
    predicted_rate = _mean_numeric("predicted_failure_likelihood_linear")
    commitment_ratio = _mean_numeric("Commitment_Ratio")
    stability_score = _mean_numeric("Forecast_Stability_Score")
    forecast_total = _sum_numeric("Forecast_Spend")
    actual_total = _sum_numeric("Actual_Spend")
    committed_total = _sum_numeric("Committed_Spend")
    variance_mean = _mean_numeric("Variance")

    high_risk_share = float("nan")
    if "predicted_failure_likelihood_linear" in programme_view.columns:
        risk_series = pd.Series(
            pd.to_numeric(programme_view["predicted_failure_likelihood_linear"], errors="coerce"),
            index=programme_view.index,
        ).dropna()
        if not risk_series.empty:
            high_risk_share = float((risk_series >= 0.5).mean())

    budget_pressure = 0
    budget_flags: list[str] = []
    if pd.notna(forecast_total) and pd.notna(committed_total) and forecast_total > 0:
        forecast_vs_committed_gap = (forecast_total - committed_total) / forecast_total
        if forecast_vs_committed_gap > 0.20:
            budget_pressure += 1
            budget_flags.append("forecasted spend is running materially ahead of committed cover")
    if pd.notna(forecast_total) and pd.notna(actual_total) and forecast_total > 0:
        forecast_vs_actual_gap = (forecast_total - actual_total) / forecast_total
        if forecast_vs_actual_gap > 0.15:
            budget_pressure += 1
            budget_flags.append("actual delivery is lagging the current forecast profile")
    if pd.notna(commitment_ratio) and commitment_ratio < 0.60:
        budget_pressure += 1
        budget_flags.append("commitment coverage remains light")
    if pd.notna(stability_score) and stability_score < 0.45:
        budget_pressure += 1
        budget_flags.append("planning stability is weak")
    if pd.notna(variance_mean) and variance_mean < 0:
        budget_pressure += 1
        budget_flags.append("recent spend variance is trending unfavourably")

    alerts = calculate_risk_alerts(programme_view, "predicted_failure_likelihood_linear")
    risk_labels = {
        "supplier_delay_risk": "supplier delivery risk",
        "cost_volatility": "cost volatility",
        "demand_spike": "demand pressure",
    }
    active_risks = [
        (risk_labels[key], value)
        for key, value in alerts.items()
        if value.get("status") in {"Red", "Amber"}
    ]
    active_risks.sort(key=lambda item: item[1].get("score", 0.0), reverse=True)

    overview_state = "broadly controlled"
    if (
        (pd.notna(predicted_rate) and predicted_rate >= 0.55)
        or (pd.notna(high_risk_share) and high_risk_share >= 0.45)
        or budget_pressure >= 3
    ):
        overview_state = "under clear pressure"
    elif (
        (pd.notna(predicted_rate) and predicted_rate >= 0.35)
        or (pd.notna(high_risk_share) and high_risk_share >= 0.25)
        or budget_pressure >= 2
    ):
        overview_state = "manageable but exposed"

    budget_confidence = "high"
    completion_phrase = "the programme is likely to remain within budget if current controls hold"
    if budget_pressure >= 3 or (pd.notna(predicted_rate) and predicted_rate >= 0.55):
        budget_confidence = "low"
        completion_phrase = "the programme is at material risk of overshooting budget without intervention"
    elif budget_pressure >= 2 or (pd.notna(predicted_rate) and predicted_rate >= 0.35):
        budget_confidence = "moderate"
        completion_phrase = "staying within budget is still achievable, but only with tighter control over upcoming periods"

    trend_text = "near-term risk looks broadly stable"
    trend_source = None
    time_candidates = [
        ("Forecast_Period", lambda s: pd.to_datetime(s.astype(str) + "-01", errors="coerce")),
        ("Forecast_Period_End_Date", lambda s: pd.to_datetime(s, errors="coerce")),
        ("Forecast_Version_Date", lambda s: pd.to_datetime(s, errors="coerce")),
    ]
    for candidate, converter in time_candidates:
        if candidate not in programme_view.columns:
            continue
        temp = programme_view.copy()
        temp["summary_period"] = converter(temp[candidate])
        temp["predicted_failure_likelihood_linear"] = pd.to_numeric(
            temp["predicted_failure_likelihood_linear"], errors="coerce"
        )
        temp = temp.dropna(subset=["summary_period", "predicted_failure_likelihood_linear"])
        if temp.empty:
            continue
        period_trend = (
            temp.groupby(pd.Grouper(key="summary_period", freq="ME"))["predicted_failure_likelihood_linear"]
            .mean()
            .dropna()
        )
        if len(period_trend) >= 3:
            recent = float(period_trend.tail(min(3, len(period_trend))).mean())
            previous = float(period_trend.iloc[:-min(3, len(period_trend))].tail(min(3, max(len(period_trend) - 1, 1))).mean()) if len(period_trend) > 3 else float(period_trend.iloc[:-1].mean())
            delta = recent - previous
            if delta > 0.05:
                trend_text = "near-term risk is building"
            elif delta < -0.05:
                trend_text = "near-term risk is easing"
            trend_source = candidate
            break

    driver = metrics.get("feature_used", "forecast discipline")
    driver_reason, _ = get_driver_description(str(driver))

    risk_sentence = f"The main watch-outs are {driver_reason.lower()} and general delivery discipline."
    if active_risks:
        lead_risks = [name for name, _ in active_risks[:2]]
        joined_risks = " and ".join(lead_risks) if len(lead_risks) == 2 else lead_risks[0]
        risk_sentence = f"The biggest upcoming risks are {joined_risks}."
        top_indicators = active_risks[0][1].get("indicators", [])[:2]
        if top_indicators:
            risk_sentence += f" Early warning signals already point to {' and '.join(indicator.lower() for indicator in top_indicators)}."

    budget_detail = ""
    if budget_flags:
        budget_detail = f" This is mainly because {budget_flags[0]}"
        if len(budget_flags) > 1:
            budget_detail += f", while {budget_flags[1]}"
        budget_detail += "."

    next_period_sentence = ""
    if pd.notna(next_month_failure_prob):
        if next_month_failure_prob >= 0.55:
            next_period_sentence = " The next reporting window already looks like a potential pressure point."
        elif next_month_failure_prob >= 0.35:
            next_period_sentence = " The next reporting window should be watched closely for slippage."

    source_sentence = ""
    if trend_source:
        source_sentence = f" This outlook is based on the recent trend in {trend_source.replace('_', ' ').lower()}."

    return (
        f"Overall, the programme looks {overview_state}: {completion_phrase}. "
        f"Budget confidence is {budget_confidence} rather than fully secure.{budget_detail} "
        f"{risk_sentence} At portfolio level, {trend_text}.{next_period_sentence}"
        f"{source_sentence}"
    ).strip()


def _mean_numeric_from_df(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    series = pd.Series(pd.to_numeric(df[column], errors="coerce"), index=df.index)
    return float(series.mean()) if series.notna().any() else float("nan")


def _sum_numeric_from_df(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    series = pd.Series(pd.to_numeric(df[column], errors="coerce"), index=df.index)
    return float(series.sum()) if series.notna().any() else float("nan")


def _describe_probability_trend(df: pd.DataFrame, value_col: str = "predicted_failure_likelihood_linear") -> tuple[str, str | None]:
    if value_col not in df.columns or df.empty:
        return "risk is broadly stable", None

    time_candidates = [
        ("Forecast_Period", lambda s: pd.to_datetime(s.astype(str) + "-01", errors="coerce")),
        ("Forecast_Period_End_Date", lambda s: pd.to_datetime(s, errors="coerce")),
        ("Forecast_Version_Date", lambda s: pd.to_datetime(s, errors="coerce")),
    ]
    for candidate, converter in time_candidates:
        if candidate not in df.columns:
            continue
        temp = df.copy()
        temp["summary_period"] = converter(temp[candidate])
        temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
        temp = temp.dropna(subset=["summary_period", value_col])
        if temp.empty:
            continue
        period_trend = (
            temp.groupby(pd.Grouper(key="summary_period", freq="ME"))[value_col]
            .mean()
            .dropna()
        )
        if len(period_trend) < 2:
            continue
        lookback = min(3, len(period_trend))
        recent = float(period_trend.tail(lookback).mean())
        previous_slice = period_trend.iloc[:-lookback] if len(period_trend) > lookback else period_trend.iloc[:-1]
        if previous_slice.empty:
            previous = float(period_trend.iloc[:-1].mean()) if len(period_trend) > 1 else recent
        else:
            previous = float(previous_slice.tail(min(3, len(previous_slice))).mean())
        delta = recent - previous
        if delta > 0.05:
            return "risk is building across upcoming periods", candidate
        if delta < -0.05:
            return "risk is easing compared with the earlier reporting cycle", candidate
        return "risk is broadly stable", candidate

    return "risk is broadly stable", None


def build_persona_summary_card(summary_text: str, accent: str = "#c7d2fe", background: str = "#eef2ff") -> html.Div:
    return html.Div(
        [
            html.H4("AI Summary", style={"margin": "0 0 8px 0"}),
            html.P(summary_text, style={"margin": 0, "lineHeight": "1.45"}),
        ],
        style={
            "backgroundColor": background,
            "border": f"1px solid {accent}",
            "borderRadius": "10px",
            "padding": "14px",
            "marginBottom": "14px",
        },
    )


def build_commercial_manager_summary(
    supplier_analysis_view: pd.DataFrame,
    contract_summary: pd.DataFrame,
    profile_summary: pd.DataFrame,
    commodity_summary: pd.DataFrame,
    supplier_watchlist: pd.DataFrame,
    risk_alerts: dict,
) -> str:
    if supplier_analysis_view.empty:
        return "There is not enough filtered supplier information to describe the current commercial position."

    predicted_rate = _mean_numeric_from_df(supplier_analysis_view, "predicted_failure_likelihood_linear")
    best_contract = contract_summary.sort_values("avg_failed_proposal_probability", ascending=True).iloc[0]["category"] if not contract_summary.empty else None
    riskiest_profile = profile_summary.sort_values("avg_failed_proposal_probability", ascending=False).iloc[0]["category"] if not profile_summary.empty else None
    riskiest_commodity = commodity_summary.sort_values("avg_failed_proposal_probability", ascending=False).iloc[0]["category"] if not commodity_summary.empty else None
    watch_supplier = supplier_watchlist.sort_values("avg_failed_proposal_probability", ascending=False).iloc[0]["Supplier_ID"] if not supplier_watchlist.empty else None

    status = "commercial exposure looks contained"
    if pd.notna(predicted_rate) and predicted_rate >= 0.55:
        status = "commercial exposure is elevated and supplier choices need tightening"
    elif pd.notna(predicted_rate) and predicted_rate >= 0.35:
        status = "commercial exposure is manageable but uneven across supplier choices"

    risk_themes = []
    if risk_alerts.get("supplier_delay_risk", {}).get("status") in {"Red", "Amber"}:
        risk_themes.append("supplier reliability is the main near-term concern")
    if risk_alerts.get("cost_volatility", {}).get("status") in {"Red", "Amber"}:
        risk_themes.append("commercial volatility is still feeding through into forecast risk")
    if not risk_themes:
        risk_themes.append("no single commercial lever is flashing red, but weaker categories still need active monitoring")

    option_text = ""
    if best_contract is not None:
        option_text = f" The data suggests {best_contract} is currently the most resilient contract structure"
        if riskiest_profile is not None:
            option_text += f", while {riskiest_profile} suppliers need tighter commercial oversight"
        option_text += "."

    hotspot_text = ""
    if watch_supplier is not None or riskiest_commodity is not None:
        parts = []
        if watch_supplier is not None:
            parts.append(f"supplier {watch_supplier}")
        if riskiest_commodity is not None:
            parts.append(f"{riskiest_commodity} demand")
        hotspot_text = f" The biggest commercial hotspots are {' and '.join(parts)}."

    return (
        f"Overall, {status}. {risk_themes[0].capitalize()}.{option_text}{hotspot_text} "
        f"This means procurement and contracting effort should be focused on the categories where terms, supplier profile and material mix are most likely to move the risk position."
    ).strip()


def build_cfo_summary(cfo_view: pd.DataFrame) -> str:
    if cfo_view.empty:
        return "There is not enough portfolio information to provide a finance-level outlook."

    forecast_total = _sum_numeric_from_df(cfo_view, "Forecast_Spend")
    actual_total = _sum_numeric_from_df(cfo_view, "Actual_Spend")
    predicted_rate = _mean_numeric_from_df(cfo_view, "predicted_failure_likelihood_linear")
    expected_at_risk_spend = float("nan")
    concentration_text = ""

    if {"Forecast_Spend", "predicted_failure_likelihood_linear", "Programme_ID"}.issubset(cfo_view.columns):
        finance_df = cfo_view.copy()
        finance_df["Forecast_Spend"] = pd.to_numeric(finance_df["Forecast_Spend"], errors="coerce")
        finance_df["predicted_failure_likelihood_linear"] = pd.to_numeric(finance_df["predicted_failure_likelihood_linear"], errors="coerce")
        finance_df = finance_df.dropna(subset=["Forecast_Spend", "predicted_failure_likelihood_linear"])
        if not finance_df.empty:
            finance_df["expected_at_risk_spend"] = (
                finance_df["Forecast_Spend"] * finance_df["predicted_failure_likelihood_linear"]
            )
            expected_at_risk_spend = float(finance_df["expected_at_risk_spend"].sum())
            programme_risk = (
                finance_df.groupby("Programme_ID", dropna=False)["expected_at_risk_spend"]
                .sum()
                .sort_values(ascending=False)
            )
            top_programmes = programme_risk.head(3).index.astype(str).tolist()
            if top_programmes:
                concentration_text = f" Risk is concentrated in {', '.join(top_programmes)}."

    budget_state = "portfolio affordability looks broadly intact"
    if pd.notna(predicted_rate) and predicted_rate >= 0.55:
        budget_state = "portfolio affordability is under clear pressure"
    elif pd.notna(predicted_rate) and predicted_rate >= 0.35:
        budget_state = "portfolio affordability is still manageable, but headroom is narrowing"

    exposure_text = ""
    if pd.notna(expected_at_risk_spend):
        exposure_text = f" The current risk-adjusted spend exposure is about GBP {expected_at_risk_spend:,.0f}"
        if pd.notna(forecast_total) and forecast_total > 0:
            exposure_text += f", equivalent to roughly {expected_at_risk_spend / forecast_total:.0%} of forecasted portfolio value"
        exposure_text += "."

    delivery_gap_text = ""
    if pd.notna(forecast_total) and pd.notna(actual_total) and forecast_total > 0:
        gap = (forecast_total - actual_total) / forecast_total
        if gap > 0.15:
            delivery_gap_text = " Actual delivery remains behind the forecast curve, which reduces confidence that the current financial plan will land cleanly."

    return (
        f"Overall, {budget_state}.{exposure_text}{concentration_text}{delivery_gap_text} "
        f"From a finance perspective, the immediate question is not only whether the programme can finish within budget, but whether the current risk concentration leaves enough headroom if any of the largest exposures deteriorate further."
    ).strip()


def build_project_controls_summary(
    programme_view: pd.DataFrame,
    risk_alerts: dict,
    top_risk_drivers: list[dict[str, str]],
) -> str:
    if programme_view.empty:
        return "There is not enough operational data to describe current control effectiveness."

    trend_text, trend_source = _describe_probability_trend(programme_view)
    stability_score = _mean_numeric_from_df(programme_view, "Forecast_Stability_Score")
    high_risk_share = float("nan")
    if "predicted_failure_likelihood_linear" in programme_view.columns:
        risk_series = pd.Series(pd.to_numeric(programme_view["predicted_failure_likelihood_linear"], errors="coerce"), index=programme_view.index).dropna()
        if not risk_series.empty:
            high_risk_share = float((risk_series >= 0.5).mean())

    hotspot_text = ""
    if {"Supplier_ID", "Region", "predicted_failure_likelihood_linear"}.issubset(programme_view.columns):
        hotspot_df = programme_view.copy()
        hotspot_df["predicted_failure_likelihood_linear"] = pd.to_numeric(hotspot_df["predicted_failure_likelihood_linear"], errors="coerce")
        hotspot_df = hotspot_df.dropna(subset=["predicted_failure_likelihood_linear"])
        if not hotspot_df.empty:
            top_hotspot = (
                hotspot_df.groupby(["Supplier_ID", "Region"], dropna=False)["predicted_failure_likelihood_linear"]
                .mean()
                .sort_values(ascending=False)
                .head(1)
            )
            if not top_hotspot.empty:
                supplier_id, region = top_hotspot.index[0]
                hotspot_text = f" The main operational hotspot is supplier {supplier_id} in {region}."

    control_state = "control performance looks steady"
    if pd.notna(stability_score) and stability_score < 0.45:
        control_state = "control performance is under strain because the forecast is still being reworked too often"
    elif pd.notna(high_risk_share) and high_risk_share >= 0.35:
        control_state = "control performance is mixed, with too much of the portfolio still sitting in elevated risk territory"

    driver_text = ""
    if top_risk_drivers:
        first_driver = top_risk_drivers[0]
        driver_text = f" The control story is being driven mainly by {first_driver['reason'].lower()}."

    alert_text = ""
    red_alerts = [
        name.replace("_", " ")
        for name, value in risk_alerts.items()
        if value.get("status") == "Red"
    ]
    if red_alerts:
        alert_text = f" Immediate intervention is needed around {' and '.join(red_alerts)}."

    source_text = f" This view is anchored on the trend in {trend_source.replace('_', ' ').lower()}." if trend_source else ""

    return (
        f"Overall, {control_state}; {trend_text}.{driver_text}{alert_text}{hotspot_text}{source_text} "
        f"For operations and controls teams, the priority is to reduce avoidable forecast churn and act early where supplier or delivery signals are starting to drift."
    ).strip()


def summarize_dimension_risk(
    df: pd.DataFrame,
    column: str,
    metric_col: str,
    *,
    top_n: int = 8,
    ascending: bool = False,
) -> pd.DataFrame:
    if column not in df.columns or metric_col not in df.columns:
        return pd.DataFrame()

    cols = [column, metric_col]
    if "Supplier_ID" in df.columns:
        cols.append("Supplier_ID")

    work = df[cols].copy()
    work[column] = work[column].fillna("Unknown").astype(str)
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work[work[metric_col].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    if "Supplier_ID" in work.columns:
        summary = (
            work.groupby(column, dropna=False)
            .agg(
                avg_failed_proposal_probability=(metric_col, "mean"),
                supplier_count=("Supplier_ID", pd.Series.nunique),
                record_count=(metric_col, "size"),
            )
            .reset_index()
        )
    else:
        summary = (
            work.groupby(column, dropna=False)
            .agg(
                avg_failed_proposal_probability=(metric_col, "mean"),
                record_count=(metric_col, "size"),
            )
            .reset_index()
        )
        summary["supplier_count"] = np.nan

    summary = summary.rename(columns={column: "category"})
    summary["avg_failed_proposal_probability_pct"] = summary["avg_failed_proposal_probability"] * 100
    return summary.sort_values("avg_failed_proposal_probability", ascending=ascending).head(top_n)


def summarize_supplier_watchlist(
    df: pd.DataFrame,
    metric_col: str,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    if "Supplier_ID" not in df.columns or metric_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work[work[metric_col].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    aggregations: dict[str, tuple[str, str]] = {
        "avg_failed_proposal_probability": (metric_col, "mean"),
        "record_count": (metric_col, "size"),
    }
    for extra_col in ["Commodity", "Contract_Type", "Supplier_Profile", "Region"]:
        if extra_col in work.columns:
            aggregations[extra_col] = (extra_col, "first")

    summary = work.groupby("Supplier_ID", dropna=False).agg(**aggregations).reset_index()
    summary["avg_failed_proposal_probability_pct"] = summary["avg_failed_proposal_probability"] * 100
    return summary.sort_values("avg_failed_proposal_probability", ascending=False).head(top_n)


def build_supplier_bar_chart(
    summary_df: pd.DataFrame,
    *,
    title: str,
    yaxis_title: str,
    avg_risk_pct: float,
    colorscale: list[list[float | str]] | list[str],
) -> go.Figure:
    chart_df = summary_df.sort_values("avg_failed_proposal_probability_pct", ascending=True).copy()
    chart_df["delta_vs_average"] = chart_df["avg_failed_proposal_probability_pct"] - avg_risk_pct
    chart_df["bar_label"] = chart_df.apply(
        lambda row: f"{row['avg_failed_proposal_probability_pct']:.1f}% ({row['delta_vs_average']:+.1f} pts)",
        axis=1,
    )
    chart_df["difference_label"] = chart_df["delta_vs_average"].apply(lambda v: f"{v:+.1f} pts vs current average")
    if "supplier_count" in chart_df.columns:
        chart_df["supplier_count_label"] = chart_df["supplier_count"].apply(
            lambda v: "" if pd.isna(v) else f"Suppliers analysed: {int(v)}"
        )
    else:
        chart_df["supplier_count_label"] = ""

    fig = px.bar(
        chart_df,
        x="avg_failed_proposal_probability_pct",
        y="category",
        orientation="h",
        color="avg_failed_proposal_probability_pct",
        color_continuous_scale=colorscale,
        text="bar_label",
        custom_data=["difference_label", "record_count", "supplier_count_label"],
        title=title,
    )
    fig.update_traces(
        textposition="auto",
        cliponaxis=True,
        constraintext="both",
        marker_line_color="#ffffff",
        marker_line_width=1.5,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Failed proposal probability: %{x:.1f}%<br>"
            "%{customdata[0]}<br>"
            "Records analysed: %{customdata[1]}<br>"
            "%{customdata[2]}"
            "<extra></extra>"
        ),
    )
    fig.add_vline(
        x=avg_risk_pct,
        line_dash="dash",
        line_color="#475569",
        line_width=1.5,
        annotation_text=f"Current average {avg_risk_pct:.1f}%",
        annotation_position="top right",
        annotation_font_color="#334155",
    )
    fig.update_layout(
        xaxis_title="Average failed proposal probability (%)",
        yaxis_title=yaxis_title,
        coloraxis_showscale=False,
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        font={"family": "Segoe UI, Arial, sans-serif", "size": 12, "color": "#1f2937"},
        margin={"l": 10, "r": 50, "t": 60, "b": 50},
        title_font={"size": 18, "color": "#111827"},
        uniformtext={"minsize": 9, "mode": "hide"},
        xaxis={
            "showgrid": True,
            "gridcolor": "#e2e8f0",
            "zeroline": False,
            "ticksuffix": "%",
        },
        yaxis={"showgrid": False},
    )
    return fig


def build_supplier_heatmap(
    supplier_heatmap: pd.DataFrame,
    *,
    title: str,
) -> go.Figure:
    matrix = supplier_heatmap.pivot(index="Region", columns="Contract_Type", values="avg_failed_proposal_probability_pct")
    row_order = matrix.mean(axis=1).sort_values(ascending=False).index.tolist()
    col_order = matrix.mean(axis=0).sort_values(ascending=False).index.tolist()
    matrix = matrix.loc[row_order, col_order]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=[
                [0.0, "#eef2ff"],
                [0.25, "#c4b5fd"],
                [0.5, "#8b5cf6"],
                [0.75, "#6366f1"],
                [1.0, "#312e81"],
            ],
            hovertemplate=(
                "<b>%{y}</b> / %{x}<br>"
                "Failed proposal probability: %{z:.1f}%<extra></extra>"
            ),
            colorbar={"title": "Failed proposal %", "ticksuffix": "%"},
            xgap=6,
            ygap=6,
        )
    )
    fig.update_traces(text=np.round(matrix.values, 1), texttemplate="%{text:.1f}%", textfont={"color": "white", "size": 12})
    fig.update_layout(
        title=title,
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        font={"family": "Segoe UI, Arial, sans-serif", "size": 12, "color": "#1f2937"},
        margin={"l": 10, "r": 10, "t": 60, "b": 30},
        title_font={"size": 18, "color": "#111827"},
        xaxis={"title": "Contract type", "side": "top", "tickangle": 0, "showgrid": False},
        yaxis={"title": "Region", "showgrid": False},
    )
    return fig


def serialize_workflow_result(result: dict) -> dict:
    return {
        "explained_df": result["explained_df"].to_json(date_format="iso", orient="split"),
        "loadings_df": result["loadings_df"].to_json(date_format="iso", orient="split"),
        "predictions_df": result["predictions_df"].to_json(date_format="iso", orient="split"),
        "selected_info": result["selected_info"],
        "metrics": result["metrics"],
    }


def deserialize_workflow_result(payload: dict) -> dict:
    return {
        "explained_df": pd.read_json(io.StringIO(payload["explained_df"]), orient="split"),
        "loadings_df": pd.read_json(io.StringIO(payload["loadings_df"]), orient="split"),
        "predictions_df": pd.read_json(io.StringIO(payload["predictions_df"]), orient="split"),
        "selected_info": payload["selected_info"],
        "metrics": payload["metrics"],
    }


def run_workflow(df: pd.DataFrame, target_col: str, n_components: str, train_frac: float, threshold: float) -> dict:
    work = prepare_dataframe(df)

    if target_col not in work.columns:
        raise ValueError(f"Selected target column not found: {target_col}")
    if "Forecast_Period_End_Date" not in work.columns:
        raise ValueError("Required column missing: Forecast_Period_End_Date")
    if "event_id" not in work.columns:
        raise ValueError("Could not build event_id. Include Programme_ID, Supplier_ID, Commodity, Forecast_Period.")

    feature_cols = [c for c in get_numeric_features(work) if c != target_col]
    if not feature_cols:
        raise ValueError("No numeric features available for PCA after removing target column.")

    explained_df, loadings_df, _scores_df, selected_info = find_influential_factor(
        df=work,
        feature_cols=feature_cols,
        n_components=parse_n_components(n_components),
        target_col=target_col,
    )

    predictions_df, metrics = train_test_linear(
        df=work,
        factor=selected_info["selected_factor"],
        target_col=target_col,
        train_frac=float(train_frac),
        threshold=float(threshold),
    )

    return {
        "explained_df": explained_df,
        "loadings_df": loadings_df,
        "selected_info": selected_info,
        "predictions_df": predictions_df,
        "metrics": metrics,
    }


def get_llm_access_token() -> str | None:
    if not LLM_AVAILABLE:
        return None

    def _read_yaml_value(path: Path, keys: list[str]) -> str | None:
        if not path.exists():
            return None

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return None

        parsed: dict[str, str] = {}
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            parsed[k.strip()] = v.strip().strip("\"'")

        for key in keys:
            value = parsed.get(key)
            if value:
                return value
        return None

    try:
        from dotenv import load_dotenv
        load_dotenv()
        client_id = os.getenv("CLIENT_ID")
        client_secret = os.getenv("CLIENT_SECRET")

        # Fallback to gaia_api_key.yaml when env vars are not set.
        if not client_id:
            client_id = _read_yaml_value(GAIA_CRED_PATH, ["CLIENT_ID", "client_id"])
        if not client_secret:
            client_secret = _read_yaml_value(GAIA_CRED_PATH, ["CLIENT_SECRET", "client_secret"])

        if not client_id or not client_secret:
            return None
        token = get_access_token(client_id, client_secret)
        return token
    except Exception as e:
        print(f"Could not retrieve LLM access token: {e}")
        return None


def generate_system_prompt(target_col: str) -> str:
    """Generate a concise business-focused system prompt for recommendation generation."""
    return f"""You are a senior business advisor for supply-chain forecasting.

Use PCA and prediction outputs to suggest practical improvements for target metric: {target_col}.

Return ONLY valid JSON as a list with up to 3 objects.
Each object must include exactly these keys:
- improvement
- effort
- expected_improvement
- implementation_steps

Rules:
- Keep each field concise and business-friendly.
- effort must be one of: Low, Medium, High.
- implementation_steps should be a short sentence, non-technical.
- Do not include any text outside JSON."""


def save_system_prompt(target_col: str) -> Path:
    """Generate and save the system prompt to a file."""
    prompt_content = generate_system_prompt(target_col)
    prompt_path = PROMPTS_DIR / f"system_prompt_{target_col}.txt"
    prompt_path.write_text(prompt_content, encoding="utf-8")
    return prompt_path


def get_llm_recommendations(
    top_factors: list[str],
    target_col: str,
    avg_risk: float,
    metrics: dict,
) -> list[dict] | None:
    """Call LLM to generate up to 3 concise recommendations in table-ready structure."""
    if not LLM_AVAILABLE:
        return None

    try:
        access_token = get_llm_access_token()
        if not access_token:
            return None

        system_prompt_path = save_system_prompt(target_col)
        system_prompt = system_prompt_path.read_text(encoding="utf-8")

        user_prompt = f"""Business context:
- target metric: {target_col}
- average predicted value: {avg_risk:.3f}
- top influencing factors: {', '.join(top_factors[:5])}
- key driver from model: {metrics.get('feature_used', 'N/A')}

Provide a maximum of 3 high-impact improvements for business leaders.
Return JSON only."""

        with with_genai_cwd():
            response = make_LLM_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                access_token=access_token,
                temp=1.0,
                model="anthropic/claude-haiku-4-5",
                enable_rag=False,
                max_completion_tokens=1000,
            )
        if not response:
            return None

        parsed: list[dict] = []
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                obj = [obj]
            if isinstance(obj, list):
                for item in obj[:3]:
                    if not isinstance(item, dict):
                        continue
                    parsed.append(
                        {
                            "improvement": str(item.get("improvement", "")).strip(),
                            "effort": str(item.get("effort", "Medium")).strip() or "Medium",
                            "expected_improvement": str(item.get("expected_improvement", "")).strip(),
                            "implementation_steps": str(item.get("implementation_steps", "")).strip(),
                        }
                    )
        except Exception:
            return None

        return parsed[:3] if parsed else None
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None


def build_graphs(
    result: dict,
    target_col: str,
    selected_programme: str | None,
    selected_region: str | None,
    selected_contract_type: str | None,
    selected_supplier_profile: str | None,
    min_risk_filter: float,
) -> list:
    explained_df = result["explained_df"]
    loadings_df = result["loadings_df"]
    selected_info = result["selected_info"]
    predictions_df = result["predictions_df"]
    metrics = result["metrics"]

    selected_component = selected_info["selected_component"]
    selected_factor = selected_info["selected_factor"]
    _ = int(metrics.get("test_rows", len(predictions_df)))
    card_style = {
        "border": "1px solid #ddd",
        "borderRadius": "8px",
        "padding": "10px",
        "backgroundColor": "#fafafa",
    }

    top_factors = (
        loadings_df[[selected_component]]
        .rename(columns={selected_component: "loading"})
        .assign(abs_loading=lambda x: x["loading"].abs())
        .sort_values("abs_loading", ascending=False)
        .head(10)
        .index.tolist()
    )

    top_factors_with_loadings = (
        loadings_df[[selected_component]]
        .rename(columns={selected_component: "loading"})
        .assign(abs_loading=lambda x: x["loading"].abs())
        .sort_values("abs_loading", ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"index": "driver"})
    )
    top_factors_with_loadings["rank"] = range(1, len(top_factors_with_loadings) + 1)
    max_abs_loading = top_factors_with_loadings["abs_loading"].max()
    threshold_high = max_abs_loading * 0.6
    threshold_med = max_abs_loading * 0.3
    top_factors_with_loadings["impact"] = top_factors_with_loadings["abs_loading"].apply(
        lambda x: "High" if x >= threshold_high else ("Medium" if x >= threshold_med else "Low")
    )
    top_factors_with_loadings["impact_class"] = top_factors_with_loadings["impact"].apply(
        lambda x: "impact-high" if x == "High" else ("impact-medium" if x == "Medium" else "impact-low")
    )
    top_factors_with_loadings["reason"] = top_factors_with_loadings["driver"].apply(
        lambda x: get_driver_description(x)[0]
    )
    top_factors_with_loadings["impact_explanation"] = top_factors_with_loadings["driver"].apply(
        lambda x: get_driver_description(x)[1]
    )

    top10_data = (
        loadings_df[[selected_component]]
        .rename(columns={selected_component: "loading"})
        .assign(feature=loadings_df.index, abs_loading=lambda x: x["loading"].abs())
        .sort_values("abs_loading", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    
    # Create color gradient from light to dark blue based on influence strength
    colors = [
        "#023E8A",  # Dark blue
        "#0353A4",  #
        "#0468B8",  #
        "#057DCD",  #
        "#0892E3",  #
        "#1BA1F2",  #
        "#48AEFF",  #
        "#77BAFF",  #
        "#A8D4FF",  #
        "#D4E9FF",  # Light blue
    ]
    
    top10_factors_fig = px.bar(
        top10_data,
        x="abs_loading",
        y="feature",
        orientation="h",
        title="Top 10 Drivers Impacting Forecast Accuracy",
    )
    top10_factors_fig.update_traces(
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b><br>Influence Strength: %{x:.3f}<extra></extra>",
    )
    top10_factors_fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Influence Strength",
        yaxis_title="Key Business Factor",
        plot_bgcolor="#f5f5f5",
        paper_bgcolor="#ffffff",
    )

    variance_fig = px.bar(
        explained_df,
        x="component",
        y="explained_variance_ratio",
        title="PCA Explained Variance Ratio",
    )

    top_loadings = (
        loadings_df[[selected_component]]
        .rename(columns={selected_component: "loading"})
        .assign(abs_loading=lambda x: x["loading"].abs())
        .sort_values("abs_loading", ascending=False)
        .head(12)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    hist_fig = px.histogram(
        predictions_df,
        x="predicted_failure_likelihood_linear",
        nbins=30,
        color=target_col,
        barmode="overlay",
        opacity=0.6,
        title="Prediction Distribution",
    )

    avg_risk = float(predictions_df["predicted_failure_likelihood_linear"].mean())

    programme_view = predictions_df.copy()
    if selected_programme and "Programme_ID" in programme_view.columns:
        programme_view = programme_view[programme_view["Programme_ID"].astype(str) == str(selected_programme)].copy()
    if selected_region and "Region" in programme_view.columns:
        programme_view = programme_view[programme_view["Region"].astype(str) == str(selected_region)].copy()
    if selected_contract_type and "Contract_Type" in programme_view.columns:
        programme_view = programme_view[programme_view["Contract_Type"].astype(str) == str(selected_contract_type)].copy()
    if selected_supplier_profile and "Supplier_Profile" in programme_view.columns:
        programme_view = programme_view[programme_view["Supplier_Profile"].astype(str) == str(selected_supplier_profile)].copy()

    if "predicted_failure_likelihood_linear" in programme_view.columns:
        programme_view["predicted_failure_likelihood_linear"] = pd.to_numeric(
            programme_view["predicted_failure_likelihood_linear"],
            errors="coerce",
        )
        if float(min_risk_filter) > 0:
            programme_view = programme_view[
                programme_view["predicted_failure_likelihood_linear"] >= float(min_risk_filter)
            ].copy()

    filter_note = ""
    if programme_view.empty:
        filter_note = "No records matched the selected filters. Showing full portfolio instead."
        programme_view = predictions_df.copy()

    supplier_analysis_view = programme_view.copy()
    probability_col = "predicted_failure_likelihood_linear"
    supplier_scope = f"Programme {selected_programme}" if selected_programme else "the current portfolio"
    avg_risk_pct = avg_risk * 100

    contract_summary = summarize_dimension_risk(
        supplier_analysis_view,
        "Contract_Type",
        probability_col,
        top_n=8,
    )
    profile_summary = summarize_dimension_risk(
        supplier_analysis_view,
        "Supplier_Profile",
        probability_col,
        top_n=8,
    )
    commodity_summary = summarize_dimension_risk(
        supplier_analysis_view,
        "Commodity",
        probability_col,
        top_n=8,
    )
    region_summary = summarize_dimension_risk(
        supplier_analysis_view,
        "Region",
        probability_col,
        top_n=8,
    )
    supplier_watchlist = summarize_supplier_watchlist(
        supplier_analysis_view,
        probability_col,
        top_n=10,
    )

    contract_fig = None
    if not contract_summary.empty:
        contract_fig = build_supplier_bar_chart(
            contract_summary,
            title="Commercial model comparison",
            yaxis_title="Contract type",
            avg_risk_pct=avg_risk_pct,
            colorscale=[
                [0.0, "#ede9fe"],
                [0.35, "#c4b5fd"],
                [0.7, "#8b5cf6"],
                [1.0, "#5b21b6"],
            ],
        )

    profile_fig = None
    if not profile_summary.empty:
        profile_fig = build_supplier_bar_chart(
            profile_summary,
            title="Supplier profile comparison",
            yaxis_title="Supplier profile",
            avg_risk_pct=avg_risk_pct,
            colorscale=[
                [0.0, "#e0f2fe"],
                [0.35, "#7dd3fc"],
                [0.7, "#0ea5e9"],
                [1.0, "#0f3d8a"],
            ],
        )

    commodity_fig = None
    if not commodity_summary.empty:
        commodity_fig = build_supplier_bar_chart(
            commodity_summary,
            title="Material / commodity comparison",
            yaxis_title="Material / commodity",
            avg_risk_pct=avg_risk_pct,
            colorscale=[
                [0.0, "#dcfce7"],
                [0.35, "#86efac"],
                [0.7, "#22c55e"],
                [1.0, "#166534"],
            ],
        )

    supplier_watchlist_fig = None
    if not supplier_watchlist.empty:
        watchlist_df = supplier_watchlist.sort_values("avg_failed_proposal_probability", ascending=True).copy()
        watchlist_df["delta_vs_average"] = watchlist_df["avg_failed_proposal_probability_pct"] - avg_risk_pct
        watchlist_df["bar_label"] = watchlist_df.apply(
            lambda row: f"{row['avg_failed_proposal_probability_pct']:.1f}% ({row['delta_vs_average']:+.1f} pts)",
            axis=1,
        )
        watchlist_df["supplier_summary"] = watchlist_df.apply(
            lambda row: " | ".join(
                [
                    str(row[col])
                    for col in ["Commodity", "Contract_Type", "Supplier_Profile", "Region"]
                    if col in watchlist_df.columns and pd.notna(row[col])
                ]
            ) or "Supplier detail available",
            axis=1,
        )
        supplier_watchlist_fig = px.bar(
            watchlist_df,
            x="avg_failed_proposal_probability_pct",
            y="Supplier_ID",
            orientation="h",
            color="avg_failed_proposal_probability_pct",
            color_continuous_scale=[
                [0.0, "#fee2e2"],
                [0.35, "#fca5a5"],
                [0.7, "#ef4444"],
                [1.0, "#991b1b"],
            ],
            title="Supplier watchlist",
            text="bar_label",
            custom_data=["supplier_summary", "record_count"],
        )
        supplier_watchlist_fig.update_traces(
            textposition="auto",
            cliponaxis=True,
            constraintext="both",
            marker_line_color="#ffffff",
            marker_line_width=1.5,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Failed proposal probability: %{x:.1f}%<br>"
                "Supplier context: %{customdata[0]}<br>"
                "Records analysed: %{customdata[1]}<extra></extra>"
            ),
        )
        supplier_watchlist_fig.add_vline(
            x=avg_risk_pct,
            line_dash="dash",
            line_color="#475569",
            line_width=1.5,
            annotation_text=f"Current average {avg_risk_pct:.1f}%",
            annotation_position="top right",
            annotation_font_color="#334155",
        )
        supplier_watchlist_fig.update_layout(
            xaxis_title="Average failed proposal probability (%)",
            yaxis_title="Supplier",
            coloraxis_showscale=False,
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
            font={"family": "Segoe UI, Arial, sans-serif", "size": 12, "color": "#1f2937"},
            margin={"l": 10, "r": 50, "t": 60, "b": 50},
            title_font={"size": 18, "color": "#111827"},
            uniformtext={"minsize": 9, "mode": "hide"},
            xaxis={"showgrid": True, "gridcolor": "#e2e8f0", "zeroline": False, "ticksuffix": "%"},
            yaxis={"showgrid": False},
        )

    supplier_heatmap_fig = None
    if "Region" in supplier_analysis_view.columns and "Contract_Type" in supplier_analysis_view.columns:
        supplier_heatmap = (
            supplier_analysis_view.assign(**{probability_col: pd.to_numeric(supplier_analysis_view[probability_col], errors="coerce")})
            .dropna(subset=[probability_col])
            .groupby(["Region", "Contract_Type"], dropna=False)[probability_col]
            .mean()
            .reset_index(name="avg_failed_proposal_probability")
        )
        if not supplier_heatmap.empty:
            supplier_heatmap["avg_failed_proposal_probability_pct"] = supplier_heatmap["avg_failed_proposal_probability"] * 100
            supplier_heatmap_fig = build_supplier_heatmap(
                supplier_heatmap,
                title="Region and contract type interaction",
            )

    revision_risk_fig = None
    if "Revision_Number" in programme_view.columns:
        revision_trend = (
            programme_view.assign(
                Revision_Number=pd.to_numeric(programme_view["Revision_Number"], errors="coerce"),
                predicted_failure_likelihood_linear=pd.to_numeric(programme_view["predicted_failure_likelihood_linear"], errors="coerce"),
            )
            .dropna(subset=["Revision_Number", "predicted_failure_likelihood_linear"])
            .groupby("Revision_Number", dropna=False)
            .agg(
                avg_failed_probability=("predicted_failure_likelihood_linear", "mean"),
                record_count=("predicted_failure_likelihood_linear", "size"),
            )
            .reset_index()
            .sort_values("Revision_Number")
        )
        if not revision_trend.empty:
            revision_trend["avg_failed_probability_pct"] = revision_trend["avg_failed_probability"] * 100
            revision_risk_fig = px.line(
                revision_trend,
                x="Revision_Number",
                y="avg_failed_probability_pct",
                markers=True,
                title="Forecast fade by revision cycle",
            )
            revision_risk_fig.update_traces(
                line={"color": "#7c3aed", "width": 3},
                marker={"size": 8, "color": "#5b21b6"},
                hovertemplate=(
                    "Revision %{x:.0f}<br>"
                    "Failed proposal probability: %{y:.1f}%<br>"
                    "Records analysed: %{customdata[0]}<extra></extra>"
                ),
                customdata=revision_trend[["record_count"]],
            )
            revision_risk_fig.update_layout(
                xaxis_title="Forecast revision number",
                yaxis_title="Average failed proposal probability (%)",
                plot_bgcolor="#f8fafc",
                paper_bgcolor="#ffffff",
                xaxis={"showgrid": True, "gridcolor": "#e2e8f0", "dtick": 1},
                yaxis={"showgrid": True, "gridcolor": "#e2e8f0", "ticksuffix": "%"},
            )

    confidence_band_fig = None
    if "Confidence_Band" in programme_view.columns:
        confidence_summary = summarize_dimension_risk(
            programme_view,
            "Confidence_Band",
            "predicted_failure_likelihood_linear",
            top_n=10,
        )
        if not confidence_summary.empty:
            confidence_band_fig = build_supplier_bar_chart(
                confidence_summary,
                title="Confidence band impact on failed proposal probability",
                yaxis_title="Confidence band",
                avg_risk_pct=avg_risk_pct,
                colorscale=[
                    [0.0, "#dbeafe"],
                    [0.35, "#93c5fd"],
                    [0.7, "#3b82f6"],
                    [1.0, "#1e3a8a"],
                ],
            )

    cfo_view = predictions_df.copy()

    cfo_programme_fig = None
    if "Programme_ID" in cfo_view.columns:
        cfo_programme = (
            cfo_view.assign(
                predicted_failure_likelihood_linear=pd.to_numeric(cfo_view["predicted_failure_likelihood_linear"], errors="coerce"),
                Forecast_Spend=pd.to_numeric(cfo_view.get("Forecast_Spend", np.nan), errors="coerce"),
            )
            .dropna(subset=["predicted_failure_likelihood_linear"])
            .groupby("Programme_ID", dropna=False)
            .agg(
                avg_failed_probability=("predicted_failure_likelihood_linear", "mean"),
                forecast_spend=("Forecast_Spend", "sum"),
                record_count=("predicted_failure_likelihood_linear", "size"),
            )
            .reset_index()
        )
        if not cfo_programme.empty:
            cfo_programme["avg_failed_probability_pct"] = cfo_programme["avg_failed_probability"] * 100
            cfo_programme["expected_at_risk_spend"] = cfo_programme["forecast_spend"] * cfo_programme["avg_failed_probability"]
            cfo_programme = cfo_programme.sort_values("expected_at_risk_spend", ascending=False).head(10)
            cfo_programme_fig = px.bar(
                cfo_programme.sort_values("expected_at_risk_spend", ascending=True),
                x="expected_at_risk_spend",
                y="Programme_ID",
                orientation="h",
                color="avg_failed_probability_pct",
                color_continuous_scale=[
                    [0.0, "#e0e7ff"],
                    [0.35, "#a5b4fc"],
                    [0.7, "#6366f1"],
                    [1.0, "#312e81"],
                ],
                text="avg_failed_probability_pct",
                title="Programmes with highest expected at-risk spend",
                custom_data=["forecast_spend", "record_count"],
            )
            cfo_programme_fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="auto",
                cliponaxis=True,
                constraintext="both",
                hovertemplate=(
                    "Programme %{y}<br>"
                    "Expected at-risk spend: GBP %{x:,.0f}<br>"
                    "Failed proposal probability: %{marker.color:.1f}%<br>"
                    "Forecast spend: GBP %{customdata[0]:,.0f}<br>"
                    "Records analysed: %{customdata[1]}<extra></extra>"
                ),
            )
            cfo_programme_fig.update_layout(
                xaxis_title="Expected at-risk spend (GBP)",
                yaxis_title="Programme",
                coloraxis_colorbar_title="Failed proposal %",
                plot_bgcolor="#f8fafc",
                paper_bgcolor="#ffffff",
                uniformtext={"minsize": 9, "mode": "hide"},
                xaxis={"showgrid": True, "gridcolor": "#e2e8f0"},
                yaxis={"showgrid": False},
            )

    cfo_risk_spend_scatter = None
    if "Forecast_Spend" in cfo_view.columns:
        scatter_df = cfo_view.copy()
        scatter_df["Forecast_Spend"] = pd.to_numeric(scatter_df["Forecast_Spend"], errors="coerce")
        scatter_df["predicted_failure_likelihood_linear"] = pd.to_numeric(scatter_df["predicted_failure_likelihood_linear"], errors="coerce")
        scatter_df = scatter_df.dropna(subset=["Forecast_Spend", "predicted_failure_likelihood_linear"])
        if not scatter_df.empty:
            cfo_risk_spend_scatter = px.scatter(
                scatter_df,
                x="Forecast_Spend",
                y="predicted_failure_likelihood_linear",
                color="predicted_failure_likelihood_linear",
                color_continuous_scale="PuBu",
                hover_data=[c for c in ["Programme_ID", "Supplier_ID", "Commodity", "Contract_Type"] if c in scatter_df.columns],
                title="Spend versus failed proposal probability",
            )
            cfo_risk_spend_scatter.update_traces(
                marker={"size": 9, "line": {"color": "#ffffff", "width": 1}},
                hovertemplate=(
                    "Forecast spend: GBP %{x:,.0f}<br>"
                    "Failed proposal probability: %{y:.1%}<extra></extra>"
                ),
            )
            cfo_risk_spend_scatter.update_layout(
                xaxis_title="Forecast spend (GBP)",
                yaxis_title="Failed proposal probability",
                coloraxis_colorbar_title="Probability",
                plot_bgcolor="#f8fafc",
                paper_bgcolor="#ffffff",
                xaxis={"showgrid": True, "gridcolor": "#e2e8f0"},
                yaxis={"showgrid": True, "gridcolor": "#e2e8f0", "tickformat": ".0%"},
            )

    controls_stability_fig = None
    if "Forecast_Stability_Score" in programme_view.columns:
        controls_df = programme_view.copy()
        controls_df["Forecast_Stability_Score"] = pd.to_numeric(controls_df["Forecast_Stability_Score"], errors="coerce")
        controls_df["predicted_failure_likelihood_linear"] = pd.to_numeric(controls_df["predicted_failure_likelihood_linear"], errors="coerce")
        controls_df = controls_df.dropna(subset=["Forecast_Stability_Score", "predicted_failure_likelihood_linear"])
        if not controls_df.empty:
            controls_stability_fig = px.scatter(
                controls_df,
                x="Forecast_Stability_Score",
                y="predicted_failure_likelihood_linear",
                color="predicted_failure_likelihood_linear",
                color_continuous_scale="Turbo",
                title="Stability score versus failed proposal probability",
            )
            controls_stability_fig.update_traces(
                marker={"size": 8, "opacity": 0.75, "line": {"color": "#ffffff", "width": 0.7}},
                hovertemplate=(
                    "Forecast stability score: %{x:.2f}<br>"
                    "Failed proposal probability: %{y:.1%}<extra></extra>"
                ),
            )
            controls_stability_fig.update_layout(
                xaxis_title="Forecast stability score",
                yaxis_title="Failed proposal probability",
                plot_bgcolor="#f8fafc",
                paper_bgcolor="#ffffff",
                yaxis={"tickformat": ".0%", "showgrid": True, "gridcolor": "#e2e8f0"},
                xaxis={"showgrid": True, "gridcolor": "#e2e8f0"},
            )

    controls_leadtime_fig = None
    if "Days_Before_Period" in programme_view.columns:
        lead_df = programme_view.copy()
        lead_df["Days_Before_Period"] = pd.to_numeric(lead_df["Days_Before_Period"], errors="coerce")
        lead_df["predicted_failure_likelihood_linear"] = pd.to_numeric(lead_df["predicted_failure_likelihood_linear"], errors="coerce")
        lead_df = lead_df.dropna(subset=["Days_Before_Period", "predicted_failure_likelihood_linear"])
        if not lead_df.empty:
            lead_df["leadtime_bucket"] = pd.cut(
                lead_df["Days_Before_Period"],
                bins=[-np.inf, 30, 60, 90, 120, np.inf],
                labels=["<=30d", "31-60d", "61-90d", "91-120d", ">120d"],
            )
            lead_summary = (
                lead_df.groupby("leadtime_bucket", dropna=False, observed=False)["predicted_failure_likelihood_linear"]
                .mean()
                .reset_index(name="avg_failed_probability")
            )
            lead_summary["avg_failed_probability_pct"] = lead_summary["avg_failed_probability"] * 100
            controls_leadtime_fig = px.bar(
                lead_summary,
                x="leadtime_bucket",
                y="avg_failed_probability_pct",
                color="avg_failed_probability_pct",
                color_continuous_scale="Blues",
                text="avg_failed_probability_pct",
                title="Lead-time window effect on failed proposal probability",
            )
            controls_leadtime_fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="auto",
                cliponaxis=True,
                constraintext="both",
                hovertemplate=(
                    "Lead-time bucket: %{x}<br>"
                    "Failed proposal probability: %{y:.1f}%<extra></extra>"
                ),
            )
            controls_leadtime_fig.update_layout(
                xaxis_title="Days before forecast period",
                yaxis_title="Average failed proposal probability (%)",
                coloraxis_showscale=False,
                plot_bgcolor="#f8fafc",
                paper_bgcolor="#ffffff",
                uniformtext={"minsize": 9, "mode": "hide"},
                yaxis={"showgrid": True, "gridcolor": "#e2e8f0", "ticksuffix": "%"},
                xaxis={"showgrid": False},
            )

    operations_risk_trend_fig = None
    trend_source = None
    if "Forecast_Period" in programme_view.columns:
        trend_source = "Forecast_Period"
        trend_df = programme_view.copy()
        trend_df["trend_dt"] = pd.to_datetime(trend_df["Forecast_Period"].astype(str) + "-01", errors="coerce")
    elif "Forecast_Period_End_Date" in programme_view.columns:
        trend_source = "Forecast_Period_End_Date"
        trend_df = programme_view.copy()
        trend_df["trend_dt"] = pd.to_datetime(trend_df["Forecast_Period_End_Date"], errors="coerce")
    elif "Forecast_Version_Date" in programme_view.columns:
        trend_source = "Forecast_Version_Date"
        trend_df = programme_view.copy()
        trend_df["trend_dt"] = pd.to_datetime(trend_df["Forecast_Version_Date"], errors="coerce")
    else:
        trend_df = None

    if trend_df is not None:
        trend_df["predicted_failure_likelihood_linear"] = pd.to_numeric(
            trend_df["predicted_failure_likelihood_linear"],
            errors="coerce",
        )
        trend_df = trend_df.dropna(subset=["trend_dt", "predicted_failure_likelihood_linear"]).copy()
        if not trend_df.empty:
            trend_df["trend_month"] = trend_df["trend_dt"].dt.to_period("M").dt.to_timestamp()
            risk_over_time = (
                trend_df.groupby("trend_month", dropna=False)
                .agg(
                    avg_failed_probability=("predicted_failure_likelihood_linear", "mean"),
                    record_count=("predicted_failure_likelihood_linear", "size"),
                )
                .reset_index()
                .sort_values("trend_month")
            )
            risk_over_time["avg_failed_probability_pct"] = risk_over_time["avg_failed_probability"] * 100
            operations_risk_trend_fig = px.line(
                risk_over_time,
                x="trend_month",
                y="avg_failed_probability_pct",
                markers=True,
                title="Change in forecast failure probability over time",
            )
            operations_risk_trend_fig.update_traces(
                line={"color": "#1d4ed8", "width": 3},
                marker={"size": 7, "color": "#1e40af"},
                customdata=risk_over_time[["record_count"]],
                hovertemplate=(
                    "Month: %{x|%b %Y}<br>"
                    "Failed proposal probability: %{y:.1f}%<br>"
                    "Records analysed: %{customdata[0]}<extra></extra>"
                ),
            )
            operations_risk_trend_fig.update_layout(
                xaxis_title="Reporting month",
                yaxis_title="Average failed proposal probability (%)",
                plot_bgcolor="#f8fafc",
                paper_bgcolor="#ffffff",
                xaxis={"showgrid": True, "gridcolor": "#e2e8f0"},
                yaxis={"showgrid": True, "gridcolor": "#e2e8f0", "ticksuffix": "%"},
                margin={"l": 10, "r": 10, "t": 60, "b": 40},
            )
            active_filters = []
            if selected_programme:
                active_filters.append(f"Programme={selected_programme}")
            if selected_region:
                active_filters.append(f"Region={selected_region}")
            if selected_contract_type:
                active_filters.append(f"Contract={selected_contract_type}")
            if selected_supplier_profile:
                active_filters.append(f"Profile={selected_supplier_profile}")
            if float(min_risk_filter) > 0:
                active_filters.append(f"MinRisk={float(min_risk_filter):.0%}")

            filter_txt = " | ".join(active_filters) if active_filters else "All active records"
            operations_risk_trend_fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0,
                y=1.14,
                text=f"View: {filter_txt} | Records: {len(trend_df):,}",
                showarrow=False,
                align="left",
                font={"size": 11, "color": "#334155"},
            )
            if trend_source:
                operations_risk_trend_fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=1,
                    y=1.12,
                    text=f"Source: {trend_source}",
                    showarrow=False,
                    font={"size": 11, "color": "#475569"},
                )

    next_month_forecast = np.nan
    next_month_failure_prob = np.nan
    if "Forecast_Period" in programme_view.columns and "Forecast_Spend" in programme_view.columns:
        tmp = programme_view.copy()
        tmp["period_dt"] = pd.to_datetime(tmp["Forecast_Period"].astype(str) + "-01", errors="coerce")
        tmp = tmp[tmp["period_dt"].notna()]
        if not tmp.empty:
            next_period = tmp["period_dt"].max()
            next_slice = tmp[tmp["period_dt"] == next_period]
            next_month_forecast = float(pd.to_numeric(next_slice["Forecast_Spend"], errors="coerce").sum())
            next_month_failure_prob = float(next_slice["predicted_failure_likelihood_linear"].mean())

    def _risk_band(prob: float) -> tuple[str, str]:
        if np.isnan(prob):
            return "Unknown", "kpi-neutral"
        if prob <= 0.25:
            return "Green", "kpi-green"
        if prob <= 0.50:
            return "Amber", "kpi-amber"
        return "Red", "kpi-red"

    risk_label, risk_class = _risk_band(next_month_failure_prob)
    programme_director_summary_text = build_programme_director_summary(
        programme_view=programme_view,
        target_col=target_col,
        metrics=metrics,
        next_month_forecast=next_month_forecast,
        next_month_failure_prob=next_month_failure_prob,
        risk_label=risk_label,
    )
    programme_director_summary_card = build_persona_summary_card(
        programme_director_summary_text,
        accent="#c7d2fe",
        background="#eef2ff",
    )

    rec_rows = get_llm_recommendations(top_factors, target_col, avg_risk, metrics)
    if not rec_rows:
        rec_rows = [
            {
                "improvement": f"Stabilize planning around {top_factors[0]}",
                "effort": "Medium",
                "expected_improvement": "Reduce missed forecasts by 8-12%",
                "implementation_steps": "Set weekly review checkpoints and approve changes in a single governance meeting.",
            },
            {
                "improvement": f"Tighten control for {top_factors[1] if len(top_factors) > 1 else 'key cost drivers'}",
                "effort": "Low",
                "expected_improvement": "Improve consistency of monthly outcomes",
                "implementation_steps": "Assign a named owner and track a simple red/amber/green status each week.",
            },
            {
                "improvement": "Introduce early-warning escalation",
                "effort": "Low",
                "expected_improvement": "Faster response to high-risk periods",
                "implementation_steps": "Escalate automatically when predicted risk crosses the agreed threshold.",
            },
        ]

    action_items = html.Div(
        [
            html.H4("Top 3 Recommended Improvements"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Improvement"),
                                html.Th("Effort"),
                                html.Th("Expected Improvement"),
                                html.Th("Implementation Steps"),
                            ],
                            style={"backgroundColor": "#7c3aed", "color": "white", "fontWeight": "600"}
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(row["improvement"]),
                                    html.Td(row["effort"]),
                                    html.Td(row["expected_improvement"]),
                                    html.Td(row["implementation_steps"]),
                                ],
                                className="recommendation-row-" + str(row["effort"].lower()),
                            )
                            for row in rec_rows[:3]
                        ]
                    ),
                ],
                className="recommendations-table",
            ),
        ]
    )

    supplier_option_candidates: list[dict[str, str]] = []
    supplier_dimension_text = {
        "Contract_Type": "Commercial structure changes how tightly scope, pricing and delivery obligations are controlled.",
        "Supplier_Profile": "Supplier behaviour patterns are a strong signal of delivery reliability and forecast discipline.",
        "Region": "Regional logistics and market conditions can either stabilise or amplify delivery risk.",
        "Commodity": "Some material categories are structurally more volatile and harder to forecast accurately.",
    }
    for dimension, label, summary_df in [
        ("Contract_Type", "Commercial model", contract_summary),
        ("Supplier_Profile", "Supplier profile", profile_summary),
        ("Region", "Region", region_summary),
        ("Commodity", "Material / commodity", commodity_summary),
    ]:
        if summary_df.empty:
            continue
        eligible = summary_df[summary_df["record_count"] >= 3] if (summary_df["record_count"] >= 3).any() else summary_df
        best_row = eligible.sort_values("avg_failed_proposal_probability", ascending=True).iloc[0]
        improvement_pp = (avg_risk - float(best_row["avg_failed_proposal_probability"])) * 100
        supplier_option_candidates.append(
            {
                "decision_lever": label,
                "recommended_option": str(best_row["category"]),
                "avg_failed_probability": f"{best_row['avg_failed_proposal_probability']:.1%}",
                "impact_vs_average": f"{improvement_pp:+.1f} pts vs {avg_risk:.1%} current average",
                "business_readout": supplier_dimension_text[dimension],
            }
        )

    supplier_cards = []
    if not contract_summary.empty:
        best_contract = contract_summary.sort_values("avg_failed_proposal_probability", ascending=True).iloc[0]
        supplier_cards.append(
            html.Div(
                [
                    html.Div("Lowest-risk contract type"),
                    html.H4(str(best_contract["category"])),
                    html.Div(f"Avg failed proposal probability: {best_contract['avg_failed_proposal_probability']:.1%}"),
                ],
                style=card_style,
                className="supplier-insight-card supplier-card-contract",
            )
        )
    if not profile_summary.empty:
        riskiest_profile = profile_summary.sort_values("avg_failed_proposal_probability", ascending=False).iloc[0]
        supplier_cards.append(
            html.Div(
                [
                    html.Div("Highest-risk supplier profile"),
                    html.H4(str(riskiest_profile["category"])),
                    html.Div(f"Avg failed proposal probability: {riskiest_profile['avg_failed_proposal_probability']:.1%}"),
                ],
                style=card_style,
                className="supplier-insight-card supplier-card-profile",
            )
        )
    if not commodity_summary.empty:
        riskiest_commodity = commodity_summary.sort_values("avg_failed_proposal_probability", ascending=False).iloc[0]
        supplier_cards.append(
            html.Div(
                [
                    html.Div("Most exposed material / commodity"),
                    html.H4(str(riskiest_commodity["category"])),
                    html.Div(f"Avg failed proposal probability: {riskiest_commodity['avg_failed_proposal_probability']:.1%}"),
                ],
                style=card_style,
                className="supplier-insight-card supplier-card-commodity",
            )
        )

    ops_fig = None
    if "Supplier_ID" in programme_view.columns and "Region" in programme_view.columns:
        heat = (
            programme_view.groupby(["Supplier_ID", "Region"], dropna=False)["predicted_failure_likelihood_linear"]
            .mean()
            .reset_index(name="avg_predicted_risk")
        )
        heat["supplier_region"] = heat["Supplier_ID"].astype(str) + " | " + heat["Region"].astype(str)
        top_supplier_region = heat.sort_values("avg_predicted_risk", ascending=False).head(FILTER_DEFAULT_SUPPLIER_TOPN).copy()
        top_supplier_region["avg_predicted_risk_pct"] = top_supplier_region["avg_predicted_risk"] * 100

        chart_height = max(520, min(980, 26 * len(top_supplier_region) + 160))
        ops_fig = px.bar(
            top_supplier_region.sort_values("avg_predicted_risk_pct", ascending=True),
            x="avg_predicted_risk_pct",
            y="supplier_region",
            orientation="h",
            color="avg_predicted_risk_pct",
            color_continuous_scale=[
                [0.0, "#fee2e2"],
                [0.35, "#fca5a5"],
                [0.7, "#ef4444"],
                [1.0, "#991b1b"],
            ],
            title="Top supplier-region risk pairs",
            text="avg_predicted_risk_pct",
            custom_data=["Supplier_ID", "Region"],
        )
        ops_fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="auto",
            cliponaxis=True,
            constraintext="both",
            marker_line_color="#ffffff",
            marker_line_width=1,
            hovertemplate=(
                "Supplier: %{customdata[0]}<br>"
                "Region: %{customdata[1]}<br>"
                "Failed proposal probability: %{x:.1f}%<extra></extra>"
            ),
        )
        ops_fig.update_layout(
            xaxis_title="Average failed proposal probability (%)",
            yaxis_title="Supplier | Region",
            coloraxis_showscale=False,
            height=chart_height,
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
            margin={"l": 20, "r": 50, "t": 70, "b": 40},
            uniformtext={"minsize": 9, "mode": "hide"},
            xaxis={"showgrid": True, "gridcolor": "#e2e8f0", "ticksuffix": "%"},
            yaxis={"showgrid": False},
        )

    kpi_cards = html.Div(
        [
            html.Div(
                [
                    html.Div("Programme"),
                    html.H4(str(selected_programme) if selected_programme else "All Programmes"),
                ],
                style=card_style,
            ),
            html.Div(
                [
                    html.Div("Forecast For Next Month"),
                    html.H4("N/A" if np.isnan(next_month_forecast) else f"GBP {next_month_forecast:,.0f}"),
                ],
                style=card_style,
            ),
            html.Div(
                [
                    html.Div("Forecast Failure Risk"),
                    html.H4(
                        "N/A"
                        if np.isnan(next_month_failure_prob)
                        else f"{next_month_failure_prob:.0%} ({risk_label})"
                    ),
                ],
                style=card_style,
                className=risk_class,
            ),
            html.Div([html.Div("Main Driver"), html.H4(selected_factor)], style=card_style),
        ],
        style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(150px, 1fr))", "gap": "10px"},
        className="kpi-grid",
    )

    # Calculate risk alerts based on active filters so cards refresh with selections.
    risk_alerts = calculate_risk_alerts(programme_view, "predicted_failure_likelihood_linear")
    
    def _get_alert_card_class(status: str) -> str:
        if status == "Red":
            return "risk-alert-red"
        elif status == "Amber":
            return "risk-alert-amber"
        return "risk-alert-green"

    supplier_delay_children: list[Component] = []
    supplier_delay_children.append(html.H4("Supplier Delay Risk", style={"margin": "0 0 8px 0"}))
    supplier_delay_children.append(html.Div(risk_alerts["supplier_delay_risk"]["status"], className="risk-status", style={"fontSize": "14px", "fontWeight": "600", "marginBottom": "8px"}))
    supplier_delay_children.append(
        html.Ul(
            [
                html.Li(indicator, style={"fontSize": "12px", "margin": "4px 0"})
                for indicator in risk_alerts["supplier_delay_risk"]["indicators"]
            ],
            style={"paddingLeft": "20px", "margin": "0"},
        ) if risk_alerts["supplier_delay_risk"]["indicators"] else html.P("No alerts", style={"fontSize": "12px", "color": "#666", "margin": 0})
    )

    cost_volatility_children: list[Component] = []
    cost_volatility_children.append(html.H4("Cost Volatility", style={"margin": "0 0 8px 0"}))
    cost_volatility_children.append(html.Div(risk_alerts["cost_volatility"]["status"], className="risk-status", style={"fontSize": "14px", "fontWeight": "600", "marginBottom": "8px"}))
    cost_volatility_children.append(
        html.Ul(
            [
                html.Li(indicator, style={"fontSize": "12px", "margin": "4px 0"})
                for indicator in risk_alerts["cost_volatility"]["indicators"]
            ],
            style={"paddingLeft": "20px", "margin": "0"},
        ) if risk_alerts["cost_volatility"]["indicators"] else html.P("No alerts", style={"fontSize": "12px", "color": "#666", "margin": 0})
    )

    demand_spike_children: list[Component] = []
    demand_spike_children.append(html.H4("Demand Spike Risk", style={"margin": "0 0 8px 0"}))
    demand_spike_children.append(html.Div(risk_alerts["demand_spike"]["status"], className="risk-status", style={"fontSize": "14px", "fontWeight": "600", "marginBottom": "8px"}))
    demand_spike_children.append(
        html.Ul(
            [
                html.Li(indicator, style={"fontSize": "12px", "margin": "4px 0"})
                for indicator in risk_alerts["demand_spike"]["indicators"]
            ],
            style={"paddingLeft": "20px", "margin": "0"},
        ) if risk_alerts["demand_spike"]["indicators"] else html.P("No alerts", style={"fontSize": "12px", "color": "#666", "margin": 0})
    )
    
    supplier_delay_card = html.Div(
        supplier_delay_children,
        className=_get_alert_card_class(risk_alerts["supplier_delay_risk"]["status"]),
        style={
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "padding": "12px",
            "backgroundColor": "#fafafa",
        }
    )
    cost_volatility_card = html.Div(
        cost_volatility_children,
        className=_get_alert_card_class(risk_alerts["cost_volatility"]["status"]),
        style={
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "padding": "12px",
            "backgroundColor": "#fafafa",
        }
    )
    demand_spike_card = html.Div(
        demand_spike_children,
        className=_get_alert_card_class(risk_alerts["demand_spike"]["status"]),
        style={
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "padding": "12px",
            "backgroundColor": "#fafafa",
        }
    )
    risk_alert_card_children: list[Component] = [supplier_delay_card, cost_volatility_card, demand_spike_card]
    risk_alert_cards = html.Div(
        risk_alert_card_children,
        style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(200px, 1fr))", "gap": "15px", "marginBottom": "20px"}
    )

    commercial_manager_summary_text = build_commercial_manager_summary(
        supplier_analysis_view=supplier_analysis_view,
        contract_summary=contract_summary,
        profile_summary=profile_summary,
        commodity_summary=commodity_summary,
        supplier_watchlist=supplier_watchlist,
        risk_alerts=risk_alerts,
    )
    commercial_manager_summary_card = build_persona_summary_card(
        commercial_manager_summary_text,
        accent="#bfdbfe",
        background="#eff6ff",
    )

    cfo_summary_text = build_cfo_summary(cfo_view)
    cfo_summary_card = build_persona_summary_card(
        cfo_summary_text,
        accent="#c7d2fe",
        background="#f8fafc",
    )

    programme_director_blocks: list[Component] = [
        programme_director_summary_card,
        html.H4("KPI Cards"),
        kpi_cards,
        html.Br(),
        html.H4("Key Risk & Alert Summary"),
        risk_alert_cards,
        html.Br(),
        html.H4("Forecast Fade and Confidence"),
        html.Br(),
        html.H4("Top 10 Drivers of Forecast Failure"),
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Rank"),
                            html.Th("Driver"),
                            html.Th("Reason"),
                            html.Th("Impact Explanation"),
                            html.Th("Impact"),
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(str(row["rank"])),
                                html.Td(row["driver"]),
                                html.Td(row["reason"]),
                                html.Td(row["impact_explanation"]),
                                html.Td(row["impact"]),
                            ],
                            className=row["impact_class"],
                        )
                        for _, row in top_factors_with_loadings.iterrows()
                    ]
                ),
            ],
            className="drivers-table",
        ),
        html.Br(),
        html.Div(
            [
                html.H4("Top 10 Drivers Impacting Forecast Accuracy", style={"margin": "0 0 16px 0"}),
                dcc.Graph(figure=top10_factors_fig),
            ],
            style={
                "backgroundColor": "#f3e8ff",
                "padding": "16px",
                "borderRadius": "8px",
                "marginBottom": "20px",
                "border": "1px solid #e9d5ff",
            }
        ),
        action_items,
    ]
    if revision_risk_fig is not None or confidence_band_fig is not None:
        fade_row_children = []
        if revision_risk_fig is not None:
            fade_row_children.append(html.Div(dcc.Graph(figure=revision_risk_fig), style={"minWidth": 0}))
        if confidence_band_fig is not None:
            fade_row_children.append(html.Div(dcc.Graph(figure=confidence_band_fig), style={"minWidth": 0}))
        programme_director_blocks.insert(
            6,
            html.Div(
                fade_row_children,
                style={"display": "grid", "gridTemplateColumns": f"repeat({len(fade_row_children)}, minmax(0, 1fr))", "gap": "14px", "marginBottom": "14px"},
            ),
        )

    commercial_manager_blocks: list[Component] = [
        html.H4("Commercial Manager View"),
        html.P(
            f"This tab uses scored outcomes for {supplier_scope} to show how supplier choices, material categories and commercial terms shift failed proposal probability.",
            style={"marginBottom": "12px", "color": "#4b5563"},
        ),
        commercial_manager_summary_card,
    ]
    if supplier_cards:
        commercial_manager_blocks.append(
            html.Div(
                supplier_cards,
                style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(180px, 1fr))", "gap": "10px", "marginBottom": "16px"},
                className="kpi-grid",
            )
        )
    if filter_note:
        commercial_manager_blocks.append(
            html.P(filter_note, style={"marginBottom": "10px", "color": "#9a3412", "fontWeight": "600"})
        )
    if supplier_option_candidates:
        commercial_manager_blocks.extend(
            [
                html.H4("Lower-risk supplier options from current data"),
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Decision lever"),
                                    html.Th("Recommended option"),
                                    html.Th("Avg failed proposal probability"),
                                    html.Th("Impact vs current average"),
                                    html.Th("Why it matters"),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(row["decision_lever"]),
                                        html.Td(row["recommended_option"]),
                                        html.Td(row["avg_failed_probability"]),
                                        html.Td(row["impact_vs_average"]),
                                        html.Td(row["business_readout"]),
                                    ]
                                )
                                for row in supplier_option_candidates
                            ]
                        ),
                    ],
                    className="business-table",
                ),
                html.Br(),
            ]
        )
    else:
        commercial_manager_blocks.append(
            html.Div(
                "Supplier attributes are not available in the current dataset yet, so supplier option comparisons cannot be calculated.",
                style={"padding": "12px", "backgroundColor": "#fff7ed", "border": "1px solid #fdba74", "borderRadius": "8px", "marginBottom": "16px"},
            )
        )

    for left_fig, right_fig in [
        (contract_fig, profile_fig),
        (commodity_fig, supplier_watchlist_fig),
        (supplier_heatmap_fig, None),
    ]:
        row_children = []
        if left_fig is not None:
            row_children.append(html.Div(dcc.Graph(figure=left_fig), style={"minWidth": 0}))
        if right_fig is not None:
            row_children.append(html.Div(dcc.Graph(figure=right_fig), style={"minWidth": 0}))
        if row_children:
            commercial_manager_blocks.append(
                html.Div(
                    row_children,
                    style={"display": "grid", "gridTemplateColumns": f"repeat({len(row_children)}, minmax(0, 1fr))", "gap": "14px", "marginBottom": "14px"},
                )
            )

    cfo_blocks: list[Component] = [
        html.H4("CFO View"),
        html.P(
            "Portfolio-level risk exposure and funding confidence, with focus on expected at-risk spend and where intervention releases investment headroom.",
            style={"marginBottom": "12px", "color": "#4b5563"},
        ),
        cfo_summary_card,
    ]
    if cfo_programme_fig is not None:
        cfo_blocks.append(html.Div(dcc.Graph(figure=cfo_programme_fig), style={"marginBottom": "14px"}))
    if cfo_risk_spend_scatter is not None:
        cfo_blocks.append(html.Div(dcc.Graph(figure=cfo_risk_spend_scatter), style={"marginBottom": "14px"}))
    if cfo_programme_fig is None and cfo_risk_spend_scatter is None:
        cfo_blocks.append(html.P("CFO-specific exposure visuals need Programme_ID and Forecast_Spend data in the scored set."))

    risk_alerts_data = [
        {
            "factor": top_factors[i],
            "reason": top_factors_with_loadings.iloc[i]["reason"],
            "impact": top_factors_with_loadings.iloc[i]["impact_explanation"],
        }
        for i in range(min(5, len(top_factors)))
    ]
    project_controls_summary_text = build_project_controls_summary(
        programme_view=programme_view,
        risk_alerts=risk_alerts,
        top_risk_drivers=risk_alerts_data,
    )
    project_controls_summary_card = build_persona_summary_card(
        project_controls_summary_text,
        accent="#bae6fd",
        background="#f0f9ff",
    )
    
    project_controls_blocks: list[Component] = [
        html.H4("Project Controls Lead View"),
        html.P(
            "Control-focused diagnostics to understand behaviour causing forecast fade and where data/process discipline should be tightened.",
            style={"marginBottom": "12px", "color": "#4b5563"},
        ),
        project_controls_summary_card,
    ]
    if operations_risk_trend_fig is not None:
        project_controls_blocks.append(
            html.Div(dcc.Graph(figure=operations_risk_trend_fig), style={"marginBottom": "14px"})
        )
    if controls_stability_fig is not None or controls_leadtime_fig is not None:
        controls_row_children = []
        if controls_stability_fig is not None:
            controls_row_children.append(html.Div(dcc.Graph(figure=controls_stability_fig), style={"minWidth": 0}))
        if controls_leadtime_fig is not None:
            controls_row_children.append(html.Div(dcc.Graph(figure=controls_leadtime_fig), style={"minWidth": 0}))
        project_controls_blocks.append(
            html.Div(
                controls_row_children,
                style={"display": "grid", "gridTemplateColumns": f"repeat({len(controls_row_children)}, minmax(0, 1fr))", "gap": "14px", "marginBottom": "14px"},
            )
        )
    project_controls_blocks.extend([
        html.H4("Supplier / Region Risk Snapshot"),
        dcc.Graph(figure=ops_fig if ops_fig is not None else hist_fig),
        html.H4("Top 5 Risk Drivers & Explanations"),
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Risk Driver"),
                            html.Th("Reason"),
                            html.Th("Impact Explanation"),
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(alert["factor"]),
                                html.Td(alert["reason"]),
                                html.Td(alert["impact"]),
                            ]
                        )
                        for alert in risk_alerts_data
                    ]
                ),
            ],
            className="business-table",
        ),
    ])

    graphs = [
        dcc.Tabs(
            [
                dcc.Tab(label="Programme Director", className="tab-item", selected_className="tab-item--selected", children=programme_director_blocks),
                dcc.Tab(label="Commercial Manager", className="tab-item", selected_className="tab-item--selected", children=commercial_manager_blocks),
                dcc.Tab(label="CFO", className="tab-item", selected_className="tab-item--selected", children=cfo_blocks),
                dcc.Tab(label="Project Controls Lead", className="tab-item", selected_className="tab-item--selected", children=project_controls_blocks),
            ],
            className="main-tabs",
        )
    ]

    return graphs


app = dash.Dash(__name__)
app.title = "PCA Linear Forecast Dashboard"

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div("Forecast Intelligence", className="title-eyebrow"),
                html.H2("PCA + Linear Regression Dashboard", className="title-main"),
                html.P(
                    "Upload a CSV, choose a target column, and generate executive, drill-down, and operations insights.",
                    className="title-sub",
                ),
            ],
            className="header-card",
        ),
        html.Div(
            [
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and drop or ", html.A("select a CSV file")]),
                    style={
                        "width": "100%",
                        "height": "64px",
                        "lineHeight": "64px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": "10px",
                        "textAlign": "center",
                        "marginBottom": "16px",
                    },
                    className="upload-zone",
                    multiple=False,
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Business Outcome To Track"),
                                dcc.Dropdown(id="target-column", placeholder="Select a target column"),
                            ],
                            className="control-block",
                        ),
                    ],
                    className="control-grid",
                ),
                html.Details(
                    [
                        html.Summary("⚙️ Model Settings", style={"cursor": "pointer", "fontWeight": "600", "fontSize": "14px", "padding": "8px"}),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Learning Window (How much history to learn from)"),
                                        dcc.Slider(
                                            id="train-frac",
                                            min=0.5,
                                            max=0.95,
                                            step=0.05,
                                            value=0.8,
                                            marks={0.5: "50%", 0.7: "70%", 0.8: "80%", 0.9: "90%"},
                                        ),
                                    ],
                                    className="slider-block",
                                ),
                                html.Div(
                                    [
                                        html.Label("At-Risk Trigger (when to flag forecasts as at risk)"),
                                        dcc.Slider(
                                            id="threshold",
                                            min=0.05,
                                            max=0.95,
                                            step=0.01,
                                            value=0.5,
                                            marks={0.1: "10%", 0.3: "30%", 0.5: "50%", 0.7: "70%", 0.9: "90%"},
                                        ),
                                    ],
                                    className="slider-block",
                                ),
                            ],
                            style={"padding": "12px", "marginTop": "8px"}
                        ),
                    ],
                    id="settings-collapsible",
                    style={"border": "1px solid #e5e7eb", "borderRadius": "8px", "marginBottom": "12px"}
                ),
                html.Div(
                    [
                        html.Button("Run workflow", id="run-button", n_clicks=0, className="btn btn-primary"),
                        html.Button(
                            "Download predictions CSV",
                            id="download-button",
                            n_clicks=0,
                            className="btn btn-secondary",
                        ),
                        html.Button("View detailed technical analysis", id="open-tech-modal", n_clicks=0, className="btn btn-light"),
                    ],
                    className="button-row",
                ),
                dcc.Download(id="download-csv"),
            ],
            className="controls-card",
            id="controls-section",
        ),
        html.Div(id="status", className="status-banner"),
        html.Pre(id="metrics", className="metrics-panel hidden-panel"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Technical Details"),
                                html.Button("Close", id="close-tech-modal", n_clicks=0, className="btn btn-secondary"),
                            ],
                            className="modal-header",
                        ),
                        html.P("Detailed data and model diagnostics are shown below for technical teams."),
                        html.Pre(id="tech-modal-content", className="metrics-panel"),
                    ],
                    className="modal-content",
                )
            ],
            id="tech-modal",
            className="modal-overlay",
            style={"display": "none"},
        ),
        # Filters section - appears just above views after workflow runs
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Programme Filter",
                            title="Filters non-CFO views to one programme. Leave blank for full portfolio.",
                            style={"fontWeight": "600", "marginBottom": "6px", "display": "block"},
                        ),
                        dcc.Dropdown(id="programme-filter", placeholder="Select Programme ID"),
                    ],
                    style={"maxWidth": "300px"}
                ),
                html.Div(
                    [
                        html.Label(
                            "Region Filter",
                            title="Shows only records for the selected region.",
                            style={"fontWeight": "600", "marginBottom": "6px", "display": "block"},
                        ),
                        dcc.Dropdown(id="region-filter", placeholder="Select Region"),
                    ],
                    style={"maxWidth": "220px"}
                ),
                html.Div(
                    [
                        html.Label(
                            "Contract Type",
                            title="Keeps records only for the selected contract type.",
                            style={"fontWeight": "600", "marginBottom": "6px", "display": "block"},
                        ),
                        dcc.Dropdown(id="contract-filter", placeholder="Select Contract Type"),
                    ],
                    style={"maxWidth": "260px"}
                ),
                html.Div(
                    [
                        html.Label(
                            "Supplier Profile",
                            title="Filters to suppliers in the chosen profile category.",
                            style={"fontWeight": "600", "marginBottom": "6px", "display": "block"},
                        ),
                        dcc.Dropdown(id="supplier-profile-filter", placeholder="Select Supplier Profile"),
                    ],
                    style={"maxWidth": "260px"}
                ),
                html.Div(
                    [
                        html.Label(
                            "Minimum Risk Filter",
                            title="Only keeps records where predicted failure probability is greater than or equal to this threshold. 0% means no risk cutoff.",
                            style={"fontWeight": "600", "marginBottom": "6px", "display": "block"},
                        ),
                        dcc.Slider(
                            id="min-risk-filter",
                            min=0.0,
                            max=0.9,
                            step=0.05,
                            value=0.0,
                            marks={0.0: "0%", 0.3: "30%", 0.5: "50%", 0.7: "70%", 0.9: "90%"},
                        ),
                    ],
                    style={"minWidth": "260px", "maxWidth": "320px"}
                ),
                html.Div(
                    [
                        html.Label(
                            "Quick Actions",
                            title="Reset all filters back to portfolio-wide defaults.",
                            style={"fontWeight": "600", "marginBottom": "6px", "display": "block"},
                        ),
                        html.Button("Reset all filters", id="reset-filters-button", n_clicks=0, className="btn btn-light"),
                    ],
                    style={"maxWidth": "220px"}
                ),
            ],
            id="filters-section",
            style={
                "display": "none",
                "padding": "12px",
                "backgroundColor": "#f8fafc",
                "borderRadius": "8px",
                "marginBottom": "16px",
                "border": "1px solid #e5e7eb",
                "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                "gap": "12px",
                "alignItems": "end",
            }
        ),
        html.Div(id="graphs", className="graphs-panel"),
        dcc.Store(id="dataset-store"),
        dcc.Store(id="predictions-store"),
        dcc.Store(id="workflow-store"),
    ],
    className="page-shell",
)


@app.callback(
    Output("dataset-store", "data"),
    Output("target-column", "options"),
    Output("target-column", "value"),
    Output("programme-filter", "options"),
    Output("programme-filter", "value"),
    Output("region-filter", "options"),
    Output("region-filter", "value"),
    Output("contract-filter", "options"),
    Output("contract-filter", "value"),
    Output("supplier-profile-filter", "options"),
    Output("supplier-profile-filter", "value"),
    Output("status", "children"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename"),
)
def load_dataset(contents, filename):
    try:
        if contents is None:
            if not DEFAULT_INPUT_PATH.exists():
                return None, [], None, [], None, [], None, [], None, [], None, "Upload a CSV to start."
            df = pd.read_csv(DEFAULT_INPUT_PATH)
            source = str(DEFAULT_INPUT_PATH)
        else:
            df = decode_uploaded_csv(contents)
            source = filename or "uploaded file"

        enriched_for_filters = prepare_dataframe(df)
        options = target_options_from_df(df)
        programme_options = programme_options_from_df(df)
        region_options = category_options_from_df(enriched_for_filters, "Region")
        contract_options = category_options_from_df(enriched_for_filters, "Contract_Type")
        supplier_profile_options = category_options_from_df(enriched_for_filters, "Supplier_Profile")
        default_target = DEFAULT_TARGET if any(o["value"] == DEFAULT_TARGET for o in options) else (
            options[0]["value"] if options else None
        )
        default_programme = None
        default_region = None
        default_contract = None
        default_supplier_profile = None

        status = f"Loaded {source}: {len(df):,} rows"
        return (
            df.to_json(date_format="iso", orient="split"),
            options,
            default_target,
            programme_options,
            default_programme,
            region_options,
            default_region,
            contract_options,
            default_contract,
            supplier_profile_options,
            default_supplier_profile,
            status,
        )
    except Exception as exc:
        return None, [], None, [], None, [], None, [], None, [], None, f"Error loading dataset: {exc}"


@app.callback(
    Output("metrics", "children"),
    Output("graphs", "children"),
    Output("predictions-store", "data"),
    Output("workflow-store", "data"),
    Output("status", "children", allow_duplicate=True),
    Output("settings-collapsible", "open"),
    Output("filters-section", "style"),
    Input("run-button", "n_clicks"),
    State("dataset-store", "data"),
    State("target-column", "value"),
    State("programme-filter", "value"),
    State("region-filter", "value"),
    State("contract-filter", "value"),
    State("supplier-profile-filter", "value"),
    State("min-risk-filter", "value"),
    State("train-frac", "value"),
    State("threshold", "value"),
    prevent_initial_call=True,
)
def run_model(
    n_clicks,
    data_json,
    target_col,
    selected_programme,
    selected_region,
    selected_contract_type,
    selected_supplier_profile,
    min_risk_filter,
    train_frac,
    threshold,
):
    _ = n_clicks
    try:
        if not data_json:
            return "", [], None, None, "No dataset available. Upload a file first.", True, {"display": "none"}
        if not target_col:
            return "", [], None, None, "Select a target column before running.", True, {"display": "none"}

        df = pd.read_json(io.StringIO(data_json), orient="split")
        result = run_workflow(
            df=df,
            target_col=target_col,
            n_components=FIXED_N_COMPONENTS,
            train_frac=float(train_frac),
            threshold=float(threshold),
        )

        metrics = result["metrics"]
        selected = result["selected_info"]

        metrics_text = json.dumps(metrics, indent=2)
        graphs = build_graphs(
            result,
            target_col=target_col,
            selected_programme=selected_programme,
            selected_region=selected_region,
            selected_contract_type=selected_contract_type,
            selected_supplier_profile=selected_supplier_profile,
            min_risk_filter=float(min_risk_filter if min_risk_filter is not None else FILTER_DEFAULT_MIN_RISK),
        )
        workflow_payload = serialize_workflow_result(result)

        status = f"Insights ready for {target_col}. Use the tabs for business views and open detailed analysis if needed."

        predictions_json = result["predictions_df"].to_json(date_format="iso", orient="split")
        
        # Collapse settings and show filters after workflow runs
        filters_style = {
            "display": "grid",
            "padding": "12px",
            "backgroundColor": "#f8fafc",
            "borderRadius": "8px",
            "marginBottom": "16px",
            "border": "1px solid #e5e7eb",
            "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
            "gap": "12px",
            "alignItems": "end",
        }
        
        return metrics_text, graphs, predictions_json, workflow_payload, status, False, filters_style
    except Exception as exc:
        return "", [], None, None, f"Error running workflow: {exc}", True, {"display": "none"}


@app.callback(
    Output("graphs", "children", allow_duplicate=True),
    Input("programme-filter", "value"),
    Input("region-filter", "value"),
    Input("contract-filter", "value"),
    Input("supplier-profile-filter", "value"),
    Input("min-risk-filter", "value"),
    State("workflow-store", "data"),
    State("target-column", "value"),
    prevent_initial_call=True,
)
def refresh_graphs_for_programme(
    selected_programme,
    selected_region,
    selected_contract_type,
    selected_supplier_profile,
    min_risk_filter,
    workflow_payload,
    target_col,
):
    if not workflow_payload:
        return dash.no_update

    result = deserialize_workflow_result(workflow_payload)
    active_target = target_col or result.get("metrics", {}).get("target_column", DEFAULT_TARGET)
    return build_graphs(
        result,
        target_col=active_target,
        selected_programme=selected_programme,
        selected_region=selected_region,
        selected_contract_type=selected_contract_type,
        selected_supplier_profile=selected_supplier_profile,
        min_risk_filter=float(min_risk_filter if min_risk_filter is not None else FILTER_DEFAULT_MIN_RISK),
    )


@app.callback(
    Output("programme-filter", "value", allow_duplicate=True),
    Output("region-filter", "value", allow_duplicate=True),
    Output("contract-filter", "value", allow_duplicate=True),
    Output("supplier-profile-filter", "value", allow_duplicate=True),
    Output("min-risk-filter", "value", allow_duplicate=True),
    Input("reset-filters-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(n_clicks):
    _ = n_clicks
    return None, None, None, None, FILTER_DEFAULT_MIN_RISK


@app.callback(
    Output("tech-modal", "style"),
    Output("tech-modal-content", "children"),
    Input("open-tech-modal", "n_clicks"),
    Input("close-tech-modal", "n_clicks"),
    State("metrics", "children"),
    prevent_initial_call=True,
)
def toggle_tech_modal(open_clicks, close_clicks, metrics_text):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    if trigger == "open-tech-modal":
        return {"display": "flex"}, metrics_text or "No technical details available yet."
    return {"display": "none"}, metrics_text or ""


@app.callback(
    Output("download-csv", "data"),
    Input("download-button", "n_clicks"),
    State("predictions-store", "data"),
    prevent_initial_call=True,
)
def download_predictions(n_clicks, predictions_json):
    _ = n_clicks
    if not predictions_json:
        return None

    df = pd.read_json(io.StringIO(predictions_json), orient="split")
    return dcc.send_data_frame(df.to_csv, "workflow_linear_predictions_uploaded.csv", index=False)


if __name__ == "__main__":
    app.run(debug=True)


