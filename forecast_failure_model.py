import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def extract_days_from_terms(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)", str(value))
    if not match:
        return np.nan
    return float(match.group(1))


def load_data(
    forecast_path: str = "data/forecast_data.csv",
    supplier_path: str = "data/supplier_attributes.csv",
) -> pd.DataFrame:
    df = pd.read_csv(forecast_path)

    df["Forecast_Version_Date"] = pd.to_datetime(df["Forecast_Version_Date"], errors="coerce")
    df["Forecast_Period_End_Date"] = pd.to_datetime(df["Forecast_Period_End_Date"], errors="coerce")

    df["event_id"] = (
        df["Programme_ID"].astype(str)
        + "|"
        + df["Supplier_ID"].astype(str)
        + "|"
        + df["Commodity"].astype(str)
        + "|"
        + df["Forecast_Period"].astype(str)
    )

    df["forecast_version_month"] = df["Forecast_Version_Date"].dt.month
    df["forecast_version_quarter"] = df["Forecast_Version_Date"].dt.quarter
    df["forecast_period_month"] = pd.to_datetime(df["Forecast_Period"] + "-01", errors="coerce").dt.month

    # Safe features available at forecast version time.
    df["forecast_change_pct"] = np.where(
        df["Previous_Forecast_Spend"].fillna(0) != 0,
        df["Forecast_Change"] / df["Previous_Forecast_Spend"],
        0.0,
    )

    df["forecast_to_committed_ratio"] = np.where(
        df["Committed_Spend"].fillna(0) != 0,
        df["Forecast_Spend"] / df["Committed_Spend"],
        np.nan,
    )

    supplier_file = Path(supplier_path)
    if supplier_file.exists():
        supplier_df = pd.read_csv(supplier_file)
        supplier_df["Supplier_ID"] = supplier_df["Supplier_ID"].astype(str).str.strip()
        supplier_df = supplier_df.drop_duplicates(subset=["Supplier_ID"], keep="first")

        supplier_df["Payment_Terms_Days"] = supplier_df["Payment_Terms"].apply(extract_days_from_terms)
        supplier_df["Strategic_Flag_Num"] = supplier_df["Strategic_Flag"].map({"Yes": 1, "No": 0})
        supplier_df["New_Supplier_Flag_Num"] = supplier_df["New_Supplier_Flag"].map({"Yes": 1, "No": 0})

        df = df.merge(supplier_df, on="Supplier_ID", how="left")

    return df


def split_by_event_time(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_dates = df.groupby("event_id")["Forecast_Period_End_Date"].max().sort_values()

    cutoff = int(len(event_dates) * train_frac)
    train_events = set(event_dates.iloc[:cutoff].index)
    test_events = set(event_dates.iloc[cutoff:].index)

    train_df = df[df["event_id"].isin(train_events)].copy()
    test_df = df[df["event_id"].isin(test_events)].copy()

    return train_df, test_df


def build_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, categorical_features),
            ("num", numeric_pipe, numeric_features),
        ]
    )

    model = LogisticRegression(
        max_iter=2500,
        class_weight="balanced",
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast failure likelihood model with supplier attributes")
    parser.add_argument("--forecast-path", default="data/forecast_data.csv")
    parser.add_argument("--supplier-path", default="data/supplier_attributes.csv")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="forecast_failure_scored.csv")

    args = parser.parse_args()

    df = load_data(args.forecast_path, args.supplier_path)
    target = "Forecast_Failed_Flag"

    excluded = {
        target,
        "Actual_Spend",
        "Variance",
        "Absolute_Error",
        "Actual_Minus_Committed",
        "Forecast_Period_End_Date",
        "event_id",
    }

    excluded = {

    }

    candidate_features = [c for c in df.columns if c not in excluded]

    categorical_features = [
        "Programme_ID",
        "Commodity",
        "Supplier_ID",
        "Forecast_Period",
        "Forecast_Change_Direction",
        "Confidence_Band",
        "Contract_Type",
        "Supplier_Profile",
        "Region",
        "Payment_Terms",
    ]

    numeric_features = [
        "Forecast_Spend",
        "Revision_Number",
        "Previous_Forecast_Spend",
        "Forecast_Change",
        "Forecast_Stability_Score",
        "Days_Before_Period",
        "Committed_Spend",
        "Commitment_Ratio",
        "PO_Count",
        "Programme_Change_Count",
        "Programme_Scope_Churn_Index",
        "Programme_Change_Impact_Index",
        "forecast_version_month",
        "forecast_version_quarter",
        "forecast_period_month",
        "forecast_change_pct",
        "forecast_to_committed_ratio",
        "OTIF_Pct",
        "Avg_Lead_Time_Days",
        "Quality_Incidents_YTD",
        "Payment_Terms_Days",
        "Strategic_Flag_Num",
        "New_Supplier_Flag_Num",
    ]

    categorical_features = [c for c in categorical_features if c in candidate_features]
    numeric_features = [c for c in numeric_features if c in candidate_features]

    train_df, test_df = split_by_event_time(df, train_frac=args.train_frac)

    X_train = train_df[categorical_features + numeric_features]
    y_train = train_df[target]
    X_test = test_df[categorical_features + numeric_features]
    y_test = test_df[target]

    pipeline = build_pipeline(categorical_features, numeric_features)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    print("\nEvaluation")
    print("----------")
    print(f"Train rows:        {len(train_df):,}")
    print(f"Test rows:         {len(test_df):,}")
    print(f"Features used:     {len(categorical_features) + len(numeric_features)}")
    print(f"ROC AUC:           {roc_auc_score(y_test, proba):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, proba):.4f}")
    print(f"Brier Score:       {brier_score_loss(y_test, proba):.4f}")
    print()
    print(classification_report(y_test, pred, digits=4))

    scored = test_df.copy()
    scored["predicted_failure_probability"] = proba
    scored["predicted_failure_flag"] = pred
    scored.to_csv(args.output, index=False)

    print(f"Scored test set saved to: {args.output}")

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False)

    print("\nTop 20 features by absolute impact:")
    print(coef_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

