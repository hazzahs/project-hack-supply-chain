import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

def clean_data(file_path:str = 'data\\forecast_data.csv'):
    df = pd.read_csv(file_path)
    print(df.head(5))
    df["Forecast_Version_Date"] = pd.to_datetime(df["Forecast_Version_Date"])
    df["Forecast_Period_End_Date"] = pd.to_datetime(df["Forecast_Period_End_Date"])

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
    df["forecast_period_month"] = pd.to_datetime(
        df["Forecast_Period"] + "-01"
    ).dt.month

    # Derived features safe at forecast time
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

    return df

def split_by_event_time(df: pd.DataFrame, train_frac: float = 0.8):
    # Sort unique forecast events by end date, then split events
    event_dates = (
        df.groupby("event_id")["Forecast_Period_End_Date"]
        .max()
        .sort_values()
    )

    cutoff = int(len(event_dates) * train_frac)
    train_events = set(event_dates.iloc[:cutoff].index)
    test_events = set(event_dates.iloc[cutoff:].index)

    train_df = df[df["event_id"].isin(train_events)].copy()
    test_df = df[df["event_id"].isin(test_events)].copy()

    return train_df, test_df


def build_pipeline(categorical_features, numeric_features):
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
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline

def main():
    df = clean_data()

    target = "Forecast_Failed_Flag"

    # Exclude leakage and post-outcome columns
    excluded = {
        target,
        "Actual_Spend",
        "Variance",
        "Absolute_Error",
        "Actual_Minus_Committed",
        "Forecast_Period_End_Date",
        "recreated_flag",
        "event_id",
    }

    candidate_features = [c for c in df.columns if c not in excluded]

    categorical_features = [
        "Programme_ID",
        "Commodity",
        "Supplier_ID",
        "Forecast_Period",
        "Forecast_Change_Direction",
        "Confidence_Band",
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
    ]

    # Keep only columns that actually exist
    categorical_features = [c for c in categorical_features if c in candidate_features]
    numeric_features = [c for c in numeric_features if c in candidate_features]

    train_df, test_df = split_by_event_time(df)

    X_train = train_df[categorical_features + numeric_features]
    y_train = train_df[target]

    X_test = test_df[categorical_features + numeric_features]
    y_test = test_df[target]

    pipeline = build_pipeline(categorical_features, numeric_features)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\nEvaluation")
    print("----------")
    print(f"ROC AUC:          {roc_auc_score(y_test, proba):.4f}")
    print(f"Average Precision:{average_precision_score(y_test, proba):.4f}")
    print(f"Brier Score:      {brier_score_loss(y_test, proba):.4f}")
    print()
    print(classification_report(y_test, pred, digits=4))

    # Example: attach probabilities to test rows
    scored = test_df.copy()
    scored["predicted_failure_probability"] = proba
    scored["predicted_failure_flag"] = pred

    print("\nSample scored rows:")
    print(
        scored[
            [
                "Programme_ID",
                "Supplier_ID",
                "Commodity",
                "Forecast_Period",
                "Forecast_Version_Date",
                "Forecast_Failed_Flag",
                "predicted_failure_probability",
                "predicted_failure_flag",
            ]
        ]
        .sort_values("predicted_failure_probability", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    # Optional: inspect logistic regression coefficients
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
    }).sort_values("coefficient", ascending=False)

    print("\nTop features increasing failure odds:")
    print(coef_df.head(15).to_string(index=False))

    print("\nTop features decreasing failure odds:")
    print(coef_df.tail(15).sort_values("coefficient").to_string(index=False))


if __name__ == "__main__":
    main()