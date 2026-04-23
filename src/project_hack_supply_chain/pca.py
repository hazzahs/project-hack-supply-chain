import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .paths import DATA_DIR


def clean_data(file_path: str = str(DATA_DIR / "forecast_data.csv")) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df["Forecast_Version_Date"] = pd.to_datetime(df["Forecast_Version_Date"], errors="coerce")
    df["Forecast_Period_End_Date"] = pd.to_datetime(df["Forecast_Period_End_Date"], errors="coerce")

    df["forecast_version_month"] = df["Forecast_Version_Date"].dt.month
    df["forecast_version_quarter"] = df["Forecast_Version_Date"].dt.quarter
    df["forecast_period_month"] = pd.to_datetime(
        df["Forecast_Period"] + "-01", errors="coerce"
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


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    target = "Forecast_Failed_Flag"

    # Exclude leakage and post-outcome columns
    excluded = {
        target,
        "Actual_Spend",
        "Variance",
        "Absolute_Error",
        "Actual_Minus_Committed",
        "Forecast_Period_End_Date",
    }

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

    return [c for c in numeric_features if c in df.columns and c not in excluded]


def build_pca_pipeline(n_components: float | int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )


def parse_n_components(value: str) -> float | int:
    try:
        return int(value)
    except ValueError:
        return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA analysis for forecast failure data")
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "forecast_data.csv"),
        help="Path to forecast CSV file",
    )
    parser.add_argument(
        "--n-components",
        default="0.95",
        help="PCA n_components as int (count) or float (variance target)",
    )
    parser.add_argument(
        "--top-n-loadings",
        type=int,
        default=10,
        help="Top absolute loadings to show per component",
    )
    parser.add_argument(
        "--output-prefix",
        default="pca_output",
        help="Prefix for output CSV files",
    )

    args = parser.parse_args()
    n_components = parse_n_components(args.n_components)

    df = clean_data(args.input)
    feature_cols = get_numeric_features(df)

    if not feature_cols:
        raise ValueError("No numeric features available for PCA after filtering.")

    X = df[feature_cols].copy()

    pipeline = build_pca_pipeline(n_components=n_components)
    scores = pipeline.fit_transform(X)

    pca = pipeline.named_steps["pca"]
    component_names = [f"PC{i + 1}" for i in range(scores.shape[1])]

    explained_df = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=component_names,
    )

    top_rows: list[dict] = []
    for component in component_names:
        top_features = (
            loadings_df[component].abs().sort_values(ascending=False).head(args.top_n_loadings)
        )
        for feature_name in top_features.index:
            loading = loadings_df.loc[feature_name, component]
            top_rows.append(
                {
                    "component": component,
                    "feature": feature_name,
                    "loading": loading,
                    "abs_loading": abs(loading),
                }
            )

    top_loadings_df = pd.DataFrame(top_rows).sort_values(
        ["component", "abs_loading"], ascending=[True, False]
    )

    scores_df = pd.DataFrame(scores, columns=component_names, index=df.index)
    if "Forecast_Failed_Flag" in df.columns:
        scores_df["Forecast_Failed_Flag"] = df["Forecast_Failed_Flag"].values

    print("\nPCA completed")
    print("-------------")
    print(f"Rows processed: {len(df):,}")
    print(f"Numeric features used ({len(feature_cols)}): {feature_cols}")
    print(f"Components retained: {len(component_names)}")

    print("\nExplained variance by component:")
    print(explained_df.to_string(index=False))

    print("\nTop feature loadings per component:")
    print(top_loadings_df.to_string(index=False))

    if "Forecast_Failed_Flag" in scores_df.columns:
        group_means = scores_df.groupby("Forecast_Failed_Flag")[component_names].mean()
        print("\nMean principal component scores by Forecast_Failed_Flag:")
        print(group_means.to_string())

    explained_path = f"{args.output_prefix}_variance.csv"
    loadings_path = f"{args.output_prefix}_loadings.csv"
    scores_path = f"{args.output_prefix}_scores.csv"
    top_loadings_path = f"{args.output_prefix}_top_loadings.csv"

    explained_df.to_csv(explained_path, index=False)
    loadings_df.to_csv(loadings_path)
    scores_df.to_csv(scores_path, index=False)
    top_loadings_df.to_csv(top_loadings_path, index=False)

    print("\nSaved output files:")
    print(f"- {explained_path}")
    print(f"- {loadings_path}")
    print(f"- {scores_path}")
    print(f"- {top_loadings_path}")


if __name__ == "__main__":
    main()

