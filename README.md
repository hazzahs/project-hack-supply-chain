# project-hack-supply-chain

## Project Overview

## Forecast Failure Logistic Model (with Supplier Features)

Train a logistic regression model for `Forecast_Failed_Flag` using forecast-time-safe fields plus supplier attributes from `data/supplier_attributes.csv`.

```powershell
python .\forecast_failure_model.py
python .\forecast_failure_model.py --train-frac 0.8 --threshold 0.45 --output scored_with_supplier_features.csv
```

## PCA Analysis

Run PCA on forecast features (using forecast-time-safe feature engineering) and inspect components linked to `Forecast_Failed_Flag`.

```powershell
python .\pca_analysis.py
python .\pca_analysis.py --n-components 5 --top-n-loadings 15 --output-prefix forecast_pca
```

The script writes these files:
- `*_variance.csv`
- `*_loadings.csv`
- `*_scores.csv`
- `*_top_loadings.csv`
