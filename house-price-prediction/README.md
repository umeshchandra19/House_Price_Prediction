# House Price Prediction (Python, scikit-learn)

An end-to-end regression project: preprocessing, feature engineering, cross-validated model selection, and a
comparison of **Linear Regression** and a **Decision Tree** (plus a Random Forest as an extra) on 20,640 housing records.

* **Report (3 pages, plain language):** [`report/House_Price_Prediction_Report.pdf`](report/House_Price_Prediction_Report.pdf)

## Results (held-out test data, 4,128 areas)

| Model | Explains (R-squared) | Typical error | Within 20% of true price |
|---|---|---|---|
| Linear Regression | 0.658 | $67k | 49% |
| **Decision Tree (tuned)** - selected by cross-validation | **0.746** | **$57k** | **62%** |
| Decision Tree (untuned) | 0.617 | $71k | 59% |
| Random Forest (extra, not part of the selection) | 0.822 | $48k | 71% |

* An untuned tree over-fits badly (training R-squared 1.00 vs 0.63 on new data); limiting its depth fixes most of that.
* The three engineered features (rooms per household, bedrooms per room, people per household) help only slightly (see `results/ablation_features.csv`).
* "Accuracy" for a regression model is not one number, so three are reported: R-squared, typical error in dollars (RMSE) and the share of predictions within 20% of the true price.
* All numbers come from `python -m housepred train` (seeded, deterministic) and are saved in `results/`.

## Quick start

```bash
pip install -r requirements.txt
python -m housepred train            # about 5 minutes on one CPU core; use --quick for a fast smoke test (~1 min)
python -m housepred predict          # predict for an example area
python -m housepred predict --median-income 8.0 --ocean-proximity "NEAR BAY" --latitude 37.8 --longitude -122.3
pytest -q                            # 10 unit tests
python report/build_report.py        # rebuild the PDF from results/
```

`predict` options (all optional, defaults describe an average Los Angeles area): `--longitude --latitude --housing-median-age
--total-rooms --total-bedrooms --population --households --median-income --ocean-proximity`.

## What the pipeline does

1. **Audit** the data (missing values, duplicates, the $500,001 price ceiling) -> `results/data_audit.json`.
2. **Split** 80/20, stratified on income so train and test look alike. The test set is used once, at the end.
3. **Pipeline** (so nothing leaks from validation folds): add 3 ratio features -> fill missing values with the median ->
   scale numbers -> one-hot encode ocean proximity -> model.
4. **Compare** models with 5-fold cross-validation on the training data. The Decision Tree's depth and leaf size are tuned by
   grid search *inside* each fold (nested cross-validation), so its score is not optimistic.
5. **Select** between Linear Regression and the Decision Tree using cross-validation only, then score on the test set once.
6. **Analyse**: feature ablation, depth curve (over-fitting), feature importance, errors by location and price band.

## Data

`data/housing.csv` is the 1990 California census housing data (Pace & Barry, 1997), 20,640 areas x 10 columns. This copy comes
from the `handson-ml2` repository by Aurelien Geron (Apache-2.0): <https://github.com/ageron/handson-ml2>.
Each row is a small neighbourhood, not a single house, and prices above about $500k are capped at $500,001.

## Repository layout

```
housepred/   data.py  features.py  pipeline.py  metrics.py  train.py  plots.py  cli.py
data/        housing.csv
results/     model_comparison.csv  ablation_features.csv  error_analysis.csv  depth_curve.csv  ... figures/
models/      best_model.joblib   (the selected tuned Decision Tree pipeline)
report/      build_report.py + the PDF report
tests/       pytest suite
```

## Limitations

1990 data on neighbourhoods (not individual houses), a price ceiling that limits accuracy on expensive areas, a single random
split plus cross-validation, and location handled only as latitude/longitude. See the report for next steps.
