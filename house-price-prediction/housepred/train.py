"""Train, cross-validate, compare and evaluate the models.

Order of operations (the test set is only used in step 4):
  1. audit the data and split it 80/20 (stratified on income)
  2. 5-fold cross-validation on the TRAINING part to compare models
  3. tune the Decision Tree's settings inside the training part
  4. one final score of each fitted model on the untouched test part
  5. ablation (are the new features worth it?), error analysis, plots
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.tree import DecisionTreeRegressor

from . import plots
from .data import DEFAULT_DATA, PRICE_CAP, audit, load_data, split
from .metrics import report, within_pct
from .pipeline import build_pipeline

SCORING = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
REQUIRED = ["Linear Regression", "Decision Tree (tuned)"]   # the two models of the original project


def _cv_row(name, model, X, y, cv, train_score=True):
    t0 = time.time()
    s = cross_validate(model, X, y, cv=cv, scoring=SCORING, return_train_score=train_score, n_jobs=1)
    return {"model": name,
            "cv_r2_mean": s["test_r2"].mean(), "cv_r2_std": s["test_r2"].std(),
            "cv_rmse_mean": -s["test_rmse"].mean(), "cv_rmse_std": s["test_rmse"].std(),
            "cv_mae_mean": -s["test_mae"].mean(),
            "train_r2_mean": s["train_r2"].mean() if train_score else float("nan"),
            "cv_seconds": round(time.time() - t0, 1)}


def run(data_path=DEFAULT_DATA, results_dir="results", models_dir="models", seed=42,
        drop_capped=False, quick=False):
    results_dir, models_dir = Path(results_dir), Path(models_dir)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. data ------------------------------------------------------------------------------
    df = load_data(data_path)
    info = audit(df)
    (results_dir / "data_audit.json").write_text(json.dumps(info, indent=2))
    X_train, X_test, y_train, y_test = split(df, seed=seed, drop_capped=drop_capped)
    print(f"Rows: {len(df):,} | train {len(X_train):,} | test {len(X_test):,} | missing: {info['missing_values']}")
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    # 2-3. candidate models ----------------------------------------------------------------
    grid = {"model__max_depth": [3, 5, 7, 9, 11, None] if not quick else [5, 9],
            "model__min_samples_leaf": [1, 10, 30] if not quick else [10]}
    tuned = GridSearchCV(build_pipeline(DecisionTreeRegressor(random_state=seed)), grid, cv=cv,
                         scoring="neg_root_mean_squared_error", n_jobs=1)
    candidates = {
        "Linear Regression (raw features)": build_pipeline(LinearRegression(), engineer=False),
        "Linear Regression": build_pipeline(LinearRegression()),
        "Decision Tree (untuned)": build_pipeline(DecisionTreeRegressor(random_state=seed)),
        "Decision Tree (tuned)": tuned,
        "Random Forest (extra)": build_pipeline(RandomForestRegressor(
            n_estimators=40 if quick else 100, min_samples_leaf=2, random_state=seed, n_jobs=1)),
    }
    cv_rows = []
    for name, model in candidates.items():
        print(f"  cross-validating {name} ...")
        # tuned tree: tuning happens inside every outer fold, so its CV score is honest (nested CV)
        cv_rows.append(_cv_row(name, model, X_train, y_train, cv, train_score="Forest" not in name))
    cv_table = pd.DataFrame(cv_rows)

    # 4. fit on the full training part, score once on the test part ------------------------------
    test_rows, fitted = [], {}
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = (model, pred)
        test_rows.append({"model": name, **{f"test_{k}": v for k, v in report(y_test, pred).items()},
                          "fit_seconds": round(time.time() - t0, 1)})
    test_table = pd.DataFrame(test_rows)
    comparison = cv_table.merge(test_table, on="model")
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)

    best_params = tuned.best_params_
    # pick between the two project models using cross-validation only (never the test set)
    req = comparison[comparison.model.isin(REQUIRED)].sort_values("cv_rmse_mean")
    selected = req.iloc[0]["model"]
    best_model, best_pred = fitted[selected]
    joblib.dump(best_model, models_dir / "best_model.joblib")
    print(f"Selected (by cross-validation): {selected}")

    # 5a. ablation: do the engineered features help? ---------------------------------------------
    depth = best_params["model__max_depth"]
    leaf = best_params["model__min_samples_leaf"]
    abl = []
    for label, est_fn in [("Linear Regression", lambda: LinearRegression()),
                          ("Decision Tree (tuned settings)", lambda: DecisionTreeRegressor(
                              max_depth=depth, min_samples_leaf=leaf, random_state=seed))]:
        for eng in (False, True):
            row = _cv_row(label, build_pipeline(est_fn(), engineer=eng), X_train, y_train, cv)
            abl.append({"model": label, "new_features": "yes" if eng else "no",
                        "cv_r2": row["cv_r2_mean"], "cv_rmse": row["cv_rmse_mean"]})
    ablation = pd.DataFrame(abl)
    ablation.to_csv(results_dir / "ablation_features.csv", index=False)

    # 5b. error analysis of the selected model --------------------------------------------------
    ea = X_test.copy()
    ea["actual"], ea["pred"] = y_test.values, best_pred
    rows = []
    for key, grp in ea.groupby("ocean_proximity"):
        if len(grp) >= 30:
            r = report(grp["actual"], grp["pred"])
            rows.append({"group": f"Location: {key}", "n": len(grp), **r})
    bands = pd.cut(ea["actual"], [0, 150_000, 300_000, np.inf], labels=["cheap (<$150k)", "middle ($150-300k)", "expensive (>$300k)"])
    for key, grp in ea.groupby(bands, observed=True):
        r = report(grp["actual"], grp["pred"])
        r["r2"] = float("nan")   # R-squared is meaningless inside a narrow price band
        rows.append({"group": f"True price: {key}", "n": len(grp), **r})
    unc = ea[ea["actual"] < PRICE_CAP]
    rows.append({"group": "Without the capped $500k rows", "n": len(unc), **report(unc["actual"], unc["pred"])})
    rows.append({"group": "All test rows", "n": len(ea), **report(ea["actual"], ea["pred"])})
    errors = pd.DataFrame(rows)
    errors.to_csv(results_dir / "error_analysis.csv", index=False)

    # 5c. depth curve (shows under- and over-fitting) ------------------------------------------------
    depths = list(range(1, 21)) if not quick else [2, 5, 10, 20]
    curve = []
    for d in depths:
        s = cross_validate(build_pipeline(DecisionTreeRegressor(max_depth=d, random_state=seed)),
                           X_train, y_train, cv=cv, scoring="r2", return_train_score=True)
        curve.append({"max_depth": d, "train_r2": s["train_score"].mean(), "cv_r2": s["test_score"].mean()})
    curve = pd.DataFrame(curve)
    curve.to_csv(results_dir / "depth_curve.csv", index=False)

    # 5d. feature importance / coefficients ---------------------------------------------------------------
    def names(pipe):
        return [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    dt_pipe = tuned.best_estimator_
    imp = pd.Series(dt_pipe.named_steps["model"].feature_importances_, index=names(dt_pipe)).sort_values(ascending=False)
    lr_pipe = fitted["Linear Regression"][0]
    coef = pd.Series(lr_pipe.named_steps["model"].coef_, index=names(lr_pipe))
    imp.to_csv(results_dir / "tree_feature_importance.csv", header=["importance"], index_label="feature")
    coef.to_csv(results_dir / "linear_coefficients.csv", header=["coefficient_per_1_std"], index_label="feature")

    # 6. plots ------------------------------------------------------------------------------------
    figs = results_dir / "figures"
    plots.fig_model_comparison(comparison, figs / "fig1_model_comparison.png")
    plots.fig_pred_vs_actual(y_test, fitted["Linear Regression"][1], fitted["Decision Tree (tuned)"][1],
                             figs / "fig2_predicted_vs_actual.png")
    plots.fig_depth_curve(curve, figs / "fig3_tree_depth.png")
    plots.fig_importance(imp, coef, figs / "fig4_feature_importance.png")

    summary = {
        "n_rows": info["n_rows"], "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "selected_model": selected, "best_tree_params": {k: (None if v is None else int(v)) for k, v in best_params.items()},
        "seed": seed, "drop_capped": drop_capped, "quick": quick,
        "sklearn_note": "numbers in results/ come from this run",
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(comparison[["model", "cv_r2_mean", "cv_rmse_mean", "test_r2", "test_rmse", "test_within_20pct"]].round(3).to_string(index=False))
    return summary
