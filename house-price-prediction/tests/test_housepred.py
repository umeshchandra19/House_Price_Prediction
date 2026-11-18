import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from housepred.data import PRICE_CAP, TARGET, audit, load_data, split
from housepred.features import ENGINEERED, add_features
from housepred.metrics import report, within_pct
from housepred.pipeline import build_pipeline


@pytest.fixture(scope="module")
def df():
    return load_data()


def test_data_size_and_known_quirks(df):
    a = audit(df)
    assert a["n_rows"] > 10_000                       # "10,000+ records"
    assert a["missing_values"] == {"total_bedrooms": 207}
    assert a["capped_price_rows"] > 0


def test_split_is_disjoint_and_stratified(df):
    Xtr, Xte, ytr, yte = split(df, seed=1)
    assert len(Xtr) + len(Xte) == len(df)
    assert set(Xtr.index).isdisjoint(Xte.index)
    assert TARGET not in Xtr.columns
    assert abs(len(Xte) / len(df) - 0.2) < 0.01
    # train and test have a similar share of high-income areas
    assert abs((Xtr.median_income > 6).mean() - (Xte.median_income > 6).mean()) < 0.01


def test_drop_capped_removes_capped_rows(df):
    _, _, ytr, yte = split(df, drop_capped=True)
    assert (ytr < PRICE_CAP).all() and (yte < PRICE_CAP).all()


def test_add_features_values_and_zero_division():
    X = pd.DataFrame({"total_rooms": [1000.0, 10.0], "total_bedrooms": [200.0, 5.0],
                      "population": [500.0, 20.0], "households": [250.0, 0.0]})
    out = add_features(X)
    assert set(ENGINEERED) <= set(out.columns)
    assert out.loc[0, "rooms_per_household"] == 4.0
    assert out.loc[0, "bedrooms_per_room"] == 0.2
    assert np.isnan(out.loc[1, "rooms_per_household"])       # no crash, no infinity
    assert "rooms_per_household" not in X.columns            # input is not modified


@pytest.mark.parametrize("estimator", [LinearRegression(), DecisionTreeRegressor(max_depth=4, random_state=0)])
def test_pipeline_fits_and_predicts(df, estimator):
    Xtr, Xte, ytr, yte = split(df)
    pipe = build_pipeline(estimator).fit(Xtr.iloc[:2000], ytr.iloc[:2000])
    pred = pipe.predict(Xte.iloc[:200])
    assert pred.shape == (200,) and np.isfinite(pred).all()


def test_missing_values_are_handled_at_prediction_time(df):
    Xtr, Xte, ytr, _ = split(df)
    pipe = build_pipeline(LinearRegression()).fit(Xtr.iloc[:2000], ytr.iloc[:2000])
    row = Xte.iloc[[0]].copy()
    row["total_bedrooms"] = np.nan
    assert np.isfinite(pipe.predict(row)).all()


def test_imputer_learns_from_training_data_only(df):
    Xtr, Xte, ytr, _ = split(df)
    sub = Xtr.iloc[:3000]
    pipe = build_pipeline(LinearRegression()).fit(sub, ytr.iloc[:3000])
    imp = pipe.named_steps["prep"].named_transformers_["num"].named_steps["fill_missing"]
    expected = add_features(sub).select_dtypes("number")["total_bedrooms"].median()
    names = list(pipe.named_steps["prep"].transformers_[0][2])
    assert imp.statistics_[names.index("total_bedrooms")] == pytest.approx(expected)


def test_unknown_category_does_not_crash(df):
    Xtr, Xte, ytr, _ = split(df)
    pipe = build_pipeline(LinearRegression()).fit(Xtr.iloc[:2000], ytr.iloc[:2000])
    row = Xte.iloc[[0]].copy()
    row["ocean_proximity"] = "ON A BOAT"
    assert np.isfinite(pipe.predict(row)).all()


def test_metrics():
    y = np.array([100.0, 200.0, 300.0])
    p = np.array([110.0, 150.0, 300.0])
    assert within_pct(y, p, 0.20) == pytest.approx(2 / 3)
    r = report(y, y)
    assert r["r2"] == 1.0 and r["rmse"] == 0.0 and r["within_20pct"] == 1.0
