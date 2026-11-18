"""Loading, auditing and splitting the housing data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "housing.csv"
TARGET = "median_house_value"
PRICE_CAP = 500001.0   # the census file caps every house value above ~$500k at this number


def load_data(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def audit(df: pd.DataFrame) -> dict:
    """A quick health check of the raw data, saved to results/data_audit.json."""
    missing = df.isna().sum()
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_values": {c: int(v) for c, v in missing.items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "capped_price_rows": int((df[TARGET] >= PRICE_CAP).sum()),
        "capped_price_share_pct": round(100 * float((df[TARGET] >= PRICE_CAP).mean()), 2),
        "price_min": float(df[TARGET].min()),
        "price_median": float(df[TARGET].median()),
        "price_max": float(df[TARGET].max()),
        "ocean_proximity_counts": {k: int(v) for k, v in df["ocean_proximity"].value_counts().items()},
    }


def income_band(income: pd.Series) -> pd.Series:
    """Five income groups, used to keep train and test sets alike (stratified split)."""
    return pd.cut(income, bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, drop_capped: bool = False):
    """Return X_train, X_test, y_train, y_test. The test set is touched only once, at the very end."""
    if drop_capped:
        df = df[df[TARGET] < PRICE_CAP]
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed,
                            stratify=income_band(df["median_income"]))
