"""Metrics, with a plain-English accuracy measure next to the usual ones."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def within_pct(y_true, y_pred, pct: float = 0.20) -> float:
    """Share of houses whose predicted price is within +-pct of the true price."""
    y_true = np.asarray(y_true, float)
    return float(np.mean(np.abs(y_pred - y_true) <= pct * y_true))


def report(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "within_20pct": within_pct(y_true, y_pred, 0.20),
    }
