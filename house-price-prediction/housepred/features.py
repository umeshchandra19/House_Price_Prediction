"""Feature engineering: three ratios that describe a neighbourhood better than raw totals."""
from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add rooms per household, bedrooms per room and people per household.

    A block with 5,000 rooms means little by itself; 5 rooms per household means a lot.
    Division by zero becomes NaN and is filled later by the imputer in the pipeline.
    """
    X = X.copy()
    X["rooms_per_household"] = X["total_rooms"] / X["households"].replace(0, np.nan)
    X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"].replace(0, np.nan)
    X["population_per_household"] = X["population"] / X["households"].replace(0, np.nan)
    return X
