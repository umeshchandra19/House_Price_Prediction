"""Model pipelines: preprocessing + estimator in one object.

Keeping the preprocessing inside the pipeline matters: during cross-validation the
medians, scaling numbers and categories are learned from the training folds only,
so nothing leaks from the validation fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import add_features


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("fill_missing", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude=np.number)),
    ])


def build_pipeline(estimator, engineer: bool = True) -> Pipeline:
    steps = []
    if engineer:
        steps.append(("features", FunctionTransformer(add_features)))
    steps += [("prep", make_preprocessor()), ("model", estimator)]
    return Pipeline(steps)
