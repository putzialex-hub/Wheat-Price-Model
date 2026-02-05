from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import ModelSpec


@dataclass
class TrainedModel:
    pipeline: Pipeline
    feature_names_: list[str]


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    spec: ModelSpec,
    model_type: str = "tree",
    ridge_alpha: float | None = None,
) -> TrainedModel:
    """
    Production-friendly default:
    - imputes missing values
    - handles categorical columns
    - robust GBDT (HistGradientBoostingRegressor)
    """
    Xc = X.copy()

    cat_cols = [c for c in Xc.columns if Xc[c].dtype == "object"]
    num_cols = [c for c in Xc.columns if c not in cat_cols]

    if model_type == "ridge":
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )
    else:
        pre = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )

    if model_type == "ridge":
        alpha = ridge_alpha if ridge_alpha is not None else spec.ridge_alpha
        model = Ridge(alpha=alpha, random_state=spec.random_state)
    else:
        model = HistGradientBoostingRegressor(
            max_depth=spec.max_depth,
            learning_rate=spec.learning_rate,
            max_iter=spec.n_estimators,
            random_state=spec.random_state,
        )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(Xc, y)

    # feature names after transform are not trivially accessible; store raw names
    return TrainedModel(pipeline=pipe, feature_names_=list(Xc.columns))


def predict(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    return model.pipeline.predict(X)
