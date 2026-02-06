from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import ModelSpec


@dataclass
class TrainedModel:
    pipeline: Pipeline
    feature_names_: list[str]

@dataclass
class QuantileModels:
    pipelines: dict[float, Pipeline]
    feature_names_: list[str]


def _build_preprocessor(Xc: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    cat_cols = [c for c in Xc.columns if Xc[c].dtype == "object"]
    num_cols = [c for c in Xc.columns if c not in cat_cols]

    if scale_numeric:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        num_step = ("num", num_pipe, num_cols)
    else:
        num_step = ("num", SimpleImputer(strategy="median"), num_cols)

    return ColumnTransformer(
        transformers=[
            num_step,
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

    pre = _build_preprocessor(Xc, scale_numeric=(model_type == "ridge"))

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


def train_quantile_models(
    X: pd.DataFrame,
    y: pd.Series,
    quantiles: list[float] | None = None,
) -> QuantileModels:
    """
    Train quantile GradientBoostingRegressor models on delta targets.
    """
    Xc = X.copy()
    qs = quantiles or [0.1, 0.5, 0.9]
    pipelines: dict[float, Pipeline] = {}
    for q in qs:
        pre = _build_preprocessor(Xc, scale_numeric=False)
        model = GradientBoostingRegressor(loss="quantile", alpha=q, random_state=42)
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(Xc, y)
        pipelines[q] = pipe
    return QuantileModels(pipelines=pipelines, feature_names_=list(Xc.columns))


def predict_quantiles(models: QuantileModels, X: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for q, pipe in models.pipelines.items():
        out[f"delta_p{int(q*100)}"] = pipe.predict(X)
    return pd.DataFrame(out, index=X.index)
