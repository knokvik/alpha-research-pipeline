"""Model specifications and fold-level prediction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    """A logged model or hyperparameter variant."""

    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_model_specs(random_seed: int = 7) -> list[ModelSpec]:
    """Return a small, explicit benchmark set."""

    return [
        ModelSpec("linear_ridge_alpha_1", "linear", {"alpha": 1.0}),
        ModelSpec(
            "boosting_hist_depth_3",
            "boosting",
            {"max_iter": 120, "max_leaf_nodes": 15, "learning_rate": 0.06, "random_state": random_seed},
        ),
    ]


def build_estimator(spec: ModelSpec) -> Pipeline:
    """Create a scikit-learn estimator from a model spec."""

    if spec.kind == "linear":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(**spec.params)),
            ]
        )

    if spec.kind == "boosting":
        model = _build_boosting_model(spec.params)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )

    raise ValueError(f"Unsupported model kind: {spec.kind}")


def predict_fold(
    spec: ModelSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "forward_return",
) -> pd.DataFrame:
    """Fit a model on one fold and return test predictions."""

    estimator = build_estimator(spec)
    estimator.fit(train[feature_columns], train[target_column])
    predictions = test[["date", "asset", target_column]].copy()
    predictions["prediction"] = estimator.predict(test[feature_columns])
    predictions["model"] = spec.name
    predictions["variant"] = spec.name
    return predictions


def daily_rank_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rank IC for each date."""

    required = {"date", "prediction", "forward_return"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for date, group in predictions.groupby("date"):
        if group["prediction"].nunique() < 2 or group["forward_return"].nunique() < 2:
            ic = np.nan
        else:
            ic = spearmanr(group["prediction"], group["forward_return"], nan_policy="omit").correlation
        rows.append({"date": date, "rank_ic": ic})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _build_boosting_model(params: dict[str, Any]) -> Any:
    """Prefer optional boosting libraries, then fall back to sklearn."""

    try:
        from lightgbm import LGBMRegressor

        lightgbm_params = {
            "n_estimators": params.get("max_iter", 120),
            "num_leaves": params.get("max_leaf_nodes", 15),
            "learning_rate": params.get("learning_rate", 0.06),
            "random_state": params.get("random_state", 7),
            "verbosity": -1,
        }
        return LGBMRegressor(**lightgbm_params)
    except Exception:
        return HistGradientBoostingRegressor(**params)
