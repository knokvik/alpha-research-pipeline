"""Forward-return labels and dataset assembly."""

from __future__ import annotations

import pandas as pd

from alpha_pipeline.data import validate_prices
from alpha_pipeline.features import assert_no_feature_lookahead, factor_columns


def build_forward_return_labels(prices: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    """Create forward-return labels with explicit outcome windows."""

    if horizon_days < 1:
        raise ValueError("horizon_days must be positive.")

    frame = validate_prices(prices).sort_values(["asset", "date"]).reset_index(drop=True)
    grouped = frame.groupby("asset", group_keys=False)
    labels = frame[["date", "asset"]].copy()
    labels["label_start_date"] = grouped["date"].shift(-1)
    labels["label_end_date"] = grouped["date"].shift(-horizon_days)
    labels["forward_return"] = grouped["close"].shift(-horizon_days) / frame["close"] - 1.0
    return labels.dropna(subset=["label_start_date", "label_end_date", "forward_return"]).sort_values(
        ["date", "asset"]
    ).reset_index(drop=True)


def assemble_model_dataset(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join features and labels while preserving validation metadata."""

    assert_no_feature_lookahead(features)
    feature_cols = factor_columns(features)
    if not feature_cols:
        raise ValueError("Feature frame has no model factor columns.")

    required_labels = {"date", "asset", "label_start_date", "label_end_date", "forward_return"}
    missing = required_labels.difference(labels.columns)
    if missing:
        raise ValueError(f"Label frame is missing required columns: {sorted(missing)}")

    dataset = features.merge(labels, on=["date", "asset"], how="inner")
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["label_start_date"] = pd.to_datetime(dataset["label_start_date"])
    dataset["label_end_date"] = pd.to_datetime(dataset["label_end_date"])
    if (dataset["label_start_date"] <= dataset["date"]).any():
        raise ValueError("Labels must begin strictly after the feature date.")

    dataset = dataset.dropna(subset=feature_cols + ["forward_return"])
    return dataset.sort_values(["date", "asset"]).reset_index(drop=True)
