from __future__ import annotations

import pandas as pd

from alpha_pipeline.data import SyntheticPriceLoader
from alpha_pipeline.features import build_factor_frame, cross_sectional_normalize, factor_columns
from alpha_pipeline.labels import assemble_model_dataset, build_forward_return_labels


def test_feature_lag_prevents_last_close_lookahead() -> None:
    prices = SyntheticPriceLoader(n_assets=8, n_days=90, seed=11).load_prices()
    modified = prices.copy()
    last_date = modified["date"].max()
    modified.loc[modified["date"].eq(last_date), "close"] *= 100.0

    original_features = build_factor_frame(prices)
    modified_features = build_factor_frame(modified)
    columns = factor_columns(original_features)
    before_last = original_features["date"] < last_date

    pd.testing.assert_frame_equal(
        original_features.loc[before_last, ["date", "asset", *columns]].reset_index(drop=True),
        modified_features.loc[before_last, ["date", "asset", *columns]].reset_index(drop=True),
    )


def test_cross_sectional_normalization_is_by_date() -> None:
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "asset": ["A", "B", "A", "B"],
            "as_of_date": pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-02"] * 2),
            "factor": [1.0, 3.0, 100.0, 104.0],
        }
    )

    normalized = cross_sectional_normalize(features)

    means = normalized.groupby("date")["factor"].mean().round(12)
    assert means.eq(0.0).all()


def test_forward_labels_are_strictly_after_feature_date() -> None:
    prices = SyntheticPriceLoader(n_assets=5, n_days=90, seed=3).load_prices()
    features = build_factor_frame(prices)
    labels = build_forward_return_labels(prices, horizon_days=5)
    dataset = assemble_model_dataset(features, labels)

    assert (dataset["label_start_date"] > dataset["date"]).all()
    assert (dataset["label_end_date"] >= dataset["label_start_date"]).all()
    assert dataset["forward_return"].notna().all()


def test_model_factor_columns_exclude_labels_and_metadata() -> None:
    prices = SyntheticPriceLoader(n_assets=5, n_days=90, seed=4).load_prices()
    features = build_factor_frame(prices)
    labels = build_forward_return_labels(prices, horizon_days=5)
    dataset = assemble_model_dataset(features, labels)

    columns = set(factor_columns(dataset))

    assert "forward_return" not in columns
    assert "label_start_date" not in columns
    assert "label_end_date" not in columns
