"""Cross-sectional factor construction with explicit lagging."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_pipeline.data import validate_prices


IDENTIFIER_COLUMNS = {
    "date",
    "asset",
    "as_of_date",
    "sector",
    "label_start_date",
    "label_end_date",
    "forward_return",
    "prediction",
    "fold_id",
    "test_start",
    "test_end",
}


def factor_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric factor columns from a feature frame."""

    return [
        column
        for column in frame.columns
        if column not in IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def build_factor_frame(prices: pd.DataFrame, lag_periods: int = 1) -> pd.DataFrame:
    """Build lagged cross-sectional factors from long-form OHLCV prices."""

    if lag_periods < 1:
        raise ValueError("lag_periods must be at least 1 to avoid same-day lookahead.")

    frame = validate_prices(prices)
    frame = frame.sort_values(["asset", "date"]).reset_index(drop=True)
    grouped = frame.groupby("asset", group_keys=False)

    close = frame["close"]
    volume = frame["volume"].replace(0, np.nan)
    returns_1d = grouped["close"].pct_change()
    dollar_volume = close * volume

    features = frame[["date", "asset", "as_of_date"]].copy()
    if "sector" in frame.columns:
        features["sector"] = frame["sector"].values

    features["momentum_5"] = grouped["close"].pct_change(5)
    features["momentum_21"] = grouped["close"].pct_change(21)
    features["momentum_63"] = grouped["close"].pct_change(63)
    features["reversal_5"] = -features["momentum_5"]
    features["volatility_21"] = returns_1d.groupby(frame["asset"]).rolling(21).std().reset_index(level=0, drop=True)
    features["volatility_63"] = returns_1d.groupby(frame["asset"]).rolling(63).std().reset_index(level=0, drop=True)
    features["dollar_volume_21"] = dollar_volume.groupby(frame["asset"]).rolling(21).mean().reset_index(level=0, drop=True)
    features["amihud_21"] = (
        (returns_1d.abs() / dollar_volume)
        .groupby(frame["asset"])
        .rolling(21)
        .mean()
        .reset_index(level=0, drop=True)
    )
    features["price_to_ma_20"] = close / grouped["close"].rolling(20).mean().reset_index(level=0, drop=True) - 1.0
    features["price_to_ma_60"] = close / grouped["close"].rolling(60).mean().reset_index(level=0, drop=True) - 1.0
    features["range_21"] = ((frame["high"] - frame["low"]) / close).groupby(frame["asset"]).rolling(21).mean().reset_index(level=0, drop=True)
    features["volume_z_20"] = _rolling_zscore(volume, frame["asset"], 20)
    features["max_return_21"] = returns_1d.groupby(frame["asset"]).rolling(21).max().reset_index(level=0, drop=True)
    features["downside_vol_21"] = (
        returns_1d.where(returns_1d < 0.0, 0.0)
        .groupby(frame["asset"])
        .rolling(21)
        .std()
        .reset_index(level=0, drop=True)
    )
    features["quality_proxy_63"] = -features["volatility_63"]
    features["value_proxy"] = -np.log(close)

    raw_feature_columns = factor_columns(features)
    features[raw_feature_columns] = grouped_lag(features, raw_feature_columns, lag_periods)
    return cross_sectional_normalize(features)


def grouped_lag(frame: pd.DataFrame, columns: list[str], periods: int) -> pd.DataFrame:
    """Lag columns within each asset."""

    return frame.groupby("asset", group_keys=False)[columns].shift(periods)


def cross_sectional_normalize(
    features: pd.DataFrame,
    winsorize_quantiles: tuple[float, float] = (0.01, 0.99),
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Winsorize and z-score factors independently for each date."""

    frame = features.copy()
    columns = factor_columns(frame)
    lower_q, upper_q = winsorize_quantiles
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError("winsorize_quantiles must be ordered values in [0, 1].")

    def normalize_group(group: pd.DataFrame) -> pd.DataFrame:
        values = group[columns].copy()
        lower = values.quantile(lower_q)
        upper = values.quantile(upper_q)
        clipped = values.clip(lower=lower, upper=upper, axis=1)
        centered = clipped - clipped.mean(axis=0)
        scale = clipped.std(axis=0, ddof=0).replace(0.0, np.nan)
        group.loc[:, columns] = centered / scale
        return group

    frame = pd.concat(
        [normalize_group(group.copy()) for _, group in frame.groupby("date", sort=False)],
        ignore_index=True,
    )
    frame[columns] = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


def assert_no_feature_lookahead(features: pd.DataFrame) -> None:
    """Raise if any feature row is timestamped after its market date."""

    required = {"date", "asset", "as_of_date"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Feature frame is missing required columns: {sorted(missing)}")
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
    if (frame["as_of_date"] > frame["date"]).any():
        raise ValueError("Feature as_of_date cannot be after feature date.")


def _rolling_zscore(values: pd.Series, assets: pd.Series, window: int) -> pd.Series:
    rolling = values.groupby(assets).rolling(window)
    mean = rolling.mean().reset_index(level=0, drop=True)
    std = rolling.std().reset_index(level=0, drop=True).replace(0.0, np.nan)
    return (values - mean) / std
