"""Cost-aware long-short portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_pipeline.data import validate_prices


def select_rebalance_dates(dates: pd.Series, frequency: str = "W-FRI") -> pd.DatetimeIndex:
    """Select the last available trading date in each rebalance bucket."""

    unique_dates = pd.Series(pd.to_datetime(dates).drop_duplicates().sort_values())
    selected = unique_dates.set_axis(unique_dates).resample(frequency).max()
    return pd.DatetimeIndex(selected.dropna().sort_values())


def make_long_short_weights(
    predictions: pd.DataFrame,
    long_quantile: float = 0.8,
    short_quantile: float = 0.2,
) -> pd.DataFrame:
    """Create dollar-neutral weights from cross-sectional predictions."""

    if not 0.0 < short_quantile < long_quantile < 1.0:
        raise ValueError("Expected 0 < short_quantile < long_quantile < 1.")

    required = {"date", "asset", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {sorted(missing)}")

    rows: list[pd.DataFrame] = []
    for date, group in predictions.groupby("date"):
        group = group[["date", "asset", "prediction"]].dropna().copy()
        if len(group) < 4:
            continue
        long_cut = group["prediction"].quantile(long_quantile)
        short_cut = group["prediction"].quantile(short_quantile)
        longs = group["prediction"] >= long_cut
        shorts = group["prediction"] <= short_cut
        weights = pd.Series(0.0, index=group.index)
        if longs.any():
            weights.loc[longs] = 0.5 / int(longs.sum())
        if shorts.any():
            weights.loc[shorts] = -0.5 / int(shorts.sum())
        out = group[["date", "asset"]].copy()
        out["weight"] = weights.values
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["date", "asset", "weight"])
    return pd.concat(rows, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)


def backtest_long_short(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_frequency: str = "W-FRI",
    transaction_cost_bps: float = 5.0,
    long_quantile: float = 0.8,
    short_quantile: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a close-to-close long-short backtest with flat turnover costs."""

    price_frame = validate_prices(prices)
    prediction_frame = predictions.copy()
    prediction_frame["date"] = pd.to_datetime(prediction_frame["date"])
    rebalance_dates = select_rebalance_dates(prediction_frame["date"], rebalance_frequency)
    rebalance_predictions = prediction_frame[prediction_frame["date"].isin(rebalance_dates)]
    weights = make_long_short_weights(rebalance_predictions, long_quantile, short_quantile)
    if weights.empty:
        raise ValueError("No portfolio weights were created from predictions.")

    returns = price_frame.sort_values(["asset", "date"]).copy()
    returns["asset_return"] = returns.groupby("asset")["close"].pct_change()
    returns_matrix = returns.pivot(index="date", columns="asset", values="asset_return").sort_index().fillna(0.0)

    weight_matrix = weights.pivot(index="date", columns="asset", values="weight").sort_index().fillna(0.0)
    weight_matrix = weight_matrix.reindex(returns_matrix.index).ffill().fillna(0.0)
    shifted_weights = weight_matrix.shift(1).fillna(0.0)
    aligned_returns = returns_matrix.reindex(columns=shifted_weights.columns).fillna(0.0)

    gross_return = (shifted_weights * aligned_returns).sum(axis=1)
    weight_diff = weight_matrix.diff().abs()
    if len(weight_diff):
        weight_diff.iloc[0] = weight_matrix.iloc[0].abs()
    turnover = weight_diff.sum(axis=1)
    cost = turnover * (transaction_cost_bps / 10_000.0)
    net_return = gross_return - cost
    equity_curve = (1.0 + net_return).cumprod()

    daily = pd.DataFrame(
        {
            "date": returns_matrix.index,
            "gross_return": gross_return.values,
            "transaction_cost": cost.values,
            "net_return": net_return.values,
            "turnover": turnover.values,
            "equity_curve": equity_curve.values,
            "gross_exposure": shifted_weights.abs().sum(axis=1).values,
            "net_exposure": shifted_weights.sum(axis=1).values,
        }
    )
    first_rebalance = weights["date"].min()
    daily = daily[daily["date"] >= first_rebalance].reset_index(drop=True)
    return daily, weights
