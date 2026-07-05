from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_pipeline.portfolio import backtest_long_short
from alpha_pipeline.stats import TrialLedger, deflated_sharpe_ratio


def _toy_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=8)
    rows = []
    for asset_idx, asset in enumerate(["A", "B", "C", "D"]):
        close = 10.0 + asset_idx + np.arange(len(dates)) * (0.2 + 0.05 * asset_idx)
        for date, price in zip(dates, close):
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000,
                }
            )
    return pd.DataFrame(rows)


def _toy_predictions() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=8)
    rows = []
    scores = {"A": 0.4, "B": 0.2, "C": -0.2, "D": -0.4}
    for date in dates:
        for asset, score in scores.items():
            rows.append({"date": date, "asset": asset, "prediction": score, "forward_return": 0.01})
    return pd.DataFrame(rows)


def test_transaction_costs_reduce_net_returns_and_turnover_is_stable() -> None:
    prices = _toy_prices()
    predictions = _toy_predictions()

    no_cost, _ = backtest_long_short(predictions, prices, rebalance_frequency="D", transaction_cost_bps=0.0)
    with_cost, _ = backtest_long_short(predictions, prices, rebalance_frequency="D", transaction_cost_bps=10.0)

    assert with_cost["net_return"].sum() < no_cost["net_return"].sum()
    assert with_cost["turnover"].ge(0).all()


def test_deflated_sharpe_decreases_as_trial_benchmark_gets_harder() -> None:
    returns = pd.Series([0.01, -0.002, 0.006, 0.003, -0.001, 0.008] * 20)
    one_trial = deflated_sharpe_ratio(returns, [0.5], observed_sharpe=1.0)
    many_trials = deflated_sharpe_ratio(returns, [0.2, 0.7, 1.1, 1.5, 1.9], observed_sharpe=1.0)

    assert many_trials.deflated_sharpe < one_trial.deflated_sharpe


def test_trial_ledger_records_all_variants() -> None:
    ledger = TrialLedger()
    ledger.add("linear", {"alpha": 1.0}, {"sharpe": 0.4})
    ledger.add("boosting", {"depth": 3}, {"sharpe": 0.7})

    assert ledger.to_dict()["n_trials"] == 2
    assert set(ledger.to_frame()["variant"]) == {"linear", "boosting"}
