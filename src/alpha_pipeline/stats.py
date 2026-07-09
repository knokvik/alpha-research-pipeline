"""Performance statistics, deflated Sharpe, and variant logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from alpha_pipeline.io import read_json, write_json


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PerformanceSummary:
    """Core cost-adjusted strategy statistics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    average_turnover: float
    hit_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Observed Sharpe adjusted for the number and dispersion of tested variants."""

    raw_sharpe: float
    benchmark_sharpe: float
    deflated_sharpe: float
    probability: float
    n_trials: int
    skewness: float
    excess_kurtosis: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class TrialLedger:
    """Append-only record of every tested strategy variant."""

    trials: list[dict[str, Any]] = field(default_factory=list)

    def add(self, variant: str, parameters: dict[str, Any], metrics: dict[str, Any]) -> None:
        self.trials.append({"variant": variant, "parameters": parameters, "metrics": metrics})

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for trial in self.trials:
            row = {"variant": trial["variant"], **trial["parameters"], **trial["metrics"]}
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {"n_trials": len(self.trials), "trials": self.trials}

    def write(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def read(cls, path: str | Path) -> "TrialLedger":
        payload = read_json(path)
        return cls(trials=list(payload.get("trials", [])))


def compute_performance(daily_returns: pd.DataFrame, return_column: str = "net_return") -> PerformanceSummary:
    """Compute standard performance metrics from a daily backtest frame."""

    if return_column not in daily_returns.columns:
        raise ValueError(f"Missing return column: {return_column}")
    returns = pd.to_numeric(daily_returns[return_column], errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    annualized_return = float(equity.iloc[-1] ** (TRADING_DAYS_PER_YEAR / max(len(returns), 1)) - 1.0) if len(equity) else 0.0
    annualized_vol = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)) if returns.std(ddof=0) > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    turnover = daily_returns["turnover"] if "turnover" in daily_returns.columns else pd.Series(dtype=float)
    return PerformanceSummary(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_vol,
        sharpe=sharpe,
        max_drawdown=float(drawdown.min()) if len(drawdown) else 0.0,
        average_turnover=float(turnover.mean()) if len(turnover) else 0.0,
        hit_rate=float((returns > 0).mean()) if len(returns) else 0.0,
    )


def information_coefficient_summary(rank_ic: pd.DataFrame) -> dict[str, float]:
    """Summarize daily rank IC values."""

    if "rank_ic" not in rank_ic.columns:
        raise ValueError("rank_ic frame must contain a rank_ic column.")
    values = pd.to_numeric(rank_ic["rank_ic"], errors="coerce").dropna()
    mean = float(values.mean()) if len(values) else 0.0
    std = float(values.std(ddof=0)) if len(values) else 0.0
    return {
        "mean_rank_ic": mean,
        "rank_ic_std": std,
        "icir": float(mean / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0,
        "positive_ic_rate": float((values > 0).mean()) if len(values) else 0.0,
    }


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes: list[float] | np.ndarray,
    observed_sharpe: float | None = None,
) -> DeflatedSharpeResult:
    """Estimate a deflated Sharpe result using a multiple-testing benchmark."""

    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()
    sharpes = np.asarray(trial_sharpes, dtype=float)
    sharpes = sharpes[np.isfinite(sharpes)]
    if len(sharpes) == 0:
        sharpes = np.array([0.0])

    raw = float(observed_sharpe) if observed_sharpe is not None else _annualized_sharpe(clean_returns)
    benchmark = _expected_max_sharpe(sharpes)
    n = max(int(clean_returns.shape[0]), 2)
    skewness = float(skew(clean_returns, bias=False)) if len(clean_returns) > 2 else 0.0
    excess_kurt = float(kurtosis(clean_returns, fisher=True, bias=False)) if len(clean_returns) > 3 else 0.0
    denominator_term = max(1.0 - skewness * raw + ((excess_kurt + 3.0) - 1.0) * raw**2 / 4.0, 1e-12)
    z_score = (raw - benchmark) * np.sqrt(n - 1.0) / np.sqrt(denominator_term)
    probability = float(norm.cdf(z_score))

    return DeflatedSharpeResult(
        raw_sharpe=raw,
        benchmark_sharpe=float(benchmark),
        deflated_sharpe=float(raw - benchmark),
        probability=probability,
        n_trials=int(len(sharpes)),
        skewness=skewness,
        excess_kurtosis=excess_kurt,
    )


def estimate_probability_of_backtest_overfitting(fold_scores: pd.DataFrame) -> float | None:
    """Estimate PBO from train/test variant rankings when both scores are available."""

    required = {"variant", "fold_id", "train_score", "test_score"}
    if not required.issubset(fold_scores.columns):
        return None
    overfit_count = 0
    evaluated_count = 0
    for _, group in fold_scores.groupby("fold_id"):
        group = group.dropna(subset=["train_score", "test_score"])
        if group["variant"].nunique() < 2:
            continue
        in_sample_winner = group.sort_values("train_score", ascending=False).iloc[0]
        test_rank_pct = group["test_score"].rank(pct=True).loc[in_sample_winner.name]
        overfit_count += int(test_rank_pct <= 0.5)
        evaluated_count += 1
    if evaluated_count == 0:
        return None
    return float(overfit_count / evaluated_count)


def _annualized_sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=0)
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0


def _expected_max_sharpe(sharpes: np.ndarray) -> float:
    if len(sharpes) <= 1:
        return 0.0
    gamma = 0.5772156649015329
    mean = float(np.mean(sharpes))
    std = float(np.std(sharpes, ddof=1))
    if std == 0.0:
        return mean
    n_trials = len(sharpes)
    return mean + std * (
        (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials)
        + gamma * norm.ppf(1.0 - 1.0 / (np.e * n_trials))
    )
