"""End-to-end experiment orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_pipeline.config import ExperimentConfig
from alpha_pipeline.data import SyntheticPriceLoader, Universe, make_data_quality_report
from alpha_pipeline.features import build_factor_frame, factor_columns
from alpha_pipeline.io import ensure_dir, write_frame, write_json
from alpha_pipeline.labels import assemble_model_dataset, build_forward_return_labels
from alpha_pipeline.models import daily_rank_ic, default_model_specs, predict_fold
from alpha_pipeline.portfolio import backtest_long_short
from alpha_pipeline.stats import (
    TrialLedger,
    compute_performance,
    deflated_sharpe_ratio,
    estimate_probability_of_backtest_overfitting,
    information_coefficient_summary,
)
from alpha_pipeline.validation import PurgedWalkForwardSplitter, write_fold_manifest


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Paths written by an experiment run."""

    output_dir: str
    metrics_path: str
    trial_ledger_path: str
    predictions_path: str
    returns_path: str
    weights_path: str
    rank_ic_path: str
    folds_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> ExperimentArtifacts:
    """Run the reproducible demo pipeline and persist dashboard-ready artifacts."""

    out = ensure_dir(output_dir)
    prices = SyntheticPriceLoader(
        n_assets=config.n_assets,
        n_days=config.n_days,
        start_date=config.start_date,
        seed=config.random_seed,
    ).load_prices()
    universe = Universe(
        name="synthetic_liquid_equity_universe",
        assets=tuple(sorted(prices["asset"].unique())),
        survivorship_bias_free=False,
        notes="Deterministic synthetic universe for reproducible pipeline verification.",
    )
    quality_report = make_data_quality_report(prices, universe)

    features = build_factor_frame(prices)
    labels = build_forward_return_labels(prices, horizon_days=config.label_horizon_days)
    dataset = assemble_model_dataset(features, labels)
    feature_cols = factor_columns(dataset)

    splitter = PurgedWalkForwardSplitter(
        train_window_days=config.train_window_days,
        test_window_days=config.test_window_days,
        step_days=config.step_days,
        embargo_days=config.embargo_days,
    )
    folds = list(splitter.split(dataset))
    if not folds:
        raise RuntimeError("No validation folds were produced.")

    model_specs = [
        spec for spec in default_model_specs(config.random_seed) if spec.kind in set(config.model_variants)
    ]
    if not model_specs:
        raise RuntimeError("No model variants selected.")

    ledger = TrialLedger()
    all_predictions: list[pd.DataFrame] = []
    all_rank_ic: list[pd.DataFrame] = []
    all_returns: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []
    fold_scores: list[dict[str, Any]] = []
    variant_metrics: dict[str, Any] = {}

    for spec in model_specs:
        variant_predictions: list[pd.DataFrame] = []
        for fold, train, test in folds:
            predictions = predict_fold(spec, train, test, feature_cols)
            predictions["fold_id"] = fold.fold_id
            predictions["test_start"] = fold.test_start
            predictions["test_end"] = fold.test_end
            variant_predictions.append(predictions)

            fold_ic = daily_rank_ic(predictions)
            fold_scores.append(
                {
                    "variant": spec.name,
                    "fold_id": fold.fold_id,
                    "test_score": float(fold_ic["rank_ic"].mean(skipna=True)),
                    "train_score": float("nan"),
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "n_train_rows": fold.n_train_rows,
                    "n_test_rows": fold.n_test_rows,
                }
            )

        predictions = pd.concat(variant_predictions, ignore_index=True)
        rank_ic = daily_rank_ic(predictions)
        rank_ic["variant"] = spec.name
        daily_returns, weights = backtest_long_short(
            predictions,
            prices,
            rebalance_frequency=config.rebalance_frequency,
            transaction_cost_bps=config.transaction_cost_bps,
            long_quantile=config.long_quantile,
            short_quantile=config.short_quantile,
        )
        daily_returns["variant"] = spec.name
        weights["variant"] = spec.name

        performance = compute_performance(daily_returns)
        ic_summary = information_coefficient_summary(rank_ic)
        metrics = {**performance.to_dict(), **ic_summary}
        ledger.add(spec.name, spec.to_dict(), metrics)

        all_predictions.append(predictions)
        all_rank_ic.append(rank_ic)
        all_returns.append(daily_returns)
        all_weights.append(weights)
        variant_metrics[spec.name] = {
            "model": spec.to_dict(),
            "performance": performance.to_dict(),
            "information_coefficient": ic_summary,
        }

    trial_sharpes = [trial["metrics"]["sharpe"] for trial in ledger.trials]
    returns_frame = pd.concat(all_returns, ignore_index=True)
    for variant, daily in returns_frame.groupby("variant"):
        observed = variant_metrics[variant]["performance"]["sharpe"]
        dsr = deflated_sharpe_ratio(daily["net_return"], trial_sharpes, observed_sharpe=observed)
        variant_metrics[variant]["deflated_sharpe"] = dsr.to_dict()

    fold_scores_frame = pd.DataFrame(fold_scores)
    pbo = estimate_probability_of_backtest_overfitting(fold_scores_frame)
    best_variant = max(
        variant_metrics,
        key=lambda name: variant_metrics[name]["deflated_sharpe"]["deflated_sharpe"],
    )

    metrics_payload = {
        "config": asdict(config),
        "data_quality": quality_report.to_dict(),
        "feature_columns": feature_cols,
        "best_variant": best_variant,
        "probability_of_backtest_overfitting": pbo,
        "variants": variant_metrics,
    }

    write_json(out / "config.json", asdict(config))
    write_json(out / "data_quality.json", quality_report.to_dict())
    write_json(out / "metrics.json", metrics_payload)
    write_fold_manifest(out / "folds.json", [fold for fold, _, _ in folds])
    ledger.write(out / "trial_ledger.json")
    write_frame(out / "prices.parquet", prices)
    write_frame(out / "features.parquet", features)
    write_frame(out / "labels.parquet", labels)
    write_frame(out / "dataset.parquet", dataset)
    write_frame(out / "predictions.parquet", pd.concat(all_predictions, ignore_index=True))
    write_frame(out / "rank_ic.parquet", pd.concat(all_rank_ic, ignore_index=True))
    write_frame(out / "returns.parquet", returns_frame)
    write_frame(out / "weights.parquet", pd.concat(all_weights, ignore_index=True))
    write_frame(out / "fold_scores.parquet", fold_scores_frame)

    return ExperimentArtifacts(
        output_dir=str(out),
        metrics_path=str(out / "metrics.json"),
        trial_ledger_path=str(out / "trial_ledger.json"),
        predictions_path=str(out / "predictions.parquet"),
        returns_path=str(out / "returns.parquet"),
        weights_path=str(out / "weights.parquet"),
        rank_ic_path=str(out / "rank_ic.parquet"),
        folds_path=str(out / "folds.json"),
    )
