"""Configuration objects for alpha research experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level knobs for a reproducible research run."""

    experiment_name: str = "demo_synthetic_cross_section"
    n_assets: int = 40
    n_days: int = 900
    start_date: str = "2018-01-01"
    label_horizon_days: int = 5
    rebalance_frequency: str = "W-FRI"
    transaction_cost_bps: float = 5.0
    long_quantile: float = 0.8
    short_quantile: float = 0.2
    train_window_days: int = 504
    test_window_days: int = 63
    step_days: int = 63
    embargo_days: int = 5
    random_seed: int = 7
    model_variants: tuple[str, ...] = field(default_factory=lambda: ("linear", "boosting"))

