"""Command-line entrypoint for reproducible demo experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha_pipeline.config import ExperimentConfig
from alpha_pipeline.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cross-sectional alpha research demo.")
    parser.add_argument("--output", default="artifacts/demo", help="Directory for experiment artifacts.")
    parser.add_argument("--assets", type=int, default=ExperimentConfig.n_assets)
    parser.add_argument("--days", type=int, default=ExperimentConfig.n_days)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.random_seed)
    parser.add_argument("--cost-bps", type=float, default=ExperimentConfig.transaction_cost_bps)
    args = parser.parse_args()

    config = ExperimentConfig(
        n_assets=args.assets,
        n_days=args.days,
        random_seed=args.seed,
        transaction_cost_bps=args.cost_bps,
    )
    artifacts = run_experiment(config, Path(args.output))
    print(f"Wrote experiment artifacts to {artifacts.output_dir}")
    print(f"Metrics: {artifacts.metrics_path}")


if __name__ == "__main__":
    main()
