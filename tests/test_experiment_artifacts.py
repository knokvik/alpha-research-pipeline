from __future__ import annotations

from pathlib import Path

from alpha_pipeline.config import ExperimentConfig
from alpha_pipeline.experiments import run_experiment
from alpha_pipeline.memo import render_memo


def test_demo_experiment_writes_dashboard_ready_artifacts(tmp_path: Path) -> None:
    config = ExperimentConfig(n_assets=10, n_days=180, train_window_days=90, test_window_days=30, step_days=30)

    artifacts = run_experiment(config, tmp_path / "demo")
    memo = render_memo(artifacts.output_dir)

    assert Path(artifacts.metrics_path).exists()
    assert Path(artifacts.predictions_path).exists()
    assert Path(artifacts.returns_path).exists()
    assert Path(artifacts.trial_ledger_path).exists()
    assert "Deflated Sharpe" in memo
