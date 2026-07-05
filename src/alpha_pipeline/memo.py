"""Research memo generation from persisted experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_memo(experiment_dir: str | Path) -> str:
    """Render a concise Markdown research memo for one experiment."""

    root = Path(experiment_dir)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    best_variant = metrics["best_variant"]
    best = metrics["variants"][best_variant]
    quality = metrics["data_quality"]
    performance = best["performance"]
    ic = best["information_coefficient"]
    dsr = best["deflated_sharpe"]

    limitations = "\n".join(f"- {item}" for item in quality["limitations"])
    variant_rows = []
    for variant, payload in metrics["variants"].items():
        variant_rows.append(
            "| {variant} | {sharpe:.2f} | {dsr:.2f} | {prob:.1%} | {ic:.3f} |".format(
                variant=variant,
                sharpe=payload["performance"]["sharpe"],
                dsr=payload["deflated_sharpe"]["deflated_sharpe"],
                prob=payload["deflated_sharpe"]["probability"],
                ic=payload["information_coefficient"]["mean_rank_ic"],
            )
        )
    variants = "\n".join(variant_rows)

    return f"""# Cross-Sectional Alpha Research Memo

## Research Question

Can a small cross-sectional factor set produce a statistically defensible out-of-sample long-short signal after walk-forward validation, transaction costs, and multiple-testing correction?

## Data

- Universe: {quality["n_assets"]} synthetic liquid equities.
- Sample: {quality["start_date"]} to {quality["end_date"]}.
- Rows: {quality["n_rows"]:,}.
- Survivorship-bias-free: {"yes" if quality["survivorship_bias_free"] else "no"}.

### Limitations

{limitations}

## Method

The pipeline builds lagged price/volume factors, normalizes them cross-sectionally by date, predicts forward returns with linear and boosting models, validates with purged walk-forward folds, and trades a dollar-neutral long-short quantile portfolio with flat transaction costs.

## Results

- Best variant: `{best_variant}`.
- Raw Sharpe: {performance["sharpe"]:.2f}.
- Deflated Sharpe: {dsr["deflated_sharpe"]:.2f}.
- DSR probability: {dsr["probability"]:.1%}.
- Trials disclosed: {dsr["n_trials"]}.
- Mean rank IC: {ic["mean_rank_ic"]:.3f}.
- ICIR: {ic["icir"]:.2f}.
- Average turnover: {performance["average_turnover"]:.3f}.
- Max drawdown: {performance["max_drawdown"]:.1%}.

| Variant | Raw Sharpe | Deflated Sharpe | DSR Probability | Mean Rank IC |
|---|---:|---:|---:|---:|
{variants}

## Interpretation

The useful result is not a standalone Sharpe ratio. The useful result is the gap between raw performance and the deflated result after disclosing all tested variants. A production-quality extension should replace the synthetic universe with point-in-time prices and fundamentals, repeat the same validation protocol, and keep the variant ledger intact.

## What Would Most Likely Break Live

- Hidden survivorship or restatement bias in non-point-in-time data.
- Turnover rising when the universe is made more realistic.
- Signal decay under a market regime absent from the validation sample.
- Model selection pressure if failed variants stop being logged.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a research memo from experiment artifacts.")
    parser.add_argument("experiment_dir", nargs="?", default="artifacts/demo")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    memo = render_memo(args.experiment_dir)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(memo, encoding="utf-8")
    else:
        print(memo)


if __name__ == "__main__":
    main()
