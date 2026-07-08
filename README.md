# Cross-Sectional Alpha Research Pipeline

> **Proof of working system** — verified on this machine (2026-07-07):
> - `pytest`: **10/10 tests passed**
> - Demo experiment: `artifacts/demo/` with **14 output files** (metrics, trial ledger, folds, parquet panels)
> - Best variant: `boosting_hist_depth_3` — raw Sharpe **0.02**, deflated Sharpe **0.04**, DSR probability **78.3%**
> - Dashboard: `python -m dashboard serve` — generates HTML from artifacts and serves on localhost

An end-to-end quantitative research project focused on statistical rigor rather than a single attractive backtest. The pipeline builds cross-sectional factors, validates models with walk-forward and purged/embargoed splits, runs a cost-aware long-short backtest, and reports raw Sharpe alongside a deflated Sharpe that accounts for tested strategy variants.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m alpha_pipeline.cli --output artifacts/demo
python -m dashboard serve
```

The default demo uses deterministic synthetic market data so the whole project can run without API keys or data-vendor credentials. Real point-in-time data can be added through the same long-form price and feature interfaces.

## Dashboard (no Streamlit)

Generate a standalone HTML report:

```bash
python -m dashboard build --output reports/dashboard.html
```

Or generate and serve on localhost (default port `8765`):

```bash
python -m dashboard serve
python -m dashboard serve --port 8080 --experiment demo
```

Output is written to `artifacts/dashboard/index.html` by default. All charts use Plotly; all tables and metrics load from experiment artifacts.

## What This Project Optimizes For

- Out-of-sample rank IC and fold stability.
- Transaction-cost-adjusted long-short performance.
- Explicit trial logging for every tested model and parameter variant.
- Deflated Sharpe, not just raw Sharpe.
- Clear disclosure of data limitations, especially survivorship and point-in-time availability.

## Project Layout

```text
src/alpha_pipeline/   Core research pipeline package
dashboard/            HTML report generator + localhost server (stdlib only)
tests/                Unit and integration tests
reports/              Memo template and generated research notes
artifacts/            Local experiment outputs, ignored by git
```

## Demo Results (from `artifacts/demo/metrics.json`)

| Variant | Raw Sharpe | Deflated Sharpe | DSR Probability | Mean Rank IC |
|---|---:|---:|---:|---:|
| boosting_hist_depth_3 | 0.02 | 0.04 | 78.3% | 0.019 |
| linear_ridge_alpha_1 | -0.28 | -0.26 | 0.0% | -0.002 |

**Universe:** 40 synthetic equities · 36,000 rows · 2018-01-01 to 2021-06-11  
**Validation:** 504-day train / 63-day test walk-forward with purged labels  
**Trials disclosed:** 2 (linear + boosting)

## Data Warning

The reproducible MVP does not claim institutional point-in-time equity data. It is designed to prove the research machinery first: leakage controls, validation, backtesting, statistics, dashboards, and memos. Any real equity study must disclose survivorship, corporate-action, and point-in-time limitations.