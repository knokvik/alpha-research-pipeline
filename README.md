# Cross-Sectional Alpha Research Pipeline

An end-to-end quantitative research project focused on statistical rigor rather than a single attractive backtest. The pipeline builds cross-sectional factors, validates models with walk-forward and purged/embargoed splits, runs a cost-aware long-short backtest, and reports raw Sharpe alongside a deflated Sharpe that accounts for tested strategy variants.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m alpha_pipeline.cli --output artifacts/demo
streamlit run dashboard/app.py
```

The default demo uses deterministic synthetic market data so the whole project can run without API keys or data-vendor credentials. Real point-in-time data can be added through the same long-form price and feature interfaces.

## What This Project Optimizes For

- Out-of-sample rank IC and fold stability.
- Transaction-cost-adjusted long-short performance.
- Explicit trial logging for every tested model and parameter variant.
- Deflated Sharpe, not just raw Sharpe.
- Clear disclosure of data limitations, especially survivorship and point-in-time availability.

## Project Layout

```text
src/alpha_pipeline/   Core research pipeline package
dashboard/            Streamlit output documentation console (Overview, Reports, Artifacts)
tests/                Unit and integration tests
reports/              Memo template and generated research notes
artifacts/            Local experiment outputs, ignored by git
```

## Data Warning

The reproducible MVP does not claim institutional point-in-time equity data. It is designed to prove the research machinery first: leakage controls, validation, backtesting, statistics, dashboards, and memos. Any real equity study must disclose survivorship, corporate-action, and point-in-time limitations.
