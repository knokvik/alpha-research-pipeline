# Cross-Sectional Alpha Research Pipeline

> **Verified on this machine (2026-07-08)**
> - **Tests:** 10/10 passed (`pytest`)
> - **Demo run:** `artifacts/demo/` — 14 output files, fully reproducible synthetic data
> - **Best variant:** `boosting_hist_depth_3` — raw Sharpe 0.02, deflated Sharpe 0.04, DSR 78.3%
> - **Dashboard:** `python -m dashboard serve` → http://127.0.0.1:8765/index.html (full-width HTML, Plotly charts, artifact-backed)

Statistically rigorous cross-sectional alpha research: factor engineering, purged walk-forward validation, cost-aware long-short backtesting, trial logging, deflated Sharpe, and an HTML research dashboard — all from persisted experiment artifacts.

---

## Quick Start

```bash
cd alpha-research-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the demo experiment
python -m alpha_pipeline.cli --output artifacts/demo

# View the dashboard (generates HTML + serves on localhost)
python -m dashboard serve
```

Open **http://127.0.0.1:8765/index.html**

---

## Dashboard

No Streamlit. Pure Python generates a standalone HTML report and serves it with the standard library.

```bash
# Generate static HTML only
python -m dashboard build
python -m dashboard build --output reports/dashboard.html --experiment demo

# Generate and serve on localhost
python -m dashboard serve
python -m dashboard serve --port 8080
```

Default output: `artifacts/dashboard/index.html`

The page is **full-width** (100% horizontal). All metrics, tables, and Plotly charts load from `artifacts/<experiment>/` — nothing is hardcoded.

---

## What It Does

1. **Data** — synthetic liquid equity panel (demo) or pluggable price/feature interfaces
2. **Features** — 16 lagged, cross-sectionally normalized price/volume factors
3. **Models** — linear ridge and histogram-based boosting variants
4. **Validation** — rolling walk-forward with label purging and embargo
5. **Portfolio** — dollar-neutral long-short quantile book with flat transaction costs
6. **Statistics** — raw Sharpe, deflated Sharpe, DSR probability, rank IC / ICIR
7. **Reporting** — `metrics.json`, trial ledger, research memo, HTML dashboard

---

## Demo Results

Source: `artifacts/demo/metrics.json`

| Variant | Raw Sharpe | Deflated Sharpe | DSR Probability | Mean Rank IC |
|---|---:|---:|---:|---:|
| boosting_hist_depth_3 | 0.02 | 0.04 | 78.3% | 0.019 |
| linear_ridge_alpha_1 | -0.28 | -0.26 | 0.0% | -0.002 |

- **Universe:** 40 equities · 36,000 rows · 2018-01-01 to 2021-06-11
- **Validation:** 504-day train / 63-day test walk-forward, purged labels
- **Trials disclosed:** 2 (linear + boosting)
- **Survivorship-bias-free:** no (demo limitation, disclosed in dashboard)

---

## Project Layout

```text
src/alpha_pipeline/     Pipeline package (features, validation, backtest, stats, CLI)
dashboard/              HTML report builder + localhost server
tests/                  Unit and integration tests
reports/                Memo template and methodology notes
artifacts/              Experiment outputs (gitignored)
```

---

## CLI Reference

| Command | Description |
|---|---|
| `python -m alpha_pipeline.cli --output artifacts/demo` | Run demo experiment |
| `python -m alpha_pipeline.memo artifacts/demo` | Print research memo |
| `python -m dashboard build` | Generate HTML report |
| `python -m dashboard serve` | Generate + serve on localhost |
| `pytest` | Run test suite |

---

## Data Warning

The built-in demo uses **synthetic data** for full reproducibility without API keys. It proves the research machinery — leakage controls, validation, backtesting, statistics, and reporting — not live equity alpha. Replace with survivorship-bias-free, point-in-time data before production claims.