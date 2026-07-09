## Cross-Sectional Alpha Research Pipeline

> **Verified output from `artifacts/demo/` (2026-07-08)**
> - **Tests:** 10/10 passed (`pytest`)
> - **Artifacts:** 14 files (metrics, trial ledger, folds, parquet panels)
> - **Best variant:** `boosting_hist_depth_3`
> - **Raw Sharpe:** 0.02 · **Deflated Sharpe:** 0.04 · **DSR probability:** 78.3%
> - **Mean rank IC:** 0.019 · **ICIR:** 1.91 · **Trials disclosed:** 2
> - **Dashboard:** `python -m dashboard serve` → HTML report from `metrics.json`

Statistically rigorous cross-sectional alpha research: factor engineering, purged walk-forward validation, cost-aware long-short backtesting, trial logging, deflated Sharpe, and an HTML research dashboard — all from persisted experiment artifacts.

---

### Research Question

Can a small cross-sectional factor set produce a statistically defensible out-of-sample long-short signal after walk-forward validation, transaction costs, and multiple-testing correction?

---

### Quick Start

```bash
cd alpha-research-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m alpha_pipeline.cli --output artifacts/demo
python -m dashboard serve
```

Open **http://127.0.0.1:8765/index.html**

---

### Demo Output

Source: `artifacts/demo/metrics.json` and `reports/generated/demo_memo.md`

**Data**
- Universe: 40 synthetic liquid equities
- Sample: 2018-01-01 to 2021-06-11
- Rows: 36,000
- Survivorship-bias-free: no

**Best variant: `boosting_hist_depth_3`**
- Raw Sharpe: 0.02
- Deflated Sharpe: 0.04
- DSR probability: 78.3%
- Mean rank IC: 0.019
- ICIR: 1.91
- Average turnover: 0.205
- Max drawdown: -9.1%

| Variant | Raw Sharpe | Deflated Sharpe | DSR Probability | Mean Rank IC |
|---|---:|---:|---:|---:|
| boosting_hist_depth_3 | 0.02 | 0.04 | 78.3% | 0.019 |
| linear_ridge_alpha_1 | -0.28 | -0.26 | 0.0% | -0.002 |

**Validation**
- Train window: 504 days · Test window: 63 days · Step: 63 days · Embargo: 5 days
- Rebalance: W-FRI · Transaction cost: 5.0 bps
- Models tested: linear, boosting

**Interpretation**

The useful result is not a standalone Sharpe ratio. The useful result is the gap between raw performance and the deflated result after disclosing all tested variants.

---

### Dashboard

```bash
python -m dashboard build --output reports/dashboard.html
python -m dashboard serve
python -m dashboard serve --port 8080
```

Default output: `artifacts/dashboard/index.html` — full-width layout with 2rem horizontal padding, Plotly charts, artifact-backed tables.

---

### What It Does

1. **Data** — synthetic liquid equity panel (demo) or pluggable price/feature interfaces
2. **Features** — 16 lagged, cross-sectionally normalized price/volume factors
3. **Models** — linear ridge and histogram-based boosting variants
4. **Validation** — rolling walk-forward with label purging and embargo
5. **Portfolio** — dollar-neutral long-short quantile book with flat transaction costs
6. **Statistics** — raw Sharpe, deflated Sharpe, DSR probability, rank IC / ICIR
7. **Reporting** — `metrics.json`, trial ledger, research memo, HTML dashboard

---

### Project Layout

```text
src/alpha_pipeline/     Pipeline package (features, validation, backtest, stats, CLI)
dashboard/              HTML report builder + localhost server
tests/                  Unit and integration tests
reports/                Memo template and generated research notes
artifacts/              Experiment outputs (gitignored)
```

---

### CLI Reference

| Command | Description |
|---|---|
| `python -m alpha_pipeline.cli --output artifacts/demo` | Run demo experiment |
| `python -m alpha_pipeline.memo artifacts/demo` | Print research memo |
| `python -m dashboard build` | Generate HTML report |
| `python -m dashboard serve` | Generate + serve on localhost |
| `pytest` | Run test suite |

---

### Data Warning

The built-in demo uses **synthetic data** for full reproducibility without API keys. It proves the research machinery — leakage controls, validation, backtesting, statistics, and reporting — not live equity alpha. Replace with survivorship-bias-free, point-in-time data before production claims.