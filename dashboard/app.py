"""Streamlit dashboard for alpha research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from alpha_pipeline.memo import render_memo


DEFAULT_ARTIFACT_ROOT = Path("artifacts")


st.set_page_config(page_title="Alpha Research Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_artifacts(experiment_dir: str) -> dict[str, object]:
    root = Path(experiment_dir)
    return {
        "metrics": json.loads((root / "metrics.json").read_text(encoding="utf-8")),
        "ledger": json.loads((root / "trial_ledger.json").read_text(encoding="utf-8")),
        "returns": pd.read_parquet(root / "returns.parquet"),
        "rank_ic": pd.read_parquet(root / "rank_ic.parquet"),
        "predictions": pd.read_parquet(root / "predictions.parquet"),
        "weights": pd.read_parquet(root / "weights.parquet"),
        "fold_scores": pd.read_parquet(root / "fold_scores.parquet"),
    }


def discover_experiments(root: Path = DEFAULT_ARTIFACT_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if (path / "metrics.json").exists())


def metric_card(label: str, value: object) -> None:
    st.metric(label, value)


experiments = discover_experiments()
if not experiments:
    st.title("Alpha Research Dashboard")
    st.warning("No experiment artifacts found. Run `python -m alpha_pipeline.cli --output artifacts/demo`.")
    st.stop()

selected = st.sidebar.selectbox("Experiment", experiments, format_func=lambda path: path.name)
artifacts = load_artifacts(str(selected))
metrics = artifacts["metrics"]
ledger = artifacts["ledger"]
returns = artifacts["returns"]
rank_ic = artifacts["rank_ic"]
fold_scores = artifacts["fold_scores"]
weights = artifacts["weights"]

best_variant = metrics["best_variant"]
best_metrics = metrics["variants"][best_variant]
best_returns = returns[returns["variant"] == best_variant].copy()
best_rank_ic = rank_ic[rank_ic["variant"] == best_variant].copy()

st.title("Cross-Sectional Alpha Research")
st.caption(f"Best variant: {best_variant}")

top = st.columns(5)
with top[0]:
    metric_card("Raw Sharpe", f"{best_metrics['performance']['sharpe']:.2f}")
with top[1]:
    metric_card("Deflated Sharpe", f"{best_metrics['deflated_sharpe']['deflated_sharpe']:.2f}")
with top[2]:
    metric_card("DSR Probability", f"{best_metrics['deflated_sharpe']['probability']:.1%}")
with top[3]:
    metric_card("Mean Rank IC", f"{best_metrics['information_coefficient']['mean_rank_ic']:.3f}")
with top[4]:
    metric_card("Variants Tried", ledger["n_trials"])

quality = metrics["data_quality"]
with st.expander("Data Coverage And Bias Warnings", expanded=True):
    qcols = st.columns(5)
    qcols[0].metric("Assets", quality["n_assets"])
    qcols[1].metric("Rows", f"{quality['n_rows']:,}")
    qcols[2].metric("Start", quality["start_date"])
    qcols[3].metric("End", quality["end_date"])
    qcols[4].metric("PIT/SBF", "Yes" if quality["survivorship_bias_free"] else "No")
    for limitation in quality["limitations"]:
        st.warning(limitation)

left, right = st.columns((2, 1))
with left:
    st.subheader("Net Equity Curve")
    st.plotly_chart(
        px.line(best_returns, x="date", y="equity_curve", color="variant"),
        use_container_width=True,
    )
with right:
    st.subheader("Drawdown")
    drawdown = best_returns["equity_curve"] / best_returns["equity_curve"].cummax() - 1.0
    drawdown_frame = best_returns[["date", "variant"]].copy()
    drawdown_frame["drawdown"] = drawdown
    st.plotly_chart(px.area(drawdown_frame, x="date", y="drawdown"), use_container_width=True)

tabs = st.tabs(["Model Comparison", "Factor IC", "Folds", "Costs", "Trial Ledger", "Memo"])

with tabs[0]:
    comparison_rows = []
    for variant, payload in metrics["variants"].items():
        comparison_rows.append(
            {
                "variant": variant,
                **payload["performance"],
                **payload["information_coefficient"],
                **payload["deflated_sharpe"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(comparison, x="variant", y=["sharpe", "deflated_sharpe"], barmode="group"),
        use_container_width=True,
    )

with tabs[1]:
    st.plotly_chart(px.line(rank_ic, x="date", y="rank_ic", color="variant"), use_container_width=True)
    st.dataframe(rank_ic.groupby("variant")["rank_ic"].describe().reset_index(), use_container_width=True)

with tabs[2]:
    st.dataframe(fold_scores, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(fold_scores, x="fold_id", y="test_score", color="variant", barmode="group"), use_container_width=True)

with tabs[3]:
    st.plotly_chart(
        px.line(best_returns, x="date", y=["gross_return", "net_return", "transaction_cost"]),
        use_container_width=True,
    )
    st.plotly_chart(px.line(best_returns, x="date", y="turnover"), use_container_width=True)
    latest_weights = weights[weights["variant"] == best_variant].sort_values("date").groupby("asset").tail(1)
    st.dataframe(latest_weights.sort_values("weight"), use_container_width=True, hide_index=True)

with tabs[4]:
    st.dataframe(pd.DataFrame(ledger["trials"]), use_container_width=True, hide_index=True)

with tabs[5]:
    memo = render_memo(Path(selected))
    st.download_button("Download Memo", memo, file_name=f"{selected.name}_research_memo.md")
    st.markdown(memo)
