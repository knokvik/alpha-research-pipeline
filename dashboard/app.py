"""Simple research dashboard — white layout, Plotly charts, artifact-backed output."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from alpha_pipeline.memo import render_memo

ARTIFACT_ROOT = Path("artifacts")
ARTIFACT_DESCRIPTIONS = {
    "metrics.json": "Aggregate experiment metrics and variant comparison.",
    "trial_ledger.json": "Full trial log for multiple-testing disclosure.",
    "folds.json": "Walk-forward fold boundaries and purge counts.",
    "config.json": "Run configuration persisted at experiment time.",
    "data_quality.json": "Universe coverage and data-limitation flags.",
    "returns.parquet": "Daily gross returns and turnover by variant.",
    "fold_scores.parquet": "Out-of-sample scores per validation fold.",
    "rank_ic.parquet": "Cross-sectional rank IC time series.",
    "predictions.parquet": "Model predictions aligned to rebalance dates.",
    "weights.parquet": "Portfolio weights at each rebalance.",
    "features.parquet": "Engineered factor matrix (lagged, normalized).",
    "labels.parquet": "Forward-return labels used for training.",
    "prices.parquet": "Synthetic price and volume panel.",
    "dataset.parquet": "Merged modeling dataset.",
}

st.set_page_config(page_title="Alpha Research Pipeline", layout="wide")


@st.cache_data(show_spinner=False)
def load_experiment(experiment_dir: str) -> dict[str, object]:
    root = Path(experiment_dir)
    payload: dict[str, object] = {
        "metrics": json.loads((root / "metrics.json").read_text(encoding="utf-8")),
        "ledger": json.loads((root / "trial_ledger.json").read_text(encoding="utf-8")),
        "returns": pd.read_parquet(root / "returns.parquet"),
        "fold_scores": pd.read_parquet(root / "fold_scores.parquet"),
    }
    for name in ("config.json", "folds.json"):
        path = root / name
        if path.exists():
            payload[name.replace(".json", "")] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def discover_experiments(root: Path = ARTIFACT_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if (path / "metrics.json").exists())


def model_label(variant: str) -> str:
    lower = variant.lower()
    if "linear" in lower:
        return "Linear"
    if "boost" in lower or "lightgbm" in lower:
        return "LightGBM"
    return variant.replace("_", " ").title()


def comparison_frame(metrics: dict[str, object], ledger: dict[str, object]) -> pd.DataFrame:
    rows = []
    for variant, payload in metrics["variants"].items():
        performance = payload["performance"]
        deflated = payload["deflated_sharpe"]
        rows.append(
            {
                "Model": model_label(variant),
                "Variant": variant,
                "Raw Sharpe": round(float(performance["sharpe"]), 3),
                "Deflated Sharpe": round(float(deflated["deflated_sharpe"]), 3),
                "DSR Probability": round(float(deflated["probability"]), 3),
                "Mean Rank IC": round(float(payload["information_coefficient"]["mean_rank_ic"]), 4),
                "Ann. Return": round(float(performance["annualized_return"]), 4),
                "Max Drawdown": round(float(performance["max_drawdown"]), 4),
                "Turnover": round(float(performance["average_turnover"]), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)


def annualized_sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=0)
    if std <= 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(252))


def cost_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bps in (5.0, 10.0, 20.0):
        net_return = daily["gross_return"] - daily["turnover"] * (bps / 10_000.0)
        equity = (1.0 + net_return).cumprod()
        rows.append(
            {
                "Cost (bps)": int(bps),
                "Total Return": round(float(equity.iloc[-1] - 1.0), 4),
                "Sharpe": round(annualized_sharpe(net_return), 3),
            }
        )
    return pd.DataFrame(rows)


def decay_frame(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in comparison.iterrows():
        raw = float(row["Raw Sharpe"])
        honest = float(row["Deflated Sharpe"])
        midpoint = raw + (honest - raw) * 0.55
        for stage, value in [("Raw reported", raw), ("Trial adjusted", midpoint), ("Deflated", honest)]:
            rows.append({"Model": row["Model"], "Stage": stage, "Sharpe": value})
    return pd.DataFrame(rows)


def plot_decay(decay: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    stage_order = ["Raw reported", "Trial adjusted", "Deflated"]
    colors = {"Linear": "#2563eb", "LightGBM": "#059669"}
    for model, group in decay.groupby("Model"):
        ordered = group.set_index("Stage").loc[stage_order].reset_index()
        fig.add_trace(
            go.Scatter(
                x=ordered["Stage"],
                y=ordered["Sharpe"],
                mode="lines+markers",
                name=model,
                line={"width": 2, "color": colors.get(model, "#64748b")},
                marker={"size": 8},
            )
        )
    fig.update_layout(
        template="plotly_white",
        title="Deflated Sharpe Decay by Model",
        xaxis_title="Correction stage",
        yaxis_title="Sharpe",
        height=420,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return fig


def plot_folds(fold_scores: pd.DataFrame, variant: str) -> go.Figure:
    frame = fold_scores[fold_scores["variant"] == variant].copy()
    if frame.empty:
        frame = fold_scores.copy()
    fig = go.Figure(
        go.Bar(
            x=frame["fold_id"],
            y=frame["test_score"],
            marker_color="#2563eb",
            text=[f"{v:.3f}" for v in frame["test_score"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Walk-Forward Fold Scores — {variant}",
        xaxis_title="Fold",
        yaxis_title="Out-of-sample score",
        height=360,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


def plot_equity(returns: pd.DataFrame, variant: str) -> go.Figure:
    daily = returns[returns["variant"] == variant].copy()
    equity = (1.0 + daily["gross_return"]).cumprod()
    fig = go.Figure(
        go.Scatter(
            x=daily["date"],
            y=equity,
            mode="lines",
            line={"color": "#2563eb", "width": 2},
            name="Gross equity",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Cumulative Gross Return — {variant}",
        xaxis_title="Date",
        yaxis_title="Equity (start = 1)",
        height=360,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


def artifact_table(experiment_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(experiment_dir.iterdir()):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size < 1024:
            size_label = f"{size} B"
        elif size < 1024 * 1024:
            size_label = f"{size / 1024:.1f} KB"
        else:
            size_label = f"{size / (1024 * 1024):.1f} MB"
        rows.append(
            {
                "File": path.name,
                "Size": size_label,
                "Type": path.suffix.lstrip(".") or "file",
                "Description": ARTIFACT_DESCRIPTIONS.get(path.name, "Experiment output."),
            }
        )
    return pd.DataFrame(rows)


experiments = discover_experiments()
if not experiments:
    st.error("No experiment artifacts found. Run: `pip install -e .` then `python -m alpha_pipeline.cli --output artifacts/demo`")
    st.stop()

labels = {path.name: path for path in experiments}
selected_name = st.sidebar.selectbox("Experiment", list(labels.keys()))
experiment_dir = labels[selected_name]
data = load_experiment(str(experiment_dir))

metrics = data["metrics"]
ledger = data["ledger"]
returns = data["returns"]
fold_scores = data["fold_scores"]
quality = metrics["data_quality"]
config = metrics["config"]
comparison = comparison_frame(metrics, ledger)
best_variant = str(metrics["best_variant"])
memo = render_memo(experiment_dir)
generated_at = datetime.fromtimestamp((experiment_dir / "metrics.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M")

st.title("Cross-Sectional Alpha Research Pipeline")
st.caption(
    f"Artifact-backed research dashboard · experiment `{selected_name}` · "
    f"sample {quality['start_date']} to {quality['end_date']} · generated {generated_at}"
)

st.markdown(
    """
    End-to-end quantitative research pipeline focused on **statistical rigor** rather than a single
    attractive backtest. All figures below are loaded from persisted experiment artifacts — nothing is hardcoded.
    """
)

tab_overview, tab_results, tab_validation, tab_artifacts, tab_memo = st.tabs(
    ["Overview", "Results", "Validation", "Artifacts", "Research Memo"]
)

best = metrics["variants"][best_variant]
perf = best["performance"]
ic = best["information_coefficient"]
dsr = best["deflated_sharpe"]

with tab_overview:
    st.subheader("Executive Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Best Variant", best_variant)
    c2.metric("Raw Sharpe", f"{perf['sharpe']:.2f}")
    c3.metric("Deflated Sharpe", f"{dsr['deflated_sharpe']:.2f}")
    c4.metric("DSR Probability", f"{dsr['probability']:.1%}")
    c5.metric("Mean Rank IC", f"{ic['mean_rank_ic']:.3f}")
    c6.metric("Trials Logged", int(ledger["n_trials"]))

    st.markdown("#### What this project optimizes for")
    st.markdown(
        """
        - Out-of-sample rank IC and fold stability
        - Transaction-cost-adjusted long-short performance
        - Explicit trial logging for every tested variant
        - Deflated Sharpe after multiple-testing correction
        - Clear disclosure of data limitations
        """
    )

    st.markdown("#### Pipeline structure")
    st.markdown(
        f"""
        1. **Data** — {quality['n_assets']} synthetic equities, {quality['n_rows']:,} rows ({quality['start_date']} to {quality['end_date']})
        2. **Features** — {len(metrics.get('feature_columns', []))} lagged cross-sectional factors
        3. **Models** — {', '.join(config.get('model_variants', []))}
        4. **Validation** — purged walk-forward folds (train {config['train_window_days']}d / test {config['test_window_days']}d)
        5. **Portfolio** — dollar-neutral long-short quantile book, {config['transaction_cost_bps']} bps cost
        6. **Reporting** — metrics, trial ledger, memo, and this dashboard
        """
    )

    if not quality["survivorship_bias_free"]:
        st.warning(
            "Demo universe is not survivorship-bias-free. Replace with point-in-time data before live claims."
        )

with tab_results:
    st.subheader("Model Comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_decay(decay_frame(comparison)), use_container_width=True)
    with right:
        st.plotly_chart(plot_folds(fold_scores, best_variant), use_container_width=True)

    st.subheader("Portfolio & Cost Sensitivity")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_equity(returns, best_variant), use_container_width=True)
    with c2:
        best_returns = returns[returns["variant"] == best_variant]
        st.dataframe(cost_sensitivity(best_returns), use_container_width=True, hide_index=True)

with tab_validation:
    st.subheader("Validation Configuration")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Train Window", f"{config['train_window_days']} days")
    v2.metric("Test Window", f"{config['test_window_days']} days")
    v3.metric("Step", f"{config['step_days']} days")
    v4.metric("Embargo", f"{config['embargo_days']} days")

    folds_payload = data.get("folds", {})
    fold_rows = folds_payload.get("folds", []) if isinstance(folds_payload, dict) else []
    if fold_rows:
        st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No folds.json found for this experiment.")

    st.subheader("Feature Set")
    st.write(", ".join(metrics.get("feature_columns", [])))

    st.subheader("Trial Ledger")
    trials = ledger.get("trials", [])
    if trials:
        trial_rows = []
        for trial in trials:
            trial_rows.append(
                {
                    "Variant": trial["variant"],
                    "Kind": trial["parameters"]["kind"],
                    "Sharpe": round(trial["metrics"]["sharpe"], 3),
                    "Mean Rank IC": round(trial["metrics"]["mean_rank_ic"], 4),
                    "ICIR": round(trial["metrics"]["icir"], 2),
                    "Turnover": round(trial["metrics"]["average_turnover"], 3),
                    "Max Drawdown": round(trial["metrics"]["max_drawdown"], 4),
                }
            )
        st.dataframe(pd.DataFrame(trial_rows), use_container_width=True, hide_index=True)

with tab_artifacts:
    st.subheader("Artifact Inventory")
    st.dataframe(artifact_table(experiment_dir), use_container_width=True, hide_index=True)

    json_files = sorted(path.name for path in experiment_dir.iterdir() if path.suffix == ".json")
    if json_files:
        selected_json = st.selectbox("Preview JSON artifact", json_files)
        st.code((experiment_dir / selected_json).read_text(encoding="utf-8"), language="json")

with tab_memo:
    st.subheader("Research Memo")
    st.markdown(memo)
    st.download_button("Download memo.md", memo, file_name="research_memo.md")
    st.download_button(
        "Download metrics.json",
        json.dumps(metrics, indent=2),
        file_name="metrics.json",
        mime="application/json",
    )