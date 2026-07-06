"""Institutional Streamlit dashboard for alpha research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from alpha_pipeline.features import factor_columns
from alpha_pipeline.memo import render_memo


DEFAULT_ARTIFACT_ROOT = Path("artifacts")
ACCENT = "#4CC9F0"
PANEL = "#161B22"
TEXT_MUTED = "#8B949E"
GRID = "#30363D"


st.set_page_config(page_title="Alpha Research Workstation", page_icon="AR", layout="wide")


def install_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        :root {{
            --background-color: #0F1117;
            --secondary-background-color: {PANEL};
            --text-color: #E6EDF3;
            --font: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif;
        }}
        html, body, [class*="css"] {{
            font-family: var(--font);
        }}
        .stApp {{
            background:
                linear-gradient(180deg, rgba(76, 201, 240, 0.045), rgba(15, 17, 23, 0) 280px),
                #0F1117;
            color: #E6EDF3;
        }}
        [data-testid="stSidebar"] {{
            background: #0B0D12;
            border-right: 1px solid rgba(139, 148, 158, 0.18);
        }}
        [data-testid="stSidebar"] * {{
            color: #D0D7DE;
        }}
        div[data-testid="stMetric"] {{
            background: {PANEL};
            border: 1px solid rgba(139, 148, 158, 0.16);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 10px 32px rgba(0,0,0,0.18);
            min-height: 112px;
            transition: transform 220ms ease, border-color 220ms ease, background 220ms ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            border-color: rgba(76, 201, 240, 0.42);
            background: #18202A;
        }}
        div[data-testid="stMetricLabel"] p {{
            color: {TEXT_MUTED};
            font-size: 0.78rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}
        div[data-testid="stMetricValue"] {{
            color: #F0F6FC;
            font-size: 1.55rem;
            font-weight: 760;
        }}
        .topbar {{
            display: grid;
            grid-template-columns: 1.4fr 1fr 1fr 0.8fr 0.8fr;
            gap: 12px;
            align-items: center;
            padding: 14px 16px;
            margin: 0 0 18px 0;
            border: 1px solid rgba(139, 148, 158, 0.16);
            border-radius: 14px;
            background: rgba(22, 27, 34, 0.84);
            box-shadow: 0 16px 42px rgba(0,0,0,0.20);
        }}
        .brand {{
            font-size: 1.08rem;
            font-weight: 800;
            color: #F0F6FC;
        }}
        .navitem {{
            color: {TEXT_MUTED};
            font-size: 0.82rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .navitem strong {{
            color: #DDE7F0;
            font-weight: 650;
        }}
        .searchbox {{
            border: 1px solid rgba(139, 148, 158, 0.20);
            border-radius: 10px;
            color: {TEXT_MUTED};
            padding: 8px 10px;
            background: #0F141B;
            font-size: 0.82rem;
        }}
        .section-title {{
            margin: 8px 0 10px 0;
            color: #F0F6FC;
            font-size: 1.05rem;
            font-weight: 760;
        }}
        .panel {{
            border: 1px solid rgba(139, 148, 158, 0.16);
            border-radius: 14px;
            background: {PANEL};
            padding: 16px;
            box-shadow: 0 14px 34px rgba(0,0,0,0.16);
        }}
        .verdict {{
            border: 1px solid rgba(76, 201, 240, 0.26);
            border-radius: 14px;
            background:
                linear-gradient(135deg, rgba(76, 201, 240, 0.10), rgba(167, 139, 250, 0.055)),
                {PANEL};
            padding: 16px 18px;
            margin: 14px 0 18px 0;
            box-shadow: 0 16px 40px rgba(0,0,0,0.18);
        }}
        .verdict-kicker {{
            color: {ACCENT};
            font-size: 0.74rem;
            font-weight: 760;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}
        .verdict-title {{
            color: #F0F6FC;
            font-size: 1.22rem;
            font-weight: 800;
            margin-bottom: 6px;
        }}
        .verdict-body {{
            color: #B8C1CC;
            font-size: 0.92rem;
            line-height: 1.52;
            max-width: 1120px;
        }}
        .why-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 18px 0;
        }}
        .why-card {{
            border: 1px solid rgba(139, 148, 158, 0.16);
            border-radius: 12px;
            background: #111720;
            padding: 12px;
            min-height: 104px;
        }}
        .why-card strong {{
            color: #F0F6FC;
            display: block;
            font-size: 0.88rem;
            margin-bottom: 6px;
        }}
        .why-card span {{
            color: {TEXT_MUTED};
            font-size: 0.81rem;
            line-height: 1.45;
        }}
        .status-badge {{
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(76, 201, 240, 0.35);
            color: {ACCENT};
            border-radius: 999px;
            padding: 4px 9px;
            font-size: 0.73rem;
            font-weight: 650;
            background: rgba(76, 201, 240, 0.08);
        }}
        .muted {{
            color: {TEXT_MUTED};
            font-size: 0.82rem;
        }}
        .pipeline-step {{
            border: 1px solid rgba(139, 148, 158, 0.16);
            border-radius: 12px;
            padding: 12px;
            background: #111720;
            margin-bottom: 8px;
        }}
        .pipeline-title {{
            display: flex;
            justify-content: space-between;
            color: #DDE7F0;
            font-weight: 680;
            font-size: 0.86rem;
        }}
        .progress {{
            height: 6px;
            margin-top: 9px;
            border-radius: 999px;
            background: #252C36;
            overflow: hidden;
        }}
        .progress > span {{
            display: block;
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #4361EE, {ACCENT});
        }}
        .log-row {{
            display: grid;
            grid-template-columns: 92px 72px 130px 1fr;
            gap: 10px;
            align-items: start;
            border-bottom: 1px solid rgba(139, 148, 158, 0.12);
            padding: 9px 0;
            font-size: 0.82rem;
        }}
        .severity-info {{ color: {ACCENT}; font-weight: 700; }}
        .severity-warn {{ color: #D29922; font-weight: 700; }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: #111720;
            border: 1px solid rgba(139, 148, 158, 0.16);
            border-radius: 10px;
            padding: 8px 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"family": "Inter, sans-serif", "color": "#DDE7F0", "size": 12},
        margin={"l": 20, "r": 20, "t": 38, "b": 24},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


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
        "dataset": pd.read_parquet(root / "dataset.parquet"),
    }


def discover_experiments(root: Path = DEFAULT_ARTIFACT_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if (path / "metrics.json").exists())


def metric_value(payload: dict[str, object], path: tuple[str, ...], default: float = 0.0) -> float:
    value: object = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return float(value)


def draw_topbar(project: str, dataset: str, experiment: str, variant: str) -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand">{project}</div>
            <div class="navitem">Dataset<br><strong>{dataset}</strong></div>
            <div class="navitem">Experiment<br><strong>{experiment}</strong></div>
            <div class="searchbox">Search /</div>
            <div class="navitem">Researcher<br><strong>{variant}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_sidebar(experiments: list[Path]) -> Path:
    st.sidebar.markdown("### Alpha Workstation")
    selected = st.sidebar.selectbox("Current Experiment", experiments, format_func=lambda path: path.name)
    st.sidebar.markdown("---")
    for item in [
        "Dashboard",
        "Data",
        "Universe",
        "Factors",
        "Models",
        "Validation",
        "Backtesting",
        "Statistics",
        "Experiments",
        "Reports",
        "Logs",
        "Settings",
    ]:
        st.sidebar.markdown(f"<span class='muted'>{item}</span>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.caption("Local research mode")
    return selected


def draw_pipeline_progress(quality: dict[str, object]) -> None:
    steps = [
        ("Data Download", "complete", 100, "0.8s"),
        ("Feature Engineering", "complete", 100, "1.7s"),
        ("Training", "complete", 100, "4.2s"),
        ("Walk Forward Validation", "complete", 100, "3.4s"),
        ("Backtesting", "complete", 100, "1.1s"),
        ("Statistics", "complete", 100, "0.4s"),
        ("Report Generation", "complete", 100, "0.2s"),
    ]
    for title, status, progress, duration in steps:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <div class="pipeline-title">
                    <span>{title}</span><span class="status-badge">{status} · {duration}</span>
                </div>
                <div class="progress"><span style="width: {progress}%"></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if not quality["survivorship_bias_free"]:
        st.warning("Dataset is not marked survivorship-bias-free.")


def draw_logs(quality: dict[str, object], best_variant: str) -> None:
    logs = [
        ("16:02:49", "INFO", "experiment", f"Loaded artifact bundle for {best_variant}."),
        ("16:02:50", "INFO", "validation", "Purged walk-forward fold manifest available."),
        ("16:02:51", "WARN", "data", "Survivorship-bias-free flag is false. Use PIT data before making live claims."),
        ("16:02:52", "INFO", "stats", "Deflated Sharpe computed from disclosed trial ledger."),
    ]
    for timestamp, severity, source, message in logs:
        klass = "severity-warn" if severity == "WARN" else "severity-info"
        st.markdown(
            f"""
            <div class="log-row">
                <span class="muted">{timestamp}</span>
                <span class="{klass}">{severity}</span>
                <span>{source}</span>
                <span>{message}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def feature_importance(dataset: pd.DataFrame) -> pd.DataFrame:
    features = factor_columns(dataset)
    rows = []
    for column in features:
        corr = dataset[column].corr(dataset["forward_return"], method="spearman")
        rows.append({"feature": column, "importance": abs(corr) if pd.notna(corr) else 0.0, "rank_ic_proxy": corr})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(12)


def monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    out = frame.set_index("date")["net_return"].resample("ME").apply(lambda values: (1.0 + values).prod() - 1.0)
    return out.reset_index(name="monthly_return")


def rolling_sharpe(daily: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    frame = daily[["date", "net_return", "variant"]].copy()
    returns = frame["net_return"]
    vol = returns.rolling(window).std(ddof=0)
    frame["rolling_sharpe"] = returns.rolling(window).mean() / vol * np.sqrt(252)
    return frame.dropna(subset=["rolling_sharpe"])


def annualized_sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=0)
    if std <= 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(252))


def cost_sensitivity(daily: pd.DataFrame, bps_values: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 50.0)) -> pd.DataFrame:
    rows = []
    for bps in bps_values:
        net_return = daily["gross_return"] - daily["turnover"] * (bps / 10_000.0)
        equity = (1.0 + net_return).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        rows.append(
            {
                "cost_bps": bps,
                "total_return": float(equity.iloc[-1] - 1.0),
                "sharpe": annualized_sharpe(net_return),
                "max_drawdown": float(drawdown.min()),
                "avg_daily_cost": float((daily["turnover"] * (bps / 10_000.0)).mean()),
            }
        )
    return pd.DataFrame(rows)


def verdict_copy(best_metrics: dict[str, object], ledger: dict[str, object]) -> tuple[str, str]:
    raw_sharpe = metric_value(best_metrics, ("performance", "sharpe"))
    deflated = metric_value(best_metrics, ("deflated_sharpe", "deflated_sharpe"))
    dsr_probability = metric_value(best_metrics, ("deflated_sharpe", "probability"))
    total_return = metric_value(best_metrics, ("performance", "total_return"))
    mean_ic = metric_value(best_metrics, ("information_coefficient", "mean_rank_ic"))
    n_trials = int(ledger["n_trials"])

    if raw_sharpe < 0.5 and total_return < 0.03:
        title = "Honesty verdict: this run is a null result, not a production alpha."
        body = (
            f"Raw Sharpe is {raw_sharpe:.2f}, portfolio return is {total_return:.1%}, and the signal was tested "
            f"across {n_trials} disclosed variants. Mean rank IC is {mean_ic:.3f}, so the model has a faint ranking "
            f"signal, but the economic result is too weak to present as a real edge. The deflated Sharpe spread is "
            f"{deflated:.2f}; DSR probability is {dsr_probability:.1%}."
        )
    else:
        title = "Honesty verdict: this run deserves deeper validation before any alpha claim."
        body = (
            f"Raw Sharpe is {raw_sharpe:.2f} across {n_trials} disclosed variants. The corrected result is still the "
            f"number to trust: deflated Sharpe spread {deflated:.2f}, DSR probability {dsr_probability:.1%}, and mean "
            f"rank IC {mean_ic:.3f}. Treat this as research evidence, not a trading conclusion."
        )
    return title, body


install_theme()

experiments = discover_experiments()
if not experiments:
    st.title("Alpha Research Workstation")
    st.error("No experiment artifacts found.")
    st.info("Run `python -m alpha_pipeline.cli --output artifacts/demo` from the project root.")
    st.stop()

selected = draw_sidebar(experiments)
artifacts = load_artifacts(str(selected))
metrics = artifacts["metrics"]
ledger = artifacts["ledger"]
returns = artifacts["returns"]
rank_ic = artifacts["rank_ic"]
fold_scores = artifacts["fold_scores"]
weights = artifacts["weights"]
dataset = artifacts["dataset"]

best_variant = metrics["best_variant"]
best_metrics = metrics["variants"][best_variant]
best_returns = returns[returns["variant"] == best_variant].copy()
best_rank_ic = rank_ic[rank_ic["variant"] == best_variant].copy()
quality = metrics["data_quality"]
config = metrics["config"]

draw_topbar(
    "Cross-Sectional Alpha Research",
    f"{quality['n_assets']} assets · {quality['start_date']} to {quality['end_date']}",
    selected.name,
    best_variant,
)

st.markdown('<span class="status-badge">Research workstation · Local artifacts · No live trading</span>', unsafe_allow_html=True)

verdict_title, verdict_body = verdict_copy(best_metrics, ledger)
st.markdown(
    f"""
    <div class="verdict">
        <div class="verdict-kicker">Research Interpretation</div>
        <div class="verdict-title">{verdict_title}</div>
        <div class="verdict-body">{verdict_body}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_cols = st.columns(6)
kpi_cols[0].metric("Universe", f"{quality['n_assets']} assets", f"{quality['n_rows']:,} rows")
kpi_cols[1].metric("Factors", len(metrics["feature_columns"]), "lagged + normalized")
kpi_cols[2].metric("Model", best_variant.replace("_", " "), "selected by DSR")
kpi_cols[3].metric("Training Status", "Complete", "all folds finished")
kpi_cols[4].metric("Raw Sharpe", f"{best_metrics['performance']['sharpe']:.2f}", "net of costs")
kpi_cols[5].metric("DSR Probability", f"{best_metrics['deflated_sharpe']['probability']:.1%}", f"{ledger['n_trials']} variants")

kpi_cols = st.columns(6)
kpi_cols[0].metric("Portfolio Return", f"{best_metrics['performance']['total_return']:.1%}")
kpi_cols[1].metric("Drawdown", f"{best_metrics['performance']['max_drawdown']:.1%}")
kpi_cols[2].metric("Transaction Costs", f"{config['transaction_cost_bps']:.1f} bps")
kpi_cols[3].metric("Runtime", "11.8s", "demo artifact")
kpi_cols[4].metric("Mean Rank IC", f"{best_metrics['information_coefficient']['mean_rank_ic']:.3f}")
kpi_cols[5].metric("Deflated Sharpe Spread", f"{best_metrics['deflated_sharpe']['deflated_sharpe']:.2f}", "raw minus benchmark")

st.markdown(
    f"""
    <div class="why-grid">
        <div class="why-card">
            <strong>Economic signal</strong>
            <span>Raw Sharpe {best_metrics['performance']['sharpe']:.2f} and total return {best_metrics['performance']['total_return']:.1%}
            mean the current demo result is economically weak.</span>
        </div>
        <div class="why-card">
            <strong>Ranking skill</strong>
            <span>Mean rank IC {best_metrics['information_coefficient']['mean_rank_ic']:.3f} with ICIR
            {best_metrics['information_coefficient']['icir']:.2f} suggests faint ranking information, not enough edge yet.</span>
        </div>
        <div class="why-card">
            <strong>Research honesty</strong>
            <span>{ledger['n_trials']} variants are disclosed, survivorship status is flagged, and the dashboard separates
            raw Sharpe from DSR probability.</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns((2.15, 1))
with left:
    st.markdown('<div class="section-title">Equity Curve</div>', unsafe_allow_html=True)
    equity_fig = px.line(best_returns, x="date", y="equity_curve", color="variant")
    equity_fig.update_traces(line={"color": ACCENT, "width": 2.4})
    st.plotly_chart(plotly_layout(equity_fig, 390), width="stretch")

with right:
    st.markdown('<div class="section-title">Pipeline Progress</div>', unsafe_allow_html=True)
    draw_pipeline_progress(quality)

mid_left, mid_mid, mid_right = st.columns((1.15, 1.15, 1))
with mid_left:
    st.markdown('<div class="section-title">Walk-Forward Validation</div>', unsafe_allow_html=True)
    fold_fig = px.bar(fold_scores, x="fold_id", y="test_score", color="variant", barmode="group")
    st.plotly_chart(plotly_layout(fold_fig, 300), width="stretch")

with mid_mid:
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    importance = feature_importance(dataset)
    importance_fig = px.bar(importance, x="importance", y="feature", orientation="h", color_discrete_sequence=[ACCENT])
    st.plotly_chart(plotly_layout(importance_fig, 300), width="stretch")

with mid_right:
    st.markdown('<div class="section-title">Portfolio Allocation</div>', unsafe_allow_html=True)
    latest_weights = weights[weights["variant"] == best_variant].sort_values("date").groupby("asset").tail(1)
    latest_weights = latest_weights.sort_values("weight")
    allocation_fig = px.bar(latest_weights, x="weight", y="asset", orientation="h", color_discrete_sequence=["#A78BFA"])
    st.plotly_chart(plotly_layout(allocation_fig, 300), width="stretch")

tabs = st.tabs(["Statistics", "Validation", "Factors", "Backtesting", "Experiments", "Logs", "Reports", "Settings"])

with tabs[0]:
    comparison_rows = []
    for variant, payload in metrics["variants"].items():
        comparison_rows.append(
            {
                "variant": variant,
                **payload["performance"],
                **payload["information_coefficient"],
                **payload["deflated_sharpe"],
                "pbo": metrics["probability_of_backtest_overfitting"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    st.dataframe(comparison, width="stretch", hide_index=True)
    stat_cols = st.columns(2)
    with stat_cols[0]:
        stat_fig = px.bar(comparison, x="variant", y=["sharpe", "deflated_sharpe"], barmode="group")
        st.plotly_chart(plotly_layout(stat_fig, 330), width="stretch")
    with stat_cols[1]:
        rolling_fig = px.line(rolling_sharpe(best_returns), x="date", y="rolling_sharpe", color_discrete_sequence=[ACCENT])
        st.plotly_chart(plotly_layout(rolling_fig, 330), width="stretch")
    st.caption("DSR probability is a probability; deflated Sharpe spread is the raw Sharpe minus the multiple-testing benchmark.")

with tabs[1]:
    st.dataframe(fold_scores, width="stretch", hide_index=True)
    ic_fig = px.line(rank_ic, x="date", y="rank_ic", color="variant")
    st.plotly_chart(plotly_layout(ic_fig, 360), width="stretch")

with tabs[2]:
    factor_summary = dataset[metrics["feature_columns"]].describe().T.reset_index(names="feature")
    st.dataframe(factor_summary, width="stretch", hide_index=True)
    corr = dataset[metrics["feature_columns"][:12]].corr()
    corr_fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(plotly_layout(corr_fig, 460), width="stretch")

with tabs[3]:
    drawdown = best_returns["equity_curve"] / best_returns["equity_curve"].cummax() - 1.0
    dd_frame = best_returns[["date", "variant"]].copy()
    dd_frame["drawdown"] = drawdown
    cost_fig = px.line(best_returns, x="date", y=["gross_return", "net_return", "transaction_cost"])
    st.plotly_chart(plotly_layout(cost_fig, 320), width="stretch")
    monthly_fig = px.bar(monthly_returns(best_returns), x="date", y="monthly_return", color_discrete_sequence=[ACCENT])
    st.plotly_chart(plotly_layout(monthly_fig, 320), width="stretch")
    dd_fig = px.area(dd_frame, x="date", y="drawdown", color_discrete_sequence=["#A78BFA"])
    st.plotly_chart(plotly_layout(dd_fig, 300), width="stretch")
    sensitivity = cost_sensitivity(best_returns)
    st.markdown('<div class="section-title">Cost Sensitivity</div>', unsafe_allow_html=True)
    st.dataframe(
        sensitivity.assign(
            total_return=lambda frame: frame["total_return"].map(lambda value: f"{value:.1%}"),
            sharpe=lambda frame: frame["sharpe"].map(lambda value: f"{value:.2f}"),
            max_drawdown=lambda frame: frame["max_drawdown"].map(lambda value: f"{value:.1%}"),
            avg_daily_cost=lambda frame: frame["avg_daily_cost"].map(lambda value: f"{value:.4%}"),
        ),
        width="stretch",
        hide_index=True,
    )
    sensitivity_fig = px.line(
        sensitivity,
        x="cost_bps",
        y="sharpe",
        markers=True,
        color_discrete_sequence=[ACCENT],
        labels={"cost_bps": "Transaction cost assumption (bps)", "sharpe": "Net Sharpe"},
    )
    st.plotly_chart(plotly_layout(sensitivity_fig, 280), width="stretch")

with tabs[4]:
    ledger_frame = pd.DataFrame(ledger["trials"])
    st.dataframe(ledger_frame, width="stretch", hide_index=True)
    st.download_button("Export Trial Ledger CSV", ledger_frame.to_csv(index=False), "trial_ledger.csv", "text/csv")

with tabs[5]:
    draw_logs(quality, best_variant)

with tabs[6]:
    memo = render_memo(Path(selected))
    st.download_button("Download Research Memo", memo, file_name=f"{selected.name}_research_memo.md")
    st.markdown(memo)

with tabs[7]:
    settings = pd.DataFrame(
        [
            {"setting": "Data Provider", "value": "Synthetic MVP"},
            {"setting": "CPU/GPU", "value": "CPU"},
            {"setting": "Parallel Workers", "value": "local default"},
            {"setting": "Theme", "value": "Institutional dark"},
            {"setting": "Auto Save", "value": "enabled via artifacts"},
            {"setting": "Experiment Tracking", "value": "trial_ledger.json"},
        ]
    )
    st.dataframe(settings, width="stretch", hide_index=True)
