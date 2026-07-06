"""DeflatedAlpha single-page research console."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from html import escape
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


DEFAULT_ARTIFACT_ROOT = Path("artifacts")
BG = "#0F1417"
PANEL = "#171D22"
BORDER = "#262E35"
AMBER = "#D4A24C"
SAGE = "#7FA087"
RUST = "#C1594A"
TEXT = "#E8E6E1"
MUTED = "#8B939E"


st.set_page_config(
    page_title="DeflatedAlpha Research Console",
    page_icon="DA",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def install_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Libre+Baskerville:wght@400;700&family=Inter:wght@400;500;600;700;800&display=swap');
        :root {{
            --bg: {BG};
            --panel: {PANEL};
            --border: {BORDER};
            --amber: {AMBER};
            --sage: {SAGE};
            --rust: {RUST};
            --text: {TEXT};
            --muted: {MUTED};
        }}
        .stApp {{
            background: {BG};
            color: {TEXT};
        }}
        .block-container {{
            max-width: 1560px;
            padding: 12px 22px 18px 22px;
        }}
        html, body, [class*="css"] {{
            font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .da-title, .section-display, .panel-title {{
            font-family: "Libre Baskerville", Georgia, serif;
            letter-spacing: 0.02em;
        }}
        .num, [data-testid="stMetricValue"], [data-testid="stMetricDelta"], table {{
            font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace !important;
        }}
        .block {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 0;
            box-shadow: none;
        }}
        .top-nav {{
            height: 56px;
            display: grid;
            grid-template-columns: 1.2fr 1fr 1.25fr;
            align-items: center;
            gap: 16px;
            padding: 0 16px;
            border-bottom: 1px solid {BORDER};
        }}
        .brand {{
            display: flex;
            align-items: baseline;
            gap: 10px;
            min-width: 0;
        }}
        .logo {{
            color: {TEXT};
            font-weight: 800;
            letter-spacing: 0.16em;
            font-size: 0.96rem;
        }}
        .version, .handle, .nav-muted {{
            color: {MUTED};
            font-size: 0.78rem;
        }}
        .dataset-pill {{
            justify-self: center;
            color: {TEXT};
            border: 1px solid {BORDER};
            background: #12181C;
            padding: 8px 14px;
            font-size: 0.84rem;
            min-width: 270px;
            text-align: center;
        }}
        .right-nav {{
            justify-self: end;
            display: flex;
            align-items: center;
            gap: 10px;
            color: {MUTED};
            font-size: 0.8rem;
        }}
        .demo-badge {{
            color: {SAGE};
            border: 1px solid rgba(127, 160, 135, 0.45);
            padding: 4px 8px;
            font-family: "IBM Plex Mono", monospace;
            font-weight: 700;
        }}
        .nav-button {{
            border: 1px solid {BORDER};
            padding: 6px 9px;
            color: {TEXT};
            background: #12181C;
            font-size: 0.78rem;
        }}
        .tab-strip {{
            height: 40px;
            display: flex;
            align-items: end;
            gap: 24px;
            padding: 0 16px;
            border-bottom: 1px solid {BORDER};
            color: {MUTED};
            font-size: 0.86rem;
        }}
        .tab-active {{
            color: {TEXT};
            border-bottom: 2px solid {AMBER};
            padding-bottom: 10px;
        }}
        .section-header {{
            height: 56px;
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            padding: 0 16px;
            border-bottom: 1px solid {BORDER};
        }}
        .section-display {{
            color: {TEXT};
            font-size: 1.08rem;
            font-weight: 700;
        }}
        .timestamp {{
            color: {MUTED};
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.74rem;
            margin-left: 12px;
        }}
        .header-tools {{
            display: flex;
            align-items: center;
            gap: 10px;
            color: {MUTED};
            font-size: 0.78rem;
        }}
        .toggle {{
            border: 1px solid {BORDER};
            display: inline-flex;
            overflow: hidden;
        }}
        .toggle span {{
            padding: 5px 9px;
        }}
        .toggle .on {{
            color: {AMBER};
            background: #1E2328;
        }}
        .hero-shell {{
            min-height: 55vh;
            display: grid;
            grid-template-columns: 320px minmax(0, 1fr);
            border-bottom: 1px solid {BORDER};
        }}
        .hero-sidebar {{
            border-right: 1px solid {BORDER};
            padding: 18px 16px;
            background: #13191D;
        }}
        .hero-label {{
            color: {MUTED};
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
        }}
        .hero-stat {{
            border-top: 1px solid {BORDER};
            padding-top: 14px;
            margin-top: 18px;
        }}
        .hero-stat .k {{
            color: {MUTED};
            font-size: 0.75rem;
        }}
        .hero-stat .v {{
            color: {AMBER};
            font-family: "IBM Plex Mono", monospace;
            font-size: 1.45rem;
            font-weight: 700;
        }}
        .hero-chart {{
            padding: 12px 14px 10px 14px;
            min-width: 0;
        }}
        .overlay-legend {{
            color: {MUTED};
            font-size: 0.78rem;
            margin-top: -30px;
            padding-left: 12px;
            position: relative;
            z-index: 2;
        }}
        .raw-dot, .honest-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .raw-dot {{ background: {AMBER}; }}
        .honest-dot {{ background: {SAGE}; }}
        .lower-grid {{
            display: grid;
            grid-template-columns: 40% 40% 20%;
            min-height: 320px;
            border-bottom: 1px solid {BORDER};
        }}
        .lower-panel {{
            padding: 14px;
            border-right: 1px solid {BORDER};
            min-width: 0;
        }}
        .lower-panel:last-child {{
            border-right: none;
        }}
        .panel-title {{
            color: {TEXT};
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .subtabs {{
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
            color: {MUTED};
            font-size: 0.76rem;
        }}
        .subtab-active {{
            color: {AMBER};
            border-bottom: 1px solid {AMBER};
        }}
        .risk-card {{
            border-left: 3px solid {RUST};
            background: #151A1E;
            padding: 12px;
            min-height: 112px;
            color: {TEXT};
        }}
        .risk-card ul {{
            margin: 8px 0 0 18px;
            padding: 0;
            color: {MUTED};
            font-size: 0.8rem;
            line-height: 1.5;
        }}
        .mini-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.76rem;
        }}
        .mini-table th, .mini-table td {{
            border-bottom: 1px solid {BORDER};
            padding: 7px 4px;
            text-align: right;
        }}
        .mini-table th:first-child, .mini-table td:first-child {{
            text-align: left;
        }}
        .footer {{
            height: 56px;
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            padding: 0 16px;
            color: {MUTED};
            font-size: 0.78rem;
        }}
        .footer-links {{
            display: flex;
            gap: 18px;
            color: {TEXT};
        }}
        div[data-testid="stCheckbox"] label p {{
            color: {TEXT};
            font-size: 0.88rem;
        }}
        div[data-testid="stRadio"] label p {{
            font-size: 0.78rem;
        }}
        div[data-testid="stDownloadButton"] button {{
            background: #12181C;
            border: 1px solid {BORDER};
            color: {TEXT};
            border-radius: 0;
            height: 34px;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
        }}
        @media (max-width: 900px) {{
            .top-nav, .section-header, .footer {{
                grid-template-columns: 1fr;
                height: auto;
                gap: 8px;
                padding: 12px;
            }}
            .right-nav, .dataset-pill {{
                justify-self: start;
            }}
            .hero-shell {{
                grid-template-columns: 1fr;
            }}
            .hero-sidebar {{
                border-right: none;
                border-bottom: 1px solid {BORDER};
            }}
            .lower-grid {{
                grid-template-columns: 1fr;
            }}
            .lower-panel {{
                border-right: none;
                border-bottom: 1px solid {BORDER};
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_artifacts(experiment_dir: str) -> dict[str, object]:
    root = Path(experiment_dir)
    return {
        "metrics": json.loads((root / "metrics.json").read_text(encoding="utf-8")),
        "ledger": json.loads((root / "trial_ledger.json").read_text(encoding="utf-8")),
        "returns": pd.read_parquet(root / "returns.parquet"),
        "fold_scores": pd.read_parquet(root / "fold_scores.parquet"),
    }


def discover_experiments(root: Path = DEFAULT_ARTIFACT_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if (path / "metrics.json").exists())


def model_label(variant: str) -> str:
    lower = variant.lower()
    if "linear" in lower:
        return "Linear"
    if "boost" in lower or "lightgbm" in lower:
        return "LightGBM"
    if "baseline" in lower:
        return "Baseline"
    if "nn" in lower or "neural" in lower:
        return "Small NN"
    return variant.replace("_", " ").title()


def comparison_frame(metrics: dict[str, object], ledger: dict[str, object]) -> pd.DataFrame:
    rows = []
    for variant, payload in metrics["variants"].items():
        performance = payload["performance"]
        deflated = payload["deflated_sharpe"]
        rows.append(
            {
                "model": model_label(variant),
                "variant": variant,
                "raw": float(performance["sharpe"]),
                "deflated": float(deflated["deflated_sharpe"]),
                "ann_return": float(performance["annualized_return"]),
                "portfolio_return": float(performance["total_return"]),
                "dsr_probability": float(deflated["probability"]),
                "mean_ic": float(payload["information_coefficient"]["mean_rank_ic"]),
                "variants_tried": int(ledger["n_trials"]),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def cost_sensitivity(daily: pd.DataFrame, bps_values: tuple[float, ...] = (5.0, 10.0, 20.0)) -> pd.DataFrame:
    rows = []
    for bps in bps_values:
        net_return = daily["gross_return"] - daily["turnover"] * (bps / 10_000.0)
        equity = (1.0 + net_return).cumprod()
        sharpe = annualized_sharpe(net_return)
        rows.append({"bps": bps, "ret": float(equity.iloc[-1] - 1.0), "sharpe": sharpe})
    return pd.DataFrame(rows)


def annualized_sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=0)
    if std <= 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(252))


def build_decay_frame(comparison: pd.DataFrame, active_models: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in comparison.iterrows():
        if row["model"] not in active_models:
            continue
        raw = float(row["raw"])
        honest = float(row["deflated"])
        midpoint = raw + (honest - raw) * 0.55
        for x, stage, value in [
            (0, "Raw reported", raw),
            (1, "Trial adjusted", midpoint),
            (2, "Honest / deflated", honest),
        ]:
            rows.append(
                {
                    "model": row["model"],
                    "variant": row["variant"],
                    "stage_index": x,
                    "stage": stage,
                    "score": value,
                    "raw": raw,
                    "honest": honest,
                }
            )
    return pd.DataFrame(rows)


def plot_decay(decay: pd.DataFrame, selected_model: str) -> go.Figure:
    fig = go.Figure()
    if decay.empty:
        fig.update_layout(title="No selected model curves")
        return fig

    for model, group in decay.groupby("model"):
        line_color = AMBER if model == selected_model else SAGE
        opacity = 1.0 if model == selected_model else 0.48
        fig.add_trace(
            go.Scatter(
                x=group["stage_index"],
                y=group["score"],
                mode="lines+markers",
                name=model,
                line={"color": line_color, "width": 3 if model == selected_model else 2},
                marker={"size": 8, "color": line_color},
                opacity=opacity,
                customdata=group[["stage", "raw", "honest"]],
                hovertemplate="<b>%{fullData.name}</b><br>%{customdata[0]}: %{y:.3f}<br>Raw: %{customdata[1]:.3f}<br>Honest: %{customdata[2]:.3f}<extra></extra>",
            )
        )
        raw_point = group[group["stage_index"] == 0].iloc[0]
        honest_point = group[group["stage_index"] == 2].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[raw_point["score"]],
                mode="markers",
                showlegend=False,
                marker={"size": 13, "color": AMBER, "line": {"color": BG, "width": 2}},
                hovertemplate=f"<b>{model}</b><br>Raw score: %{{y:.3f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[2],
                y=[honest_point["score"]],
                mode="markers",
                showlegend=False,
                marker={"size": 13, "color": SAGE, "line": {"color": BG, "width": 2}},
                hovertemplate=f"<b>{model}</b><br>Honest / deflated: %{{y:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        title={
            "text": "DEFLATED SHARPE DECAY CHART",
            "font": {"family": "Libre Baskerville, Georgia, serif", "size": 20, "color": TEXT},
        },
        font={"family": "IBM Plex Mono, monospace", "color": TEXT, "size": 12},
        margin={"l": 52, "r": 28, "t": 64, "b": 54},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0.42},
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=["Raw reported", "Trial adjusted", "Honest / deflated"],
        gridcolor=BORDER,
        zerolinecolor=BORDER,
    )
    fig.update_yaxes(title="Sharpe score", gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def plot_folds(fold_scores: pd.DataFrame, selected_variant: str) -> go.Figure:
    frame = fold_scores[fold_scores["variant"] == selected_variant].copy()
    if frame.empty:
        frame = fold_scores.copy()
    worst_idx = frame["test_score"].idxmin()
    colors = [RUST if idx == worst_idx else SAGE for idx in frame.index]
    fig = go.Figure(
        go.Bar(
            x=frame["fold_id"],
            y=frame["test_score"],
            marker={"color": colors, "line": {"color": BORDER, "width": 1}},
            customdata=frame[["test_start", "test_end", "variant"]],
            hovertemplate="Fold %{x}<br>OOS score: %{y:.4f}<br>%{customdata[0]} to %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=238,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"family": "IBM Plex Mono, monospace", "color": TEXT, "size": 11},
        margin={"l": 34, "r": 12, "t": 10, "b": 34},
    )
    fig.update_xaxes(title="Fold", gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(title="OOS score", gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def render_top_nav(dataset_label: str, memo: str) -> None:
    left, center, right = st.columns([1.15, 1.0, 1.25])
    with left:
        st.markdown(
            """
            <div class="block top-nav" style="border-right:none;">
                <div class="brand">
                    <span class="logo">DEFLATEDALPHA</span>
                    <span class="version">v1.0.0</span>
                    <span class="handle">@niraj</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with center:
        st.markdown(
            f"""
            <div class="block top-nav" style="border-left:none;border-right:none;">
                <div class="dataset-pill">{escape(dataset_label)} ▾</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="block top-nav" style="border-left:none;">
                <div class="right-nav">
                    <span class="demo-badge">DEMO</span>
                    <span class="nav-button">GitHub</span>
                    <span class="nav-button">Search ⌘K</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button("Export Report", memo, file_name="deflatedalpha_report.md", key="nav_export")


def render_tab_strip() -> None:
    st.markdown(
        """
        <div class="block tab-strip">
            <span class="tab-active">Main</span>
            <span>Validation</span>
            <span>Factors</span>
            <span>Backtesting</span>
            <span>Experiments</span>
            <span>Logs</span>
            <span>Reports</span>
            <span>Settings</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(date_range: str, timestamp: str) -> None:
    st.markdown(
        f"""
        <div class="block section-header">
            <div>
                <span class="section-display">RESEARCH RESULTS</span>
                <span class="timestamp">{escape(timestamp)}</span>
            </div>
            <div class="header-tools">
                <span>{escape(date_range)}</span>
                <span class="toggle"><span class="on">Chart</span><span>Table</span></span>
                <span class="nav-button">Fullscreen</span>
                <span class="nav-button">Pin</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(repo_label: str) -> None:
    st.markdown(
        f"""
        <div class="block footer">
            <div>{escape(repo_label)} · built from RESULTS.json — no hardcoded figures</div>
            <div class="footer-links">
                <span>Docs</span><span>Changelog</span><span>GitHub</span><span>Contact</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


install_theme()

experiments = discover_experiments()
if not experiments:
    st.error("No experiment artifacts found. Run `python -m alpha_pipeline.cli --output artifacts/demo`.")
    st.stop()

selected_experiment = experiments[0]
artifacts = load_artifacts(str(selected_experiment))
metrics = artifacts["metrics"]
ledger = artifacts["ledger"]
returns = artifacts["returns"]
fold_scores = artifacts["fold_scores"]
quality = metrics["data_quality"]
config = metrics["config"]
memo = render_memo(selected_experiment)
comparison = comparison_frame(metrics, ledger)

best_variant = str(metrics["best_variant"])
best_model = model_label(best_variant)
dataset_label = f"Synthetic · {quality['start_date']}–{quality['end_date']}"
date_range = f"{quality['start_date']} → {quality['end_date']}"
timestamp = datetime.fromtimestamp((selected_experiment / "metrics.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

render_top_nav(dataset_label, memo)
render_tab_strip()
render_section_header(date_range, timestamp)

label_to_variant = {model_label(row["variant"]): row["variant"] for _, row in comparison.iterrows()}
desired_models = ["Baseline", "Linear", "LightGBM", "Small NN"]

st.markdown('<div class="block hero-shell">', unsafe_allow_html=True)
hero_left, hero_right = st.columns([0.235, 0.765], gap="small")
with hero_left:
    st.markdown('<div class="hero-sidebar"><div class="hero-label">Toggle variants</div>', unsafe_allow_html=True)
    active_models: list[str] = []
    for label in desired_models:
        available = label in label_to_variant
        default = label in label_to_variant
        checked = st.checkbox(
            label,
            value=default,
            disabled=not available,
            key=f"decay_toggle_{label}",
            help=None if available else "Not present in RESULTS.json",
        )
        if available and checked:
            active_models.append(label)
    if not active_models:
        active_models = [best_model]
    raw = comparison.loc[comparison["variant"] == best_variant, "raw"].iloc[0]
    honest = comparison.loc[comparison["variant"] == best_variant, "deflated"].iloc[0]
    st.markdown(
        f"""
        <div class="hero-stat">
            <div class="k">Reported raw score</div>
            <div class="v num">{raw:.2f}</div>
            <div class="k">Honest / deflated score</div>
            <div class="v num" style="color:{SAGE};">{honest:.2f}</div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown('<div class="hero-chart">', unsafe_allow_html=True)
    decay = build_decay_frame(comparison, active_models)
    st.plotly_chart(plot_decay(decay, best_model), width="stretch", config={"displaylogo": False})
    st.markdown(
        """
        <div class="overlay-legend">
            <span class="raw-dot"></span>raw score&nbsp;&nbsp;&nbsp;
            <span class="honest-dot"></span>honest/deflated score
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="block lower-grid">', unsafe_allow_html=True)
panel_5a, panel_5b, panel_5c = st.columns([0.4, 0.4, 0.2], gap="small")

with panel_5a:
    st.markdown('<div class="lower-panel"><div class="panel-title">MODEL COMPARISON</div>', unsafe_allow_html=True)
    model_filter = st.radio("Model filter", ["All", "LightGBM", "Linear"], horizontal=True, label_visibility="collapsed")
    table = comparison.copy()
    if model_filter != "All":
        table = table[table["model"] == model_filter].copy()
    table = table.reset_index(drop=True)
    default_selected = best_variant if best_variant in set(table["variant"]) else str(table["variant"].iloc[0])
    table["selected"] = np.where(table["variant"] == default_selected, ">", "")
    display = table[["selected", "model", "raw", "deflated", "ann_return", "portfolio_return", "dsr_probability"]].rename(
        columns={
            "selected": "",
            "model": "model",
            "raw": "raw",
            "deflated": "deflated",
            "ann_return": "ann return",
            "portfolio_return": "portfolio return",
            "dsr_probability": "DSR probability",
        }
    )
    event = st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=188,
        on_select="rerun",
        selection_mode="single-row",
        key="model_comparison_select",
    )
    selected_variant = default_selected
    selected_rows = getattr(event, "selection", {}).get("rows", []) if event is not None else []
    if selected_rows:
        selected_variant = str(table.iloc[selected_rows[0]]["variant"])
    selected_model = model_label(selected_variant)
    st.markdown(
        f'<div class="nav-muted">Selected row highlights <span style="color:{AMBER};">{escape(selected_model)}</span> across folds.</div></div>',
        unsafe_allow_html=True,
    )

with panel_5b:
    st.markdown('<div class="lower-panel"><div class="panel-title">WALK-FORWARD FOLDS</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_folds(fold_scores, selected_variant), width="stretch", config={"displaylogo": False})
    selected_folds = fold_scores[fold_scores["variant"] == selected_variant]
    if not selected_folds.empty:
        worst = selected_folds.loc[selected_folds["test_score"].idxmin()]
        st.markdown(
            f'<div class="nav-muted">Worst fold auto-flagged: <span style="color:{RUST};">fold {int(worst["fold_id"])}</span> · score <span class="num">{worst["test_score"]:.4f}</span></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("</div>", unsafe_allow_html=True)

with panel_5c:
    st.markdown('<div class="lower-panel"><div class="panel-title">RISK & DISCLOSURES</div>', unsafe_allow_html=True)
    n_trials = int(ledger["n_trials"])
    best_returns = returns[returns["variant"] == best_variant].copy()
    sensitivity = cost_sensitivity(best_returns)
    rows = "".join(
        f"<tr><td>{row.bps:.0f} bps</td><td>{row.ret:.1%}</td><td>{row.sharpe:.2f}</td></tr>"
        for row in sensitivity.itertuples(index=False)
    )
    st.markdown(
        f"""
        <div class="risk-card">
            <strong>{n_trials} variants tried</strong>
            <ul>
                <li>Dataset is not marked survivorship-bias-free.</li>
                <li>Result is a demo research artifact, not a live alpha claim.</li>
                <li>Worst fold is computed from validation output.</li>
            </ul>
        </div>
        <table class="mini-table">
            <thead><tr><th>cost</th><th>return</th><th>sharpe</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
render_footer("/Users/nirajrajendranaphade/Documents/alpha-research-pipeline")
