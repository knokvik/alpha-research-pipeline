"""DeflatedAlpha research console — documents pipeline output from artifacts."""

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

DOC_TABS = ("Overview", "Reports", "Artifacts", "Validation", "Experiments")
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from alpha_pipeline.memo import render_memo


DEFAULT_ARTIFACT_ROOT = Path("artifacts")
THEMES = {
    "Dark": {
        "BG": "#0F1417",
        "PANEL": "#171D22",
        "PANEL_ALT": "#13191D",
        "CONTROL": "#12181C",
        "BORDER": "#262E35",
        "AMBER": "#D4A24C",
        "SAGE": "#7FA087",
        "RUST": "#C1594A",
        "TEXT": "#E8E6E1",
        "MUTED": "#8B939E",
        "CHART_TEMPLATE": "plotly_dark",
    },
    "Light": {
        "BG": "#F3F0E8",
        "PANEL": "#FBF8EF",
        "PANEL_ALT": "#EFE9DC",
        "CONTROL": "#FFFDF7",
        "BORDER": "#CFC7B9",
        "AMBER": "#A96F1D",
        "SAGE": "#54775D",
        "RUST": "#A5483D",
        "TEXT": "#20242A",
        "MUTED": "#68717C",
        "CHART_TEMPLATE": "plotly_white",
    },
}
BG = THEMES["Dark"]["BG"]
PANEL = THEMES["Dark"]["PANEL"]
PANEL_ALT = THEMES["Dark"]["PANEL_ALT"]
CONTROL = THEMES["Dark"]["CONTROL"]
BORDER = THEMES["Dark"]["BORDER"]
AMBER = THEMES["Dark"]["AMBER"]
SAGE = THEMES["Dark"]["SAGE"]
RUST = THEMES["Dark"]["RUST"]
TEXT = THEMES["Dark"]["TEXT"]
MUTED = THEMES["Dark"]["MUTED"]
CHART_TEMPLATE = THEMES["Dark"]["CHART_TEMPLATE"]


st.set_page_config(
    page_title="DeflatedAlpha Research Console",
    page_icon="DA",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def install_theme(theme_name: str) -> None:
    global BG, PANEL, PANEL_ALT, CONTROL, BORDER, AMBER, SAGE, RUST, TEXT, MUTED, CHART_TEMPLATE
    palette = THEMES[theme_name]
    BG = palette["BG"]
    PANEL = palette["PANEL"]
    PANEL_ALT = palette["PANEL_ALT"]
    CONTROL = palette["CONTROL"]
    BORDER = palette["BORDER"]
    AMBER = palette["AMBER"]
    SAGE = palette["SAGE"]
    RUST = palette["RUST"]
    TEXT = palette["TEXT"]
    MUTED = palette["MUTED"]
    CHART_TEMPLATE = palette["CHART_TEMPLATE"]
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {BG};
            --panel: {PANEL};
            --panel-alt: {PANEL_ALT};
            --control: {CONTROL};
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
        html, body, [class*="css"], .stApp, button, input, textarea, select,
        .da-title, .section-display, .panel-title, .num,
        [data-testid="stMetricValue"], [data-testid="stMetricDelta"], table {{
            font-family: "SF Mono", "SFMono-Regular", ui-monospace, Menlo, Monaco, Consolas, monospace !important;
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
            background: {CONTROL};
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
            font-weight: 700;
        }}
        .nav-button {{
            border: 1px solid {BORDER};
            padding: 6px 9px;
            color: {TEXT};
            background: {CONTROL};
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
            background: {PANEL_ALT};
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
            background: {PANEL_ALT};
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
            background: {PANEL_ALT};
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
            background: {CONTROL};
            border: 1px solid {BORDER};
            color: {TEXT};
            border-radius: 0;
            height: 34px;
        }}
        div[data-testid="stRadio"] {{
            border: 1px solid {BORDER};
            background: {PANEL};
            padding: 4px 8px;
        }}
        div[data-testid="stRadio"] label {{
            color: {TEXT};
        }}
        div[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p,
        div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {{
            color: {MUTED};
        }}
        div[data-testid="stDataFrame"] * {{
            font-family: "SF Mono", "SFMono-Regular", ui-monospace, Menlo, Monaco, Consolas, monospace !important;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
        }}
        .doc-shell {{
            border-bottom: 1px solid {BORDER};
        }}
        .doc-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
            min-height: 62vh;
        }}
        .doc-main, .doc-side {{
            padding: 16px;
            min-width: 0;
        }}
        .doc-main {{
            border-right: 1px solid {BORDER};
        }}
        .doc-memo {{
            background: {PANEL_ALT};
            border: 1px solid {BORDER};
            border-left: 3px solid {AMBER};
            padding: 18px 20px;
            max-height: 68vh;
            overflow-y: auto;
            font-size: 0.82rem;
            line-height: 1.55;
        }}
        .doc-memo h1 {{
            color: {TEXT};
            font-size: 1.05rem;
            letter-spacing: 0.06em;
            margin: 0 0 14px 0;
        }}
        .doc-memo h2 {{
            color: {AMBER};
            font-size: 0.88rem;
            margin: 18px 0 8px 0;
        }}
        .doc-memo p, .doc-memo li {{
            color: {MUTED};
        }}
        .doc-memo strong {{
            color: {TEXT};
        }}
        .doc-memo table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.76rem;
        }}
        .doc-memo th, .doc-memo td {{
            border-bottom: 1px solid {BORDER};
            padding: 6px 4px;
            text-align: right;
        }}
        .doc-memo th:first-child, .doc-memo td:first-child {{
            text-align: left;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 14px;
        }}
        .stat-card {{
            background: {PANEL_ALT};
            border: 1px solid {BORDER};
            padding: 10px 12px;
        }}
        .stat-card .k {{
            color: {MUTED};
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .stat-card .v {{
            color: {AMBER};
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 4px;
        }}
        .stat-card .v.sage {{ color: {SAGE}; }}
        .stat-card .v.rust {{ color: {RUST}; }}
        .artifact-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78rem;
        }}
        .artifact-table th, .artifact-table td {{
            border-bottom: 1px solid {BORDER};
            padding: 9px 6px;
            text-align: left;
        }}
        .artifact-table th {{
            color: {MUTED};
            font-weight: 600;
        }}
        .artifact-table td.size {{
            text-align: right;
            color: {AMBER};
        }}
        .artifact-table tr:hover td {{
            background: {PANEL_ALT};
        }}
        .json-viewer {{
            background: {PANEL_ALT};
            border: 1px solid {BORDER};
            padding: 12px;
            font-size: 0.74rem;
            color: {MUTED};
            max-height: 320px;
            overflow: auto;
            white-space: pre-wrap;
        }}
        .feature-pills {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }}
        .feature-pill {{
            border: 1px solid {BORDER};
            background: {CONTROL};
            color: {TEXT};
            padding: 4px 8px;
            font-size: 0.72rem;
        }}
        .section-note {{
            color: {MUTED};
            font-size: 0.76rem;
            margin-bottom: 12px;
        }}
        div[data-testid="stRadio"][aria-label="Section"] {{
            border: none;
            background: transparent;
            padding: 0;
        }}
        div[data-testid="stRadio"][aria-label="Section"] > div {{
            border-bottom: 1px solid {BORDER};
            background: {PANEL};
            padding: 0 16px;
            min-height: 40px;
            align-items: end;
        }}
        div[data-testid="stRadio"][aria-label="Section"] label {{
            border-bottom: 2px solid transparent;
            padding: 10px 2px;
            margin-right: 22px;
            color: {MUTED};
        }}
        div[data-testid="stRadio"][aria-label="Section"] label[data-checked="true"] {{
            color: {TEXT};
            border-bottom-color: {AMBER};
        }}
        div[data-testid="stRadio"][aria-label="Section"] label p {{
            font-size: 0.86rem !important;
        }}
        div[data-testid="stRadio"][aria-label="Section"] [data-testid="stMarkdownContainer"] {{
            display: none;
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
            .doc-grid {{
                grid-template-columns: 1fr;
            }}
            .doc-main {{
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
    payload: dict[str, object] = {
        "metrics": json.loads((root / "metrics.json").read_text(encoding="utf-8")),
        "ledger": json.loads((root / "trial_ledger.json").read_text(encoding="utf-8")),
        "returns": pd.read_parquet(root / "returns.parquet"),
        "fold_scores": pd.read_parquet(root / "fold_scores.parquet"),
    }
    for name in ("config.json", "data_quality.json", "folds.json"):
        path = root / name
        if path.exists():
            payload[name.replace(".json", "")] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def artifact_inventory(experiment_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(experiment_dir.iterdir()):
        if not path.is_file():
            continue
        rows.append(
            {
                "file": path.name,
                "size": human_size(path.stat().st_size),
                "bytes": path.stat().st_size,
                "kind": path.suffix.lstrip(".") or "file",
                "description": ARTIFACT_DESCRIPTIONS.get(path.name, "Experiment output artifact."),
            }
        )
    return pd.DataFrame(rows)


def load_methodology() -> str:
    path = PROJECT_ROOT / "reports" / "methodology.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "Methodology notes are not available."


def memo_to_html(memo: str) -> str:
    html_parts: list[str] = []
    in_list = False
    in_table = False
    for line in memo.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            if in_table:
                html_parts.append("</table>")
                in_table = False
            html_parts.append(f"<h1>{escape(stripped[2:])}</h1>")
            continue
        if stripped.startswith("## "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            if in_table:
                html_parts.append("</table>")
                in_table = False
            html_parts.append(f"<h2>{escape(stripped[3:])}</h2>")
            continue
        if stripped.startswith("- "):
            if in_table:
                html_parts.append("</table>")
                in_table = False
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{escape(stripped[2:])}</li>")
            continue
        if stripped.startswith("|"):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if cells and all(cell.replace("-", "").replace(":", "") == "" for cell in cells):
                continue
            if not in_table:
                html_parts.append("<table>")
                in_table = True
            row_tag = "th" if html_parts[-1] == "<table>" else "td"
            html_parts.append(
                "<tr>" + "".join(f"<{row_tag}>{escape(cell)}</{row_tag}>" for cell in cells) + "</tr>"
            )
            continue
        if not stripped:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            if in_table:
                html_parts.append("</table>")
                in_table = False
            continue
        if in_list:
            html_parts.append("</ul>")
            in_list = False
        if in_table:
            html_parts.append("</table>")
            in_table = False
        html_parts.append(f"<p>{escape(stripped)}</p>")
    if in_list:
        html_parts.append("</ul>")
    if in_table:
        html_parts.append("</table>")
    return "\n".join(html_parts)


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
        template=CHART_TEMPLATE,
        height=520,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        title={
            "text": "DEFLATED SHARPE DECAY CHART",
            "font": {"family": "SF Mono, SFMono-Regular, ui-monospace, Menlo, monospace", "size": 20, "color": TEXT},
        },
        font={"family": "SF Mono, SFMono-Regular, ui-monospace, Menlo, monospace", "color": TEXT, "size": 12},
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
        template=CHART_TEMPLATE,
        height=238,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"family": "SF Mono, SFMono-Regular, ui-monospace, Menlo, monospace", "color": TEXT, "size": 11},
        margin={"l": 34, "r": 12, "t": 10, "b": 34},
    )
    fig.update_xaxes(title="Fold", gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(title="OOS score", gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def render_top_nav(experiment_names: list[str], selected_name: str, memo: str) -> str:
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
            """
            <div class="block top-nav" style="border-left:none;border-right:none;">
                <div class="dataset-pill">Experiment output ▾</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        chosen = st.selectbox(
            "Experiment",
            experiment_names,
            index=experiment_names.index(selected_name),
            label_visibility="collapsed",
            key="experiment_select",
        )
    with right:
        st.markdown(
            """
            <div class="block top-nav" style="border-left:none;">
                <div class="right-nav">
                    <span class="demo-badge">OUTPUT</span>
                    <span class="nav-button">Docs</span>
                    <span class="nav-button">Export</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.radio(
            "Theme",
            ["Dark", "Light"],
            horizontal=True,
            key="theme_mode",
            label_visibility="collapsed",
        )
        st.download_button("Export Report", memo, file_name="deflatedalpha_report.md", key="nav_export")
    return chosen


def render_tab_strip() -> str:
    return st.radio("Section", DOC_TABS, horizontal=True, label_visibility="collapsed", key="doc_tab")


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


def render_footer(experiment_name: str, artifact_count: int) -> None:
    st.markdown(
        f"""
        <div class="block footer">
            <div>{escape(experiment_name)} · {artifact_count} artifacts · all figures from metrics.json</div>
            <div class="footer-links">
                <span>Reports</span><span>Methodology</span><span>Artifacts</span><span>Export</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, tone: str = "amber") -> str:
    tone_class = {"amber": "", "sage": " sage", "rust": " rust"}.get(tone, "")
    return (
        f'<div class="stat-card"><div class="k">{escape(label)}</div>'
        f'<div class="v{tone_class}">{escape(value)}</div></div>'
    )


def render_overview_tab(
    comparison: pd.DataFrame,
    best_variant: str,
    best_model: str,
    fold_scores: pd.DataFrame,
    returns: pd.DataFrame,
    ledger: dict[str, object],
) -> None:
    label_to_variant = {model_label(row["variant"]): row["variant"] for _, row in comparison.iterrows()}
    desired_models = ["Baseline", "Linear", "LightGBM", "Small NN"]

    st.markdown('<div class="block hero-shell">', unsafe_allow_html=True)
    hero_left, hero_right = st.columns([0.235, 0.765], gap="small")
    with hero_left:
        st.markdown('<div class="hero-sidebar"><div class="hero-label">Toggle variants</div>', unsafe_allow_html=True)
        active_models: list[str] = []
        for label in desired_models:
            available = label in label_to_variant
            checked = st.checkbox(
                label,
                value=available,
                disabled=not available,
                key=f"decay_toggle_{label}",
                help=None if available else "Not present in metrics.json",
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


def render_reports_tab(memo: str, metrics: dict[str, object], experiment_dir: Path) -> None:
    best = metrics["variants"][metrics["best_variant"]]
    methodology = load_methodology()
    st.markdown('<div class="block doc-shell">', unsafe_allow_html=True)
    left, right = st.columns([1.4, 1.0], gap="small")
    with left:
        st.markdown('<div class="doc-main">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">RESEARCH MEMO</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">Auto-generated from experiment artifacts — export for documentation.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="doc-memo">{memo_to_html(memo)}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="doc-side">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">OUTPUT SUMMARY</div>', unsafe_allow_html=True)
        cards = "".join(
            [
                render_stat_card("Best variant", str(metrics["best_variant"])),
                render_stat_card("Raw Sharpe", f"{best['performance']['sharpe']:.2f}"),
                render_stat_card("Deflated Sharpe", f"{best['deflated_sharpe']['deflated_sharpe']:.2f}", "sage"),
                render_stat_card("DSR probability", f"{best['deflated_sharpe']['probability']:.1%}", "sage"),
                render_stat_card("Trials disclosed", str(best["deflated_sharpe"]["n_trials"])),
                render_stat_card("Mean rank IC", f"{best['information_coefficient']['mean_rank_ic']:.3f}"),
            ]
        )
        st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">METHODOLOGY</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="json-viewer">{escape(methodology)}</div>', unsafe_allow_html=True)
        metrics_json = json.dumps(metrics, indent=2)
        st.download_button("Download metrics.json", metrics_json, file_name="metrics.json", key="reports_metrics")
        st.download_button("Download memo.md", memo, file_name="research_memo.md", key="reports_memo")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_artifacts_tab(experiment_dir: Path) -> None:
    inventory = artifact_inventory(experiment_dir)
    st.markdown('<div class="block doc-shell"><div class="doc-main" style="border-right:none;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">ARTIFACT INVENTORY</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Every file written by the pipeline run — sizes and descriptions for documentation.</div>',
        unsafe_allow_html=True,
    )
    rows = "".join(
        (
            f"<tr><td>{escape(row.file)}</td>"
            f'<td class="size">{escape(row.size)}</td>'
            f"<td>{escape(row.kind)}</td>"
            f"<td>{escape(row.description)}</td></tr>"
        )
        for row in inventory.itertuples(index=False)
    )
    st.markdown(
        f"""
        <table class="artifact-table">
            <thead><tr><th>file</th><th>size</th><th>type</th><th>description</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

    json_files = [path for path in sorted(experiment_dir.iterdir()) if path.suffix == ".json"]
    if json_files:
        st.markdown('<div class="block" style="padding:14px;">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">JSON PREVIEW</div>', unsafe_allow_html=True)
        selected = st.selectbox("Artifact file", [path.name for path in json_files], key="artifact_preview")
        preview_path = experiment_dir / selected
        preview_text = preview_path.read_text(encoding="utf-8")
        st.markdown(f'<div class="json-viewer">{escape(preview_text)}</div>', unsafe_allow_html=True)
        st.download_button(f"Download {selected}", preview_text, file_name=selected, key="artifact_download")
        st.markdown("</div>", unsafe_allow_html=True)


def render_validation_tab(metrics: dict[str, object], artifacts: dict[str, object]) -> None:
    config = metrics.get("config", artifacts.get("config", {}))
    quality = metrics["data_quality"]
    folds_payload = artifacts.get("folds", {})
    fold_rows = folds_payload.get("folds", []) if isinstance(folds_payload, dict) else []
    features = metrics.get("feature_columns", [])

    st.markdown('<div class="block doc-shell">', unsafe_allow_html=True)
    left, right = st.columns([1.0, 1.0], gap="small")
    with left:
        st.markdown('<div class="doc-main">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">VALIDATION CONFIG</div>', unsafe_allow_html=True)
        cards = "".join(
            [
                render_stat_card("Train window", f"{config.get('train_window_days', '—')} days"),
                render_stat_card("Test window", f"{config.get('test_window_days', '—')} days"),
                render_stat_card("Step", f"{config.get('step_days', '—')} days"),
                render_stat_card("Embargo", f"{config.get('embargo_days', '—')} days"),
                render_stat_card("Rebalance", str(config.get("rebalance_frequency", "—"))),
                render_stat_card("Txn cost", f"{config.get('transaction_cost_bps', '—')} bps"),
            ]
        )
        st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">DATA QUALITY</div>', unsafe_allow_html=True)
        quality_cards = "".join(
            [
                render_stat_card("Assets", str(quality["n_assets"])),
                render_stat_card("Rows", f"{quality['n_rows']:,}"),
                render_stat_card("Survivorship-free", "yes" if quality["survivorship_bias_free"] else "no", "rust"),
                render_stat_card("Missing prices", str(quality["missing_price_rows"]), "sage"),
            ]
        )
        st.markdown(f'<div class="stat-grid">{quality_cards}</div>', unsafe_allow_html=True)
        limitations = "".join(f"<li>{escape(item)}</li>" for item in quality["limitations"])
        st.markdown(f'<div class="risk-card"><ul>{limitations}</ul></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="doc-side">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">WALK-FORWARD FOLDS</div>', unsafe_allow_html=True)
        if fold_rows:
            fold_frame = pd.DataFrame(fold_rows)
            st.dataframe(
                fold_frame[
                    ["fold_id", "train_start", "train_end", "test_start", "test_end", "n_train_rows", "n_test_rows", "purged_rows"]
                ],
                hide_index=True,
                width="stretch",
                height=280,
            )
        else:
            st.info("No folds.json found for this experiment.")
        st.markdown('<div class="panel-title">FEATURE SET</div>', unsafe_allow_html=True)
        pills = "".join(f'<span class="feature-pill">{escape(name)}</span>' for name in features)
        st.markdown(f'<div class="feature-pills">{pills}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_experiments_tab(ledger: dict[str, object], comparison: pd.DataFrame) -> None:
    trials = ledger.get("trials", [])
    st.markdown('<div class="block doc-shell"><div class="doc-main" style="border-right:none;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">TRIAL LEDGER</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-note">{int(ledger["n_trials"])} variants tested and logged — required for deflated Sharpe.</div>',
        unsafe_allow_html=True,
    )
    if trials:
        rows = []
        for trial in trials:
            metrics = trial["metrics"]
            rows.append(
                {
                    "variant": trial["variant"],
                    "kind": trial["parameters"]["kind"],
                    "sharpe": metrics["sharpe"],
                    "mean_rank_ic": metrics["mean_rank_ic"],
                    "icir": metrics["icir"],
                    "turnover": metrics["average_turnover"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )
        trial_frame = pd.DataFrame(rows)
        st.dataframe(trial_frame, hide_index=True, width="stretch", height=220)
    st.markdown('<div class="panel-title" style="margin-top:18px;">VARIANT RANKING</div>', unsafe_allow_html=True)
    ranking = comparison[["model", "variant", "raw", "deflated", "dsr_probability", "mean_ic"]].copy()
    st.dataframe(ranking, hide_index=True, width="stretch", height=180)
    st.markdown("</div></div>", unsafe_allow_html=True)


theme_name = st.session_state.get("theme_mode", "Dark")
if theme_name not in THEMES:
    theme_name = "Dark"
install_theme(theme_name)

experiments = discover_experiments()
if not experiments:
    st.error("No experiment artifacts found. Run `python -m alpha_pipeline.cli --output artifacts/demo`.")
    st.stop()

experiment_labels = {path.name: path for path in experiments}
experiment_names = list(experiment_labels.keys())
default_name = experiment_names[0]
default_experiment = experiment_labels[default_name]
bootstrap_memo = render_memo(default_experiment)

selected_name = render_top_nav(experiment_names, default_name, bootstrap_memo)
selected_experiment = experiment_labels[selected_name]

artifacts = load_artifacts(str(selected_experiment))
metrics = artifacts["metrics"]
ledger = artifacts["ledger"]
returns = artifacts["returns"]
fold_scores = artifacts["fold_scores"]
quality = metrics["data_quality"]
memo = render_memo(selected_experiment)
comparison = comparison_frame(metrics, ledger)
inventory = artifact_inventory(selected_experiment)

best_variant = str(metrics["best_variant"])
best_model = model_label(best_variant)
date_range = f"{quality['start_date']} → {quality['end_date']}"
timestamp = datetime.fromtimestamp((selected_experiment / "metrics.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
active_tab = render_tab_strip()
render_section_header(date_range, timestamp)

if active_tab == "Overview":
    render_overview_tab(comparison, best_variant, best_model, fold_scores, returns, ledger)
elif active_tab == "Reports":
    render_reports_tab(memo, metrics, selected_experiment)
elif active_tab == "Artifacts":
    render_artifacts_tab(selected_experiment)
elif active_tab == "Validation":
    render_validation_tab(metrics, artifacts)
elif active_tab == "Experiments":
    render_experiments_tab(ledger, comparison)

render_footer(selected_name, len(inventory))
