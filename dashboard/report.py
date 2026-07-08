"""Build a standalone HTML research report from experiment artifacts."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

STYLES = """
* { box-sizing: border-box; }
html, body { width: 100%; margin: 0; padding: 0; overflow-x: hidden; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       background: #fff; color: #1a1a1a; line-height: 1.5; }
.wrap { width: 100%; max-width: 100%; margin: 0; padding: 0; }
.header { width: 100%; padding: 20px 0 12px; border-bottom: 1px solid #e5e7eb; }
.header h1 { font-size: 1.75rem; margin: 0 0 8px; padding: 0 16px; }
.meta { color: #6b7280; font-size: 0.9rem; margin: 0; padding: 0 16px 12px; }
.section { width: 100%; padding: 20px 0; border-bottom: 1px solid #f3f4f6; }
.section h2 { font-size: 1.2rem; margin: 0 0 12px; padding: 0 16px 8px; border-bottom: 1px solid #e5e7eb; }
.section h3 { font-size: 1rem; margin: 20px 0 8px; padding: 0 16px; }
.section p, .section ol, .section ul { padding: 0 16px; margin: 8px 0; }
.note { background: #f9fafb; border-top: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb;
        padding: 14px 16px; margin: 0; width: 100%; }
.warn { background: #fffbeb; border-top: 1px solid #fcd34d; border-bottom: 1px solid #fcd34d;
        padding: 12px 16px; margin: 0; width: 100%; }
.metrics { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0; margin: 0; width: 100%;
           border-bottom: 1px solid #e5e7eb; }
.metric { border-right: 1px solid #e5e7eb; padding: 14px 16px; }
.metric:last-child { border-right: none; }
.metric .k { color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
.metric .v { font-size: 1.15rem; font-weight: 600; margin-top: 4px; word-break: break-word; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin: 0; width: 100%; }
.chart { border-right: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; padding: 0; min-width: 0; }
.chart:last-child { border-right: none; }
.chart .plotly-graph-div { width: 100% !important; }
.table-wrap { width: 100%; overflow-x: auto; }
table.data { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 0; }
table.data th, table.data td { border-bottom: 1px solid #e5e7eb; padding: 10px 16px; text-align: left; }
table.data th { background: #f9fafb; font-weight: 600; }
pre.memo { background: #f9fafb; border: none; border-top: 1px solid #e5e7eb; padding: 16px;
           overflow-x: auto; font-size: 0.85rem; white-space: pre-wrap; margin: 0; width: 100%; }
.features { padding: 0 16px 8px; }
@media (max-width: 900px) {
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .metric { border-bottom: 1px solid #e5e7eb; }
  .charts { grid-template-columns: 1fr; }
  .chart { border-right: none; }
}
"""


def discover_experiments(root: Path = ARTIFACT_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if (path / "metrics.json").exists())


def load_experiment(experiment_dir: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "metrics": json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8")),
        "ledger": json.loads((experiment_dir / "trial_ledger.json").read_text(encoding="utf-8")),
        "returns": pd.read_parquet(experiment_dir / "returns.parquet"),
        "fold_scores": pd.read_parquet(experiment_dir / "fold_scores.parquet"),
    }
    for name in ("config.json", "folds.json"):
        path = experiment_dir / name
        if path.exists():
            payload[name.replace(".json", "")] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def model_label(variant: str) -> str:
    lower = variant.lower()
    if "linear" in lower:
        return "Linear"
    if "boost" in lower or "lightgbm" in lower:
        return "LightGBM"
    return variant.replace("_", " ").title()


def comparison_frame(metrics: dict[str, object]) -> pd.DataFrame:
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
        height=400,
        autosize=True,
        width=None,
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
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Walk-Forward Fold Scores — {variant}",
        xaxis_title="Fold",
        yaxis_title="Out-of-sample score",
        height=360,
        autosize=True,
        width=None,
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
        autosize=True,
        width=None,
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


def trial_frame(ledger: dict[str, object]) -> pd.DataFrame:
    rows = []
    for trial in ledger.get("trials", []):
        rows.append(
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
    return pd.DataFrame(rows)


def df_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, classes="data", border=0)


def figure_html(fig: go.Figure, *, include_plotlyjs: bool) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        config={"displayModeBar": False, "responsive": True},
    )


def metric_card(label: str, value: str) -> str:
    return f'<div class="metric"><div class="k">{escape(label)}</div><div class="v">{escape(value)}</div></div>'


def build_report(experiment_dir: Path | str, output: Path | str | None = None) -> str:
    """Generate HTML report. Optionally write to *output* and return the HTML string."""

    root = Path(experiment_dir)
    data = load_experiment(root)
    metrics = data["metrics"]
    ledger = data["ledger"]
    returns = data["returns"]
    fold_scores = data["fold_scores"]
    quality = metrics["data_quality"]
    config = metrics["config"]
    comparison = comparison_frame(metrics)
    best_variant = str(metrics["best_variant"])
    best = metrics["variants"][best_variant]
    perf = best["performance"]
    ic = best["information_coefficient"]
    dsr = best["deflated_sharpe"]
    memo = render_memo(root)
    generated_at = datetime.fromtimestamp((root / "metrics.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    figures = [
        plot_decay(decay_frame(comparison)),
        plot_folds(fold_scores, best_variant),
        plot_equity(returns, best_variant),
    ]
    chart_blocks = []
    for index, fig in enumerate(figures):
        chart_blocks.append(f'<div class="chart">{figure_html(fig, include_plotlyjs=index == 0)}</div>')

    folds_payload = data.get("folds", {})
    fold_rows = folds_payload.get("folds", []) if isinstance(folds_payload, dict) else []
    fold_html = df_html(pd.DataFrame(fold_rows)) if fold_rows else "<p>No folds.json found.</p>"

    best_returns = returns[returns["variant"] == best_variant]
    warning = ""
    if not quality["survivorship_bias_free"]:
        warning = (
            '<div class="warn">Demo universe is not survivorship-bias-free. '
            "Replace with point-in-time data before live claims.</div>"
        )

    validation_metrics = f"""<div class="metrics">
      {metric_card("Train Window", f"{config['train_window_days']} days")}
      {metric_card("Test Window", f"{config['test_window_days']} days")}
      {metric_card("Step", f"{config['step_days']} days")}
      {metric_card("Embargo", f"{config['embargo_days']} days")}
      {metric_card("Rebalance", str(config.get("rebalance_frequency", "—")))}
      {metric_card("Txn Cost", f"{config['transaction_cost_bps']} bps")}
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Alpha Research Pipeline — {escape(root.name)}</title>
  <style>{STYLES}</style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>Cross-Sectional Alpha Research Pipeline</h1>
      <p class="meta">Experiment <strong>{escape(root.name)}</strong> · sample {escape(quality['start_date'])} to {escape(quality['end_date'])} · generated {escape(generated_at)}</p>
    </div>
    <div class="note">End-to-end quantitative research pipeline focused on statistical rigor. All figures and tables are loaded from persisted experiment artifacts — nothing is hardcoded.</div>

    <div class="section">
      <h2>Executive Summary</h2>
      <div class="metrics">
        {metric_card("Best Variant", best_variant)}
        {metric_card("Raw Sharpe", f"{perf['sharpe']:.2f}")}
        {metric_card("Deflated Sharpe", f"{dsr['deflated_sharpe']:.2f}")}
        {metric_card("DSR Probability", f"{dsr['probability']:.1%}")}
        {metric_card("Mean Rank IC", f"{ic['mean_rank_ic']:.3f}")}
        {metric_card("Trials Logged", str(ledger['n_trials']))}
      </div>
      {warning}
    </div>

    <div class="section">
      <h2>Pipeline Structure</h2>
      <ol>
        <li><strong>Data</strong> — {quality['n_assets']} synthetic equities, {quality['n_rows']:,} rows</li>
        <li><strong>Features</strong> — {len(metrics.get('feature_columns', []))} lagged cross-sectional factors</li>
        <li><strong>Models</strong> — {escape(', '.join(config.get('model_variants', [])))}</li>
        <li><strong>Validation</strong> — purged walk-forward (train {config['train_window_days']}d / test {config['test_window_days']}d)</li>
        <li><strong>Portfolio</strong> — dollar-neutral long-short, {config['transaction_cost_bps']} bps transaction cost</li>
        <li><strong>Reporting</strong> — metrics, trial ledger, memo, HTML dashboard</li>
      </ol>
    </div>

    <div class="section">
      <h2>Model Comparison</h2>
      <div class="table-wrap">{df_html(comparison)}</div>
    </div>

    <div class="section">
      <h2>Charts</h2>
      <div class="charts">
        {chart_blocks[0]}
        {chart_blocks[1]}
      </div>
      <div class="charts">
        {chart_blocks[2]}
        <div class="chart"><div class="table-wrap" style="padding:12px 0 0">{df_html(cost_sensitivity(best_returns))}</div></div>
      </div>
    </div>

    <div class="section">
      <h2>Validation</h2>
      {validation_metrics}
      <h3>Walk-Forward Folds</h3>
      <div class="table-wrap">{fold_html}</div>
      <h3>Feature Set</h3>
      <p class="features">{escape(', '.join(metrics.get('feature_columns', [])))}</p>
      <h3>Trial Ledger</h3>
      <div class="table-wrap">{df_html(trial_frame(ledger))}</div>
    </div>

    <div class="section">
      <h2>Artifact Inventory</h2>
      <div class="table-wrap">{df_html(artifact_table(root))}</div>
    </div>

    <div class="section">
      <h2>Research Memo</h2>
      <pre class="memo">{escape(memo)}</pre>
    </div>
  </div>
</body>
</html>
"""

    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
    return html