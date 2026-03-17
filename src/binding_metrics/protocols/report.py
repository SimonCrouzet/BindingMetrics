"""HTML report generation for binding metrics CSV outputs.

Usage:
    binding-metrics-report --input scores.csv --output report.html
    binding-metrics-report --input scores.csv --output report.html --config my_report.json
    binding-metrics-report --input scores.csv --output report.html --title "My Experiment" --top-n 20
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default report configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    # Columns treated as metadata (excluded from histograms / summary stats)
    "metadata_columns": [
        "sample_id", "success", "error_message",
        "peptide_chain", "receptor_chain", "design_chain",
    ],
    # Ordered list of report sections to render
    "sections": ["summary", "success_rate", "histograms", "top_table"],
    # Named plots from PLOT_REGISTRY to render; null = auto-detect all numeric columns
    "plots": None,
    # Top-N table: number of rows to show
    "top_n": 10,
    # Column to rank by for the top table; null = first numeric column
    "rank_by": None,
    # Whether lower is better (True) or higher is better (False) for rank_by
    "rank_ascending": True,
    # Report title (overridden by --title CLI flag if provided)
    "title": "Binding Metrics Report",
}


def _merge_config(user_config: dict) -> dict:
    """Deep-merge user config on top of defaults."""
    cfg = dict(_DEFAULT_CONFIG)
    cfg.update(user_config)
    return cfg


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _make_histogram(series, spec) -> str:
    """Render a histogram for *series* using *spec* and return a base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = series.dropna()
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(data, bins=spec.bins, color=spec.color, edgecolor="white", linewidth=0.5)
    ax.set_title(spec.title, fontsize=11)
    ax.set_xlabel(spec.x_label, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    encoded = _fig_to_base64(fig)
    plt.close(fig)
    return encoded


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_summary(df, meta_cols: set[str]) -> str:
    import pandas as pd
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in meta_cols]
    if not numeric_cols:
        return "<p>No numeric metric columns found.</p>"
    stats = df[numeric_cols].describe().T[["count", "mean", "std", "min", "max"]]
    stats = stats.round(4)
    return stats.to_html(classes="metric-table", border=0)


def _render_success_rate(df) -> str:
    if "success" not in df.columns:
        return "<p><em>No <code>success</code> column found.</em></p>"
    total = len(df)
    n_ok = int(df["success"].sum()) if df["success"].dtype == bool else int((df["success"] == True).sum())
    n_fail = total - n_ok
    pct = 100 * n_ok / total if total else 0
    return (
        f"<p><strong>Total samples:</strong> {total} &nbsp;|&nbsp; "
        f"<strong>Success:</strong> {n_ok} ({pct:.1f}%) &nbsp;|&nbsp; "
        f"<strong>Failed:</strong> {n_fail}</p>"
    )


def _render_histograms(df, cfg: dict, meta_cols: set[str]) -> str:
    from binding_metrics.protocols.plots import PLOT_REGISTRY, PlotSpec

    plot_names: list[str] | None = cfg.get("plots")

    if plot_names is not None:
        # Explicit list: resolve names through registry
        specs: list[PlotSpec] = []
        for name in plot_names:
            if name not in PLOT_REGISTRY:
                print(f"warning: unknown plot name {name!r}, skipping", file=sys.stderr)
                continue
            specs.append(PLOT_REGISTRY[name])
    else:
        # Auto-detect: use registry entries whose column is present, then fallback for the rest
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in meta_cols]
        registry_cols = {spec.column: spec for spec in PLOT_REGISTRY.values()}
        specs = []
        seen: set[str] = set()
        for col in numeric_cols:
            if col in registry_cols:
                specs.append(registry_cols[col])
            else:
                specs.append(PlotSpec(column=col, title=col.replace("_", " ").title()))
            seen.add(col)

    imgs: list[str] = []
    for spec in specs:
        if spec.column not in df.columns:
            continue
        b64 = _make_histogram(df[spec.column], spec)
        imgs.append(f'<div class="plot-card"><img src="data:image/png;base64,{b64}" alt="{spec.title}"></div>')

    if not imgs:
        return "<p>No numeric columns to plot.</p>"
    return '<div class="plot-grid">' + "".join(imgs) + "</div>"


def _render_top_table(df, cfg: dict, meta_cols: set[str]) -> str:
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in meta_cols]
    if not numeric_cols:
        return "<p>No numeric columns available for ranking.</p>"

    rank_by: str = cfg.get("rank_by") or numeric_cols[0]
    if rank_by not in df.columns:
        return f"<p>Column {rank_by!r} not found in data.</p>"

    ascending: bool = cfg.get("rank_ascending", True)
    top_n: int = int(cfg.get("top_n", 10))

    id_cols = [c for c in ["sample_id"] if c in df.columns]
    display_cols = id_cols + [c for c in numeric_cols if c not in id_cols]

    top = (
        df[display_cols + ([rank_by] if rank_by not in display_cols else [])]
        .sort_values(rank_by, ascending=ascending)
        .head(top_n)
        .round(4)
    )
    return (
        f"<p><em>Ranked by <code>{rank_by}</code> "
        f"({'ascending' if ascending else 'descending'}), top {top_n}</em></p>"
        + top.to_html(classes="metric-table", border=0, index=False)
    )


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_SECTION_TITLES = {
    "summary": "Summary Statistics",
    "success_rate": "Success Rate",
    "histograms": "Metric Distributions",
    "top_table": "Top Samples",
}

_CSS = """
body { font-family: system-ui, sans-serif; max-width: 1300px; margin: 0 auto; padding: 24px; color: #222; }
h1 { border-bottom: 2px solid #ddd; padding-bottom: 8px; }
h2 { margin-top: 2em; color: #444; }
table.metric-table { border-collapse: collapse; width: 100%; font-size: 0.85em; }
table.metric-table th, table.metric-table td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
table.metric-table th { background: #f5f5f5; font-weight: 600; }
table.metric-table tr:nth-child(even) { background: #fafafa; }
.plot-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; margin-top: 12px; }
.plot-card img { width: 100%; border: 1px solid #eee; border-radius: 4px; }
.meta { font-size: 0.8em; color: #888; margin-bottom: 1.5em; }
"""


def _build_html(df, cfg: dict, meta_cols: set[str], source_path: str) -> str:
    import datetime

    title = cfg.get("title", "Binding Metrics Report")
    sections = cfg.get("sections", _DEFAULT_CONFIG["sections"])

    section_html = ""
    for sec in sections:
        heading = _SECTION_TITLES.get(sec, sec.replace("_", " ").title())
        if sec == "summary":
            body = _render_summary(df, meta_cols)
        elif sec == "success_rate":
            body = _render_success_rate(df)
        elif sec == "histograms":
            body = _render_histograms(df, cfg, meta_cols)
        elif sec == "top_table":
            body = _render_top_table(df, cfg, meta_cols)
        else:
            body = f"<p><em>Unknown section: {sec!r}</em></p>"
        section_html += f"<h2>{heading}</h2>\n{body}\n"

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Source: {source_path} &nbsp;|&nbsp; {len(df)} samples &nbsp;|&nbsp; Generated: {now}</p>
{section_html}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an HTML report from a binding metrics CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output HTML file")
    parser.add_argument("--config", "-c", type=Path, default=None, help="JSON config file (overrides defaults)")
    parser.add_argument("--title", type=str, default=None, help="Report title")
    parser.add_argument("--top-n", type=int, default=None, help="Number of rows in top table")
    parser.add_argument("--rank-by", type=str, default=None, help="Column to rank top table by")
    parser.add_argument("--rank-descending", action="store_true", help="Rank highest first (default: lowest first)")
    args = parser.parse_args()

    try:
        import pandas as pd
        import matplotlib  # noqa: F401 — fail early if missing
    except ImportError as e:
        print(
            f"error: {e}\nInstall report dependencies with: pip install binding-metrics[report]",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load user config and merge with defaults
    user_config: dict = {}
    if args.config:
        if not args.config.exists():
            print(f"error: config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        with open(args.config) as f:
            user_config = json.load(f)

    cfg = _merge_config(user_config)

    # CLI flags take precedence over config file
    if args.title:
        cfg["title"] = args.title
    if args.top_n is not None:
        cfg["top_n"] = args.top_n
    if args.rank_by:
        cfg["rank_by"] = args.rank_by
    if args.rank_descending:
        cfg["rank_ascending"] = False

    df = pd.read_csv(args.input)
    meta_cols: set[str] = set(cfg.get("metadata_columns", []))

    html = _build_html(df, cfg, meta_cols, source_path=str(args.input))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"report written to {args.output}  ({len(df)} samples)")
