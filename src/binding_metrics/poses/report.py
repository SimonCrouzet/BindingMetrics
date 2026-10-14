"""Writers of a pose-set result: JSON, CSV, a Markdown summary and a self-contained HTML page.

``write_pose_report(result, fmt, path)`` writes one file from a ``PoseSetResult`` (or the
dictionary that ``json.loads`` gives back from its JSON). No writer changes the result.

``json``
    The canonical dictionary (``PoseSetResult.to_json``), UTF-8, keys in the order of the
    result. A NaN or infinity that a registry metric returned is written as the token ``NaN`` or
    ``Infinity``, and ``nonfinite_fields`` lists where they are. ``include_timing=False``
    leaves the ``timing`` block out.

``csv``
    One row per pose, scalar columns only (``PoseSetResult.to_flat_rows``): a nested key is
    joined to its parent with ``.`` (``interface.delta_sasa``, ``native.iptm``,
    ``frame.receptor_rmsd_angstrom``), a short list of scalars is one column per item
    (``frame.transform.translation[0]``), and a list longer than ``max_list_items`` or a list of
    dictionaries (``residue_table``) is not in the CSV: it stays in the JSON. A key is never
    escaped, so a key with a dot (the clash cutoff ``2.2``) gives ``...clash_atoms.2.2``. The
    first columns are ``id``, ``kind``, ``status``, ``reason``, ``content_hash``; the others
    follow in the order they first occur over the poses in id order. A missing value is an empty
    cell, NaN is ``nan``, ``True`` and ``False`` are ``True`` and ``False``, a float is written
    with ``repr`` (it reads back to the same number). UTF-8, one ``\\n`` after each line.

``md``
    ``PoseSetResult.summary()``: a table per group of poses (the back-end by default) with the
    status, the receptor RMSD of the frame and the headline value of each requested metric, the
    failed poses and the poses with a missing value, each with its reason, the agreement block
    (pairwise peptide RMSD inside and between groups, clusters, medoids, interface overlap), the
    aggregates of the headline values per group, the representatives and the warnings.

``html``
    The same content as one page with inline styles: no script, link, external font or image.
    Every text is escaped.

Markdown and HTML are built from the same blocks. A file depends only on the result (its
``provenance`` and ``timing`` included), so the same result gives the same bytes.

Importing this module imports the standard library and the pose-set result module.
"""

from __future__ import annotations

import csv
import html
import io
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping

from binding_metrics.metrics.registry import METRICS_BY_NAME
from binding_metrics.poses.aggregate import group_key
from binding_metrics.poses.score import REPRESENTATIVE_META_KEYS, PoseSetResult

logger = logging.getLogger(__name__)

#: The formats of ``write_pose_report`` and the suffix that a caller usually gives each.
FORMATS = ("json", "csv", "md", "html")

#: Headline key of the registry metrics that return a bundle (no ``headline_key`` declared).
_BUNDLE_HEADLINES = {"interface": "delta_sasa", "prediction": "iptm"}

_NONE_TEXT = "n/a"


# ---------------------------------------------------------------------------- the writers


def write_pose_report(
    result: PoseSetResult | Mapping[str, Any],
    fmt: str = "json",
    path: str | os.PathLike | None = None,
    *,
    include_timing: bool = True,
    max_list_items: int = 16,
) -> Path:
    """Write a pose-set result to one file.

    Args:
        result: A ``PoseSetResult``, or the dictionary read back from its JSON (checked with
            ``PoseSetResult.from_dict``).
        fmt: ``"json"`` (default), ``"csv"``, ``"md"`` or ``"html"`` (see the module
            docstring).
        path: The file to write. Missing parent directories are created; an existing file is
            replaced. Nothing is written anywhere else.
        include_timing: ``json`` only: False leaves out the ``timing`` block.
        max_list_items: ``csv`` only: the longest list of scalars that becomes columns.

    Returns:
        The path that was written, as a ``Path``.

    Raises:
        ValueError: If ``fmt`` is not one of ``FORMATS``, ``path`` is missing, or ``result`` is
            a dictionary that is not a pose-set result of the schema version this code reads.
        TypeError: If ``result`` is neither a ``PoseSetResult`` nor a dictionary.
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {list(FORMATS)}, got {fmt!r}")
    if path is None:
        raise ValueError("path is required: the file to write")
    result = _as_result(result)
    target = Path(path)
    if fmt == "json":
        text = result.to_json(include_timing=include_timing) + "\n"
    elif fmt == "csv":
        text = _csv_text(result, max_list_items)
    elif fmt == "md":
        text = summary_markdown(result)
    else:
        text = summary_html(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    logger.info("pose-set report (%s) written to %s", fmt, target)
    return target


def _as_result(result: Any) -> PoseSetResult:
    if isinstance(result, PoseSetResult):
        return result
    if isinstance(result, Mapping):
        return PoseSetResult.from_dict(result)
    raise TypeError(
        f"result must be a PoseSetResult or the dictionary of one, got {type(result).__name__}"
    )


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    return str(value)


def _csv_text(result: PoseSetResult, max_list_items: int) -> str:
    rows = result.to_flat_rows(max_list_items=max_list_items)
    columns = list(rows[0]) if rows else []
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_csv_cell(row[name]) for name in columns])
    return buffer.getvalue()


# ---------------------------------------------------------------------------- the content

# A block is a tuple: ("title", text), ("heading", level, text), ("text", text),
# ("list", [text, ...]) or ("table", header, rows, aligns) with aligns "l" or "r" per column.
Block = tuple


def _number(value: Any) -> str:
    """A value for a table cell: four significant digits, ``n/a`` for a missing one."""
    if value is None:
        return _NONE_TEXT
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.4g}"
    return str(value)


def _dig(block: Any, path: str) -> Any:
    """The value at a dotted path of nested dictionaries, None where there is none."""
    value = block
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _headline_columns(result: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``[(metric name, headline key)]`` for the requested metrics, in request order.

    The key is the registry ``headline_key``, else the entry of ``_BUNDLE_HEADLINES``, else the
    first top-level number of the metric's block in the first pose that has one.
    """
    columns = []
    rows = result["poses"]
    for name in result["parameters"].get("metrics", []):
        spec = METRICS_BY_NAME.get(name)
        key = spec.headline_key if spec is not None else None
        key = key or _BUNDLE_HEADLINES.get(name)
        if key is None:
            for row in rows:
                block = row.get(name)
                if isinstance(block, Mapping):
                    key = next(
                        (
                            k
                            for k, v in block.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)
                        ),
                        None,
                    )
                    if key is not None:
                        break
        if key is not None:
            columns.append((name, key))
    return columns


def _groups(result: Mapping[str, Any]) -> dict[str, list[dict]]:
    """The rows by group, groups in sorted order (the group of ``group_by`` of the call)."""
    group_by = tuple(result["parameters"].get("group_by", ("backend",)))
    groups: dict[str, list[dict]] = {}
    for row in result["poses"]:
        groups.setdefault(group_key(row, group_by) if group_by else "all", []).append(row)
    return {name: groups[name] for name in sorted(groups)}


def _seed_of(row: Mapping[str, Any]) -> str:
    provenance = row.get("provenance") or {}
    for key in ("seed", "seed_index"):
        if provenance.get(key) is not None:
            return str(provenance[key])
    return _NONE_TEXT


def _pose_tables(result: Mapping[str, Any], headline: list[tuple[str, str]]) -> list[Block]:
    blocks: list[Block] = [("heading", 2, "Poses")]
    header = ["id", "seed", "status", "receptor RMSD (A)"] + [f"{m}.{k}" for m, k in headline]
    aligns = ["l", "r", "l", "r"] + ["r"] * len(headline)
    for group, rows in _groups(result).items():
        table = []
        for row in rows:
            frame = row.get("frame") or {}
            table.append(
                [
                    row["id"],
                    _seed_of(row),
                    row["status"],
                    _number(frame.get("receptor_rmsd_angstrom")),
                ]
                + [_number(_dig(row.get(metric), key)) for metric, key in headline]
            )
        blocks += [
            ("heading", 3, f"{group} ({len(rows)} pose{'s' if len(rows) != 1 else ''})"),
            ("table", header, table, aligns),
        ]
    return blocks


def _problem_tables(result: Mapping[str, Any]) -> list[Block]:
    failed, partial = [], []
    for row in result["poses"]:
        if row["status"] == "failed":
            stage = next(iter(row["reasons"]), "")
            failed.append([row["id"], stage, row["reason"] or ""])
        elif row["status"] == "partial":
            partial += [[row["id"], name, text] for name, text in row["reasons"].items()]
    blocks: list[Block] = [("heading", 2, "Failed poses")]
    if failed:
        blocks.append(("table", ["id", "stage", "reason"], failed, ["l", "l", "l"]))
    else:
        blocks.append(("text", "No pose failed."))
    if partial:
        blocks += [
            ("heading", 2, "Poses with a missing value"),
            ("table", ["id", "stage or metric", "reason"], partial, ["l", "l", "l"]),
        ]
    return blocks


def _statistic_rows(kind: str, statistics: Mapping[str, Any]) -> list[list[str]]:
    def line(scope: str, entry: Mapping[str, Any]) -> list[str]:
        return [
            kind,
            scope,
            _number(entry.get("n_defined")),
            _number(entry.get("min")),
            _number(entry.get("mean")),
            _number(entry.get("max")),
        ]

    rows = [line("all poses", statistics["overall"])]
    rows += [line(f"within {group}", entry) for group, entry in statistics["within"].items()]
    rows += [line(f"between {pair}", entry) for pair, entry in statistics["between"].items()]
    return rows


def _agreement_blocks(agreement: Mapping[str, Any]) -> list[Block]:
    blocks: list[Block] = [("heading", 2, "Agreement")]
    if agreement.get("status") != "ok":
        reason = agreement.get("reason") or "not computed"
        return blocks + [("text", f"Agreement was not computed: {reason}.")]
    ids = agreement["pose_ids"]
    text = (
        f"{len(ids)} poses compared in the frame of the reference; the clusters and the medoids "
        f"use the {agreement['kind']} atoms (complete linkage), the spread is of the alpha carbons."
    )
    if agreement["excluded"]:
        text += f" Left out: {', '.join(sorted(agreement['excluded']))}."
    blocks.append(("text", text))
    header = ["atoms", "scope", "pairs", "min (A)", "mean (A)", "max (A)"]
    aligns = ["l", "l", "r", "r", "r", "r"]
    pose_rows: list[list[str]] = []
    for kind, entry in agreement["rmsd"].items():
        pose_rows += _statistic_rows(kind, entry["statistics"])
    blocks += [
        ("heading", 3, "Peptide RMSD, no refit"),
        ("table", header, pose_rows, aligns),
    ]
    if agreement.get("shape_rmsd"):
        shape_rows: list[list[str]] = []
        for kind, entry in agreement["shape_rmsd"].items():
            shape_rows += _statistic_rows(kind, entry["statistics"])
        blocks += [
            ("heading", 3, "Shape RMSD, peptides refitted"),
            ("table", header, shape_rows, aligns),
        ]
    reasons = {
        entry["statistics"]["between_reason"]
        for entry in agreement["rmsd"].values()
        if entry["statistics"].get("between_reason")
    }
    if reasons:
        blocks.append(("text", f"Between groups: {', '.join(sorted(reasons))}."))
    clusters = [
        [
            cutoff,
            _number(entry["n_clusters"]),
            "-" if entry["sizes"] is None else ", ".join(str(n) for n in entry["sizes"]),
            _number(entry["largest_cluster_fraction"]),
            "-" if entry["mixed_clusters"] is None else str(len(entry["mixed_clusters"])),
        ]
        for cutoff, entry in agreement["clusters"].items()
    ]
    blocks += [
        ("heading", 3, "Clusters"),
        (
            "table",
            ["cutoff (A)", "clusters", "sizes", "largest fraction", "mixed clusters"],
            clusters,
            ["r", "r", "l", "r", "r"],
        ),
    ]
    medoids = agreement["medoids"]
    items = [f"all poses: {_number(medoids['overall']['pose_id'])}"]
    items += [f"{group}: {_number(m['pose_id'])}" for group, m in medoids["per_group"].items()]
    items += [
        f"distance between the medoids of {pair}: {_number(entry['distance_angstrom'])} A"
        for pair, entry in medoids["between_groups"].items()
    ]
    blocks += [("heading", 3, "Medoids"), ("list", items)]
    contacts = agreement.get("contacts") or {}
    overlap = []
    for name, entry in (contacts.get("interface_jaccard") or {}).items():
        overlap.append((f"interface, {name}", entry["statistics"]["overall"]))
    if contacts.get("contact_map_jaccard"):
        overlap.append(("contact map", contacts["contact_map_jaccard"]["statistics"]["overall"]))
    if overlap:
        blocks += [
            ("heading", 3, "Interface overlap (Jaccard index)"),
            (
                "table",
                ["set", "pairs", "min", "mean", "max"],
                [
                    [
                        name,
                        _number(e.get("n_defined")),
                        _number(e.get("min")),
                        _number(e.get("mean")),
                        _number(e.get("max")),
                    ]
                    for name, e in overlap
                ],
                ["l", "r", "r", "r", "r"],
            ),
        ]
    spread = agreement.get("spread") or {}
    items = [f"all poses: {_number(spread.get('all', {}).get('mean_spread_angstrom'))} A"]
    items += [
        f"{group}: {_number(entry.get('mean_spread_angstrom'))} A"
        for group, entry in (spread.get("per_group") or {}).items()
    ]
    blocks += [("heading", 3, "Mean spread of the alpha carbons per position"), ("list", items)]
    return blocks


def _aggregate_blocks(result: Mapping[str, Any], headline: list[tuple[str, str]]) -> list[Block]:
    statistics = result["parameters"].get("aggregate", ["mean", "std", "min", "max", "median"])
    paths = ["frame.receptor_rmsd_angstrom"] + [f"{metric}.{key}" for metric, key in headline]
    native = sorted(
        {
            f"native.{name}"
            for row in result["poses"]
            for name, value in (row.get("native") or {}).items()
            if isinstance(value, (int, float))
        }
    )
    header = ["key", "n_ok / n_total", *statistics]
    aligns = ["l", "r"] + ["r"] * len(statistics)
    blocks: list[Block] = [("heading", 2, "Aggregates")]
    for group, entries in result["aggregates"].items():
        table = []
        for path in paths + native:
            entry = entries.get(path)
            if entry is None:
                continue
            table.append(
                [path, f"{entry['n_ok']} / {entry['n_total']}"]
                + [_number(entry.get(name)) for name in statistics]
            )
        if table:
            blocks += [("heading", 3, group), ("table", header, table, aligns)]
    if len(blocks) == 1:
        blocks.append(("text", "No numeric value to aggregate."))
    return blocks


def _representative_blocks(representatives: Mapping[str, Any]) -> list[Block]:
    items = [
        f"{group}: {_number(pose_id)}"
        for group, pose_id in representatives.items()
        if group not in REPRESENTATIVE_META_KEYS
    ]
    blocks: list[Block] = [
        ("heading", 2, "Representatives"),
        ("text", f"Rule: {representatives.get('rule')}."),
        ("list", items),
    ]
    reasons = representatives.get("reasons") or {}
    if reasons:
        blocks.append(("list", [f"no representative for {k}: {v}" for k, v in reasons.items()]))
    caveats = representatives.get("caveats") or {}
    if caveats:
        blocks.append(("list", [f"caveat for {k}: {v}" for k, v in caveats.items()]))
    return blocks


def _blocks(result: PoseSetResult) -> list[Block]:
    rows = result["poses"]
    groups = _groups(result)
    status = {s: sum(1 for row in rows if row["status"] == s) for s in ("ok", "partial", "failed")}
    construct = result.get("construct")
    references = result["references"]
    reference = (
        f"the first readable pose ({references['implicit_pose_id']})"
        if references["implicit"]
        else ", ".join(references["variants"]) or "none"
    )
    blocks: list[Block] = [
        ("title", f"Pose set: {construct}" if construct else "Pose set"),
        (
            "text",
            f"{len(rows)} poses in {len(groups)} group{'s' if len(groups) != 1 else ''}: "
            f"{status['ok']} ok, {status['partial']} partial, {status['failed']} failed. "
            f"Binder chain {_number(result['binder_chain'])}, target chain "
            f"{_number(result['target_chain'])}. Reference: {reference}.",
        ),
        (
            "text",
            f"Result schema version {result['schema_version']}, binding_metrics "
            f"{_number(result['package_version'])}, input key {result['input_key'][:12]}.",
        ),
    ]
    headline = _headline_columns(result)
    blocks += _pose_tables(result, headline)
    blocks += _problem_tables(result)
    blocks += _agreement_blocks(result["agreement"])
    blocks += _aggregate_blocks(result, headline)
    blocks += _representative_blocks(result["representatives"])
    notes = list(result["warnings"])
    if result["nonfinite_fields"]:
        notes.append(
            f"{len(result['nonfinite_fields'])} value(s) are NaN or infinite in the JSON; "
            "nonfinite_fields lists them."
        )
    if notes:
        blocks += [("heading", 2, "Warnings"), ("list", notes)]
    return blocks


# ---------------------------------------------------------------------------- Markdown


def _md_cell(text: Any) -> str:
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _md_table(header: list[str], rows: list[list[str]], aligns: list[str]) -> list[str]:
    rule = ["---:" if align == "r" else "---" for align in aligns]
    lines = ["| " + " | ".join(_md_cell(h) for h in header) + " |", "|" + "|".join(rule) + "|"]
    lines += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in rows]
    return lines


def summary_markdown(result: PoseSetResult | Mapping[str, Any]) -> str:
    """The Markdown summary of a result (what ``PoseSetResult.summary()`` returns).

    Contents: the poses per group with their headline values, the failed poses with their
    reasons, the agreement block, the aggregates, the representatives and the warnings. It ends
    with one newline.
    """
    lines: list[str] = []
    for block in _blocks(_as_result(result)):
        kind = block[0]
        if kind == "title":
            lines += [f"# {block[1]}", ""]
        elif kind == "heading":
            lines += ["#" * block[1] + f" {block[2]}", ""]
        elif kind == "text":
            lines += [block[1], ""]
        elif kind == "list":
            lines += [f"- {item}" for item in block[1]] + [""]
        else:
            lines += _md_table(block[1], block[2], block[3]) + [""]
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------- HTML

_CSS = """\
:root{--bg:#ffffff;--fg:#1b1f24;--line:#d0d7de;--head:#f3f5f7;--accent:#0b4f8a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#10151b;--fg:#e6eaee;\
--line:#2d3640;--head:#18202a;--accent:#6cb2f0}}
:root[data-theme="dark"]{--bg:#10151b;--fg:#e6eaee;--line:#2d3640;--head:#18202a;\
--accent:#6cb2f0}
body{background:var(--bg);color:var(--fg);margin:0 auto;padding:24px 16px 48px;max-width:70rem;\
font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
h1{font-size:1.6rem;margin:0 0 .5rem;color:var(--accent)}
h2{font-size:1.2rem;margin:2rem 0 .5rem;border-bottom:1px solid var(--line);padding-bottom:.25rem}
h3{font-size:1rem;margin:1.25rem 0 .4rem}
p,li{color:var(--fg)}
.wrap{overflow-x:auto}
table{border-collapse:collapse;margin:.25rem 0 1rem;font-size:.9rem}
th,td{border:1px solid var(--line);padding:.25rem .6rem;text-align:left;vertical-align:top}
th{background:var(--head)}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
"""


def _html_table(header: list[str], rows: list[list[str]], aligns: list[str]) -> str:
    def cell(tag: str, text: Any, align: str) -> str:
        css = ' class="r"' if align == "r" else ""
        return f"<{tag}{css}>{html.escape(str(text), quote=True)}</{tag}>"

    head = "".join(cell("th", h, a) for h, a in zip(header, aligns))
    body = "\n".join(
        "<tr>" + "".join(cell("td", c, a) for c, a in zip(row, aligns)) + "</tr>" for row in rows
    )
    return (
        f'<div class="wrap"><table>\n<thead><tr>{head}</tr></thead>\n'
        f"<tbody>\n{body}\n</tbody>\n</table></div>"
    )


def summary_html(result: PoseSetResult | Mapping[str, Any]) -> str:
    """The summary as one HTML page with inline styles and nothing fetched from a network.

    Same content as ``summary_markdown``. Every text is escaped; no script, link, external
    font or image.
    """
    blocks = _blocks(_as_result(result))
    title = next(block[1] for block in blocks if block[0] == "title")
    parts = []
    for block in blocks:
        kind = block[0]
        if kind == "title":
            parts.append(f"<h1>{html.escape(block[1], quote=True)}</h1>")
        elif kind == "heading":
            parts.append(f"<h{block[1]}>{html.escape(block[2], quote=True)}</h{block[1]}>")
        elif kind == "text":
            parts.append(f"<p>{html.escape(block[1], quote=True)}</p>")
        elif kind == "list":
            items = "\n".join(f"<li>{html.escape(item, quote=True)}</li>" for item in block[1])
            parts.append(f"<ul>\n{items}\n</ul>")
        else:
            parts.append(_html_table(block[1], block[2], block[3]))
    body = "\n".join(parts)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title, quote=True)}</title>\n"
        f"<style>\n{_CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


__all__ = [
    "FORMATS",
    "summary_html",
    "summary_markdown",
    "write_pose_report",
]
