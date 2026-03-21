"""Final pipeline step: write results to disk as JSON or CSV, with optional Markdown summary.

Used internally by ``binding-metrics-run`` and as a standalone tool.

Usage:
    binding-metrics-report --results path/to/*_results.json [--format json|csv] [--summary]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Scorecard thresholds used by the Markdown summary.
#
# Each entry:
#   key       : tuple of keys to traverse from the root results dict
#   label     : display name
#   unit      : appended to the value in the table
#   direction : "lower" or "higher" (informational)
#   green(v)  : callable → True for best band
#   amber(v)  : callable → True for middle band; else RED
#
# Energies from OpenMM are in kJ/mol.
# SASA-derived ΔG (interface module) is in kcal/mol.
# ---------------------------------------------------------------------------
_THRESHOLDS: list[dict] = [
    # MD RMSD (Å): < 2 excellent, 2–5 moderate, > 5 poor
    dict(key=("relax", "rmsd_md_final"), label="MD RMSD", unit="Å", direction="lower",
         green=lambda v: v < 2.0, amber=lambda v: v < 5.0),
    # Peptide RMSF mean (Å): < 1 rigid, 1–2 moderate, > 2 flexible
    dict(key=("relax", "peptide_rmsf_mean"), label="RMSF mean", unit="Å", direction="lower",
         green=lambda v: v < 1.0, amber=lambda v: v < 2.0),
    # OpenMM interaction energy (kJ/mol): < −40 strong, −40–0 moderate, > 0 repulsive
    dict(key=("energy", "relaxed_interaction_energy"), label="E_int", unit="kJ/mol", direction="lower",
         green=lambda v: v < -40.0, amber=lambda v: v < 0.0),
    # ΔSASA (Å²): > 1000 well-buried, 500–1000 moderate, < 500 poor
    dict(key=("interface", "delta_sasa"), label="ΔSASA", unit="Å²", direction="higher",
         green=lambda v: v > 1000.0, amber=lambda v: v > 500.0),
    # H-bond count: ≥ 5 good, 2–4 acceptable, < 2 poor
    dict(key=("interface", "hbonds"), label="H-bonds", unit="", direction="higher",
         green=lambda v: v >= 5, amber=lambda v: v >= 2),
    # Salt bridges: ≥ 2 good, 1 acceptable, 0 poor
    dict(key=("interface", "saltbridges"), label="Salt bridges", unit="", direction="higher",
         green=lambda v: v >= 2, amber=lambda v: v >= 1),
    # Ramachandran favoured %: > 95 excellent, 80–95 acceptable, < 80 poor
    dict(key=("geometry", "ramachandran", "ramachandran_favoured_pct"),
         label="Rama favoured", unit="%", direction="higher",
         green=lambda v: v > 95.0, amber=lambda v: v > 80.0),
    # ω outlier fraction: < 0.05 good, 0.05–0.20 acceptable, > 0.20 poor
    dict(key=("geometry", "omega", "omega_outlier_fraction"),
         label="ω outlier frac", unit="", direction="lower",
         green=lambda v: v < 0.05, amber=lambda v: v < 0.20),
    # Coulomb energy (kJ/mol): < −100 favourable, −100–0 moderate, > 0 poor
    dict(key=("electrostatics", "coulomb_energy_kJ"), label="Coulomb E", unit="kJ/mol", direction="lower",
         green=lambda v: v < -100.0, amber=lambda v: v < 0.0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nested_get(d: dict, *keys) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _rag(value: Any, spec: dict) -> str:
    if value is None:
        return "⬜"
    try:
        if spec["green"](value):
            return "🟢"
        if spec["amber"](value):
            return "🟡"
        return "🔴"
    except Exception:
        return "⬜"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(h), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |"
        for row in rows
    )
    return f"{head}\n{sep}\n{body}"


# ---------------------------------------------------------------------------
# CSV flattening — extracts scalar leaf values from the nested results dict
# ---------------------------------------------------------------------------

def _flatten(results: dict) -> dict[str, Any]:
    """Return a flat {column: value} dict suitable for one CSV row."""
    flat: dict[str, Any] = {
        "sample_id": results.get("sample_id"),
        "input": results.get("input"),
        "total_elapsed_s": results.get("total_elapsed_s"),
    }

    def _add(prefix: str, d: Any) -> None:
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            if k in ("per_residue", "charged_atoms_peptide", "charged_atoms_receptor",
                     "interface_residues_peptide", "interface_residues_receptor",
                     "peptide_rmsf_per_residue"):
                continue  # skip list fields
            if isinstance(v, dict):
                _add(f"{prefix}_{k}", v)
            elif not isinstance(v, list):
                flat[f"{prefix}_{k}"] = v

    for section in ("relax", "energy", "interface", "geometry", "electrostatics", "openfold"):
        if section in results:
            _add(section, results[section])

    return flat


# ---------------------------------------------------------------------------
# Markdown summary sections
# ---------------------------------------------------------------------------

def _md_header(results: dict) -> str:
    sid = results.get("sample_id", "unknown")
    inp = results.get("input", "—")
    total = results.get("total_elapsed_s")
    elapsed = f"{total:.1f} s" if total is not None else "—"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"# Binding Metrics Report\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Sample ID | `{sid}` |\n"
        f"| Input | `{inp}` |\n"
        f"| Elapsed | {elapsed} |\n"
        f"| Generated | {now} |\n"
    )


def _is_skipped(section: dict | None) -> bool:
    return isinstance(section, dict) and section.get("skipped") is True


def _md_cyclic(relax: dict | None) -> str | None:
    """Return a cyclic topology section, or None if linear."""
    bonds = (relax or {}).get("cyclic_bonds")
    if not bonds:
        return None
    lines = ["## Cyclic topology\n"]
    lines.append(_md_table(
        ["Bond type", "Atom 1", "Atom 2"],
        [[b["type"], b["atom1"], b["atom2"]] for b in bonds],
    ))
    lines.append(
        f"\n> ℹ️ {len(bonds)} closure bond(s) detected. "
        "Ramachandran and ω outlier scores should be interpreted with this in mind."
    )
    return "\n".join(lines) + "\n"


def _md_relax(relax: dict | None) -> str:
    lines = ["## Relaxation\n"]
    if _is_skipped(relax):
        return lines[0] + "_Skipped._\n"
    if not relax or not relax.get("success", False):
        return lines[0] + f"_Failed: {(relax or {}).get('error_message', 'missing')}_\n"
    lines.append(_md_table(
        ["Metric", "Value", "Unit"],
        [
            ["Min. energy",   _fmt(relax.get("potential_energy_minimized")), "kJ/mol"],
            ["MD avg energy", _fmt(relax.get("potential_energy_md_avg")),    "kJ/mol"],
            ["MD energy std", _fmt(relax.get("potential_energy_md_std")),    "kJ/mol"],
            ["RMSD (final)",  _fmt(relax.get("rmsd_md_final")),              "Å"],
            ["RMSF mean",     _fmt(relax.get("peptide_rmsf_mean")),          "Å"],
            ["RMSF max",      _fmt(relax.get("peptide_rmsf_max")),           "Å"],
        ],
    ))
    rmsf_raw = relax.get("peptide_rmsf_per_residue")
    if rmsf_raw:
        try:
            vals: list[float] = json.loads(rmsf_raw) if isinstance(rmsf_raw, str) else list(rmsf_raw)
            flagged = [f"res{i+1}={v:.2f}" for i, v in enumerate(vals) if v > 1.5]
            if flagged:
                lines.append(f"\n⚠️ **High RMSF (> 1.5 Å):** {', '.join(flagged)}")
        except Exception:
            pass
    return "\n".join(lines) + "\n"


def _md_energy(energy: dict | None) -> str:
    lines = ["## Interaction Energy\n"]
    if _is_skipped(energy):
        return lines[0] + "_Skipped._\n"
    if not energy or not energy.get("success", True):
        return lines[0] + f"_Failed: {(energy or {}).get('error_message', 'absent')}_\n"
    lines.append(_md_table(
        ["Metric", "Value", "Unit"],
        [
            ["E_int",          _fmt(energy.get("relaxed_interaction_energy")), "kJ/mol"],
            ["E_complex",      _fmt(energy.get("relaxed_e_complex")),          "kJ/mol"],
            ["E_peptide",      _fmt(energy.get("relaxed_e_peptide")),          "kJ/mol"],
            ["E_receptor",     _fmt(energy.get("relaxed_e_receptor")),         "kJ/mol"],
            ["Contacts",       _fmt(energy.get("num_contacts")),               ""],
            ["Close contacts", _fmt(energy.get("num_close_contacts")),         ""],
        ],
    ))
    return "\n".join(lines) + "\n"


def _md_interface(iface: dict | None) -> str:
    lines = ["## Interface\n"]
    if _is_skipped(iface):
        return lines[0] + "_Skipped._\n"
    if not iface:
        return lines[0] + "_Absent._\n"
    lines.append(_md_table(
        ["Metric", "Value", "Unit"],
        [
            ["ΔSASA",               _fmt(iface.get("delta_sasa")),              "Å²"],
            ["ΔG_int",              _fmt(iface.get("delta_g_int")),             "kcal/mol"],
            ["ΔG_int",              _fmt(iface.get("delta_g_int_kJ")),          "kJ/mol"],
            ["Polar area",          _fmt(iface.get("polar_area")),              "Å²"],
            ["Apolar area",         _fmt(iface.get("apolar_area")),             "Å²"],
            ["Fraction polar",      _fmt(iface.get("fraction_polar"), 4),       ""],
            ["H-bonds",             _fmt(iface.get("hbonds")),                  ""],
            ["Salt bridges",        _fmt(iface.get("saltbridges")),             ""],
            ["Interface res (pep)", _fmt(iface.get("n_interface_residues_peptide")),  ""],
            ["Interface res (rec)", _fmt(iface.get("n_interface_residues_receptor")), ""],
        ],
    ))
    # Per-residue buried SASA for peptide
    peptide_chain = iface.get("peptide_chain", "B")
    pep_res = sorted(
        [r for r in (iface.get("per_residue") or []) if r.get("chain") == peptide_chain],
        key=lambda r: r.get("res_id", 0),
    )
    if pep_res:
        lines.append("\n**Per-residue buried SASA — peptide**\n")
        lines.append(_md_table(
            ["Residue", "Buried SASA (Å²)", "Polar (Å²)", "Apolar (Å²)", "ΔG_res (kcal/mol)"],
            [[f"{r['res_name']}:{r['res_id']}", _fmt(r.get("buried_sasa"), 2),
              _fmt(r.get("polar_area"), 2), _fmt(r.get("apolar_area"), 2),
              _fmt(r.get("delta_g_res"), 4)] for r in pep_res],
        ))
    return "\n".join(lines) + "\n"


def _md_geometry(geo: dict | None) -> str:
    lines = ["## Geometry\n"]
    if _is_skipped(geo):
        return lines[0] + "_Skipped._\n"
    if not geo:
        return lines[0] + "_Absent._\n"
    rama = geo.get("ramachandran") or {}
    omega = geo.get("omega") or {}
    lines.append("**Ramachandran**\n")
    lines.append(_md_table(
        ["Metric", "Value"],
        [
            ["Favoured", f"{_fmt(rama.get('ramachandran_favoured_pct'), 1)} %"],
            ["Allowed",  f"{_fmt(rama.get('ramachandran_allowed_pct'),  1)} %"],
            ["Outliers", f"{_fmt(rama.get('ramachandran_outlier_pct'),  1)} % "
                         f"(n={rama.get('ramachandran_outlier_count', 0)})"],
            ["Residues evaluated", _fmt(rama.get("n_residues_evaluated"))],
        ],
    ))
    for r in [r for r in (rama.get("per_residue") or []) if r.get("region") == "outlier"]:
        lines.append(f"\n⚠️ **Rama outlier:** {r['res_name']}{r['res_id']} "
                     f"(φ={r['phi']:.1f}°, ψ={r['psi']:.1f}°)")
    lines.append("\n**Peptide-bond planarity (ω)**\n")
    lines.append(_md_table(
        ["Metric", "Value"],
        [
            ["Mean deviation",   f"{_fmt(omega.get('omega_mean_dev'), 1)} °"],
            ["Max deviation",    f"{_fmt(omega.get('omega_max_dev'),  1)} °"],
            ["Outlier fraction", _fmt(omega.get("omega_outlier_fraction"), 3)],
            ["Outlier count",    _fmt(omega.get("omega_outlier_count"))],
        ],
    ))
    for r in [r for r in (omega.get("per_residue") or []) if r.get("is_outlier")]:
        lines.append(f"\n⚠️ **ω outlier:** {r['res_name']}{r['res_id']} "
                     f"(ω={r['omega']:.1f}°, dev={r['deviation']:.1f}°)")
    return "\n".join(lines) + "\n"


def _md_electrostatics(elec: dict | None) -> str:
    lines = ["## Electrostatics\n"]
    if _is_skipped(elec):
        return lines[0] + "_Skipped._\n"
    if not elec:
        return lines[0] + "_Absent._\n"
    lines.append(_md_table(
        ["Metric", "Value", "Unit"],
        [
            ["Coulomb E",     _fmt(elec.get("coulomb_energy_kJ")),   "kJ/mol"],
            ["Coulomb E",     _fmt(elec.get("coulomb_energy_kcal")), "kcal/mol"],
            ["Charged pairs", _fmt(elec.get("n_charged_pairs")),     ""],
            ["Attractive",    _fmt(elec.get("n_attractive")),        ""],
            ["Repulsive",     _fmt(elec.get("n_repulsive")),         ""],
        ],
    ))
    return "\n".join(lines) + "\n"


def _md_openfold(of: dict | None) -> str:
    lines = ["## OpenFold\n"]
    if _is_skipped(of):
        return lines[0] + "_Skipped._\n"
    if not of:
        return lines[0] + "_Absent._\n"
    rows = [
        ["avg pLDDT",  _fmt(of.get("avg_plddt"), 2)],
        ["pTM",        _fmt(of.get("ptm"), 3)],
        ["ipTM",       _fmt(of.get("iptm"), 3)],
        ["gPDE",       f"{_fmt(of.get('gpde'), 2)} Å"],
    ]
    refold_rmsd = of.get("binder_ca_rmsd")
    if refold_rmsd is not None:
        rows.append(["Refolding RMSD", f"{_fmt(refold_rmsd, 2)} Å"])
    lines.append(_md_table(["Metric", "Value"], rows))
    # per-residue low pLDDT warning
    plddt_per_res = of.get("binder_plddt_per_residue")
    if plddt_per_res is not None:
        try:
            import numpy as np
            arr = np.asarray(plddt_per_res, dtype=float)
            low_idx = [i for i, v in enumerate(arr) if v < 70]
            if low_idx:
                low_strs = [f"res{i+1} ({plddt_per_res[i]:.1f})" for i in low_idx]
                lines.append(f"\n⚠️ **Low binder pLDDT (< 70):** {', '.join(low_strs)}")
        except Exception:
            pass
    return "\n".join(lines) + "\n"


def _md_scorecard(results: dict) -> str:
    rows = []
    for spec in _THRESHOLDS:
        value = _nested_get(results, *spec["key"])
        rows.append([
            _rag(value, spec),
            spec["label"],
            _fmt(value),
            spec["unit"],
            "↓" if spec["direction"] == "lower" else "↑",
        ])
    lines = ["## Summary Scorecard\n"]
    lines.append(_md_table(["", "Metric", "Value", "Unit", "Better"], rows))
    lines.append("\n🟢 OK  🟡 AMBER  🔴 RED  ⬜ N/A")
    return "\n".join(lines) + "\n"


def _build_summary(results: dict) -> str:
    cyclic = _md_cyclic(results.get("relax"))
    sections = [
        _md_header(results),
        _md_relax(results.get("relax")),
        *([] if cyclic is None else [cyclic]),
        _md_energy(results.get("energy")),
        _md_interface(results.get("interface")),
        _md_geometry(results.get("geometry")),
        _md_electrostatics(results.get("electrostatics")),
    ]
    if "openfold" in results:
        sections.append(_md_openfold(results["openfold"]))
    sections.append(_md_scorecard(results))
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_report(
    results: dict,
    output_dir: Path,
    sample_id: str,
    fmt: str = "json",
    summary: bool = False,
) -> Path:
    """Write pipeline results to *output_dir*.

    Args:
        results:    Aggregated results dict from the pipeline.
        output_dir: Directory to write outputs into.
        sample_id:  Used as the file name stem.
        fmt:        Primary output format — ``"json"`` (default) or ``"csv"``.
        summary:    If True, also write a human-readable ``*_report.md``.

    Returns:
        Path to the primary output file (JSON or CSV).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        out_path = output_dir / f"{sample_id}_results.csv"
        flat = _flatten(results)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flat.keys()))
            writer.writeheader()
            writer.writerow(flat)
    else:  # json (default)
        out_path = output_dir / f"{sample_id}_results.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

    if summary:
        md_path = output_dir / f"{sample_id}_report.md"
        md_path.write_text(_build_summary(results), encoding="utf-8")
        print(f"  summary   → {md_path}")

    return out_path


# ---------------------------------------------------------------------------
# CLI (standalone use)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export binding-metrics results to JSON or CSV, with optional Markdown summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", "-r", type=Path, required=True,
                        help="Path to a *_results.json file")
    parser.add_argument("--format", "-f", choices=["json", "csv"], default="json",
                        dest="fmt", help="Output format")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Also write a human-readable *_report.md")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Output directory (default: same directory as --results)")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"error: file not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    with open(args.results, encoding="utf-8") as fh:
        results = json.load(fh)

    sample_id = results.get("sample_id") or args.results.stem.removesuffix("_results")
    output_dir = args.output_dir or args.results.parent

    out = write_report(results, output_dir, sample_id, fmt=args.fmt, summary=args.summary)
    print(f"  results   → {out}")
