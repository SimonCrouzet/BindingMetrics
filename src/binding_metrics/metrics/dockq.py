"""DockQ reference-based accuracy metrics for predicted complexes.

Unlike the physics-based scorers in this package (energy, SASA, electrostatics,
…) which score a *single* structure on physical plausibility, these metrics are
**reference-based**: they quantify how close a *predicted* complex is to a known
*native* (reference) structure. They only make sense for benchmarking or
retrospective validation — never for scoring a design in isolation — and are
therefore gated behind an explicit ``--reference`` argument.

This module wraps the canonical DockQ tool (Basu & Wallner, *PLoS ONE* 11:e0161879,
2016; https://github.com/bjornwallner/DockQ, MIT licensed) and reports the
CAPRI-style quantities used by CAPRI/CASP assessors:

    - DockQ    : composite score in [0, 1] combining the three below
    - fnat     : fraction of native residue-residue contacts recovered
    - fnonnat  : fraction of predicted contacts that are non-native
    - iRMSD    : interface backbone RMSD (Å)   — CAPRI "i-RMSD"
    - LRMSD    : ligand RMSD after receptor superposition (Å) — CAPRI "L-RMSD"

DockQ runs its own optimal chain-mapping search, so antibody-antigen complexes
whose chains are named differently or permuted between model and reference
(H/L chains, VHH, antigen) are matched automatically — something the exact-key
matching in ``comparison.py`` does not do.

We invoke the ``DockQ`` command-line tool with ``--json`` rather than calling the
Python API: the CLI performs the full automatic chain-mapping search and emits
structured JSON, which is stable across DockQ releases.

Requires the DockQ package::

    pip install DockQ

Usage::

    binding-metrics-dockq --model predicted.cif --reference native.cif
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# DockQ-score thresholds for CAPRI quality classes (Basu & Wallner 2016).
# These bins are the standard shortcut mapping from a DockQ score to the CAPRI
# Incorrect / Acceptable / Medium / High categories.
_CAPRI_ACCEPTABLE = 0.23
_CAPRI_MEDIUM = 0.49
_CAPRI_HIGH = 0.80


def capri_class(dockq: float) -> str:
    """Map a DockQ score to its CAPRI quality class.

    Args:
        dockq: DockQ score in [0, 1].

    Returns:
        One of "Incorrect", "Acceptable", "Medium", "High".
    """
    if dockq >= _CAPRI_HIGH:
        return "High"
    if dockq >= _CAPRI_MEDIUM:
        return "Medium"
    if dockq >= _CAPRI_ACCEPTABLE:
        return "Acceptable"
    return "Incorrect"


def _dockq_command() -> list[str]:
    """Locate the DockQ executable, raising an informative error if absent.

    Returns:
        The command prefix as a list (e.g. ``["DockQ"]``).

    Raises:
        ImportError: if DockQ is not installed / not on PATH.
    """
    exe = shutil.which("DockQ")
    if exe is not None:
        return [exe]
    raise ImportError(
        "DockQ is required for reference-based CAPRI metrics but was not found "
        "on PATH. Install with: pip install DockQ"
    )


def _parse_dockq_json(data: dict) -> dict:
    """Normalise DockQ's ``--json`` output into a flat result dict.

    Pure function (no I/O) so it can be unit-tested without DockQ installed.

    Args:
        data: Parsed JSON produced by ``DockQ ... --json``.

    Returns:
        Dictionary with keys:
            - dockq (float): global DockQ score, averaged over interfaces
            - capri_class (str): CAPRI class of the global score
            - n_interfaces (int)
            - best_mapping (str | None): the chosen chain mapping
            - interfaces (list[dict]): per-interface breakdown, each with
              chains, DockQ, fnat, fnonnat, iRMSD, LRMSD, clashes, capri_class
    """
    per_interface = data.get("best_result", {}) or {}

    interfaces = []
    for chain_pair, res in per_interface.items():
        dq = res.get("DockQ")
        if isinstance(chain_pair, (tuple, list)):
            chains = "".join(chain_pair)
        else:
            chains = str(chain_pair)
        interfaces.append(
            {
                "chains": chains,
                "DockQ": dq,
                "fnat": res.get("fnat"),
                "fnonnat": res.get("fnonnat"),
                "iRMSD": res.get("iRMSD"),
                "LRMSD": res.get("LRMSD"),
                "clashes": res.get("clashes"),
                "capri_class": capri_class(dq) if dq is not None else None,
            }
        )

    # "GlobalDockQ" is the mean across interfaces; fall back to "best_dockq".
    global_dockq = data.get("GlobalDockQ")
    if global_dockq is None:
        global_dockq = data.get("best_dockq")

    return {
        "dockq": global_dockq,
        "capri_class": capri_class(global_dockq) if global_dockq is not None else None,
        "n_interfaces": len(interfaces),
        "best_mapping": data.get("best_mapping_str"),
        "interfaces": interfaces,
    }


def compute_dockq_metrics(
    model_path: str | Path,
    reference_path: str | Path,
    mapping: Optional[str] = None,
) -> dict:
    """Compute DockQ / CAPRI accuracy of a predicted complex against a reference.

    Runs the DockQ CLI, which performs an automatic optimal chain-mapping search
    between the model and the reference, and parses its JSON output.

    Args:
        model_path: Predicted complex (PDB or mmCIF).
        reference_path: Native / reference complex (PDB or mmCIF).
        mapping: Optional explicit chain mapping to constrain the search, in
            DockQ CLI ``model:native`` convention (e.g. ``"AB:CD"``). Wildcards
            are allowed to fix part of the mapping. If None, DockQ searches for
            the optimal mapping itself.

    Returns:
        The normalised dict from :func:`_parse_dockq_json`.

    Raises:
        ImportError: if DockQ is not installed.
        FileNotFoundError: if either input path does not exist.
        RuntimeError: if the DockQ run fails or produces no parseable output.
    """
    model_path = Path(model_path)
    reference_path = Path(reference_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model structure not found: {model_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference structure not found: {reference_path}")

    cmd = _dockq_command()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = Path(tmp.name)

    try:
        # DockQ CLI argument order is: DockQ <model> <native> ...
        full_cmd = cmd + [str(model_path), str(reference_path), "--json", str(json_path)]
        if mapping:
            full_cmd += ["--mapping", mapping]

        proc = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"DockQ failed (exit {proc.returncode}).\n"
                f"Command: {' '.join(full_cmd)}\n"
                f"stderr:\n{proc.stderr.strip()}"
            )

        try:
            data = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read DockQ JSON output at {json_path}: {exc}\n"
                f"DockQ stdout:\n{proc.stdout.strip()}"
            ) from exc
    finally:
        json_path.unlink(missing_ok=True)

    return _parse_dockq_json(data)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reference-based CAPRI accuracy of a predicted complex via DockQ "
            "(DockQ, fnat, fnonnat, i-RMSD, L-RMSD). Requires: pip install DockQ"
        )
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        required=True,
        help="Predicted complex structure (PDB or mmCIF)",
    )
    parser.add_argument(
        "--reference",
        "--native",
        "-r",
        dest="reference",
        type=Path,
        required=True,
        help="Native / reference complex structure (PDB or mmCIF)",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="Optional chain mapping in DockQ model:native convention (e.g. AB:CD)",
    )
    from binding_metrics.cli import add_log_file_arg

    add_log_file_arg(parser)
    args = parser.parse_args()

    from binding_metrics.cli import log_to_file

    with log_to_file(args.log_file):
        print("Computing DockQ reference-based metrics:")
        print(f"  Model:     {args.model}")
        print(f"  Reference: {args.reference}")
        if args.mapping:
            print(f"  Mapping:   {args.mapping}")

        result = compute_dockq_metrics(args.model, args.reference, mapping=args.mapping)

        print("\nGlobal:")
        dq = result["dockq"]
        print(f"  DockQ: {dq:.3f} ({result['capri_class']})" if dq is not None else "  DockQ: N/A")
        if result["best_mapping"]:
            print(f"  Chain mapping: {result['best_mapping']}")
        print(f"  Interfaces: {result['n_interfaces']}")

        for iface in result["interfaces"]:
            print(f"\nInterface {iface['chains']}:")
            for key in ("DockQ", "fnat", "fnonnat", "iRMSD", "LRMSD", "clashes"):
                val = iface.get(key)
                if val is None:
                    print(f"  {key}: N/A")
                elif key in ("iRMSD", "LRMSD"):
                    print(f"  {key}: {val:.3f} Å")
                elif key == "clashes":
                    print(f"  {key}: {val}")
                else:
                    print(f"  {key}: {val:.3f}")
            if iface.get("capri_class"):
                print(f"  CAPRI class: {iface['capri_class']}")


if __name__ == "__main__":
    main()
