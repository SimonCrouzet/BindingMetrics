"""Structure comparison utilities for evaluating structural changes.

Computes RMSD between two structures (e.g. initial vs. relaxed) using gemmi
for robust atom matching across structures that may differ in atom count
(e.g. after hydrogen addition or side-chain rebuilding).

Usage:
    binding-metrics-compare \\
        --initial input.cif \\
        --processed relaxed.cif \\
        --design-chain A
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from binding_metrics.metrics._common import ChainAliasAction, resolve_chain_role
from binding_metrics.utils import configure_logging


def _get_coords(
    structure,
    chain_filter: Optional[str] = None,
    backbone_only: bool = False,
) -> tuple[np.ndarray, list]:
    """Extract coordinates and atom keys from a gemmi Structure.

    Args:
        structure: gemmi Structure object; only its first model is read
        chain_filter: If given, only include atoms from this chain
        backbone_only: If True, only include backbone atoms (N, CA, C, O)

    Returns:
        Tuple of (coords array shape (N, 3), list of (chain, res_num, atom_name) keys)
    """
    backbone_atoms = {"N", "CA", "C", "O"}
    coords = []
    keys = []

    # First model only, like every other loader: concatenating the models of
    # an NMR/ensemble file would build one point set with duplicated keys.
    models = [structure[0]] if len(structure) > 0 else []
    for model in models:
        for chain in model:
            if chain_filter is not None and chain.name != chain_filter:
                continue
            for residue in chain:
                if residue.name in {"HOH", "WAT"}:
                    continue
                for atom in residue:
                    if backbone_only and atom.name not in backbone_atoms:
                        continue
                    pos = atom.pos
                    coords.append([pos.x, pos.y, pos.z])
                    keys.append((chain.name, residue.seqid.num, atom.name))

    arr = np.array(coords) if coords else np.zeros((0, 3))
    return arr, keys


def _kabsch_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute Kabsch-aligned RMSD between two coordinate sets (Kabsch, 1976).

    The optimal proper rotation is found by SVD of the covariance matrix; a
    reflection is never allowed (the determinant correction flips the last
    singular direction).

    Args:
        coords1: Reference coordinates, shape (N, 3)
        coords2: Target coordinates, shape (N, 3)

    Returns:
        RMSD in Ångström
    """
    p = coords1.copy() - coords1.mean(axis=0)
    q = coords2.copy() - coords2.mean(axis=0)

    H = p.T @ q
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # R is the rotation for column vectors (q ~ R @ p_i); the coordinates are
    # row vectors, so the same rotation is applied as p @ R.T.
    p_rot = p @ R.T
    return float(np.sqrt(np.mean(np.sum((p_rot - q) ** 2, axis=1))))


def _matched_rmsd(
    coords1: np.ndarray,
    keys1: list,
    coords2: np.ndarray,
    keys2: list,
) -> Optional[float]:
    """Compute RMSD after matching atoms by (chain, res_num, atom_name).

    If atom counts differ, only the common atoms are used.

    Args:
        coords1, keys1: Coordinates and keys for structure 1
        coords2, keys2: Coordinates and keys for structure 2

    Returns:
        RMSD in Ångström, or None if no common atoms
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return None

    if len(coords1) == len(coords2):
        return _kabsch_rmsd(coords1, coords2)

    common = set(keys1) & set(keys2)
    if not common:
        return None

    # Group atom indices by key so that keys appearing with different
    # multiplicities in the two structures (e.g. altlocs, duplicated keys)
    # are paired one-to-one up to the smaller count. This guarantees the two
    # selected index lists have equal length, which Kabsch requires.
    pos1: dict = {}
    for i, k in enumerate(keys1):
        if k in common:
            pos1.setdefault(k, []).append(i)
    pos2: dict = {}
    for i, k in enumerate(keys2):
        if k in common:
            pos2.setdefault(k, []).append(i)

    sel1: list[int] = []
    sel2: list[int] = []
    for k in sorted(common):
        n = min(len(pos1[k]), len(pos2[k]))
        sel1.extend(pos1[k][:n])
        sel2.extend(pos2[k][:n])

    if not sel1:
        return None

    c1 = coords1[sel1]
    c2 = coords2[sel2]
    return _kabsch_rmsd(c1, c2)


def _why_no_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> str:
    """Say why `_matched_rmsd` returned None for these inputs."""
    if len(coords1) == 0 and len(coords2) == 0:
        return "no atoms selected in either structure"
    if len(coords1) == 0:
        return "no atoms selected in the initial structure"
    if len(coords2) == 0:
        return "no atoms selected in the processed structure"
    return "no (chain, residue number, atom name) key is shared by the two structures"


def compute_structure_rmsd(
    initial_path: str | Path,
    processed_path: str | Path,
    design_chain: Optional[str] = None,
    *,
    binder_chain: Optional[str] = None,
) -> dict[str, Optional[float] | str]:
    """Compute RMSD between two structures (e.g. initial vs. relaxed).

    Atoms are matched by (chain, residue number, atom name) to handle
    structures that differ in hydrogen atoms or side-chain atoms. Computes
    RMSD variants for the full complex and for the designed chain only,
    both all-atom and backbone-only.

    Requires gemmi: pip install gemmi

    Args:
        initial_path: Path to the first (reference) structure
        processed_path: Path to the second (target) structure
        design_chain: Chain ID of the designed/peptide region. Auto-detected
            as the smallest protein chain if None.
        binder_chain: Alias of ``design_chain`` (same meaning); giving both
            with different IDs raises ``ValueError``.

    Returns:
        Dictionary with keys:
            - rmsd (float, Å): All-atom RMSD of full complex
            - bb_rmsd (float, Å): Backbone-only RMSD of full complex
            - rmsd_design (float, Å): All-atom RMSD of designed chain only
            - bb_rmsd_design (float, Å): Backbone-only RMSD of designed chain
            Values are None if computation failed for that variant.
            - reason (str): Only present when at least one value is None;
              names each variant that could not be computed and why.
    """
    design_chain = resolve_chain_role("design_chain", design_chain, "binder_chain", binder_chain)
    try:
        import gemmi
    except ImportError:
        raise ImportError(
            "gemmi is required for structure comparison. Install with: pip install gemmi"
        )

    initial_st = gemmi.read_structure(str(initial_path))
    processed_st = gemmi.read_structure(str(processed_path))

    # The smallest chain with any non-water residue is taken as the design
    # chain, so a one-residue ligand chain would win: pass design_chain then.
    if design_chain is None:
        chain_sizes = []
        if len(initial_st) > 0:
            for chain in initial_st[0]:
                n_res = sum(1 for r in chain if r.name not in {"HOH", "WAT"})
                if n_res > 0:
                    chain_sizes.append((chain.name, n_res))
        if chain_sizes:
            chain_sizes.sort(key=lambda x: x[1])
            design_chain = chain_sizes[0][0]

    result: dict[str, Optional[float] | str] = {
        "rmsd": None,
        "bb_rmsd": None,
        "rmsd_design": None,
        "bb_rmsd_design": None,
    }

    failures: list[str] = []

    def _variant(name: str, chain_filter: Optional[str], backbone_only: bool) -> None:
        c1, k1 = _get_coords(initial_st, chain_filter, backbone_only)
        c2, k2 = _get_coords(processed_st, chain_filter, backbone_only)
        result[name] = _matched_rmsd(c1, k1, c2, k2)
        if result[name] is None:
            failures.append(f"{name} ({_why_no_rmsd(c1, c2)})")

    _variant("rmsd", None, backbone_only=False)
    _variant("bb_rmsd", None, backbone_only=True)
    if design_chain:
        _variant("rmsd_design", design_chain, backbone_only=False)
        _variant("bb_rmsd_design", design_chain, backbone_only=True)
    else:
        failures.append("rmsd_design, bb_rmsd_design (no design chain given or detected)")

    if failures:
        result["reason"] = "not computed: " + "; ".join(failures)
    return result


def main():
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Compute RMSD between two structures (e.g. initial vs. relaxed)"
    )
    parser.add_argument(
        "--initial", "-a", type=Path, required=True, help="Initial (reference) structure"
    )
    parser.add_argument(
        "--processed", "-b", type=Path, required=True, help="Processed (target) structure"
    )
    parser.add_argument(
        "--design-chain",
        "--binder-chain",
        action=ChainAliasAction,
        type=str,
        default=None,
        help="Designed chain ID (auto-detect if omitted)",
    )
    from binding_metrics.cli import add_log_file_arg

    add_log_file_arg(parser)
    args = parser.parse_args()

    from binding_metrics.cli import log_to_file

    with log_to_file(args.log_file):
        print("Comparing structures:")
        print(f"  Initial:   {args.initial}")
        print(f"  Processed: {args.processed}")
        if args.design_chain:
            print(f"  Design chain: {args.design_chain}")

        result = compute_structure_rmsd(args.initial, args.processed, args.design_chain)

        print("\nResults:")
        for key, val in result.items():
            if key == "reason":
                continue
            if val is not None:
                print(f"  {key}: {val:.3f} Å")
            else:
                print(f"  {key}: N/A")
        if "reason" in result:
            print(f"  reason: {result['reason']}", file=sys.stderr)


if __name__ == "__main__":
    main()
