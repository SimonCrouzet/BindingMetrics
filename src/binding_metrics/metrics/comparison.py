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
from pathlib import Path
from typing import Optional

import numpy as np


def _get_coords(
    structure,
    chain_filter: Optional[str] = None,
    backbone_only: bool = False,
) -> tuple[np.ndarray, list]:
    """Extract coordinates and atom keys from a gemmi Structure.

    Args:
        structure: gemmi Structure object
        chain_filter: If given, only include atoms from this chain
        backbone_only: If True, only include backbone atoms (N, CA, C, O)

    Returns:
        Tuple of (coords array shape (N, 3), list of (chain, res_num, atom_name) keys)
    """
    backbone_atoms = {"N", "CA", "C", "O"}
    coords = []
    keys = []

    for model in structure:
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
    """Compute Kabsch-aligned RMSD between two coordinate sets.

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

    p_rot = p @ R
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

    idx1 = sorted([(k, i) for i, k in enumerate(keys1) if k in common])
    idx2 = sorted([(k, i) for i, k in enumerate(keys2) if k in common])
    c1 = coords1[[i for _, i in idx1]]
    c2 = coords2[[i for _, i in idx2]]
    return _kabsch_rmsd(c1, c2)


def compute_structure_rmsd(
    initial_path: str | Path,
    processed_path: str | Path,
    design_chain: Optional[str] = None,
) -> dict[str, Optional[float]]:
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

    Returns:
        Dictionary with keys:
            - rmsd (float, Å): All-atom RMSD of full complex
            - bb_rmsd (float, Å): Backbone-only RMSD of full complex
            - rmsd_design (float, Å): All-atom RMSD of designed chain only
            - bb_rmsd_design (float, Å): Backbone-only RMSD of designed chain
            Values are None if computation failed for that variant.
    """
    try:
        import gemmi
    except ImportError:
        raise ImportError(
            "gemmi is required for structure comparison. "
            "Install with: pip install gemmi"
        )

    initial_st = gemmi.read_structure(str(initial_path))
    processed_st = gemmi.read_structure(str(processed_path))

    # Auto-detect design chain from initial structure
    if design_chain is None:
        chain_sizes = []
        for model in initial_st:
            for chain in model:
                n_res = sum(1 for r in chain if r.name not in {"HOH", "WAT"})
                if n_res > 0:
                    chain_sizes.append((chain.name, n_res))
        if chain_sizes:
            chain_sizes.sort(key=lambda x: x[1])
            design_chain = chain_sizes[0][0]

    result: dict[str, Optional[float]] = {
        "rmsd": None,
        "bb_rmsd": None,
        "rmsd_design": None,
        "bb_rmsd_design": None,
    }

    # Full complex — all atoms
    c1, k1 = _get_coords(initial_st, backbone_only=False)
    c2, k2 = _get_coords(processed_st, backbone_only=False)
    result["rmsd"] = _matched_rmsd(c1, k1, c2, k2)

    # Full complex — backbone
    b1, bk1 = _get_coords(initial_st, backbone_only=True)
    b2, bk2 = _get_coords(processed_st, backbone_only=True)
    result["bb_rmsd"] = _matched_rmsd(b1, bk1, b2, bk2)

    if design_chain:
        # Design chain — all atoms
        d1, dk1 = _get_coords(initial_st, chain_filter=design_chain, backbone_only=False)
        d2, dk2 = _get_coords(processed_st, chain_filter=design_chain, backbone_only=False)
        result["rmsd_design"] = _matched_rmsd(d1, dk1, d2, dk2)

        # Design chain — backbone
        db1, dbk1 = _get_coords(initial_st, chain_filter=design_chain, backbone_only=True)
        db2, dbk2 = _get_coords(processed_st, chain_filter=design_chain, backbone_only=True)
        result["bb_rmsd_design"] = _matched_rmsd(db1, dbk1, db2, dbk2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute RMSD between two structures (e.g. initial vs. relaxed)"
    )
    parser.add_argument("--initial", "-a", type=Path, required=True, help="Initial (reference) structure")
    parser.add_argument("--processed", "-b", type=Path, required=True, help="Processed (target) structure")
    parser.add_argument("--design-chain", type=str, default=None, help="Designed chain ID (auto-detect if omitted)")
    args = parser.parse_args()

    print(f"Comparing structures:")
    print(f"  Initial:   {args.initial}")
    print(f"  Processed: {args.processed}")
    if args.design_chain:
        print(f"  Design chain: {args.design_chain}")

    result = compute_structure_rmsd(args.initial, args.processed, args.design_chain)

    print("\nResults:")
    for key, val in result.items():
        if val is not None:
            print(f"  {key}: {val:.3f} Å")
        else:
            print(f"  {key}: N/A")


if __name__ == "__main__":
    main()
