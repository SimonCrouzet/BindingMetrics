"""Receptor structural quality assessment.

Evaluates MolProbity-style geometry metrics (Ramachandran, rotamer quality,
Cβ deviations, backbone bonds/angles, clashscore, MolProbity composite score),
B-factor statistics, and absolute AMBER ff14SB potential energy for the receptor
chain. Handles receptor-only structures, complexes (non-receptor chains ignored),
and multi-model PDB/CIF files (all models scored independently).

The receptor chain is identified as the largest protein chain by residue count.
This module is self-contained and does not depend on the main pipeline.

The validation terms follow MolProbity (Chen et al. 2010, Acta Cryst. D66:12-21;
Williams et al. 2018, Protein Sci. 27:293-315) but are lighter approximations:
the clashscore counts heavy-atom overlaps only (no hydrogens), the rotamer term
is a chi1-only check, and the composite score is therefore indicative rather than
comparable to published MolProbity values.

Usage:
    binding-metrics-receptor-quality --input receptor.pdb
    binding-metrics-receptor-quality --input complex.cif --receptor-chain A
    binding-metrics-receptor-quality --input ensemble.pdb --output results.csv
    binding-metrics-receptor-quality --input ensemble.pdb --output results.json
"""

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from binding_metrics._constants import DEFAULT_RANDOM_SEED
from binding_metrics.metrics._common import resolve_chain_role
from binding_metrics.utils import backfill_auth_columns, configure_logging

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _import_biotite():
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb_io
        import biotite.structure.io.pdbx as pdbx

        return struc, pdbx, pdb_io
    except ImportError:
        raise ImportError(
            "biotite is required for receptor quality metrics. "
            "Install with: pip install binding-metrics[biotite]"
        )


def _import_scipy():
    try:
        from scipy.spatial import cKDTree

        return cKDTree
    except ImportError:
        raise ImportError(
            "scipy is required for clashscore computation. "
            "Install with: pip install binding-metrics[biotite]"
        )


def _import_openmm():
    try:
        import openmm
        import openmm.unit as unit
        from openmm.app import ForceField, Modeller, PDBFile, Simulation

        return openmm, unit, ForceField, Modeller, PDBFile, Simulation
    except ImportError:
        raise ImportError(
            "openmm is required for energy computation. "
            "Install with: pip install binding-metrics[simulation]"
        )


# ---------------------------------------------------------------------------
# Structure loading — all models
# ---------------------------------------------------------------------------


def _load_all_models(path: Path) -> list:
    """Load all models from a PDB or CIF file as a list of AtomArrays."""
    struc, pdbx, pdb_io = _import_biotite()
    suffix = path.suffix.lower()

    if suffix in (".cif", ".mmcif"):
        f = pdbx.CIFFile.read(str(path))
        backfill_auth_columns(f)
        structure = pdbx.get_structure(f)
    else:
        f = pdb_io.PDBFile.read(str(path))
        structure = pdb_io.get_structure(f)

    if isinstance(structure, struc.AtomArrayStack):
        return [structure[i] for i in range(structure.shape[0])]
    return [structure]


# ---------------------------------------------------------------------------
# Receptor chain detection
# ---------------------------------------------------------------------------


def _detect_receptor_chain(atoms) -> Optional[str]:
    """Return chain ID of the largest protein chain (by residue count)."""
    struc, _, _ = _import_biotite()
    aa_mask = struc.filter_amino_acids(atoms)
    aa_atoms = atoms[aa_mask]
    chain_ids = sorted(set(aa_atoms.chain_id))

    if not chain_ids:
        return None

    best_chain, best_count = None, 0
    for cid in chain_ids:
        cid_mask = aa_atoms.chain_id == cid
        n_res = len(
            set(
                zip(
                    aa_atoms.chain_id[cid_mask],
                    aa_atoms.res_id[cid_mask],
                )
            )
        )
        if n_res > best_count:
            best_count, best_chain = n_res, cid
    return best_chain


# ---------------------------------------------------------------------------
# Residue iteration helper
# ---------------------------------------------------------------------------


def _iter_residues(atoms):
    """Yield (res_name, atom_coord_dict) for each residue in an AtomArray.

    Uses biotite's get_residue_starts for correct residue boundaries.
    """
    struc, _, _ = _import_biotite()
    starts = struc.get_residue_starts(atoms)
    for i, start in enumerate(starts):
        end = int(starts[i + 1]) if i + 1 < len(starts) else len(atoms)
        res = atoms[start:end]
        res_name = str(res.res_name[0]).strip()
        atom_dict = {str(a.atom_name).strip(): a.coord.copy() for a in res}
        yield res_name, atom_dict


# ---------------------------------------------------------------------------
# VDW radii
# ---------------------------------------------------------------------------

_VDW_RADII: dict[str, float] = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "SE": 1.90,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}


def _vdw(element: str) -> float:
    return _VDW_RADII.get(element.strip().upper(), 1.80)


# ---------------------------------------------------------------------------
# Ramachandran
# ---------------------------------------------------------------------------


def _ramachandran(chain_atoms) -> dict:
    """Ramachandran backbone dihedral quality for a single AtomArray.

    Each residue with both phi and psi is placed in a favoured, allowed or
    outlier region of the Ramachandran plot (``geometry._classify_ramachandran``;
    D-residues are scored on the mirrored plot). The regions are simplified
    boxes, not the density contours of MolProbity's Top8000 data (Lovell et al.
    2003, Proteins 50:437; Williams et al. 2018).

    Returns:
        ``favoured_pct``, ``allowed_pct``, ``outlier_pct`` (float, NaN when
        nothing was evaluated), ``favoured_count``, ``allowed_count``,
        ``outlier_count`` and ``n_evaluated`` (int), and ``reason`` (str) only
        when nothing could be evaluated.
    """
    from binding_metrics.core.nonstandard import is_d_residue
    from binding_metrics.metrics.geometry import _classify_ramachandran

    struc, _, _ = _import_biotite()
    _empty = {
        "favoured_pct": np.nan,
        "allowed_pct": np.nan,
        "outlier_pct": np.nan,
        "outlier_count": 0,
        "favoured_count": 0,
        "allowed_count": 0,
        "n_evaluated": 0,
    }

    try:
        phi_rad, psi_rad, _ = struc.dihedral_backbone(chain_atoms)
    except (IndexError, struc.BadStructureError) as exc:
        # biotite cannot build the backbone when N, CA or C atoms are missing.
        return {**_empty, "reason": f"backbone dihedrals unavailable: {type(exc).__name__}"}

    phi_deg = np.degrees(phi_rad)
    psi_deg = np.degrees(psi_rad)
    ca_atoms = chain_atoms[chain_atoms.atom_name == "CA"]
    counts = {"favoured": 0, "allowed": 0, "outlier": 0}

    for i, ca in enumerate(ca_atoms):
        if i >= len(phi_deg):
            break
        phi, psi = float(phi_deg[i]), float(psi_deg[i])
        res_name = str(ca.res_name).strip()
        region = _classify_ramachandran(phi, psi, is_d=is_d_residue(res_name))
        if region is None:
            continue
        counts[region] += 1

    n_eval = sum(counts.values())
    if n_eval == 0:
        return {**_empty, "reason": "no residue with both phi and psi"}

    return {
        "favoured_pct": 100.0 * counts["favoured"] / n_eval,
        "allowed_pct": 100.0 * counts["allowed"] / n_eval,
        "outlier_pct": 100.0 * counts["outlier"] / n_eval,
        "favoured_count": counts["favoured"],
        "allowed_count": counts["allowed"],
        "outlier_count": counts["outlier"],
        "n_evaluated": n_eval,
    }


# ---------------------------------------------------------------------------
# Clashscore
# ---------------------------------------------------------------------------


# Covalent radii (Å; Cordero et al. 2008, Dalton Trans. 2832). They only decide
# whether two atoms of one residue are bonded, so a generous margin is harmless:
# every same-residue pair is already exempt from the clash count.
_COVALENT_RADII: dict[str, float] = {
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
    "SE": 1.20,
    "F": 0.57,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}
_COVALENT_DEFAULT_RADIUS = 1.0
_COVALENT_TOLERANCE_ANGSTROM = 0.4

# Peptide bond between residues that follow each other in the chain. Read off
# the coordinates rather than from residue numbers, so that numbering gaps
# (chymotrypsin numbering skips 149 and 218) do not turn a peptide bond into a
# clash. Same value as the amide threshold in ``core.cyclic``.
_PEPTIDE_BOND_MAX_ANGSTROM = 2.0

# Longest distance at which two atoms of residues that are not sequence
# neighbours count as one covalent link: disulfide, head-to-tail closure,
# lactam and isopeptide bridges, hydrocarbon staples, thioethers, lactones.
# Only these element pairs can form a link, so a close C...C or N...O contact is
# never explained away. S-S and N-C match the thresholds of ``core.cyclic``.
# The N-C limit is generous because NMR models of lactams stretch the bond (1XY4
# has 1.89 Å).
_CROSSLINK_MAX_ANGSTROM: dict[frozenset, float] = {
    frozenset(("S", "S")): 2.6,
    frozenset(("C", "N")): 2.0,
    frozenset(("C", "C")): 1.9,
    frozenset(("C", "S")): 2.2,
    frozenset(("C", "O")): 1.9,
}

# Atoms separated by at most this many covalent bonds are not scored against
# each other (the usual 1-2, 1-3 and 1-4 exclusion; a heavy-atom-only score
# has no hydrogens to tell a gauche 1-4 contact from a clash).
_MAX_EXCLUDED_BOND_SEPARATION = 3

# N/O pairs closer than the sum of their vdW radii (3.0-3.1 Å) are hydrogen
# bonds, not clashes, down to this distance; heavy atoms alone cannot see the
# hydrogen, so 2.5 Å (the short end of strong N...O and O...O bonds) is used as
# the floor.
_HBOND_HETEROATOMS = ("N", "O")
_HBOND_MIN_DISTANCE_ANGSTROM = 2.5


def _covalent_reach(
    heavy,
    heavy_elements: np.ndarray,
    coords: np.ndarray,
    pairs: np.ndarray,
    distances: np.ndarray,
):
    """Sparse matrix (scipy CSR), nonzero for heavy-atom pairs within three covalent bonds.

    The bond graph joins (1) atoms of one residue closer than the sum of their
    covalent radii plus a margin, (2) the C and N of consecutive residues that
    are within peptide-bond distance, and (3) atoms of non-neighbouring
    residues that form a covalent link (see ``_CROSSLINK_MAX_ANGSTROM``).
    ``pairs`` are the candidate atom index pairs from the spatial search and
    ``distances`` their lengths.
    """
    from scipy import sparse

    struc, _, _ = _import_biotite()
    n_atoms = len(heavy)
    starts = struc.get_residue_starts(heavy)
    residue_of = np.zeros(n_atoms, dtype=np.int64)
    residue_of[starts[1:]] = 1
    residue_of = np.cumsum(residue_of)

    i, j = pairs[:, 0], pairs[:, 1]
    res_i, res_j = residue_of[i], residue_of[j]
    same_residue = res_i == res_j
    same_chain = heavy.chain_id[i] == heavy.chain_id[j]
    names = np.asarray(heavy.atom_name)

    radius = np.array(
        [_COVALENT_RADII.get(e, _COVALENT_DEFAULT_RADIUS) for e in heavy_elements], dtype=float
    )
    intra = same_residue & (distances <= radius[i] + radius[j] + _COVALENT_TOLERANCE_ANGSTROM)

    peptide = (
        same_chain
        & (distances <= _PEPTIDE_BOND_MAX_ANGSTROM)
        & (
            ((res_j == res_i + 1) & (names[i] == "C") & (names[j] == "N"))
            | ((res_i == res_j + 1) & (names[j] == "C") & (names[i] == "N"))
        )
    )

    # Longest link distance per element pair, looked up through element codes so the
    # per-pair work stays in numpy (a receptor has about ten candidate pairs per atom).
    symbols = sorted(set(heavy_elements))
    code_of = {symbol: k for k, symbol in enumerate(symbols)}
    codes = np.array([code_of[e] for e in heavy_elements])
    limit_table = np.zeros((len(symbols), len(symbols)))
    for pair_symbols, limit in _CROSSLINK_MAX_ANGSTROM.items():
        members = sorted(pair_symbols)  # one member for a homonuclear pair such as S-S
        if members[0] in code_of and members[-1] in code_of:
            first, second = code_of[members[0]], code_of[members[-1]]
            limit_table[first, second] = limit_table[second, first] = limit
    crosslink = ~same_residue & (distances <= limit_table[codes[i], codes[j]])

    bonded = pairs[intra | peptide | crosslink]
    adjacency = sparse.coo_matrix(
        (np.ones(len(bonded), dtype=np.int32), (bonded[:, 0], bonded[:, 1])),
        shape=(n_atoms, n_atoms),
    )
    adjacency = (adjacency + adjacency.T).tocsr()
    adjacency.data[:] = 1

    reach = adjacency.copy()
    frontier = adjacency
    for _ in range(_MAX_EXCLUDED_BOND_SEPARATION - 1):
        frontier = frontier @ adjacency
        reach = reach + frontier
    return reach.tocsr()


def _clashscore(
    chain_atoms,
    clash_cutoff: float = 0.4,
    exclude_bonded: bool = True,
    exempt_hbond_pairs: bool = True,
) -> dict:
    """Heavy-atom overlap clashscore: clashing atom pairs per 1000 heavy atoms.

    Two heavy atoms clash when the sum of their van der Waals radii exceeds
    their distance by at least ``clash_cutoff`` Å (the 0.4 Å of Word et al. 1999;
    Chen et al. 2010). This is not MolProbity's clashscore: MolProbity adds
    hydrogens with Reduce and evaluates all-atom contacts with Probe
    (Williams et al. 2018), which also resolves hydrogen bonds from the
    hydrogen position. Here hydrogens are ignored, so the value is an
    approximation that is usually higher than MolProbity's for the same model.

    A pair is not scored when

    - both atoms lie in the same residue or in residues whose numbers differ by
      at most one (same chain), as before;
    - ``exclude_bonded`` is true and the atoms are separated by at most three
      covalent bonds, following the peptide bond by coordinates, and disulfide,
      head-to-tail, lactam, staple and similar links by distance (see
      ``_CROSSLINK_MAX_ANGSTROM``);
    - ``exempt_hbond_pairs`` is true and the pair is N/O to N/O at
      2.5 Å or more (a hydrogen bond, not an overlap).

    With both switches off the result equals the earlier definition, which
    counted disulfides, cyclic closures and N...O hydrogen bonds as clashes.

    Returns:
        ``clashscore`` (float, NaN below two heavy atoms), ``n_clashes`` (int)
        and ``n_heavy_atoms`` (int); ``reason`` (str) only when the score could
        not be computed.

    References:
        Word et al. 1999, J. Mol. Biol. 285:1735 (all-atom contact analysis);
        Chen et al. 2010, Acta Cryst. D66:12; Williams et al. 2018, Protein Sci.
        27:293 (MolProbity clashscore).
    """
    cKDTree = _import_scipy()

    elements = np.array([str(a.element).strip().upper() for a in chain_atoms])
    heavy_mask = ~np.isin(elements, ["H", "D", ""])
    heavy = chain_atoms[heavy_mask]
    heavy_elements = elements[heavy_mask]
    n_heavy = len(heavy)

    if n_heavy < 2:
        return {
            "clashscore": np.nan,
            "n_clashes": 0,
            "n_heavy_atoms": n_heavy,
            "reason": "fewer than two heavy atoms",
        }

    coords = heavy.coord
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=2 * max(_VDW_RADII.values()), output_type="ndarray")

    n_clashes = 0
    if len(pairs) > 0:
        i, j = pairs[:, 0], pairs[:, 1]
        chain_ids = heavy.chain_id
        res_ids = heavy.res_id.astype(np.int64)
        coords64 = coords.astype(np.float64)
        distances = np.linalg.norm(coords64[i] - coords64[j], axis=1)
        radii = np.array([_vdw(e) for e in heavy_elements])

        neighbours = (chain_ids[i] == chain_ids[j]) & (np.abs(res_ids[i] - res_ids[j]) <= 1)
        is_clash = ~neighbours & (radii[i] + radii[j] - distances >= clash_cutoff)

        if exempt_hbond_pairs:
            polar = np.isin(heavy_elements, _HBOND_HETEROATOMS)
            is_clash &= ~(polar[i] & polar[j] & (distances >= _HBOND_MIN_DISTANCE_ANGSTROM))

        if exclude_bonded and is_clash.any():
            reach = _covalent_reach(heavy, heavy_elements, coords, pairs, distances)
            candidates = pairs[is_clash]
            bonded = np.asarray(reach[candidates[:, 0], candidates[:, 1]]).ravel() > 0
            is_clash[np.flatnonzero(is_clash)[bonded]] = False

        n_clashes = int(is_clash.sum())

    return {
        "clashscore": float(1000.0 * n_clashes / n_heavy),
        "n_clashes": n_clashes,
        "n_heavy_atoms": n_heavy,
    }


# ---------------------------------------------------------------------------
# B-factor statistics
# ---------------------------------------------------------------------------

_HIGH_B_THRESHOLD = 60.0  # Å²


def _bfactor_stats(chain_atoms) -> dict:
    """Compute B-factor statistics from Cα atoms."""
    b_factors = getattr(chain_atoms, "b_factor", None)
    if b_factors is None:
        return {"available": False}

    ca_mask = chain_atoms.atom_name == "CA"
    ca_b = b_factors[ca_mask]

    if len(ca_b) == 0 or np.all(ca_b == 0.0):
        return {"available": False}

    return {
        "available": True,
        "mean_b_factor": float(np.mean(ca_b)),
        "max_b_factor": float(np.max(ca_b)),
        "min_b_factor": float(np.min(ca_b)),
        "std_b_factor": float(np.std(ca_b)),
        "n_high_b_residues": int(np.sum(ca_b > _HIGH_B_THRESHOLD)),
    }


# ---------------------------------------------------------------------------
# Rotamer quality (simplified χ1 check)
# ---------------------------------------------------------------------------

# Last atom of the χ1 dihedral (N-CA-CB-X) by residue name
_CHI1_TERMINAL: dict[str, str] = {
    "ARG": "CG",
    "ASN": "CG",
    "ASP": "CG",
    "GLN": "CG",
    "GLU": "CG",
    "HIS": "CG",
    "HID": "CG",
    "HIE": "CG",
    "HIP": "CG",
    "ILE": "CG1",
    "LEU": "CG",
    "LYS": "CG",
    "MET": "CG",
    "PHE": "CG",
    "PRO": "CG",
    "TRP": "CG",
    "TYR": "CG",
    "CYS": "SG",
    "CYX": "SG",
    "SER": "OG",
    "THR": "OG1",
    "VAL": "CG1",
}
_CHI1_OUTLIER_THRESHOLD = 40.0  # degrees from nearest canonical position
_CHI1_CANONICAL = [-60.0, 60.0, 180.0]  # g−, g+, trans


def _chi1_dist_to_canonical(chi1_deg: float) -> float:
    """Minimum angular distance (°) from chi1 to the nearest canonical position."""
    dists = []
    for p in _CHI1_CANONICAL:
        d = abs(chi1_deg - p)
        dists.append(min(d, 360.0 - d))
    return min(dists)


def _rotamer_quality(chain_atoms) -> dict:
    """Compute rotamer outliers via simplified χ1 analysis.

    A residue is flagged as a rotamer outlier when its χ1 dihedral deviates
    more than 40° from the nearest canonical rotamer position (g−/g+/trans).
    Residues without χ1 (GLY, ALA) are skipped.

    Note: this is a χ1-only approximation. Full rotamer validation requires
    the backbone-dependent Dunbrack rotamer library (Shapovalov & Dunbrack 2011,
    Structure 19:844; used by MolProbity via Lovell et al. 2000, Proteins 40:389).

    Returns:
        ``outlier_count`` and ``n_evaluated`` (int), ``outlier_pct`` (float,
        NaN when no residue has a χ1), and ``reason`` (str) only in that case.
    """
    struc, _, _ = _import_biotite()

    n_evaluated = 0
    n_outliers = 0

    for res_name, atoms in _iter_residues(chain_atoms):
        terminal = _CHI1_TERMINAL.get(res_name)
        if terminal is None:
            continue
        if not all(k in atoms for k in ("N", "CA", "CB", terminal)):
            continue

        # Dihedral N-CA-CB-X
        chi1_rad = float(
            struc.dihedral(
                atoms["N"],
                atoms["CA"],
                atoms["CB"],
                atoms[terminal],
            )
        )

        chi1_deg = float(np.degrees(chi1_rad))
        n_evaluated += 1
        if _chi1_dist_to_canonical(chi1_deg) > _CHI1_OUTLIER_THRESHOLD:
            n_outliers += 1

    if n_evaluated == 0:
        return {
            "outlier_count": 0,
            "outlier_pct": np.nan,
            "n_evaluated": 0,
            "reason": "no residue with a complete chi1 dihedral",
        }

    return {
        "outlier_count": n_outliers,
        "outlier_pct": 100.0 * n_outliers / n_evaluated,
        "n_evaluated": n_evaluated,
    }


# ---------------------------------------------------------------------------
# Cβ deviations
# ---------------------------------------------------------------------------


def _ideal_cbeta(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> Optional[np.ndarray]:
    """Compute ideal Cβ position from backbone N, CA, C using tetrahedral geometry.

    Returns the ideal CB position (numpy array, Å), or None if degenerate input.
    Coefficients -0.5774 (≈ 1/√3) and 0.8165 (≈ √(2/3)) correspond to the
    standard tetrahedral bond angle of 109.47°.
    """
    b1 = n - ca
    b2 = c - ca
    n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    b1, b2 = b1 / n1, b2 / n2

    nc = np.cross(b1, b2)
    nc_n = np.linalg.norm(nc)
    if nc_n < 1e-8:
        return None
    nc = nc / nc_n

    bc = b1 + b2
    bc_n = np.linalg.norm(bc)
    if bc_n < 1e-8:
        return None
    bc = bc / bc_n

    cb_dir = -0.5774 * bc + 0.8165 * nc
    cb_dir_n = np.linalg.norm(cb_dir)
    if cb_dir_n < 1e-8:
        return None
    return ca + 1.521 * cb_dir / cb_dir_n


def _cbeta_deviations(chain_atoms, threshold: float = 0.25) -> dict:
    """Count Cβ deviations > threshold Å from their ideal backbone-derived position.

    Follows MolProbity convention: deviations > 0.25 Å indicate backbone distortion
    (Lovell et al. 2003, Proteins 50:437). GLY is skipped (no Cβ). MolProbity
    derives the ideal Cβ from the side-chain-dependent Cα geometry; here it comes
    from the backbone N, CA, C with tetrahedral geometry, so values differ slightly.

    Returns:
        ``cb_deviation_count`` and ``cb_n_evaluated`` (int), ``cb_deviation_pct``
        (float, NaN when no residue was evaluated), and ``reason`` (str) only in
        that case.
    """
    n_evaluated = 0
    n_deviating = 0

    for res_name, atoms in _iter_residues(chain_atoms):
        if res_name == "GLY":
            continue
        if not all(k in atoms for k in ("N", "CA", "C", "CB")):
            continue

        ideal = _ideal_cbeta(atoms["N"], atoms["CA"], atoms["C"])
        if ideal is None:
            continue

        dev = float(np.linalg.norm(atoms["CB"] - ideal))
        n_evaluated += 1
        if dev > threshold:
            n_deviating += 1

    result = {
        "cb_deviation_count": n_deviating,
        "cb_n_evaluated": n_evaluated,
        "cb_deviation_pct": 100.0 * n_deviating / n_evaluated if n_evaluated > 0 else np.nan,
    }
    if n_evaluated == 0:
        result["reason"] = "no residue with N, CA, C and CB"
    return result


# ---------------------------------------------------------------------------
# Backbone geometry (bonds and angles)
# ---------------------------------------------------------------------------

# Engh & Huber (1991) ideal values: (mean, 4σ threshold)
# Bond lengths in Å, angles in degrees.
_INTRA_BONDS: dict[tuple, tuple] = {
    ("N", "CA"): (1.458, 4 * 0.019),
    ("CA", "C"): (1.525, 4 * 0.021),
    ("C", "O"): (1.229, 4 * 0.019),
}
_INTER_BOND: tuple = (1.336, 4 * 0.023)  # peptide C→N

_INTRA_ANGLES: dict[tuple, tuple] = {
    ("N", "CA", "C"): (111.2, 4 * 2.77),
    ("CA", "C", "O"): (120.8, 4 * 1.67),
}
_INTER_ANGLES: dict[tuple, tuple] = {
    ("CA", "C", "N"): (116.2, 4 * 2.01),  # angle spanning two residues
    ("C", "N", "CA"): (121.7, 4 * 1.80),
}

_PEPTIDE_BOND_MAX = 2.5  # Å — skip inter-residue pairs farther than this (chain break)


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b, in degrees."""
    v1 = a - b
    v3 = c - b
    n1, n3 = np.linalg.norm(v1), np.linalg.norm(v3)
    if n1 < 1e-8 or n3 < 1e-8:
        return np.nan
    cos_a = np.dot(v1, v3) / (n1 * n3)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _backbone_geometry(chain_atoms) -> dict:
    """Check backbone bond lengths and angles against Engh & Huber ideal values.

    Flags bonds and angles deviating > 4σ from ideal as 'bad' (Engh & Huber
    1991, Acta Cryst. A47:392; the criterion MolProbity applies).
    Inter-residue C–N bonds are skipped when the distance exceeds 2.5 Å
    (chain break or model gap).

    Returns:
        ``bad_bonds``, ``total_bonds``, ``bad_angles``, ``total_angles`` (int),
        ``bad_bonds_pct`` and ``bad_angles_pct`` (float, NaN when the total is
        zero), and ``reason`` (str) only when no bond or no angle could be
        evaluated.
    """
    residues = []
    for res_name, atoms in _iter_residues(chain_atoms):
        residues.append(atoms)

    bad_bonds = 0
    total_bonds = 0
    bad_angles = 0
    total_angles = 0

    for i, r in enumerate(residues):
        r_next = residues[i + 1] if i + 1 < len(residues) else None

        # Intra-residue backbone bonds
        for (a1, a2), (ideal, thresh) in _INTRA_BONDS.items():
            if a1 in r and a2 in r:
                d = float(np.linalg.norm(r[a2] - r[a1]))
                total_bonds += 1
                if abs(d - ideal) > thresh:
                    bad_bonds += 1

        # Inter-residue peptide C–N bond
        if r_next and "C" in r and "N" in r_next:
            d = float(np.linalg.norm(r_next["N"] - r["C"]))
            if d < _PEPTIDE_BOND_MAX:
                ideal, thresh = _INTER_BOND
                total_bonds += 1
                if abs(d - ideal) > thresh:
                    bad_bonds += 1

        # Intra-residue backbone angles
        for (a1, a2, a3), (ideal, thresh) in _INTRA_ANGLES.items():
            if all(k in r for k in (a1, a2, a3)):
                ang = _angle_deg(r[a1], r[a2], r[a3])
                if np.isfinite(ang):
                    total_angles += 1
                    if abs(ang - ideal) > thresh:
                        bad_angles += 1

        # Inter-residue backbone angles (spanning peptide bond)
        if r_next:
            # CA–C–N: current CA and C, next N
            if all(k in r for k in ("CA", "C")) and "N" in r_next:
                d_cn = float(np.linalg.norm(r_next["N"] - r["C"]))
                if d_cn < _PEPTIDE_BOND_MAX:
                    ideal, thresh = _INTER_ANGLES[("CA", "C", "N")]
                    ang = _angle_deg(r["CA"], r["C"], r_next["N"])
                    if np.isfinite(ang):
                        total_angles += 1
                        if abs(ang - ideal) > thresh:
                            bad_angles += 1

            # C–N–CA: current C, next N and CA
            if "C" in r and all(k in r_next for k in ("N", "CA")):
                d_cn = float(np.linalg.norm(r_next["N"] - r["C"]))
                if d_cn < _PEPTIDE_BOND_MAX:
                    ideal, thresh = _INTER_ANGLES[("C", "N", "CA")]
                    ang = _angle_deg(r["C"], r_next["N"], r_next["CA"])
                    if np.isfinite(ang):
                        total_angles += 1
                        if abs(ang - ideal) > thresh:
                            bad_angles += 1

    result = {
        "bad_bonds": bad_bonds,
        "total_bonds": total_bonds,
        "bad_bonds_pct": 100.0 * bad_bonds / total_bonds if total_bonds > 0 else np.nan,
        "bad_angles": bad_angles,
        "total_angles": total_angles,
        "bad_angles_pct": 100.0 * bad_angles / total_angles if total_angles > 0 else np.nan,
    }
    if total_bonds == 0 or total_angles == 0:
        result["reason"] = "no complete backbone bond or angle found"
    return result


# ---------------------------------------------------------------------------
# MolProbity composite score
# ---------------------------------------------------------------------------


def _molprobity_score(
    clashscore: float,
    rama_outlier_pct: float,
    rota_outlier_pct: float,
) -> float:
    """Approximate MolProbity score (Chen et al. 2010, Acta Cryst D66:12-21).

    Combines clashscore, Ramachandran outliers %, and rotamer outliers % into
    a single score on a scale similar to X-ray resolution (lower = better).
    Penalty terms activate above baseline noise levels (0.2% rama, 2% rotamer).

    Note: rotamer term uses simplified χ1 classification, not the full Dunbrack
    library, and the clashscore counts heavy-atom overlaps only (Williams et al.
    2018 describe the all-atom original), so the score is indicative rather than
    directly comparable to published MolProbity values.

    Returns NaN when any input is not finite.
    """
    if not all(np.isfinite(v) for v in (clashscore, rama_outlier_pct, rota_outlier_pct)):
        return np.nan
    return float(
        0.426 * np.log(1.0 + clashscore)
        + 0.33 * np.log(1.0 + max(0.0, rama_outlier_pct - 0.2) / 0.2)
        + 0.25 * np.log(1.0 + max(0.0, rota_outlier_pct - 2.0) / 2.0)
        + 0.5
    )


# ---------------------------------------------------------------------------
# Absolute MM energy
# ---------------------------------------------------------------------------


def _receptor_energy(
    chain_atoms,
    solvent_model: str = "obc2",
    device: str = "cuda",
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
) -> dict:
    """Compute absolute AMBER ff14SB potential energy for a receptor chain.

    Writes receptor atoms to a temp PDB, prepares with PDBFixer (if available),
    adds hydrogens, builds an AMBER ff14SB + implicit solvent system, and
    evaluates potential energy at the input geometry (no minimization).

    Stochastic steps (PDBFixer's rebuild of missing atoms and the jitter of new
    hydrogens) are seeded with ``random_seed``, so the same input gives the same
    energy; ``None`` leaves them unseeded. With ``device="cuda"`` the final
    single-point evaluation uses mixed precision, which is not bit-reproducible.
    """
    _nan = {
        "energy_kJ_mol": np.nan,
        "energy_per_residue_kJ_mol": np.nan,
        "n_atoms_with_h": None,
        "error": None,
    }

    try:
        openmm, unit, ForceField, Modeller, PDBFile, Simulation = _import_openmm()
        from binding_metrics.core.system import deterministic_hydrogen_placement
    except ImportError as e:
        return {**_nan, "error": str(e)}

    try:
        from pdbfixer import PDBFixer

        has_pdbfixer = True
    except ImportError:
        has_pdbfixer = False

    try:
        _, _, pdb_io = _import_biotite()
    except ImportError as e:
        return {**_nan, "error": str(e)}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
            tmp_path = Path(tmp.name)
        out_pdb = pdb_io.PDBFile()
        pdb_io.set_structure(out_pdb, chain_atoms)
        out_pdb.write(str(tmp_path))

        gb_file = "implicit/gbn2.xml" if solvent_model == "gbn2" else "implicit/obc2.xml"

        # Modeller.addHydrogens offsets each new hydrogen with Python's global
        # random module and settles it with a short minimisation on the fastest
        # platform. The seed and the Reference platform (double precision,
        # single-threaded) make that step, and so the single-point energy below,
        # repeatable.
        placement_platform = openmm.Platform.getPlatformByName("Reference")
        with deterministic_hydrogen_placement(random_seed):
            if has_pdbfixer:
                fixer = PDBFixer(filename=str(tmp_path))
                fixer.findMissingResidues()
                fixer.findNonstandardResidues()
                fixer.replaceNonstandardResidues()
                fixer.removeHeterogens(keepWater=False)
                fixer.findMissingAtoms()
                fixer.addMissingAtoms(seed=random_seed)
                # Same call as PDBFixer.addMissingHydrogens(7.4), with the platform pinned.
                mod = Modeller(fixer.topology, fixer.positions)
                mod.addHydrogens(pH=7.4, platform=placement_platform)
            else:
                pdb = PDBFile(str(tmp_path))
                ff_tmp = ForceField("amber14-all.xml", "amber14/tip3pfb.xml", gb_file)
                mod = Modeller(pdb.topology, pdb.positions)
                mod.addHydrogens(ff_tmp, pH=7.4, platform=placement_platform)
        topology, positions = mod.topology, mod.positions

        ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml", gb_file)
        system = ff.createSystem(
            topology,
            nonbondedMethod=openmm.app.NoCutoff,
            constraints=openmm.app.HBonds,
        )

        platform = openmm.Platform.getPlatformByName("CPU")
        props = {}
        if device == "cuda":
            try:
                platform = openmm.Platform.getPlatformByName("CUDA")
                props = {"CudaPrecision": "mixed"}
            except openmm.OpenMMException:
                pass  # no CUDA platform registered: evaluate on the CPU platform

        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        sim = Simulation(topology, system, integrator, platform, props)
        sim.context.setPositions(positions)
        state = sim.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        if not np.isfinite(energy):
            return {**_nan, "error": "non-finite energy (likely severe clashes)"}

        n_res = topology.getNumResidues()
        return {
            "energy_kJ_mol": float(energy),
            "energy_per_residue_kJ_mol": float(energy / max(1, n_res)),
            "n_atoms_with_h": topology.getNumAtoms(),
            "error": None,
        }

    except Exception as e:
        # Broad on purpose: PDBFixer, OpenMM and the force field raise many exception
        # types on unusual input, and the energy is one term of a multi-metric report,
        # so a failure is recorded in ``error`` instead of aborting the other terms.
        return {**_nan, "error": f"{type(e).__name__}: {e}"}

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Per-model dispatcher
# ---------------------------------------------------------------------------


def _score_model(
    atoms,
    receptor_chain: str,
    clash_cutoff: float,
    solvent_model: str,
    device: str,
    model_index: int,
    exclude_bonded: bool = True,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
) -> dict:
    """Compute all quality metrics for one model."""
    struc, _, _ = _import_biotite()

    aa_mask = struc.filter_amino_acids(atoms)
    rec_mask = aa_mask & (atoms.chain_id == receptor_chain)
    rec_atoms = atoms[rec_mask]

    if len(rec_atoms) == 0:
        return {
            "model_index": model_index,
            "n_residues": 0,
            "n_heavy_atoms": 0,
            "ramachandran": {},
            "clashes": {},
            "rotamers": {},
            "cbeta": {},
            "backbone_geometry": {},
            "b_factors": {"available": False},
            "energy": {
                "energy_kJ_mol": np.nan,
                "energy_per_residue_kJ_mol": np.nan,
                "n_atoms_with_h": None,
                "error": "no receptor atoms found",
            },
            "molprobity_score": np.nan,
        }

    n_res = len(set(zip(rec_atoms.chain_id, rec_atoms.res_id)))
    elements = np.array([str(a.element).strip().upper() for a in rec_atoms])
    n_heavy = int(np.sum(~np.isin(elements, ["H", "D", ""])))

    rama = _ramachandran(rec_atoms)
    clash = _clashscore(
        rec_atoms, clash_cutoff, exclude_bonded=exclude_bonded, exempt_hbond_pairs=exclude_bonded
    )
    rota = _rotamer_quality(rec_atoms)
    cbeta = _cbeta_deviations(rec_atoms)
    bbgeom = _backbone_geometry(rec_atoms)
    bfact = _bfactor_stats(rec_atoms)
    energy = _receptor_energy(
        rec_atoms, solvent_model=solvent_model, device=device, random_seed=random_seed
    )

    mp_score = _molprobity_score(
        clash.get("clashscore", np.nan),
        rama.get("outlier_pct", np.nan),
        rota.get("outlier_pct", np.nan),
    )

    result = {
        "model_index": model_index,
        "n_residues": n_res,
        "n_heavy_atoms": n_heavy,
        "ramachandran": rama,
        "clashes": clash,
        "rotamers": rota,
        "cbeta": cbeta,
        "backbone_geometry": bbgeom,
        "b_factors": bfact,
        "energy": energy,
        "molprobity_score": mp_score,
    }
    if not np.isfinite(mp_score):
        undefined = [
            f"{name}: {term['reason']}"
            for name, term in (("clashscore", clash), ("ramachandran", rama), ("rotamers", rota))
            if "reason" in term
        ]
        detail = "; ".join(undefined) or "a term is not finite"
        result["reason"] = f"molprobity_score undefined ({detail})"
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_receptor_quality(
    path: str | Path,
    receptor_chain: Optional[str] = None,
    clash_cutoff: float = 0.4,
    solvent_model: str = "obc2",
    device: str = "cuda",
    *,
    exclude_bonded: bool = True,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    target_chain: Optional[str] = None,
) -> dict:
    """Compute MolProbity-style structural quality metrics for a receptor chain.

    Accepts receptor-only structures or complexes (non-receptor chains ignored).
    All models in multi-model PDB/CIF files are evaluated independently.

    Type: score

    Args:
        path: Path to structure file (PDB or CIF/mmCIF).
        receptor_chain: Chain ID of the receptor (auto-detects largest chain if None).
        clash_cutoff: Minimum VDW overlap in Å to count as a clash (default 0.4 Å).
        solvent_model: Implicit solvent for energy: 'obc2' or 'gbn2'.
        device: Compute device for energy: 'cuda' or 'cpu'.
        exclude_bonded: Leave covalently linked atom pairs (disulfides, cyclic
            closures, lactams, staples, peptide bonds across numbering gaps) and
            N/O hydrogen-bond pairs out of the clash count (default True). False
            restores the earlier count, which scored them as clashes.
        random_seed: Seed for the stochastic steps of the energy term
            (hydrogen placement and PDBFixer's rebuild of missing atoms).
            Defaults to the package-wide ``DEFAULT_RANDOM_SEED`` so repeated
            calls agree; ``None`` gives fresh randomness. The energy is a single
            point at the input geometry, so it is the term most exposed to the
            hydrogen jitter.
        target_chain: Alias of ``receptor_chain``; different IDs in both raise
            ``ValueError``.

    Returns:
        Dictionary with keys:

        Metadata:
            receptor_chain (str), n_models (int)

        Per-model results — models (list[dict]), each with:
            model_index (int), n_residues (int), n_heavy_atoms (int)
            ramachandran: favoured/allowed/outlier counts and %, n_evaluated
            clashes: clashscore, n_clashes, n_heavy_atoms  [heavy-atom overlap, see _clashscore]
            rotamers: outlier_count, outlier_pct, n_evaluated  [χ1 simplified]
            cbeta: cb_deviation_count, cb_n_evaluated, cb_deviation_pct
            backbone_geometry: bad/total bonds+angles and %
            b_factors: mean/max/min/std, n_high_b_residues (>60 Å²)
            energy: energy_kJ_mol, energy_per_residue_kJ_mol, n_atoms_with_h,
                error (None on success, else "ExceptionType: message")
            molprobity_score (float): composite score (lower = better); NaN when
                the clashscore, Ramachandran or rotamer term is undefined
                (for example a chain without any χ1 residue)

            Every term dict above carries a ``reason`` string only when its
            value could not be computed; the value then keeps its NaN sentinel.

        Aggregate summary (mean over models):
            summary (dict) — all scalar metrics averaged; best_model_index

        A structure without a protein chain returns ``error`` (str) instead.

    References:
        Chen et al. 2010, Acta Cryst. D66:12 and Williams et al. 2018, Protein
        Sci. 27:293 (MolProbity); Engh & Huber 1991, Acta Cryst. A47:392
        (backbone geometry); Lovell et al. 2003, Proteins 50:437 (Cβ deviation).
    """
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain
    )
    path = Path(path)
    all_models = _load_all_models(path)

    if receptor_chain is None:
        receptor_chain = _detect_receptor_chain(all_models[0])
    if receptor_chain is None:
        return {
            "receptor_chain": None,
            "n_models": 0,
            "models": [],
            "summary": {},
            "error": "No protein chains found in structure.",
        }

    model_results = [
        _score_model(
            atoms,
            receptor_chain,
            clash_cutoff,
            solvent_model,
            device,
            idx + 1,
            exclude_bonded=exclude_bonded,
            random_seed=random_seed,
        )
        for idx, atoms in enumerate(all_models)
    ]

    return {
        "receptor_chain": str(receptor_chain),
        "n_models": len(model_results),
        "models": model_results,
        "summary": _aggregate_summary(model_results),
    }


def _aggregate_summary(model_results: list) -> dict:
    """Compute mean metrics over all models and identify the best model."""

    def _mean(key_path: str) -> float:
        vals = []
        for m in model_results:
            obj = m
            for k in key_path.split("."):
                obj = obj.get(k, {}) if isinstance(obj, dict) else {}
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                v = float(obj)
                if np.isfinite(v):
                    vals.append(v)
        return float(np.mean(vals)) if vals else np.nan

    summary = {
        "ramachandran_favoured_pct": _mean("ramachandran.favoured_pct"),
        "ramachandran_outlier_pct": _mean("ramachandran.outlier_pct"),
        "ramachandran_outlier_count": _mean("ramachandran.outlier_count"),
        "clashscore": _mean("clashes.clashscore"),
        "rotamer_outlier_pct": _mean("rotamers.outlier_pct"),
        "rotamer_outlier_count": _mean("rotamers.outlier_count"),
        "cb_deviation_count": _mean("cbeta.cb_deviation_count"),
        "bad_bonds_pct": _mean("backbone_geometry.bad_bonds_pct"),
        "bad_angles_pct": _mean("backbone_geometry.bad_angles_pct"),
        "molprobity_score": _mean("molprobity_score"),
        "energy_kJ_mol": _mean("energy.energy_kJ_mol"),
        "energy_per_residue_kJ_mol": _mean("energy.energy_per_residue_kJ_mol"),
    }

    # Best model: lowest MolProbity score
    best_idx, best_score = None, np.inf
    for m in model_results:
        mp = m.get("molprobity_score", np.nan)
        if np.isfinite(mp) and mp < best_score:
            best_score, best_idx = mp, m["model_index"]

    summary["best_model_index"] = best_idx
    return summary


# ---------------------------------------------------------------------------
# CSV / JSON export helpers
# ---------------------------------------------------------------------------


def _model_to_csv_row(filename: str, receptor_chain: str, m: dict) -> dict:
    """Flatten one model result into a flat dict suitable for CSV."""
    r = m.get("ramachandran", {})
    c = m.get("clashes", {})
    ro = m.get("rotamers", {})
    cb = m.get("cbeta", {})
    bg = m.get("backbone_geometry", {})
    b = m.get("b_factors", {})
    e = m.get("energy", {})
    return {
        "filename": filename,
        "receptor_chain": receptor_chain,
        "model_index": m.get("model_index"),
        "n_residues": m.get("n_residues"),
        "n_heavy_atoms": m.get("n_heavy_atoms"),
        # Ramachandran
        "rama_favoured_pct": r.get("favoured_pct"),
        "rama_allowed_pct": r.get("allowed_pct"),
        "rama_outlier_pct": r.get("outlier_pct"),
        "rama_outlier_count": r.get("outlier_count"),
        "rama_favoured_count": r.get("favoured_count"),
        "rama_n_evaluated": r.get("n_evaluated"),
        # Clashscore
        "clashscore": c.get("clashscore"),
        "n_clashes": c.get("n_clashes"),
        # Rotamers
        "rotamer_outlier_count": ro.get("outlier_count"),
        "rotamer_outlier_pct": ro.get("outlier_pct"),
        "rotamer_n_evaluated": ro.get("n_evaluated"),
        # Cβ deviations
        "cbeta_dev_count": cb.get("cb_deviation_count"),
        "cbeta_n_evaluated": cb.get("cb_n_evaluated"),
        "cbeta_dev_pct": cb.get("cb_deviation_pct"),
        # Backbone geometry
        "bad_bonds": bg.get("bad_bonds"),
        "total_bonds": bg.get("total_bonds"),
        "bad_bonds_pct": bg.get("bad_bonds_pct"),
        "bad_angles": bg.get("bad_angles"),
        "total_angles": bg.get("total_angles"),
        "bad_angles_pct": bg.get("bad_angles_pct"),
        # MolProbity
        "molprobity_score": m.get("molprobity_score"),
        # Energy
        "energy_kJ_mol": e.get("energy_kJ_mol"),
        "energy_per_residue_kJ_mol": e.get("energy_per_residue_kJ_mol"),
        # B-factors
        "b_factor_mean": b.get("mean_b_factor"),
        "b_factor_max": b.get("max_b_factor"),
        "b_factor_std": b.get("std_b_factor"),
        "n_high_b_residues": b.get("n_high_b_residues"),
    }


def _write_csv(result: dict, output_path: Path) -> None:
    rows = [
        _model_to_csv_row(result.get("input_filename", ""), result["receptor_chain"], m)
        for m in result["models"]
    ]
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    configure_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Assess receptor structural quality: Ramachandran, rotamer outliers, "
            "Cβ deviations, backbone bonds/angles, clashscore, MolProbity score, "
            "B-factors, and AMBER ff14SB potential energy."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input CIF or PDB file (receptor-only or complex)",
    )
    parser.add_argument(
        "--receptor-chain",
        type=str,
        default=None,
        help="Receptor chain ID (auto-detects largest chain if omitted)",
    )
    parser.add_argument(
        "--clash-cutoff", type=float, default=0.4, help="VDW overlap threshold in Å (default 0.4)"
    )
    parser.add_argument("--solvent-model", choices=["obc2", "gbn2"], default="obc2")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write results to file. Extension determines format: .csv or .json",
    )
    from binding_metrics.cli import add_log_file_arg, add_random_seed_arg

    add_random_seed_arg(parser, "hydrogen placement and atom rebuilding in the energy term")
    add_log_file_arg(parser)
    args = parser.parse_args()

    from binding_metrics.cli import log_to_file

    with log_to_file(args.log_file):
        print(f"Receptor quality assessment: {args.input}")

        result = compute_receptor_quality(
            args.input,
            receptor_chain=args.receptor_chain,
            clash_cutoff=args.clash_cutoff,
            solvent_model=args.solvent_model,
            device=args.device,
            random_seed=args.random_seed,
        )
        result["input_filename"] = args.input.name

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        print(f"  Receptor chain : {result['receptor_chain']}")
        print(f"  Models scored  : {result['n_models']}")

        for m in result["models"]:
            label = f"Model {m['model_index']}" if result["n_models"] > 1 else "Structure"
            print(f"\n  {label}  ({m['n_residues']} residues, {m['n_heavy_atoms']} heavy atoms)")

            r = m["ramachandran"]
            if r.get("n_evaluated", 0) > 0:
                print(
                    f"    Ramachandran  favoured={r['favoured_pct']:.1f}%  "
                    f"outliers={r['outlier_pct']:.1f}% ({r['outlier_count']})"
                )

            c = m["clashes"]
            if np.isfinite(c.get("clashscore", np.nan)):
                print(f"    Clashscore    {c['clashscore']:.2f}  ({c['n_clashes']} clashes)")

            ro = m["rotamers"]
            if ro.get("n_evaluated", 0) > 0:
                print(
                    f"    Rotamers      outliers={ro['outlier_pct']:.1f}% "
                    f"({ro['outlier_count']}/{ro['n_evaluated']})"
                )

            cb = m["cbeta"]
            if cb.get("cb_n_evaluated", 0) > 0:
                print(f"    Cβ deviations {cb['cb_deviation_count']} / {cb['cb_n_evaluated']}")

            bg = m["backbone_geometry"]
            if bg.get("total_bonds", 0) > 0:
                print(
                    f"    Bonds         {bg['bad_bonds']} / {bg['total_bonds']} bad "
                    f"({bg['bad_bonds_pct']:.2f}%)"
                )
                print(
                    f"    Angles        {bg['bad_angles']} / {bg['total_angles']} bad "
                    f"({bg['bad_angles_pct']:.2f}%)"
                )

            mp = m.get("molprobity_score", np.nan)
            if np.isfinite(mp):
                print(f"    MolProbity    {mp:.2f}")

            e = m["energy"]
            if e.get("error"):
                print(f"    Energy        failed: {e['error']}")
            elif np.isfinite(e.get("energy_kJ_mol", np.nan)):
                print(
                    f"    Energy        {e['energy_kJ_mol']:.1f} kJ/mol  "
                    f"({e['energy_per_residue_kJ_mol']:.1f} kJ/mol/residue)"
                )

            b = m["b_factors"]
            if b.get("available"):
                print(
                    f"    B-factors     mean={b['mean_b_factor']:.1f}  "
                    f"max={b['max_b_factor']:.1f}  high-B={b['n_high_b_residues']}"
                )

        if result["n_models"] > 1:
            s = result["summary"]
            print(f"\n  Summary (mean over {result['n_models']} models):")

            def _sfmt(k):
                return f"{s[k]:.2f}" if np.isfinite(s.get(k, np.nan)) else "N/A"

            print(f"    Ramachandran favoured : {_sfmt('ramachandran_favoured_pct')}%")
            print(f"    Ramachandran outliers : {_sfmt('ramachandran_outlier_pct')}%")
            print(f"    Clashscore            : {_sfmt('clashscore')}")
            print(f"    Rotamer outliers      : {_sfmt('rotamer_outlier_pct')}%")
            print(f"    Cβ deviations         : {s.get('cb_deviation_count', 'N/A'):.1f}")
            print(f"    Bad bonds             : {_sfmt('bad_bonds_pct')}%")
            print(f"    Bad angles            : {_sfmt('bad_angles_pct')}%")
            print(f"    MolProbity score      : {_sfmt('molprobity_score')}")
            print(f"    Energy                : {_sfmt('energy_kJ_mol')} kJ/mol")
            print(f"    Energy per residue    : {_sfmt('energy_per_residue_kJ_mol')} kJ/mol")
            if s.get("best_model_index") is not None:
                print(f"    Best model            : {s['best_model_index']}")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            suffix = args.output.suffix.lower()
            if suffix == ".csv":
                _write_csv(result, args.output)
            else:
                with open(args.output, "w") as f:
                    json.dump(result, f, indent=2, default=_json_default)
            print(f"\n  Results written to: {args.output}")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
