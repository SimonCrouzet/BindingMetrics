"""Backbone geometry metrics, shape complementarity, and void volume analysis.

Implements Ramachandran analysis, omega planarity, shape complementarity
(Lawrence & Colman, 1993), and buried void volume detection for
peptide-protein complexes.

Usage:
    binding-metrics-geometry --input complex.cif --chain A --metric ramachandran
    binding-metrics-geometry --input complex.cif --metric omega
    binding-metrics-geometry --input complex.cif --metric sc
    binding-metrics-geometry --input complex.cif --metric void
"""

import argparse
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from binding_metrics.metrics._common import resolve_chain_role
from binding_metrics.utils import backfill_auth_columns, configure_logging

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _import_biotite():
    """Lazy import of required biotite modules."""
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb_io
        import biotite.structure.io.pdbx as pdbx
        from biotite.structure.info import vdw_radius_single

        return struc, pdbx, pdb_io, vdw_radius_single
    except ImportError:
        raise ImportError(
            "biotite is required for geometry metrics. "
            "Install with: pip install binding-metrics[biotite]"
        )


def _import_scipy():
    """Lazy import of scipy spatial."""
    try:
        from scipy.ndimage import label
        from scipy.spatial import cKDTree

        return cKDTree, label
    except ImportError:
        raise ImportError(
            "scipy is required for shape complementarity and void volume metrics. "
            "Install with: pip install binding-metrics[biotite]"
        )


# ---------------------------------------------------------------------------
# Structure loading helpers
# ---------------------------------------------------------------------------


def _load_structure(path: Path):
    """Load a PDB or CIF file as a biotite AtomArray."""
    struc, pdbx, pdb_io, _ = _import_biotite()
    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        pdbx_file = pdbx.CIFFile.read(str(path))
        backfill_auth_columns(pdbx_file)
        return pdbx.get_structure(pdbx_file, model=1)
    else:
        pdb_file = pdb_io.PDBFile.read(str(path))
        return pdb_io.get_structure(pdb_file, model=1)


_HETERO_MODES = ("ignore", "keep")


def _filter_hetero(atoms, hetero: Literal["ignore", "keep"]):
    """Apply the heteroatom policy before any per-chain selection.

    Waters, ions and ligands frequently carry the chain ID of the protein
    chain they sit next to, so a chain-ID mask alone counts them as protein
    atoms (on 1CWA this moves Sc from 0.722 to 0.750 and the void interface
    atom count from 153 to 123). The policy is the one shared by the
    structure-based metrics, `interface.filter_hetero_atoms`: "ignore" keeps
    the polymer (amino acids including D- and non-canonical residues, AMBER
    protonation variants such as HID/HIE, and ACE/NME/NH2 caps), "keep"
    returns the atoms as read.

    Raises:
        ValueError: If ``hetero`` is not one of the two modes.
    """
    from binding_metrics.metrics.interface import filter_hetero_atoms

    return filter_hetero_atoms(atoms, hetero)


_NO_CHAIN_REASON = "no chain was given and no protein chain could be auto-detected"
_TWO_CHAINS_REASON = (
    "peptide and receptor chains could not be determined (fewer than two protein chains)"
)


def _empty_chain_reason(peptide_chain: str, receptor_chain: str, pep_atoms, hetero: str) -> str:
    """Reason text for a peptide/receptor pair where one chain selects no atoms."""
    missing = "peptide" if len(pep_atoms) == 0 else "receptor"
    chain = peptide_chain if missing == "peptide" else receptor_chain
    return f"{missing} chain {chain!r} has no atoms in the structure (hetero={hetero!r})"


def _auto_detect_designed_chain(atoms) -> Optional[str]:
    """Return chain ID of the smallest protein chain."""
    from binding_metrics.metrics.interface import detect_interface_chains

    pep, _ = detect_interface_chains(atoms, None)
    return pep


def _auto_detect_chains(atoms, peptide_chain=None, receptor_chain=None):
    """Return (peptide_chain, receptor_chain) with auto-detection as needed."""
    from binding_metrics.metrics.interface import detect_interface_chains

    if peptide_chain is None or receptor_chain is None:
        auto_pep, auto_rec = detect_interface_chains(atoms, peptide_chain)
        peptide_chain = peptide_chain or auto_pep
        receptor_chain = receptor_chain or auto_rec
    return peptide_chain, receptor_chain


_DEFAULT_VDW_RADIUS = 1.80


def _get_vdw(element: str) -> float:
    """Return the van der Waals radius in Å of an element (Bondi, 1964), default 1.8 Å."""
    _VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20, "P": 1.80}
    return _VDW.get(element.strip().upper(), _DEFAULT_VDW_RADIUS)


# ---------------------------------------------------------------------------
# Ramachandran analysis
# ---------------------------------------------------------------------------

# C(last)-N(first) distance below which a chain is taken to close head to tail
# (a peptide C-N bond is 1.33 A; 2.0 A leaves room for strained macrocycles).
_CLOSURE_MAX_DISTANCE = 2.0


def _terminal_backbone_coords(chain_atoms) -> dict[str, np.ndarray]:
    """N, CA and C coordinates of the first and last amino-acid residue.

    Keys are ``"N_first"``, ``"CA_first"``, ``"C_first"``, ``"N_last"``,
    ``"CA_last"`` and ``"C_last"``; an atom the residue lacks is left out. The
    dict is empty when the chain has fewer than two amino-acid residues.
    """
    struc, _, _, _ = _import_biotite()
    # waters and ions can carry the chain ID and would be taken as the last residue
    chain_atoms = chain_atoms[struc.filter_amino_acids(chain_atoms)]
    starts = struc.get_residue_starts(chain_atoms)
    if len(starts) < 2:
        return {}
    residues = {"first": chain_atoms[starts[0] : starts[1]], "last": chain_atoms[starts[-1] :]}
    coords = {}
    for which, residue in residues.items():
        for name in ("N", "CA", "C"):
            found = residue.coord[residue.atom_name == name]
            if len(found):
                coords[f"{name}_{which}"] = found[0]
    return coords


def _has_head_to_tail_closure(chain_atoms) -> bool:
    """True if the last residue's C is bonded to the first residue's N.

    ``biotite.structure.dihedral_backbone`` walks the chain sequentially, so
    the φ/ψ/ω that involve the ring-closing bond of a cyclic peptide are never
    produced. This detects such a chain so they can be computed separately.
    """
    coords = _terminal_backbone_coords(chain_atoms)
    if "N_first" not in coords or "C_last" not in coords:
        return False
    return bool(np.linalg.norm(coords["N_first"] - coords["C_last"]) < _CLOSURE_MAX_DISTANCE)


def _closing_dihedrals_deg(chain_atoms) -> Optional[tuple[float, float, float]]:
    """φ of the first residue, ψ and ω of the last residue across the ring closure.

    Uses the bond between the last residue's C and the first residue's N, so
    φ(1) = C(last)-N(1)-CA(1)-C(1), ψ(last) = N(last)-CA(last)-C(last)-N(1) and
    ω(last) = CA(last)-C(last)-N(1)-CA(1), all in degrees.

    Returns:
        ``(phi_first, psi_last, omega_last)``, or None when the chain does not
        close head to tail or a backbone atom of either end residue is missing.
    """
    struc, _, _, _ = _import_biotite()
    if not _has_head_to_tail_closure(chain_atoms):
        return None
    c = _terminal_backbone_coords(chain_atoms)
    needed = ("N_first", "CA_first", "C_first", "N_last", "CA_last", "C_last")
    if any(key not in c for key in needed):
        return None
    phi_first = struc.dihedral(c["C_last"], c["N_first"], c["CA_first"], c["C_first"])
    psi_last = struc.dihedral(c["N_last"], c["CA_last"], c["C_last"], c["N_first"])
    omega_last = struc.dihedral(c["CA_last"], c["C_last"], c["N_first"], c["CA_first"])
    return tuple(float(np.degrees(angle)) for angle in (phi_first, psi_last, omega_last))


def _apply_closing_dihedrals(chain_atoms, phi_deg, psi_deg, omega_deg) -> bool:
    """Fill the ring-closing φ, ψ and ω into the arrays from ``dihedral_backbone``.

    The arrays are indexed like the chain's Cα atoms (see the callers), so the
    first and last amino-acid residues are located through the Cα atoms that
    belong to amino acids. The arrays are modified in place.

    Returns:
        True if the chain closes head to tail and the closing angles were set.
    """
    struc, _, _, _ = _import_biotite()
    closing = _closing_dihedrals_deg(chain_atoms)
    if closing is None:
        return False
    is_amino_acid = struc.filter_amino_acids(chain_atoms)[chain_atoms.atom_name == "CA"]
    rows = np.flatnonzero(is_amino_acid)
    if len(rows) < 2 or rows[-1] >= len(phi_deg):
        return False
    phi_first, psi_last, omega_last = closing
    phi_deg[rows[0]] = phi_first
    psi_deg[rows[-1]] = psi_last
    omega_deg[rows[-1]] = omega_last
    return True


def _select_chain(atoms, chain: str):
    """Atoms of one chain.

    Raises:
        ValueError: If no atom carries ``chain``; the message lists the chain
            IDs that are present.
    """
    mask = atoms.chain_id == chain
    if not mask.any():
        available = sorted({str(c) for c in atoms.chain_id})
        raise ValueError(
            f"chain {chain!r} not found in the structure; available chains: {available}"
        )
    return atoms[mask]


def _backbone_dihedrals_deg(chain_atoms):
    """φ, ψ and ω in degrees for a chain, closing a head-to-tail ring if present.

    Returns:
        ``(phi_deg, psi_deg, omega_deg, closure_detected, closure_evaluated)``;
        the three arrays follow ``dihedral_backbone`` (indexed like the chain's
        Cα atoms), ``closure_evaluated`` is True when the ring-closing angles
        were computed and put into them.
    """
    struc, _, _, _ = _import_biotite()
    closure_detected = _has_head_to_tail_closure(chain_atoms)
    phi_deg, psi_deg, omega_deg = (np.degrees(a) for a in struc.dihedral_backbone(chain_atoms))
    closure_evaluated = closure_detected and _apply_closing_dihedrals(
        chain_atoms, phi_deg, psi_deg, omega_deg
    )
    return phi_deg, psi_deg, omega_deg, closure_detected, closure_evaluated


def _classify_ramachandran(phi: float, psi: float, is_d: bool = False) -> Optional[str]:
    """Classify a residue into a Ramachandran region (box approximation).

    The regions are hand-drawn rectangles in the (φ, ψ) plane, not the
    contours of MolProbity's reference distributions (Lovell et al., 2003;
    Williams et al., 2018), and there are no separate Gly, Pro or pre-Pro
    classes: Gly at φ > 0 is judged with the general L-amino-acid regions.

    For D-amino acids pass ``is_d=True``: φ/ψ are negated before region
    lookup so that the mirrored Ramachandran plot maps correctly onto the
    standard L-amino acid regions (D-α-helix φ≈+57°,ψ≈+47° → −57°,−47°).

    Args:
        phi: Phi dihedral in degrees.
        psi: Psi dihedral in degrees.
        is_d: True for D-amino acids.

    Returns:
        'favoured', 'allowed', or 'outlier'; None at termini (NaN input).
    """
    if np.isnan(phi) or np.isnan(psi):
        return None  # terminus, skip

    if is_d:
        phi, psi = -phi, -psi

    # Favoured boxes: rough stand-ins for the 98 % contour of the general case.
    in_alpha = (-90 <= phi <= -30) and (-80 <= psi <= 10)
    in_beta = (-180 <= phi <= -45) and ((90 <= psi <= 180) or (-180 <= psi <= -160))
    in_ppii = (-90 <= phi <= -50) and (120 <= psi <= 180)
    in_l_hel = (20 <= phi <= 90) and (0 <= psi <= 85)
    if in_alpha or in_beta or in_ppii or in_l_hel:
        return "favoured"

    # Allowed boxes: rough stand-ins for the 99.95 % contour.
    in_all_a = (-125 <= phi <= 0) and (-100 <= psi <= 30)
    in_all_b = (-180 <= phi <= -30) and ((60 <= psi <= 180) or (-180 <= psi <= -100))
    in_all_l = (0 <= phi <= 110) and (-30 <= psi <= 100)
    if in_all_a or in_all_b or in_all_l:
        return "allowed"

    return "outlier"


def compute_ramachandran(
    cif_path: str | Path,
    chain: Optional[str] = None,
    *,
    binder_chain: Optional[str] = None,
) -> dict:
    """Compute Ramachandran backbone dihedral quality metrics for a chain.

    Evaluates phi/psi dihedral angles and classifies each residue as
    favoured, allowed or outlier. MolProbity defines these classes as the
    contours that hold 98 % (favoured) and 99.95 % (allowed) of a
    high-resolution reference set, separately for general, Gly, Pro and
    pre-Pro residues (Lovell et al., 2003; Williams et al., 2018). This
    function uses hand-drawn rectangular regions that approximate the general
    case only (see `_classify_ramachandran`), so its percentages are a screen
    and will not match MolProbity's. D-residues are mirrored before the
    lookup.

    Terminal residues of a linear chain have no complete φ/ψ pair and are
    skipped. For a head-to-tail cyclic peptide (last C bonded to first N,
    detected by a C-N distance below 2 Å) φ of the first residue and ψ of the
    last residue are computed across the closing amide bond, so every residue
    is evaluated; see ``cyclic_closure_detected`` and
    ``cyclic_closure_evaluated``.

    Type: score

    Args:
        cif_path: Path to structure file (CIF or PDB)
        chain: Chain ID to evaluate (auto-detects smallest chain if None)
        binder_chain: Alias of ``chain`` (the binder is the chain evaluated by
            default); different IDs in both raise ``ValueError``.

    Returns:
        Dictionary with keys:

        Scores:
            ramachandran_favoured_pct (float): Percentage in favoured regions
            ramachandran_allowed_pct (float): Percentage in allowed regions
            ramachandran_outlier_pct (float): Percentage as outliers
            ramachandran_outlier_count (int): Number of outlier residues
            n_residues_evaluated (int): Residues with a complete φ/ψ pair
                (termini excluded unless the chain is a head-to-tail ring)
            n_d_residues (int): Evaluated residues that are D-amino acids
            cyclic_closure_detected (bool): The chain's last C is bonded to
                its first N (head-to-tail macrocycle)
            cyclic_closure_evaluated (bool): True when the ring-closing φ/ψ
                were computed and are part of these statistics

        Features:
            per_residue (list[dict]): Per-residue data with keys:
                res_id, res_name, chain, phi, psi, is_d_aa, region

        Diagnostics:
            reason (str): Only present when no residue could be evaluated
                (percentages are then NaN); says why.

    Raises:
        ValueError: If ``chain`` is given but absent from the structure; the
            message lists the chain IDs that are present.
    """
    from binding_metrics.core.nonstandard import is_d_residue

    chain = resolve_chain_role("chain", chain, "binder_chain", binder_chain)
    cif_path = Path(cif_path)
    atoms = _load_structure(cif_path)

    if chain is None:
        chain = _auto_detect_designed_chain(atoms)
    if chain is None:
        return {
            "ramachandran_favoured_pct": np.nan,
            "ramachandran_allowed_pct": np.nan,
            "ramachandran_outlier_pct": np.nan,
            "ramachandran_outlier_count": 0,
            "n_residues_evaluated": 0,
            "n_d_residues": 0,
            "cyclic_closure_detected": False,
            "cyclic_closure_evaluated": False,
            "per_residue": [],
            "reason": _NO_CHAIN_REASON,
        }

    chain_atoms = _select_chain(atoms, chain)
    phi_deg, psi_deg, _, closure_detected, closure_evaluated = _backbone_dihedrals_deg(chain_atoms)

    # dihedral_backbone yields one (phi, psi, omega) per residue, so the CA
    # atoms are used to label them; this breaks if a residue lacks a CA atom.
    ca_mask = chain_atoms.atom_name == "CA"
    ca_atoms = chain_atoms[ca_mask]

    per_residue = []
    counts = {"favoured": 0, "allowed": 0, "outlier": 0}
    n_d = 0

    for i, ca in enumerate(ca_atoms):
        if i >= len(phi_deg):
            break
        phi = float(phi_deg[i])
        psi = float(psi_deg[i])
        res_name = str(ca.res_name).strip()
        d_aa = is_d_residue(res_name)
        region = _classify_ramachandran(phi, psi, is_d=d_aa)
        if region is None:
            continue
        counts[region] += 1
        if d_aa:
            n_d += 1
        per_residue.append(
            {
                "res_id": int(ca.res_id),
                "res_name": res_name,
                "chain": str(ca.chain_id),
                "phi": phi,
                "psi": psi,
                "is_d_aa": d_aa,
                "region": region,
            }
        )

    n_eval = len(per_residue)
    if n_eval == 0:
        return {
            "ramachandran_favoured_pct": np.nan,
            "ramachandran_allowed_pct": np.nan,
            "ramachandran_outlier_pct": np.nan,
            "ramachandran_outlier_count": 0,
            "n_residues_evaluated": 0,
            "n_d_residues": n_d,
            "cyclic_closure_detected": closure_detected,
            "cyclic_closure_evaluated": closure_evaluated,
            "per_residue": per_residue,
            "reason": f"chain {chain!r} has no residue with a complete phi/psi pair",
        }

    return {
        "ramachandran_favoured_pct": 100.0 * counts["favoured"] / n_eval,
        "ramachandran_allowed_pct": 100.0 * counts["allowed"] / n_eval,
        "ramachandran_outlier_pct": 100.0 * counts["outlier"] / n_eval,
        "ramachandran_outlier_count": counts["outlier"],
        "n_residues_evaluated": n_eval,
        "n_d_residues": n_d,
        "cyclic_closure_detected": closure_detected,
        "cyclic_closure_evaluated": closure_evaluated,
        "per_residue": per_residue,
    }


# ---------------------------------------------------------------------------
# Omega planarity
# ---------------------------------------------------------------------------

# |omega| below this is counted as a cis peptide bond (MolProbity's cut-off).
_CIS_OMEGA_MAX_DEG = 30.0


def compute_omega_planarity(
    cif_path: str | Path,
    chain: Optional[str] = None,
    *,
    binder_chain: Optional[str] = None,
) -> dict:
    """Compute omega dihedral planarity metrics for peptide bonds.

    Trans peptide bonds have ω ≈ 180°; cis bonds ω ≈ 0°. Every deviation
    larger than 15° from 180° is flagged as an outlier, cis bonds included:
    the 15° cut-off is this package's heuristic, and a legitimate cis-Pro or
    N-methylated amide scores a deviation near 180° and counts as an outlier.
    ``omega_cis_count`` reports how many of the evaluated bonds are cis so
    that these can be told apart from twisted trans bonds.

    For a head-to-tail cyclic peptide (last C bonded to first N, detected by a
    C-N distance below 2 Å) the closing peptide bond is evaluated as well and
    is listed under the last residue; see ``cyclic_closure_detected`` and
    ``cyclic_closure_evaluated``.

    Type: score

    Args:
        cif_path: Path to structure file (CIF or PDB)
        chain: Chain ID to evaluate (auto-detects smallest chain if None)
        binder_chain: Alias of ``chain`` (the binder is the chain evaluated by
            default); different IDs in both raise ``ValueError``.

    Returns:
        Dictionary with keys:

        Scores:
            omega_mean_dev (float): Mean |ω - 180°| in degrees
            omega_max_dev (float): Maximum |ω - 180°| in degrees
            omega_outlier_fraction (float): Fraction of bonds with |dev| > 15°
            omega_outlier_count (int): Number of outlier peptide bonds
            n_bonds_evaluated (int): Number of non-NaN omega values
            omega_cis_count (int): Evaluated bonds with |ω| < 30°
            cyclic_closure_detected (bool): The chain's last C is bonded to
                its first N (head-to-tail macrocycle)
            cyclic_closure_evaluated (bool): True when the closing peptide
                bond's ω was computed and is part of these statistics

        Features:
            per_residue (list[dict]): Per-residue data with keys:
                res_id, res_name, chain, omega, deviation, is_outlier

        Diagnostics:
            reason (str): Only present when no peptide bond could be
                evaluated (the deviations are then NaN); says why.

    Raises:
        ValueError: If ``chain`` is given but absent from the structure; the
            message lists the chain IDs that are present.
    """
    chain = resolve_chain_role("chain", chain, "binder_chain", binder_chain)
    cif_path = Path(cif_path)
    atoms = _load_structure(cif_path)

    if chain is None:
        chain = _auto_detect_designed_chain(atoms)
    if chain is None:
        return {
            "omega_mean_dev": np.nan,
            "omega_max_dev": np.nan,
            "omega_outlier_fraction": np.nan,
            "omega_outlier_count": 0,
            "n_bonds_evaluated": 0,
            "omega_cis_count": 0,
            "cyclic_closure_detected": False,
            "cyclic_closure_evaluated": False,
            "per_residue": [],
            "reason": _NO_CHAIN_REASON,
        }

    chain_atoms = _select_chain(atoms, chain)
    _, _, omega_deg, closure_detected, closure_evaluated = _backbone_dihedrals_deg(chain_atoms)

    ca_mask = chain_atoms.atom_name == "CA"
    ca_atoms = chain_atoms[ca_mask]

    per_residue = []
    deviations = []

    for i, ca in enumerate(ca_atoms):
        if i >= len(omega_deg):
            break
        omega = float(omega_deg[i])
        if np.isnan(omega):
            continue
        # Distance from trans (180°), handling ±180° wrap
        dev = min(abs(omega - 180.0), abs(omega + 180.0))
        is_outlier = dev > 15.0
        deviations.append(dev)
        per_residue.append(
            {
                "res_id": int(ca.res_id),
                "res_name": str(ca.res_name).strip(),
                "chain": str(ca.chain_id),
                "omega": omega,
                "deviation": dev,
                "is_outlier": is_outlier,
            }
        )

    n_eval = len(deviations)
    if n_eval == 0:
        return {
            "omega_mean_dev": np.nan,
            "omega_max_dev": np.nan,
            "omega_outlier_fraction": np.nan,
            "omega_outlier_count": 0,
            "n_bonds_evaluated": 0,
            "omega_cis_count": 0,
            "cyclic_closure_detected": closure_detected,
            "cyclic_closure_evaluated": closure_evaluated,
            "per_residue": per_residue,
            "reason": f"chain {chain!r} has no peptide bond with a defined omega angle",
        }

    dev_arr = np.array(deviations)
    n_outlier = int(np.sum(dev_arr > 15.0))

    return {
        "omega_mean_dev": float(np.mean(dev_arr)),
        "omega_max_dev": float(np.max(dev_arr)),
        "omega_outlier_fraction": float(n_outlier / n_eval),
        "omega_outlier_count": n_outlier,
        "n_bonds_evaluated": n_eval,
        "omega_cis_count": sum(abs(r["omega"]) < _CIS_OMEGA_MAX_DEG for r in per_residue),
        "cyclic_closure_detected": closure_detected,
        "cyclic_closure_evaluated": closure_evaluated,
        "per_residue": per_residue,
    }


# ---------------------------------------------------------------------------
# Shape complementarity (Lawrence & Colman 1993)
# ---------------------------------------------------------------------------


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n evenly spaced points on unit sphere via Fibonacci lattice.

    The lattice is the golden-angle spiral of González (2010, Math. Geosci.
    42, 49), which spreads points almost uniformly without a random seed.

    Args:
        n: Number of points

    Returns:
        Array of shape (n, 3) with unit vectors
    """
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n, dtype=float)
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    phi_angles = 2 * np.pi * i / golden
    return np.stack(
        [
            np.sin(theta) * np.cos(phi_angles),
            np.sin(theta) * np.sin(phi_angles),
            np.cos(theta),
        ],
        axis=1,
    )


def _build_surface_dots(
    interface_atoms,
    all_same_chain_atoms,
    opposite_atoms,
    n_dots: int,
    buried_cutoff: float,
    normal_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build molecular-surface dots and smoothed outward normals.

    For each interface atom this generates ``n_dots`` points on its van der
    Waals sphere, discards points buried inside neighbouring same-chain
    atoms, and keeps only points that lie in genuine contact with the
    opposite chain (nearest opposite atom within ``buried_cutoff``). This
    contact selection isolates the buried interface patch (Lawrence &
    Colman's interface surface, minus the peripheral band) rather than the
    whole solvent-exposed shell of every interface atom.

    The outward surface normal at each dot is estimated as the direction
    from the local same-chain atomic centroid (atoms within
    ``normal_radius``) to the dot. This smoothed normal approximates the
    true molecular-surface normal far better than a single atom-centre →
    dot vector, which is dominated by per-atom curvature and collapses the
    normal dot-product for anything but head-on contacts.

    Args:
        interface_atoms: Atoms in this chain that are at the interface.
        all_same_chain_atoms: All atoms in this chain (occlusion + normals).
        opposite_atoms: All atoms in the opposite chain.
        n_dots: Number of candidate surface dots per atom.
        buried_cutoff: Max distance (Å) from a dot to the nearest opposite
            atom for the dot to count as buried interface surface.
        normal_radius: Radius (Å) of the same-chain neighbourhood used to
            estimate the smoothed outward surface normal.

    Returns:
        Tuple of (dots, normals) each as (N, 3) arrays. Returns empty arrays
        if no buried interface dots are found.
    """
    cKDTree, _ = _import_scipy()
    unit_sphere = _fibonacci_sphere(n_dots)  # (n_dots, 3)

    all_dots = []
    all_normals = []

    same_coords = all_same_chain_atoms.coord  # (n_same, 3)
    same_vdw = np.array([_get_vdw(str(a.element).strip()) for a in all_same_chain_atoms])
    if len(same_coords) == 0 or len(opposite_atoms) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    same_tree = cKDTree(same_coords)
    opp_tree = cKDTree(opposite_atoms.coord)
    max_same_vdw = float(same_vdw.max())

    for atom_a in interface_atoms:
        vdw_a = _get_vdw(str(atom_a.element).strip())
        center = atom_a.coord  # (3,)
        dots = center + vdw_a * unit_sphere  # (n_dots, 3)

        # Occlusion: drop dots that fall inside any OTHER same-chain vdW
        # sphere. The generating atom itself never occludes because its own
        # dots sit at distance vdw_a, just outside vdw_a * 0.99.
        neigh = same_tree.query_ball_point(dots, vdw_a + max_same_vdw)
        keep = np.ones(len(dots), dtype=bool)
        for i, nb in enumerate(neigh):
            if not nb:
                continue
            d_nb = np.linalg.norm(same_coords[nb] - dots[i], axis=1)
            if np.any(d_nb < same_vdw[nb] * 0.99):
                keep[i] = False
        dots = dots[keep]
        if len(dots) == 0:
            continue

        # Buried-patch filter: keep only dots in real contact with the
        # opposite chain (nearest opposite atom within buried_cutoff).
        d_opp, _ = opp_tree.query(dots, k=1)
        dots = dots[d_opp < buried_cutoff]
        if len(dots) == 0:
            continue

        # Smoothed outward surface normals: dot minus local same-chain
        # atomic centroid, with a radial fallback when no neighbours exist.
        normals = np.empty_like(dots)
        neigh_n = same_tree.query_ball_point(dots, normal_radius)
        for i, nb in enumerate(neigh_n):
            local_centroid = same_coords[nb].mean(axis=0) if nb else center
            n_vec = dots[i] - local_centroid
            if float(n_vec @ n_vec) < 1e-12:
                n_vec = dots[i] - center  # degenerate: fall back to radial
            normals[i] = n_vec
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        normals = normals / norms

        all_dots.append(dots)
        all_normals.append(normals)

    if not all_dots:
        return np.zeros((0, 3)), np.zeros((0, 3))

    return np.vstack(all_dots), np.vstack(all_normals)


def compute_shape_complementarity(
    cif_path: str | Path,
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    n_dots: int = 150,
    interface_cutoff: float = 6.0,
    buried_cutoff: float = 2.4,
    normal_radius: float = 6.0,
    weight: float = 0.5,
    *,
    binder_chain: Optional[str] = None,
    target_chain: Optional[str] = None,
    hetero: Literal["ignore", "keep"] = "ignore",
) -> dict:
    """Compute shape complementarity Sc (Lawrence & Colman, 1993, J. Mol. Biol. 234, 946).

    This is a dot-and-normal approximation of the published method, not a
    port of the CCP4 ``sc`` program: surface dots are placed on van der Waals
    spheres rather than on a molecular surface, and the outward normals are
    estimated from the local atom neighbourhood. Values are comparable
    between structures scored here; do not compare them numerically with
    CCP4 ``sc`` output.

    For the buried interface patch of each chain this samples molecular
    surface dots with smoothed outward normals, then for every dot on one
    surface finds the nearest dot on the other and scores
    ``S = (n_a · n_a') * exp(-w · d²)``, where ``n_a`` is the dot's outward
    normal, ``n_a'`` the outward normal at the nearest opposing dot (sign
    flipped so complementary, anti-parallel surfaces score positive), ``d``
    their separation and ``w`` the weight. Sc is the mean of the two
    directional medians (A→B and B→A).

    Typical values for well-formed native interfaces: ~0.65-0.75 for
    protein-protein/protease-inhibitor packing, a touch lower for peptide
    interfaces; ≲0.4 indicates flat, non-complementary surfaces.

    Type: score

    Args:
        cif_path: Path to structure file (CIF or PDB)
        peptide_chain: Chain ID of peptide (auto-detected if None)
        receptor_chain: Chain ID of receptor (auto-detected if None)
        n_dots: Number of candidate surface dots per atom (default 150)
        interface_cutoff: Distance cutoff in Å for pre-selecting interface
            atoms (default 6.0); only affects which atoms are dotted, not the
            buried-patch definition below.
        buried_cutoff: Max distance in Å from a surface dot to the nearest
            opposite-chain atom for the dot to count as buried interface
            surface (default 2.4). This isolates the genuine contact patch.
        normal_radius: Neighbourhood radius in Å for smoothed surface
            normals (default 6.0).
        weight: Gaussian distance weight w in Å⁻² for exp(-w·d²)
            (default 0.5, per Lawrence & Colman).
        hetero: "ignore" (default) keeps only amino-acid atoms, so waters,
            ions and ligands that carry a protein chain ID are dropped before
            the chain masks are built; "keep" uses every atom of the chain.
        binder_chain: Alias of ``peptide_chain``; different IDs in both raise
            ``ValueError``.
        target_chain: Alias of ``receptor_chain``, same rule.

    Returns:
        Dictionary with keys:

        Score:
            sc (float): Shape complementarity score [−1, 1]; NaN if no interface
            sc_A_to_B (float): Median score from peptide dots → receptor
            sc_B_to_A (float): Median score from receptor dots → peptide

        Features:
            n_surface_dots_A (int): Number of interface surface dots for peptide
            n_surface_dots_B (int): Number of interface surface dots for receptor
            per_dot_scores_A (np.ndarray): S_i values for peptide dots
            per_dot_scores_B (np.ndarray): S_i values for receptor dots

        Diagnostics:
            reason (str): Only present when Sc could not be computed (the
                score is then NaN); says why.
    """
    peptide_chain = resolve_chain_role("peptide_chain", peptide_chain, "binder_chain", binder_chain)
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain
    )
    cKDTree, _ = _import_scipy()
    struc, _, _, _ = _import_biotite()

    cif_path = Path(cif_path)
    atoms = _filter_hetero(_load_structure(cif_path), hetero)
    peptide_chain, receptor_chain = _auto_detect_chains(atoms, peptide_chain, receptor_chain)

    def _nan_result(reason: str) -> dict:
        return {
            "sc": np.nan,
            "sc_A_to_B": np.nan,
            "sc_B_to_A": np.nan,
            "n_surface_dots_A": 0,
            "n_surface_dots_B": 0,
            "per_dot_scores_A": np.array([]),
            "per_dot_scores_B": np.array([]),
            "reason": reason,
        }

    if peptide_chain is None or receptor_chain is None:
        return _nan_result(_TWO_CHAINS_REASON)

    pep_atoms = atoms[atoms.chain_id == peptide_chain]
    rec_atoms = atoms[atoms.chain_id == receptor_chain]

    if len(pep_atoms) == 0 or len(rec_atoms) == 0:
        return _nan_result(_empty_chain_reason(peptide_chain, receptor_chain, pep_atoms, hetero))

    # Pre-select interface atoms: nearest opposite-chain atom within cutoff.
    # KD-tree keeps this O(N log N) instead of an O(N_a·N_b) dense matrix.
    rec_tree = cKDTree(rec_atoms.coord)
    pep_tree = cKDTree(pep_atoms.coord)
    d_pep, _ = rec_tree.query(pep_atoms.coord, k=1)
    d_rec, _ = pep_tree.query(rec_atoms.coord, k=1)
    pep_iface = pep_atoms[d_pep < interface_cutoff]
    rec_iface = rec_atoms[d_rec < interface_cutoff]

    if len(pep_iface) == 0 or len(rec_iface) == 0:
        return _nan_result(
            f"no atom of one chain lies within interface_cutoff={interface_cutoff} A of the other"
        )

    dots_A, normals_A = _build_surface_dots(
        pep_iface, pep_atoms, rec_atoms, n_dots, buried_cutoff, normal_radius
    )
    dots_B, normals_B = _build_surface_dots(
        rec_iface, rec_atoms, pep_atoms, n_dots, buried_cutoff, normal_radius
    )

    if len(dots_A) == 0 or len(dots_B) == 0:
        return _nan_result(
            f"no surface dot lies within buried_cutoff={buried_cutoff} A of the opposite chain"
        )

    tree_B = cKDTree(dots_B)
    dist_AB, idx_AB = tree_B.query(dots_A, k=1)
    omega_AB = np.exp(-weight * dist_AB**2)
    # Normal dot product with L&C sign flip: complementary (anti-parallel in
    # the lab frame) surfaces yield a POSITIVE product.
    dot_product_AB = np.sum(normals_A * (-normals_B[idx_AB]), axis=1)
    scores_A = omega_AB * dot_product_AB

    tree_A = cKDTree(dots_A)
    dist_BA, idx_BA = tree_A.query(dots_B, k=1)
    omega_BA = np.exp(-weight * dist_BA**2)
    dot_product_BA = np.sum(normals_B * (-normals_A[idx_BA]), axis=1)
    scores_B = omega_BA * dot_product_BA

    sc_A_to_B = float(np.median(scores_A))
    sc_B_to_A = float(np.median(scores_B))
    sc = float(np.mean([sc_A_to_B, sc_B_to_A]))

    return {
        "sc": sc,
        "sc_A_to_B": sc_A_to_B,
        "sc_B_to_A": sc_B_to_A,
        "n_surface_dots_A": len(dots_A),
        "n_surface_dots_B": len(dots_B),
        "per_dot_scores_A": scores_A,
        "per_dot_scores_B": scores_B,
    }


# ---------------------------------------------------------------------------
# Buried void volume
# ---------------------------------------------------------------------------


def _exterior_mask(solid: np.ndarray) -> np.ndarray:
    """Compute exterior mask via flood fill from corner.

    Pads the grid, labels connected components of empty space, identifies
    the exterior component by checking the corner voxel, then returns
    the exterior mask (without padding).

    Args:
        solid: 3D boolean array where True = solid (occupied) voxel

    Returns:
        3D boolean array where True = exterior (accessible to solvent)
    """
    _, label = _import_scipy()
    padded = np.pad(solid, 1, constant_values=False)
    labeled, _ = label(~padded)
    exterior_label = labeled[0, 0, 0]  # corner is always exterior
    exterior = labeled == exterior_label
    return exterior[1:-1, 1:-1, 1:-1]


def _occupancy_grid(
    coords: np.ndarray,
    radii: np.ndarray,
    origin: np.ndarray,
    grid_spacing: float,
    grid_dims: np.ndarray,
) -> np.ndarray:
    """Mark the voxels whose centre lies inside any sphere (strict ``<`` radius).

    Args:
        coords: Sphere centres in Å, shape (N, 3).
        radii: Sphere radii in Å, shape (N,).
        origin: Coordinates of voxel (0, 0, 0) in Å.
        grid_spacing: Voxel edge in Å.
        grid_dims: Number of voxels along each axis.

    Returns:
        Boolean grid of shape ``grid_dims``; spheres are clipped at the grid edge.
    """
    occupied = np.zeros(grid_dims, dtype=bool)
    for coord, radius in zip(coords, radii):
        lo = np.floor((coord - radius - origin) / grid_spacing).astype(int)
        hi = np.ceil((coord + radius - origin) / grid_spacing).astype(int) + 1
        lo = np.clip(lo, 0, grid_dims - 1)
        hi = np.clip(hi, 0, grid_dims)
        if np.any(hi <= lo):
            continue
        gx, gy, gz = np.meshgrid(
            np.arange(lo[0], hi[0]),
            np.arange(lo[1], hi[1]),
            np.arange(lo[2], hi[2]),
            indexing="ij",
        )
        idx = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        inside = np.linalg.norm(idx * grid_spacing + origin - coord, axis=1) < radius
        idx = idx[inside]
        occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occupied


# Probe-centre voxels are voxel centres inside the allowed region, so they sit
# on average a fraction of a voxel inside its true boundary and a sweep by the
# bare probe radius under-fills the cavity. Extending the sweep by this many
# voxels removes most of the bias: against a fine-grid reference on synthetic
# cavities of 5-7 A radius the void volume agrees within about 8 % for probes
# of 1.4-3 A, where the bare radius is 7-25 % low at 0.5 A spacing.
_SWEEP_BIAS_VOXELS = 0.25


def _dilate_by_probe(
    mask: np.ndarray, grid_spacing: float, probe_radius: float, outside: bool
) -> np.ndarray:
    """Voxels swept by a probe whose centre visits the True voxels of ``mask``.

    ``outside`` is the value assumed beyond the grid edge: True for bulk
    solvent, which then sweeps into the box from every side; False for a mask
    that has nothing outside the box. With ``probe_radius=0`` the mask is
    returned unchanged.
    """
    if probe_radius <= 0 or not (mask.any() or outside):
        return mask.copy()
    from scipy.ndimage import distance_transform_edt

    sweep_radius = probe_radius + _SWEEP_BIAS_VOXELS * grid_spacing
    margin = int(np.ceil(sweep_radius / grid_spacing)) + 1
    padded = np.pad(mask, margin, constant_values=outside)
    swept = distance_transform_edt(~padded, sampling=grid_spacing) <= sweep_radius
    return swept[margin:-margin, margin:-margin, margin:-margin]


def _interface_void_mask(
    solid_complex: np.ndarray,
    inflated_pep: np.ndarray,
    inflated_rec: np.ndarray,
    grid_spacing: float,
    probe_radius: float,
) -> np.ndarray:
    """Voxels of the buried interface void (see `compute_buried_void_volume`).

    Args:
        solid_complex: Voxels inside a van der Waals sphere of either chain.
        inflated_pep: Voxels within ``vdw + probe_radius`` of a peptide atom,
            i.e. where a probe centre is not allowed with the peptide alone.
        inflated_rec: Same for the receptor.
        grid_spacing: Voxel edge in Å.
        probe_radius: Probe radius in Å.

    Returns:
        Boolean grid, True on the void voxels. The steps are: probe-centre
        positions that are closed to the bulk in the complex but open to it
        for the peptide alone and for the receptor alone form the cavity
        seeds; the probe body swept from those seeds (Richards, 1977) minus
        whatever the bulk probe can touch is the void. Requiring both chains
        alone to be open keeps only cavities that need both partners to be
        closed: a cavity walled by one chain is trivially open once that
        chain is removed and is not an interface feature.
    """
    inflated_complex = inflated_pep | inflated_rec
    bulk_complex = _exterior_mask(inflated_complex)
    open_for_each_chain = _exterior_mask(inflated_pep) & _exterior_mask(inflated_rec)
    cavity_seeds = ~inflated_complex & ~bulk_complex & open_for_each_chain
    cavity_body = _dilate_by_probe(cavity_seeds, grid_spacing, probe_radius, outside=False)
    bulk_swept = _dilate_by_probe(bulk_complex, grid_spacing, probe_radius, outside=True)
    return cavity_body & ~solid_complex & ~bulk_swept


def compute_buried_void_volume(
    cif_path: str | Path,
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    grid_spacing: float = 0.5,
    probe_radius: float = 1.4,
    interface_cutoff: float = 5.0,
    padding: float = 3.0,
    *,
    binder_chain: Optional[str] = None,
    target_chain: Optional[str] = None,
    hetero: Literal["ignore", "keep"] = "ignore",
) -> dict:
    """Compute buried void volume at the peptide-receptor interface.

    An interface void is a cavity that a solvent probe of radius
    ``probe_radius`` cannot reach from the bulk in the complex, that would be
    open if the peptide were removed, and would be open if the receptor were
    removed: space that only the two partners together close off, a sign of
    poor packing. A cavity walled by a single chain is not counted (removing
    that chain trivially opens it), and neither are interstitial gaps between
    atoms that are too narrow for the probe.

    Definition, on a regular grid. A probe centre is allowed where it stays at
    least ``vdw + probe_radius`` from every atom centre (Lee & Richards,
    1971). Allowed centres that are not connected to the outside of the grid
    in the complex, but are connected to it for the peptide alone and for the
    receptor alone, seed the void. The void is the region the probe body
    sweeps from those seeds (every empty voxel within ``probe_radius`` of a
    seed; Richards, 1977; Connolly, 1983), minus anything the bulk solvent
    probe can touch. Its volume is therefore the empty volume of the closed
    cavity, not just the volume available to the probe centre. Grid cavity
    detection in this spirit is used by VOIDOO (Kleywegt & Jones, 1994).
    With ``probe_radius=0`` the seeds are the empty spaces closed to a point
    probe.

    The result depends on the probe and the grid: compare values only at the
    same ``probe_radius`` and ``grid_spacing``. The voxel-count volume has a
    discretisation error that grows with the cavity surface and the voxel
    size (a few percent on 5 Å-radius test cavities at 0.5 Å); use a finer
    grid to check a value.

    The bounding box is set by the interface atoms (any atom of one chain
    within ``interface_cutoff`` of the other, plus ``padding``); every atom of
    the two chains that reaches into the box is used as an occluder, so that
    space shielded by atoms outside the interface is not read as open.

    Type: score

    Args:
        cif_path: Path to structure file (CIF or PDB)
        peptide_chain: Chain ID of peptide (auto-detected if None)
        receptor_chain: Chain ID of receptor (auto-detected if None)
        grid_spacing: Voxel size in Å (default 0.5)
        probe_radius: Solvent probe radius in Å (default 1.4, water); must be >= 0
        interface_cutoff: Distance cutoff for interface atom selection in Å (default 5.0)
        padding: Bounding box padding in Å (default 3.0)
        hetero: "ignore" (default) keeps only amino-acid atoms, so waters,
            ions and ligands that carry a protein chain ID are not counted as
            interface atoms; "keep" uses every atom of the chain.
        binder_chain: Alias of ``peptide_chain``; different IDs in both raise
            ``ValueError``.
        target_chain: Alias of ``receptor_chain``, same rule.

    Returns:
        Dictionary with keys:

        Score:
            void_volume_A3 (float): Void volume in Å³; lower = better-packed

        Features:
            void_grid_fraction (float): void voxels / total bounding box voxels
            interface_box_volume_A3 (float): Total interface bounding box volume in Å³
            n_interface_atoms (int): Total interface atoms considered

        Diagnostics:
            reason (str): Only present when the void volume could not be
                computed (the score is then NaN); says why.

    Raises:
        ValueError: If ``probe_radius`` is negative.
    """
    if probe_radius < 0:
        raise ValueError(f"probe_radius must be >= 0, got {probe_radius}")

    peptide_chain = resolve_chain_role("peptide_chain", peptide_chain, "binder_chain", binder_chain)
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain
    )
    cif_path = Path(cif_path)
    atoms = _filter_hetero(_load_structure(cif_path), hetero)
    peptide_chain, receptor_chain = _auto_detect_chains(atoms, peptide_chain, receptor_chain)

    def _nan_result(reason: str) -> dict:
        return {
            "void_volume_A3": np.nan,
            "void_grid_fraction": np.nan,
            "interface_box_volume_A3": np.nan,
            "n_interface_atoms": 0,
            "reason": reason,
        }

    if peptide_chain is None or receptor_chain is None:
        return _nan_result(_TWO_CHAINS_REASON)

    pep_atoms = atoms[atoms.chain_id == peptide_chain]
    rec_atoms = atoms[atoms.chain_id == receptor_chain]

    if len(pep_atoms) == 0 or len(rec_atoms) == 0:
        return _nan_result(_empty_chain_reason(peptide_chain, receptor_chain, pep_atoms, hetero))

    diff = pep_atoms.coord[:, np.newaxis, :] - rec_atoms.coord[np.newaxis, :, :]
    dist_pr = np.linalg.norm(diff, axis=-1)  # (n_pep, n_rec)
    pep_iface = pep_atoms[np.any(dist_pr < interface_cutoff, axis=1)]
    rec_iface = rec_atoms[np.any(dist_pr < interface_cutoff, axis=0)]

    n_iface = len(pep_iface) + len(rec_iface)
    if n_iface == 0:
        return _nan_result(
            f"no atom of one chain lies within interface_cutoff={interface_cutoff} A of the other"
        )

    all_iface_coords = np.vstack([pep_iface.coord, rec_iface.coord])
    min_coord = all_iface_coords.min(axis=0) - padding
    max_coord = all_iface_coords.max(axis=0) + padding

    box_size = max_coord - min_coord
    box_dims = np.ceil(box_size / grid_spacing).astype(int) + 1

    # The grid extends beyond the analysis box so that a cavity cut by the box
    # face is classified as a whole (closed or open) instead of being read as
    # open at the cut; only voxels inside the analysis box are counted.
    margin_voxels = int(np.ceil(2.0 * (_DEFAULT_VDW_RADIUS + probe_radius) / grid_spacing))
    origin = min_coord - margin_voxels * grid_spacing
    grid_dims = box_dims + 2 * margin_voxels
    grid_min = origin
    grid_max = origin + (grid_dims - 1) * grid_spacing
    in_box = tuple(slice(margin_voxels, margin_voxels + n) for n in box_dims)

    def _occluders(chain_atoms):
        """Atoms of the chain whose (probe-inflated) sphere can reach into the grid."""
        radii = np.array([_get_vdw(str(a.element).strip()) for a in chain_atoms])
        reach = radii + probe_radius
        near = np.all(
            (chain_atoms.coord >= grid_min - reach[:, None])
            & (chain_atoms.coord <= grid_max + reach[:, None]),
            axis=1,
        )
        return chain_atoms.coord[near], radii[near]

    pep_coord, pep_vdw = _occluders(pep_atoms)
    rec_coord, rec_vdw = _occluders(rec_atoms)

    def _grid(coords, radii):
        return _occupancy_grid(coords, radii, origin, grid_spacing, grid_dims)

    try:
        solid_complex = _grid(pep_coord, pep_vdw) | _grid(rec_coord, rec_vdw)
        interface_void = _interface_void_mask(
            solid_complex,
            _grid(pep_coord, pep_vdw + probe_radius),
            _grid(rec_coord, rec_vdw + probe_radius),
            grid_spacing,
            probe_radius,
        )
    except MemoryError:
        return _nan_result(
            f"the {int(np.prod(grid_dims))}-voxel grid does not fit in memory; "
            "increase grid_spacing"
        )

    void_voxels = int(interface_void[in_box].sum())
    total_voxels = int(np.prod(box_dims))
    void_volume = void_voxels * grid_spacing**3
    box_volume = total_voxels * grid_spacing**3

    return {
        "void_volume_A3": float(void_volume),
        "void_grid_fraction": float(void_voxels / total_voxels) if total_voxels > 0 else np.nan,
        "interface_box_volume_A3": float(box_volume),
        "n_interface_atoms": n_iface,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Compute backbone geometry, shape complementarity, and void metrics"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CIF/PDB file")
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Chain ID for Ramachandran/omega analysis (auto-detect if omitted)",
    )
    parser.add_argument(
        "--peptide-chain",
        type=str,
        default=None,
        help="Peptide chain ID for Sc/void (auto-detect if omitted)",
    )
    parser.add_argument(
        "--receptor-chain",
        type=str,
        default=None,
        help="Receptor chain ID for Sc/void (auto-detect if omitted)",
    )
    parser.add_argument(
        "--metric",
        choices=["ramachandran", "omega", "sc", "void"],
        default="ramachandran",
        help="Metric to compute (default: ramachandran)",
    )
    # Sc parameters
    parser.add_argument("--n-dots", type=int, default=150, help="Surface dots per atom for Sc")
    parser.add_argument(
        "--interface-cutoff", type=float, default=6.0, help="Interface atom pre-selection cutoff Å"
    )
    parser.add_argument(
        "--buried-cutoff", type=float, default=2.4, help="Buried-patch dot cutoff Å for Sc"
    )
    parser.add_argument(
        "--normal-radius", type=float, default=6.0, help="Smoothed-normal neighbourhood radius Å"
    )
    parser.add_argument(
        "--weight", type=float, default=0.5, help="Gaussian weight w (Å⁻²) for Sc exp(-w·d²)"
    )
    # Void parameters
    parser.add_argument("--grid-spacing", type=float, default=0.5, help="Grid spacing Å for void")
    parser.add_argument("--probe-radius", type=float, default=1.4, help="Probe radius Å for void")
    parser.add_argument(
        "--hetero",
        choices=list(_HETERO_MODES),
        default="ignore",
        help=(
            "Sc/void: 'ignore' drops waters, ions and ligands before chain selection; "
            "'keep' uses every atom of the chain (default: ignore)"
        ),
    )
    from binding_metrics.cli import add_log_file_arg

    add_log_file_arg(parser)
    args = parser.parse_args()

    from binding_metrics.cli import log_to_file

    with log_to_file(args.log_file):
        print(f"Computing '{args.metric}' metrics for: {args.input}")

        if args.metric == "ramachandran":
            result = compute_ramachandran(args.input, chain=args.chain)
            scalar_keys = [
                "ramachandran_favoured_pct",
                "ramachandran_allowed_pct",
                "ramachandran_outlier_pct",
                "ramachandran_outlier_count",
                "n_residues_evaluated",
            ]
        elif args.metric == "omega":
            result = compute_omega_planarity(args.input, chain=args.chain)
            scalar_keys = [
                "omega_mean_dev",
                "omega_max_dev",
                "omega_outlier_fraction",
                "omega_outlier_count",
                "n_bonds_evaluated",
            ]
        elif args.metric == "sc":
            result = compute_shape_complementarity(
                args.input,
                peptide_chain=args.peptide_chain,
                receptor_chain=args.receptor_chain,
                n_dots=args.n_dots,
                interface_cutoff=args.interface_cutoff,
                buried_cutoff=args.buried_cutoff,
                normal_radius=args.normal_radius,
                weight=args.weight,
                hetero=args.hetero,
            )
            scalar_keys = ["sc", "sc_A_to_B", "sc_B_to_A", "n_surface_dots_A", "n_surface_dots_B"]
        else:  # void
            result = compute_buried_void_volume(
                args.input,
                peptide_chain=args.peptide_chain,
                receptor_chain=args.receptor_chain,
                grid_spacing=args.grid_spacing,
                probe_radius=args.probe_radius,
                interface_cutoff=args.interface_cutoff,
                hetero=args.hetero,
            )
            scalar_keys = [
                "void_volume_A3",
                "void_grid_fraction",
                "interface_box_volume_A3",
                "n_interface_atoms",
            ]

        print(f"\n{args.metric.capitalize()} summary:")
        for key in scalar_keys:
            val = result.get(key)
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
