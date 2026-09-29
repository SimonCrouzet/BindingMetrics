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

from binding_metrics.utils import backfill_auth_columns

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
    atom count from 153 to 123). "ignore" keeps only amino-acid atoms with
    ``biotite.structure.filter_amino_acids``, which also covers D- and
    non-canonical peptide-linking residues; "keep" returns the atoms as read.

    Args:
        atoms: biotite AtomArray.
        hetero: "ignore" or "keep".

    Raises:
        ValueError: If ``hetero`` is not one of the two modes.
    """
    if hetero not in _HETERO_MODES:
        raise ValueError(f"hetero must be one of {_HETERO_MODES}, got {hetero!r}")
    if hetero == "keep":
        return atoms
    struc, _, _, _ = _import_biotite()
    return atoms[struc.filter_amino_acids(atoms)]


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
    """Return VDW radius for element string, default 1.8 Å."""
    _VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20, "P": 1.80}
    return _VDW.get(element.strip().upper(), _DEFAULT_VDW_RADIUS)


# ---------------------------------------------------------------------------
# Task 3: Ramachandran analysis
# ---------------------------------------------------------------------------


def _classify_ramachandran(phi: float, psi: float, is_d: bool = False) -> Optional[str]:
    """Classify a residue into a Ramachandran region.

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

    # Favoured regions (covers ~98% of high-quality crystallographic residues)
    in_alpha = (-90 <= phi <= -30) and (-80 <= psi <= 10)
    in_beta = (-180 <= phi <= -45) and ((90 <= psi <= 180) or (-180 <= psi <= -160))
    in_ppii = (-90 <= phi <= -50) and (120 <= psi <= 180)
    in_l_hel = (20 <= phi <= 90) and (0 <= psi <= 85)
    if in_alpha or in_beta or in_ppii or in_l_hel:
        return "favoured"

    # Allowed regions
    in_all_a = (-125 <= phi <= 0) and (-100 <= psi <= 30)
    in_all_b = (-180 <= phi <= -30) and ((60 <= psi <= 180) or (-180 <= psi <= -100))
    in_all_l = (0 <= phi <= 110) and (-30 <= psi <= 100)
    if in_all_a or in_all_b or in_all_l:
        return "allowed"

    return "outlier"


def compute_ramachandran(
    cif_path: str | Path,
    chain: Optional[str] = None,
) -> dict:
    """Compute Ramachandran backbone dihedral quality metrics for a chain.

    Evaluates phi/psi dihedral angles and classifies each residue into
    favoured, allowed, or outlier Ramachandran regions following standard
    MolProbity-style geometry validation criteria.

    Type: score

    Args:
        cif_path: Path to structure file (CIF or PDB)
        chain: Chain ID to evaluate (auto-detects smallest chain if None)

    Returns:
        Dictionary with keys:

        Scores:
            ramachandran_favoured_pct (float): Percentage in favoured regions
            ramachandran_allowed_pct (float): Percentage in allowed regions
            ramachandran_outlier_pct (float): Percentage as outliers
            ramachandran_outlier_count (int): Number of outlier residues
            n_residues_evaluated (int): Residues with complete backbone (excl. termini)

        Features:
            per_residue (list[dict]): Per-residue data with keys:
                res_id, res_name, chain, phi, psi, region
    """
    from binding_metrics.core.nonstandard import is_d_residue

    struc, _, _, _ = _import_biotite()
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
            "per_residue": [],
        }

    chain_atoms = atoms[atoms.chain_id == chain]
    phi_rad, psi_rad, _ = struc.dihedral_backbone(chain_atoms)

    phi_deg = np.degrees(phi_rad)
    psi_deg = np.degrees(psi_rad)

    # Get unique residues in order (CA atoms give one per residue)
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
            "per_residue": per_residue,
        }

    return {
        "ramachandran_favoured_pct": 100.0 * counts["favoured"] / n_eval,
        "ramachandran_allowed_pct": 100.0 * counts["allowed"] / n_eval,
        "ramachandran_outlier_pct": 100.0 * counts["outlier"] / n_eval,
        "ramachandran_outlier_count": counts["outlier"],
        "n_residues_evaluated": n_eval,
        "n_d_residues": n_d,
        "per_residue": per_residue,
    }


# ---------------------------------------------------------------------------
# Task 4: Omega planarity
# ---------------------------------------------------------------------------


def compute_omega_planarity(
    cif_path: str | Path,
    chain: Optional[str] = None,
) -> dict:
    """Compute omega dihedral planarity metrics for peptide bonds.

    Trans peptide bonds should have ω ≈ 180°; cis bonds ω ≈ 0°.
    Deviations > 15° from 180° are flagged as outliers.

    Type: score

    Args:
        cif_path: Path to structure file (CIF or PDB)
        chain: Chain ID to evaluate (auto-detects smallest chain if None)

    Returns:
        Dictionary with keys:

        Scores:
            omega_mean_dev (float): Mean |ω - 180°| in degrees
            omega_max_dev (float): Maximum |ω - 180°| in degrees
            omega_outlier_fraction (float): Fraction of bonds with |dev| > 15°
            omega_outlier_count (int): Number of outlier peptide bonds
            n_bonds_evaluated (int): Number of non-NaN omega values

        Features:
            per_residue (list[dict]): Per-residue data with keys:
                res_id, res_name, chain, omega, deviation, is_outlier
    """
    struc, _, _, _ = _import_biotite()
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
            "per_residue": [],
        }

    chain_atoms = atoms[atoms.chain_id == chain]
    _, _, omega_rad = struc.dihedral_backbone(chain_atoms)
    omega_deg = np.degrees(omega_rad)

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
            "per_residue": per_residue,
        }

    dev_arr = np.array(deviations)
    n_outlier = int(np.sum(dev_arr > 15.0))

    return {
        "omega_mean_dev": float(np.mean(dev_arr)),
        "omega_max_dev": float(np.max(dev_arr)),
        "omega_outlier_fraction": float(n_outlier / n_eval),
        "omega_outlier_count": n_outlier,
        "n_bonds_evaluated": n_eval,
        "per_residue": per_residue,
    }


# ---------------------------------------------------------------------------
# Task 5: Shape complementarity (Lawrence & Colman 1993)
# ---------------------------------------------------------------------------


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n evenly spaced points on unit sphere via Fibonacci lattice.

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
    hetero: Literal["ignore", "keep"] = "ignore",
) -> dict:
    """Compute shape complementarity Sc (Lawrence & Colman, 1993).

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
    """
    cKDTree, _ = _import_scipy()
    struc, _, _, _ = _import_biotite()

    cif_path = Path(cif_path)
    atoms = _filter_hetero(_load_structure(cif_path), hetero)
    peptide_chain, receptor_chain = _auto_detect_chains(atoms, peptide_chain, receptor_chain)

    _nan_result = {
        "sc": np.nan,
        "sc_A_to_B": np.nan,
        "sc_B_to_A": np.nan,
        "n_surface_dots_A": 0,
        "n_surface_dots_B": 0,
        "per_dot_scores_A": np.array([]),
        "per_dot_scores_B": np.array([]),
    }

    if peptide_chain is None or receptor_chain is None:
        return _nan_result

    pep_atoms = atoms[atoms.chain_id == peptide_chain]
    rec_atoms = atoms[atoms.chain_id == receptor_chain]

    if len(pep_atoms) == 0 or len(rec_atoms) == 0:
        return _nan_result

    # Pre-select interface atoms: nearest opposite-chain atom within cutoff.
    # KD-tree keeps this O(N log N) instead of an O(N_a·N_b) dense matrix.
    rec_tree = cKDTree(rec_atoms.coord)
    pep_tree = cKDTree(pep_atoms.coord)
    d_pep, _ = rec_tree.query(pep_atoms.coord, k=1)
    d_rec, _ = pep_tree.query(rec_atoms.coord, k=1)
    pep_iface = pep_atoms[d_pep < interface_cutoff]
    rec_iface = rec_atoms[d_rec < interface_cutoff]

    if len(pep_iface) == 0 or len(rec_iface) == 0:
        return _nan_result

    # Build buried-patch surface dots + smoothed normals for each chain
    dots_A, normals_A = _build_surface_dots(
        pep_iface, pep_atoms, rec_atoms, n_dots, buried_cutoff, normal_radius
    )
    dots_B, normals_B = _build_surface_dots(
        rec_iface, rec_atoms, pep_atoms, n_dots, buried_cutoff, normal_radius
    )

    if len(dots_A) == 0 or len(dots_B) == 0:
        return _nan_result

    # Compute scores A → B: for each A dot, find nearest B dot
    tree_B = cKDTree(dots_B)
    dist_AB, idx_AB = tree_B.query(dots_A, k=1)
    omega_AB = np.exp(-weight * dist_AB**2)
    # Normal dot product with L&C sign flip: complementary (anti-parallel in
    # the lab frame) surfaces yield a POSITIVE product.
    dot_product_AB = np.sum(normals_A * (-normals_B[idx_AB]), axis=1)
    scores_A = omega_AB * dot_product_AB

    # Compute scores B → A
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
# Task 6: Buried void volume
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

    Returns:
        Dictionary with keys:

        Score:
            void_volume_A3 (float): Void volume in Å³; lower = better-packed

        Features:
            void_grid_fraction (float): void voxels / total bounding box voxels
            interface_box_volume_A3 (float): Total interface bounding box volume in Å³
            n_interface_atoms (int): Total interface atoms considered

    Raises:
        ValueError: If ``probe_radius`` is negative.
    """
    if probe_radius < 0:
        raise ValueError(f"probe_radius must be >= 0, got {probe_radius}")

    cif_path = Path(cif_path)
    atoms = _filter_hetero(_load_structure(cif_path), hetero)
    peptide_chain, receptor_chain = _auto_detect_chains(atoms, peptide_chain, receptor_chain)

    _nan_result = {
        "void_volume_A3": np.nan,
        "void_grid_fraction": np.nan,
        "interface_box_volume_A3": np.nan,
        "n_interface_atoms": 0,
    }

    if peptide_chain is None or receptor_chain is None:
        return _nan_result

    pep_atoms = atoms[atoms.chain_id == peptide_chain]
    rec_atoms = atoms[atoms.chain_id == receptor_chain]

    if len(pep_atoms) == 0 or len(rec_atoms) == 0:
        return _nan_result

    diff = pep_atoms.coord[:, np.newaxis, :] - rec_atoms.coord[np.newaxis, :, :]
    dist_pr = np.linalg.norm(diff, axis=-1)  # (n_pep, n_rec)
    pep_iface = pep_atoms[np.any(dist_pr < interface_cutoff, axis=1)]
    rec_iface = rec_atoms[np.any(dist_pr < interface_cutoff, axis=0)]

    n_iface = len(pep_iface) + len(rec_iface)
    if n_iface == 0:
        return _nan_result

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

    solid_complex = _grid(pep_coord, pep_vdw) | _grid(rec_coord, rec_vdw)
    interface_void = _interface_void_mask(
        solid_complex,
        _grid(pep_coord, pep_vdw + probe_radius),
        _grid(rec_coord, rec_vdw + probe_radius),
        grid_spacing,
        probe_radius,
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
