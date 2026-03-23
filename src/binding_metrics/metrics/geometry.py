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
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _import_biotite():
    """Lazy import of required biotite modules."""
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdbx as pdbx
        import biotite.structure.io.pdb as pdb_io
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
        from scipy.spatial import cKDTree
        from scipy.ndimage import label

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
        return pdbx.get_structure(pdbx_file, model=1)
    else:
        pdb_file = pdb_io.PDBFile.read(str(path))
        return pdb_io.get_structure(pdb_file, model=1)


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


def _get_vdw(element: str) -> float:
    """Return VDW radius for element string, default 1.8 Å."""
    _VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20, "P": 1.80}
    return _VDW.get(element.strip().upper(), 1.80)


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
        per_residue.append({
            "res_id": int(ca.res_id),
            "res_name": res_name,
            "chain": str(ca.chain_id),
            "phi": phi,
            "psi": psi,
            "is_d_aa": d_aa,
            "region": region,
        })

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
        per_residue.append({
            "res_id": int(ca.res_id),
            "res_name": str(ca.res_name).strip(),
            "chain": str(ca.chain_id),
            "omega": omega,
            "deviation": dev,
            "is_outlier": is_outlier,
        })

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
    return np.stack([
        np.sin(theta) * np.cos(phi_angles),
        np.sin(theta) * np.sin(phi_angles),
        np.cos(theta),
    ], axis=1)


def _build_surface_dots(
    interface_atoms,
    all_same_chain_atoms,
    opposite_atoms,
    n_dots: int,
    interface_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build accessible surface dots for interface atoms.

    For each interface atom generates n_dots surface points, filters out
    dots occluded by same-chain atoms, and keeps only dots within
    interface_cutoff of the opposite chain.

    Args:
        interface_atoms: Atoms in this chain that are at the interface
        all_same_chain_atoms: All atoms in this chain (for occlusion check)
        opposite_atoms: All atoms in the opposite chain
        n_dots: Number of surface dots per atom
        interface_cutoff: Distance cutoff in Å for interface proximity

    Returns:
        Tuple of (dots, normals) each as (N, 3) arrays. Returns empty arrays
        if no interface dots are found.
    """
    unit_sphere = _fibonacci_sphere(n_dots)  # (n_dots, 3)

    all_dots = []
    all_normals = []

    opp_coords = opposite_atoms.coord  # (n_opp, 3)
    same_coords = all_same_chain_atoms.coord  # (n_same, 3)
    same_vdw = np.array([
        _get_vdw(str(a.element).strip()) for a in all_same_chain_atoms
    ])

    for idx_a, atom_a in enumerate(interface_atoms):
        vdw_a = _get_vdw(str(atom_a.element).strip())
        center = atom_a.coord  # (3,)
        dots = center + vdw_a * unit_sphere  # (n_dots, 3)

        # Filter occluded dots: dot must not be inside any OTHER same-chain atom
        # Build a mask: for each dot, compute distance to all same-chain atoms
        diff_same = dots[:, np.newaxis, :] - same_coords[np.newaxis, :, :]  # (n_d, n_s, 3)
        dist_same = np.linalg.norm(diff_same, axis=-1)  # (n_d, n_s)

        # Check if any same-chain atom (excluding self) occludes the dot
        # Self-atom index: find in all_same_chain_atoms
        # Simple approach: use vdw_same as threshold
        occluded = np.any(dist_same < same_vdw[np.newaxis, :] * 0.99, axis=1)
        dots = dots[~occluded]
        if len(dots) == 0:
            continue

        # Filter interface dots: within cutoff of opposite chain
        diff_opp = dots[:, np.newaxis, :] - opp_coords[np.newaxis, :, :]  # (n_d2, n_opp, 3)
        dist_opp = np.linalg.norm(diff_opp, axis=-1)  # (n_d2, n_opp)
        near_interface = np.any(dist_opp < interface_cutoff, axis=1)
        dots = dots[near_interface]
        if len(dots) == 0:
            continue

        # Outward normals: from atom center to dot
        normals = dots - center
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
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
    n_dots: int = 50,
    interface_cutoff: float = 5.0,
    sigma: float = 7.0,
) -> dict:
    """Compute shape complementarity score (Lawrence & Colman, 1993).

    Generates molecular surface dots for interface atoms of each chain,
    matches them across the interface, and computes a complementarity
    score based on surface normal alignment weighted by proximity.
    Typical values: 0.6-0.9 for well-packed interfaces, 0.0-0.4 for poor.

    Type: score

    Args:
        cif_path: Path to structure file (CIF or PDB)
        peptide_chain: Chain ID of peptide (auto-detected if None)
        receptor_chain: Chain ID of receptor (auto-detected if None)
        n_dots: Number of surface dots per atom (default 50)
        interface_cutoff: Distance cutoff in Å for interface atoms (default 5.0)
        sigma: Gaussian decay constant in Å (default 7.0)

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
    atoms = _load_structure(cif_path)
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

    # Find interface atoms: within interface_cutoff of any atom in opposite chain
    def _interface_atom_mask(chain_a_atoms, chain_b_atoms) -> np.ndarray:
        diff = chain_a_atoms.coord[:, np.newaxis, :] - chain_b_atoms.coord[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)  # (n_a, n_b)
        return np.any(dist < interface_cutoff, axis=1)

    pep_iface_mask = _interface_atom_mask(pep_atoms, rec_atoms)
    rec_iface_mask = _interface_atom_mask(rec_atoms, pep_atoms)

    pep_iface = pep_atoms[pep_iface_mask]
    rec_iface = rec_atoms[rec_iface_mask]

    if len(pep_iface) == 0 or len(rec_iface) == 0:
        return _nan_result

    # Build surface dots for each chain
    dots_A, normals_A = _build_surface_dots(pep_iface, pep_atoms, rec_atoms, n_dots, interface_cutoff)
    dots_B, normals_B = _build_surface_dots(rec_iface, rec_atoms, pep_atoms, n_dots, interface_cutoff)

    if len(dots_A) == 0 or len(dots_B) == 0:
        return _nan_result

    # Compute scores A → B: for each A dot, find nearest B dot
    tree_B = cKDTree(dots_B)
    dist_AB, idx_AB = tree_B.query(dots_A, k=1)
    omega_AB = np.exp(-(dist_AB ** 2) / (sigma ** 2))
    # Normal dot product: A outward normal · (-B outward normal)
    dot_product_AB = np.sum(normals_A * (-normals_B[idx_AB]), axis=1)
    scores_A = omega_AB * dot_product_AB

    # Compute scores B → A
    tree_A = cKDTree(dots_A)
    dist_BA, idx_BA = tree_A.query(dots_B, k=1)
    omega_BA = np.exp(-(dist_BA ** 2) / (sigma ** 2))
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


def compute_buried_void_volume(
    cif_path: str | Path,
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    grid_spacing: float = 0.5,
    probe_radius: float = 1.4,
    interface_cutoff: float = 5.0,
    padding: float = 3.0,
) -> dict:
    """Compute buried void volume at the peptide-receptor interface.

    Uses a grid-based approach to identify interface voids: regions that
    are unoccupied by atoms, inaccessible to solvent in the complex, but
    would be solvent-accessible if one chain were removed. These represent
    poorly-packed cavities at the binding interface.

    Type: score

    Args:
        cif_path: Path to structure file (CIF or PDB)
        peptide_chain: Chain ID of peptide (auto-detected if None)
        receptor_chain: Chain ID of receptor (auto-detected if None)
        grid_spacing: Voxel size in Å (default 0.5)
        probe_radius: Solvent probe radius in Å (default 1.4)
        interface_cutoff: Distance cutoff for interface atom selection in Å (default 5.0)
        padding: Bounding box padding in Å (default 3.0)

    Returns:
        Dictionary with keys:

        Score:
            void_volume_A3 (float): Void volume in Å³; lower = better-packed

        Features:
            void_grid_fraction (float): void voxels / total bounding box voxels
            interface_box_volume_A3 (float): Total interface bounding box volume in Å³
            n_interface_atoms (int): Total interface atoms considered
    """
    cif_path = Path(cif_path)
    atoms = _load_structure(cif_path)
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

    # Find interface atoms from each chain
    diff = pep_atoms.coord[:, np.newaxis, :] - rec_atoms.coord[np.newaxis, :, :]
    dist_pr = np.linalg.norm(diff, axis=-1)  # (n_pep, n_rec)
    pep_iface_mask = np.any(dist_pr < interface_cutoff, axis=1)
    rec_iface_mask = np.any(dist_pr < interface_cutoff, axis=0)

    pep_iface = pep_atoms[pep_iface_mask]
    rec_iface = rec_atoms[rec_iface_mask]

    n_iface = len(pep_iface) + len(rec_iface)
    if n_iface == 0:
        return _nan_result

    # Build bounding box around interface atoms
    all_iface_coords = np.vstack([pep_iface.coord, rec_iface.coord])
    min_coord = all_iface_coords.min(axis=0) - padding
    max_coord = all_iface_coords.max(axis=0) + padding

    box_size = max_coord - min_coord
    grid_dims = np.ceil(box_size / grid_spacing).astype(int) + 1

    # Collect VDW radii for all interface atoms
    all_iface_atoms_list = [pep_iface, rec_iface]
    all_iface_atoms_concat_coords = np.vstack([pep_iface.coord, rec_iface.coord])
    all_iface_vdw = np.array([
        _get_vdw(str(a.element).strip())
        for chain in all_iface_atoms_list for a in chain
    ])

    # Map atom coordinates to grid indices
    def _make_solid_grid(atom_coords: np.ndarray, atom_vdw: np.ndarray) -> np.ndarray:
        """Create boolean grid where True = voxel within VDW radius of any atom."""
        solid = np.zeros(grid_dims, dtype=bool)
        # Iterate over atoms, mark voxels within VDW radius
        for coord, vdw in zip(atom_coords, atom_vdw):
            # Bounding box of this atom in grid indices
            r = vdw + probe_radius
            lo = np.floor((coord - r - min_coord) / grid_spacing).astype(int)
            hi = np.ceil((coord + r - min_coord) / grid_spacing).astype(int) + 1
            lo = np.clip(lo, 0, grid_dims - 1)
            hi = np.clip(hi, 0, grid_dims)

            # Generate voxel centers in this sub-box
            xi = np.arange(lo[0], hi[0])
            yi = np.arange(lo[1], hi[1])
            zi = np.arange(lo[2], hi[2])
            if len(xi) == 0 or len(yi) == 0 or len(zi) == 0:
                continue
            gx, gy, gz = np.meshgrid(xi, yi, zi, indexing="ij")
            voxel_centers = (
                np.stack([gx, gy, gz], axis=-1).reshape(-1, 3) * grid_spacing + min_coord
            )
            d = np.linalg.norm(voxel_centers - coord, axis=1)
            inside = d < vdw  # strict VDW (no probe for solid body)
            idx = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)[inside]
            if len(idx) > 0:
                solid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return solid

    # Build solid grids
    pep_vdw = np.array([_get_vdw(str(a.element).strip()) for a in pep_iface])
    rec_vdw = np.array([_get_vdw(str(a.element).strip()) for a in rec_iface])

    solid_pep = _make_solid_grid(pep_iface.coord, pep_vdw)
    solid_rec = _make_solid_grid(rec_iface.coord, rec_vdw)
    solid_complex = solid_pep | solid_rec

    # Exterior masks (flood fill from corner)
    try:
        accessible_complex = _exterior_mask(solid_complex)
        accessible_pep = _exterior_mask(solid_pep)
        accessible_rec = _exterior_mask(solid_rec)
    except Exception:
        return _nan_result

    # Interface void: unoccupied, inaccessible in complex, but accessible in either half
    interface_void = (
        ~solid_complex
        & ~accessible_complex
        & (accessible_pep | accessible_rec)
    )

    void_voxels = int(interface_void.sum())
    total_voxels = int(np.prod(grid_dims))
    void_volume = void_voxels * grid_spacing ** 3
    box_volume = total_voxels * grid_spacing ** 3

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
        "--chain", type=str, default=None,
        help="Chain ID for Ramachandran/omega analysis (auto-detect if omitted)",
    )
    parser.add_argument(
        "--peptide-chain", type=str, default=None,
        help="Peptide chain ID for Sc/void (auto-detect if omitted)",
    )
    parser.add_argument(
        "--receptor-chain", type=str, default=None,
        help="Receptor chain ID for Sc/void (auto-detect if omitted)",
    )
    parser.add_argument(
        "--metric",
        choices=["ramachandran", "omega", "sc", "void"],
        default="ramachandran",
        help="Metric to compute (default: ramachandran)",
    )
    # Sc parameters
    parser.add_argument("--n-dots", type=int, default=50, help="Surface dots per atom for Sc")
    parser.add_argument("--interface-cutoff", type=float, default=5.0, help="Interface cutoff Å")
    parser.add_argument("--sigma", type=float, default=7.0, help="Gaussian sigma for Sc (Å)")
    # Void parameters
    parser.add_argument("--grid-spacing", type=float, default=0.5, help="Grid spacing Å for void")
    parser.add_argument("--probe-radius", type=float, default=1.4, help="Probe radius Å for void")
    from binding_metrics.cli import add_log_file_arg
    add_log_file_arg(parser)
    args = parser.parse_args()

    from binding_metrics.cli import log_to_file
    with log_to_file(args.log_file):
        print(f"Computing '{args.metric}' metrics for: {args.input}")

        if args.metric == "ramachandran":
            result = compute_ramachandran(args.input, chain=args.chain)
            scalar_keys = [
                "ramachandran_favoured_pct", "ramachandran_allowed_pct",
                "ramachandran_outlier_pct", "ramachandran_outlier_count",
                "n_residues_evaluated",
            ]
        elif args.metric == "omega":
            result = compute_omega_planarity(args.input, chain=args.chain)
            scalar_keys = [
                "omega_mean_dev", "omega_max_dev",
                "omega_outlier_fraction", "omega_outlier_count", "n_bonds_evaluated",
            ]
        elif args.metric == "sc":
            result = compute_shape_complementarity(
                args.input,
                peptide_chain=args.peptide_chain,
                receptor_chain=args.receptor_chain,
                n_dots=args.n_dots,
                interface_cutoff=args.interface_cutoff,
                sigma=args.sigma,
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
            )
            scalar_keys = [
                "void_volume_A3", "void_grid_fraction",
                "interface_box_volume_A3", "n_interface_atoms",
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
