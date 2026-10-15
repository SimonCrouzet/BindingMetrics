"""Cross-chain contacts of a structure counted a second way, to derive the expected value of a test.

A test that compares a metric with a number read off one run of the package pins that run. The
functions below apply the documented rule of a metric with code of their own, on the structure
file or on biotite and hydride called directly, so that the expected value follows from the input:

- ``coulomb_reference``: formal charges read from the columns of a PDB file, summed pair by pair;
- ``salt_bridge_reference``: charged side-chain atoms of two chains, nearest contact per residue
  pair;
- ``hbond_reference``: hydrogens from hydride, contacts from biotite's detector, one score per
  donor-acceptor heavy-atom pair.

The module imports nothing of ``binding_metrics``. biotite and hydride are imported inside
``hbond_reference`` only, so a test of the Coulomb sum or of the salt bridges needs neither.
"""

from pathlib import Path

import numpy as np

# e^2 N_A / (4 pi eps0) in kJ A / (mol e^2): e and N_A are exact in the SI, eps0 is CODATA 2018.
COULOMB_KJ_ANGSTROM = (
    1.602176634e-19**2 * 6.02214076e23 / (4 * np.pi * 8.8541878128e-12) * 1e10 / 1e3
)
KJ_PER_KCAL = 4.184  # thermochemical calorie, exact by definition

# Documented defaults of the Coulomb and salt-bridge scores.
DIELECTRIC = 4.0
COULOMB_CUTOFF_ANGSTROM = 12.0
SALT_BRIDGE_WINDOW_ANGSTROM = (0.5, 5.5)

# Formal charges at pH 7 (plain histidine is neutral, termini are uncharged), split over the
# resonance-equivalent atoms.
COULOMB_CHARGES = {
    ("LYS", "NZ"): 1.0,
    ("ARG", "NH1"): 0.5,
    ("ARG", "NH2"): 0.5,
    ("ASP", "OD1"): -0.5,
    ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5,
    ("GLU", "OE2"): -0.5,
}

# Charged side-chain atoms of the salt-bridge criterion.
POSITIVE_ATOMS = {("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE")}
NEGATIVE_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}

# An H-bond whose H...A distance is 2.0 A and whose D-H...A angle is 180 degrees scores
# -2.5 kcal/mol; the score is -k cos^2(180 deg - angle) / d_HA with k = 2.5 * 2.0 kcal A/mol.
IDEAL_HBOND_SCORE_KCAL = -2.5
IDEAL_HBOND_DISTANCE_ANGSTROM = 2.0


def coulomb_reference(pdb_path, chain_a, chain_b):
    """Coulomb energy (kJ/mol) and number of charged atom pairs between two chains of a PDB file.

    Pairs closer than the cutoff count, at the effective dielectric of the metric.
    """
    charged = {chain_a: [], chain_b: []}
    for line in Path(pdb_path).read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[21] not in charged:
            continue
        charge = COULOMB_CHARGES.get((line[17:20].strip(), line[12:16].strip()))
        if charge is not None:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            charged[line[21]].append((charge, xyz))
    pairs = [
        (charge_a * charge_b, np.linalg.norm(xyz_a - xyz_b))
        for charge_a, xyz_a in charged[chain_a]
        for charge_b, xyz_b in charged[chain_b]
    ]
    inside = [(product, r) for product, r in pairs if 0.0 < r < COULOMB_CUTOFF_ANGSTROM]
    energy = COULOMB_KJ_ANGSTROM / DIELECTRIC * sum(product / r for product, r in inside)
    return energy, len(inside)


def salt_bridge_reference(atoms, chain_a, chain_b):
    """Number of salt bridges between two chains of an AtomArray and their score (kcal/mol).

    A residue pair (one positive, one negative side chain, one in each chain) is a bridge when
    one positive-negative atom pair lies inside the window; it scores the Coulomb energy of unit
    charges at its nearest such atom pair.
    """
    low, high = SALT_BRIDGE_WINDOW_ANGSTROM
    nearest = {}
    names = list(zip(atoms.res_name.tolist(), atoms.atom_name.tolist()))
    residue = list(zip(atoms.chain_id.tolist(), atoms.res_id.tolist(), atoms.ins_code.tolist()))
    positive = [i for i, name in enumerate(names) if name in POSITIVE_ATOMS]
    negative = [i for i, name in enumerate(names) if name in NEGATIVE_ATOMS]
    for i in positive:
        for j in negative:
            if {atoms.chain_id[i], atoms.chain_id[j]} != {chain_a, chain_b}:
                continue
            r = np.linalg.norm(atoms.coord[i].astype(float) - atoms.coord[j].astype(float))
            if low < r < high:
                key = (residue[i], residue[j])
                nearest[key] = min(r, nearest.get(key, r))
    coulomb_kcal = COULOMB_KJ_ANGSTROM / KJ_PER_KCAL
    energy = -coulomb_kcal / DIELECTRIC * sum(1.0 / r for r in nearest.values())
    return len(nearest), energy


def hbond_reference(atoms, chain_a, chain_b):
    """Number of hydrogen bonds between two chains of an AtomArray and their score (kcal/mol).

    Hydrogens are added by hydride when ``atoms`` has none. biotite's Baker-Hubbard detector
    lists the contacts; of the contacts of one donor-acceptor heavy-atom pair the shortest
    H...A one is kept, and each pair scores ``-k cos^2(180 deg - angle) / d_HA``.
    """
    import biotite.structure as struc
    import hydride

    atoms = atoms.copy()
    if not np.any(atoms.element == "H"):
        if atoms.bonds is None:
            atoms.bonds = struc.connect_via_residue_names(atoms)
        if "charge" not in atoms.get_annotation_categories():
            atoms.set_annotation("charge", np.zeros(len(atoms), dtype=int))
        atoms, _ = hydride.add_hydrogen(atoms)
    coord = atoms.coord.astype(float)
    shortest = {}
    for donor, hydrogen, acceptor in struc.hbond(atoms):
        if {atoms.chain_id[donor], atoms.chain_id[acceptor]} != {chain_a, chain_b}:
            continue
        distance = np.linalg.norm(coord[hydrogen] - coord[acceptor])
        key = (donor, acceptor)
        if key not in shortest or distance < shortest[key][0]:
            shortest[key] = (distance, hydrogen)
    scale = IDEAL_HBOND_SCORE_KCAL * IDEAL_HBOND_DISTANCE_ANGSTROM
    energy = 0.0
    for (donor, acceptor), (distance, hydrogen) in shortest.items():
        angle = struc.angle(coord[donor], coord[hydrogen], coord[acceptor])
        energy += scale * np.cos(np.pi - angle) ** 2 / distance
    return len(shortest), float(energy)
