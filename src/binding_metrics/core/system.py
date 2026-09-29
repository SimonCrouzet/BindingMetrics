"""System preparation utilities for MD simulations."""

import contextlib
import logging
import os
import tempfile
from typing import Literal, Optional

import openmm.unit as unit
from openmm.app import ForceField, Modeller, PDBFile

from binding_metrics.core.forcefields import get_forcefield
from binding_metrics.utils import add_to_report, extend_report

log = logging.getLogger(__name__)

# Optional pdbfixer for structure repair
try:
    from pdbfixer import PDBFixer

    HAS_PDBFIXER = True
except ImportError:
    HAS_PDBFIXER = False


#: Default seed for every stochastic step (hydrogen placement, PDBFixer atom
#: rebuild, MD velocities and Langevin noise) so the pipeline is reproducible by
#: default. MUST be non-zero: OpenMM's ``setRandomNumberSeed(0)`` is the sentinel
#: for "choose a fresh random seed at run time", so a 0 here would silently
#: re-randomize the integrators it is fed to (e.g. PDBFixer.addMissingAtoms).
#: The specific value carries no meaning; do not "tune" it to dodge a bad
#: hydrogen placement — repair_ca_hydrogen_chirality exists to fix those. Pass
#: ``random_seed=None`` through the configs to opt back into fresh randomness.
DEFAULT_RANDOM_SEED = 1

#: Coordinates (nm) this close to 0 on every axis mark a placeholder atom written
#: by pipelines that do not model it, not a real position.
_ORIGIN_PLACEHOLDER_TOL_NM = 1e-6

#: Equilibrium Cα–HA bond length (nm) in ff14SB (``protein-CX``/``protein-H1``
#: bond in ``amber14/protein.ff14SB.xml``; Maier et al., J. Chem. Theory Comput.
#: 2015, 11, 3696-3713). A repaired HA is placed at this distance so the
#: ``constraints=HBonds`` constraint starts at its rest length.
_CA_HA_BOND_NM = 0.109

#: Below this norm the three unit vectors N, C, CB around a Cα sum to nothing:
#: the tripod is planar and has no fourth tetrahedral vertex to point HA at.
_DEGENERATE_TRIPOD_NORM = 1e-6


def _extract_custom_bonds(topology) -> list:
    """Capture non-sequential intra-chain bonds that PDBxFile.writeFile drops.

    Returns a list of (chain_id, local_res_idx, atom_name,
                        chain_id, local_res_idx, atom_name) tuples.
    Disulfides (SG-SG) are excluded — detected from rebuilt geometry after
    addMissingAtoms by rename_disulfide_cys_to_cyx.
    """
    res_local: dict[int, tuple] = {}
    for chain in topology.chains():
        for local_idx, res in enumerate(chain.residues()):
            res_local[res.index] = (chain.id, local_idx)

    bonds = []
    for bond in topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        r1, r2 = a1.residue, a2.residue
        if r1.chain.id != r2.chain.id:
            continue
        if abs(r1.index - r2.index) <= 1:
            continue
        if a1.name == "SG" and a2.name == "SG":
            continue  # disulfide — rebuilt from geometry after addMissingAtoms
        bonds.append(
            (
                res_local[r1.index][0],
                res_local[r1.index][1],
                a1.name,
                res_local[r2.index][0],
                res_local[r2.index][1],
                a2.name,
            )
        )
    return bonds


def _readd_custom_bonds(topology, custom_bonds: list):
    """Re-add custom bonds to a post-PDBFixer topology by position."""
    if not custom_bonds:
        return topology
    lookup: dict[tuple, object] = {}
    for chain in topology.chains():
        for local_idx, res in enumerate(chain.residues()):
            for atom in res.atoms():
                lookup[(chain.id, local_idx, atom.name)] = atom

    existing: set = {
        (min(b.atom1.index, b.atom2.index), max(b.atom1.index, b.atom2.index))
        for b in topology.bonds()
    }

    for ch1, li1, an1, ch2, li2, an2 in custom_bonds:
        a1 = lookup.get((ch1, li1, an1))
        a2 = lookup.get((ch2, li2, an2))
        if a1 and a2:
            key = (min(a1.index, a2.index), max(a1.index, a2.index))
            if key not in existing:
                topology.addBond(a1, a2)
                existing.add(key)
        else:
            log.warning(
                "Could not restore custom bond %s[%d].%s – %s[%d].%s after prep",
                ch1,
                li1,
                an1,
                ch2,
                li2,
                an2,
            )
    return topology


def _rebuild_connect_records(fixer) -> None:
    """Detect and register all non-standard covalent bonds from rebuilt geometry.

    Called after addMissingAtoms so that atoms previously at the origin
    (zero-coord placeholders) have been placed at correct positions.

    Currently handles:
      - Disulfide bonds: any SG–SG pair within _DISULFIDE_THRESH nm is bonded.
        Residues are kept as CYS; renaming to CYX is deferred to createSystem
        time (energy/MD) where the CYX template must be matched explicitly.

    Cyclic closure bonds (head-to-tail, lactam) are handled separately by
    patch_cyclic_topology (which uses custom_bonds as hints for strained geometry).
    """
    from binding_metrics.core.cyclic import register_ss_bonds

    fixer.topology = register_ss_bonds(fixer.topology, fixer.positions)


def _delete_zero_coord_atoms(fixer) -> "PDBFixer":
    """Remove atoms placed at the origin (zero-coordinate placeholders from pipelines
    that don't model all atoms) so PDBFixer can rebuild them via
    findMissingAtoms() + addMissingAtoms().

    Returns the original fixer unchanged if no zero-coordinate atoms are found.
    """
    from openmm.app import PDBxFile

    atoms_at_origin = [
        i
        for i, pos in enumerate(fixer.positions)
        if max(abs(pos.x), abs(pos.y), abs(pos.z)) < _ORIGIN_PLACEHOLDER_TOL_NM
    ]
    if not atoms_at_origin:
        return fixer

    log.debug("Found %d zero-coordinate atoms — removing for rebuild", len(atoms_at_origin))
    all_atoms = list(fixer.topology.atoms())
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.delete([all_atoms[i] for i in atoms_at_origin])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as tmp:
        PDBxFile.writeFile(modeller.topology, modeller.positions, tmp)
        tmp_path = tmp.name
    try:
        return PDBFixer(filename=tmp_path)
    finally:
        os.unlink(tmp_path)


def _topology_to_fixer(topology, positions) -> "PDBFixer":
    """Write topology+positions to a temp CIF and load with PDBFixer.

    CIF keeps atom and residue names and handles residue numbers beyond the
    PDB limit of 9999; PDB format truncates them. Neither keeps the caller's
    residue numbers or chain IDs: PDBxFile.writeFile renumbers residues from 1
    in each chain and assigns chain IDs A, B, C, ... in order.
    """
    if not HAS_PDBFIXER:
        raise ImportError(
            "pdbfixer is required. Install with: pip install binding-metrics[structure]"
        )
    from openmm.app import PDBxFile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as tmp:
        PDBxFile.writeFile(topology, positions, tmp)
        tmp_path = tmp.name
    try:
        fixer = PDBFixer(filename=tmp_path)
    finally:
        os.unlink(tmp_path)
    return fixer


# Standard amino acids and nucleotides recognised by AMBER ff14SB.
_STANDARD_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    # protonation variants
    "HIE",
    "HID",
    "HIP",
    "CYX",
    "ASH",
    "GLH",
    "LYN",
    # nucleotides
    "DA",
    "DC",
    "DG",
    "DT",
    "A",
    "C",
    "G",
    "T",
    "U",
}

# Common metal ions parameterised in standard force fields (no GAFF2 needed).
_METAL_ELEMENTS = {
    "Li",
    "Na",
    "K",
    "Rb",
    "Cs",
    "Mg",
    "Ca",
    "Sr",
    "Ba",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
}

_WATER_NAMES = {"HOH", "WAT", "SOL", "TIP", "TIP3", "H2O"}

#: Longest C(i)-N(i+1) distance still read as a peptide bond. A real amide bond
#: is about 1.33 A (Engh and Huber, Acta Cryst. A47, 392-400, 1991); one missing
#: residue puts the neighbours at 3.8 A or more, so 2.0 A separates the two cases
#: with room for poor geometry.
_PEPTIDE_BOND_MAX_ANGSTROM = 2.0


def _count_residue_gaps(topology, positions) -> int:
    """Count places where a chain skips residues (unresolved loops).

    A gap is a pair of consecutive amino-acid residues in one chain whose residue
    numbers jump by more than 1 and whose C and N atoms are not bonded. The
    distance test keeps a chain that is merely renumbered from being counted.
    When the C or N atom is absent, or sits at the origin as a placeholder, the
    numbering jump alone decides.

    Call it on the topology as loaded from the file. PDBFixer's own
    ``missingResidues`` cannot serve: it reads the sequence records of the input
    file, which the topology-to-CIF round trip in :func:`_topology_to_fixer` does
    not carry, and that round trip also renumbers residues from 1 in each chain.
    """
    import numpy as np

    coords_nm = np.array(positions.value_in_unit(unit.nanometer))

    def _coords_angstrom_or_none(index):
        if index is None:
            return None
        xyz = coords_nm[index]
        if np.abs(xyz).max() < _ORIGIN_PLACEHOLDER_TOL_NM:
            return None
        return xyz * 10.0

    n_gaps = 0
    for chain in topology.chains():
        previous = None
        for res in chain.residues():
            atoms = {a.name: a.index for a in res.atoms()}
            if "CA" not in atoms:
                continue  # water, ion, ligand or nucleotide
            if previous is not None:
                try:
                    jump = int(res.id) - int(previous[0].id)
                except ValueError:
                    jump = 1  # non-numeric residue id: cannot judge
                if jump > 1:
                    c_xyz = _coords_angstrom_or_none(previous[1].get("C"))
                    n_xyz = _coords_angstrom_or_none(atoms.get("N"))
                    bonded = (
                        c_xyz is not None
                        and n_xyz is not None
                        and np.linalg.norm(c_xyz - n_xyz) < _PEPTIDE_BOND_MAX_ANGSTROM
                    )
                    if not bonded:
                        n_gaps += 1
            previous = (res, atoms)
    return n_gaps


def _add_hydrogens_cyclic(
    topology,
    positions,
    custom_bonds: list,
    ph: float,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
) -> tuple:
    """Add hydrogens to a topology that contains non-sequential cyclic bonds.

    PDBFixer's addMissingHydrogens cannot handle cyclic peptides — it applies
    standard N-terminal templates that try to place H2/H3 on the N atom, which
    fails when the N is already bonded to the C-terminus carbon.  This function
    uses the cyclic-aware ForceField templates instead.
    """
    from openmm.app import ForceField, Modeller

    from binding_metrics.core.cyclic import (
        get_addh_variants,
        load_extra_xmls,
        patch_cyclic_topology,
        rename_disulfide_cys_to_cyx,
    )
    from binding_metrics.core.gaff_ncaa import parameterize_ncaa_residues
    from binding_metrics.core.nonstandard import (
        detect_nonstandard,
        load_nonstandard_xmls,
        patch_nonstandard,
    )

    # custom bonds are intra-chain, so both ends share the same chain ID.
    cyclic_chain = custom_bonds[0][0]

    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    # Phosphorylated residues → AMBER phosaa params (net −2), not GAFF.
    from binding_metrics.core import phosaa

    phosaa.register(ff)
    phosaa.ensure_hydrogen_definitions()

    # D-amino-acid / N-methyl rename first (e.g. DAL→ALA, SAR→NMG) so their
    # standard/curated templates match; must precede patch_cyclic_topology.
    ns_info = detect_nonstandard(topology, cyclic_chain)
    if not ns_info.is_empty:
        topology, positions = patch_nonstandard(topology, positions, cyclic_chain, ns_info)
        load_nonstandard_xmls(ff, ns_info)

    # patch_cyclic_topology detects the cyclic bond by distance (works even
    # without the bond already in the topology), removes C-terminal OXT and
    # terminal H atoms that PDBFixer added, and adds the N-C closure bond.
    topology, positions, bond_info = patch_cyclic_topology(topology, positions, cyclic_chain)

    # Rename any remaining SS-bonded CYS → CYX.  patch_cyclic_topology only
    # handles intra-chain disulfides; this catches inter-chain ones (e.g.
    # peptide–receptor SS bonds) that would otherwise fail in addHydrogens.
    topology, positions = rename_disulfide_cys_to_cyx(topology, positions)

    if bond_info:
        load_extra_xmls(ff, bond_info)

    # GAFF2 ExternalBond templates for exotic NCAAs (BMT/ABA/…): generates and
    # loads their templates and injects their hydrogens so addHydrogens (whose
    # internal createSystem would otherwise fail on "No template") succeeds.
    topology, positions, _ncaa_xmls = parameterize_ncaa_residues(topology, positions, ff)

    modeller = Modeller(topology, positions)
    addh_variants = (
        get_addh_variants(modeller.topology, bond_info, cyclic_chain) if bond_info else None
    )
    with deterministic_hydrogen_placement(random_seed):
        modeller.addHydrogens(
            ff,
            pH=ph,
            variants=addh_variants,
            platform=_hydrogen_placement_platform(),
        )
    return modeller.topology, modeller.positions


def _hydrogen_placement_platform():
    """The Reference platform: double-precision, single-threaded, deterministic.

    ``addHydrogens`` runs a short energy minimization to settle the new
    hydrogens. On a fast platform (CUDA/OpenCL) the reduction order is not
    deterministic, so that minimization lands in a marginally different spot each
    run — hydrogens on rotatable groups shift by up to ~0.25 Å, which is enough
    to tip the downstream main minimization into a different basin and change the
    reported energy by hundreds of kJ/mol. Running the tiny H minimization on the
    Reference platform makes prep bit-reproducible regardless of installed
    hardware; it is cheap (a few dozen steps on one small system).
    """
    from openmm import Platform

    return Platform.getPlatformByName("Reference")


@contextlib.contextmanager
def deterministic_hydrogen_placement(seed: Optional[int] = DEFAULT_RANDOM_SEED):
    """Make ``addHydrogens`` reproducible for the duration of the block.

    ``Modeller.addHydrogens`` offsets every new hydrogen by
    ``0.05 nm * Vec3(random(), random(), random())`` drawn from Python's global
    ``random`` module, which nobody seeds. Identical input therefore yields a
    slightly different structure on every run, and hence a different minimum:
    1YCR has been observed anywhere between about -14.0k and -14.6k kJ/mol
    across runs. For a tool whose output is a QC *measurement*, that
    irreproducibility is a defect in its own right.

    Seeding fixes the placement so the same input gives the same answer. The
    previous RNG state is restored on exit, so seeding here never perturbs
    randomness elsewhere in the caller's process. Pass ``seed=None`` to leave
    the global RNG untouched and get fresh randomness (opt-in via the configs'
    ``random_seed``).

    The same block also seeds any other OpenMM ``Modeller`` step that draws from
    the global ``random`` module, notably the ion placement in
    ``Modeller.addSolvent`` (see :func:`solvate`).
    """
    if seed is None:
        yield
        return

    import random

    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def repair_ca_hydrogen_chirality(topology, positions, verbose: bool = True):
    """Move Cα hydrogens that hydrogen addition placed on the wrong face.

    ``Modeller.addHydrogens`` seeds each new hydrogen at a 0.1 nm base vector
    plus ``0.05 nm * Vec3(random(), random(), random())`` drawn from Python's
    *unseeded* global ``random``, then relaxes the hydrogens with every heavy
    atom frozen (``setParticleMass(i, 0)``). When the jitter pushes HA across
    the N/CA/C plane, that frozen-heavy-atom relaxation cannot bring it back:
    HA settles into the wrong-side minimum and prep emits a Cα whose HA and CB
    share a face, which is chemically impossible.

    Left alone, this silently inverts the stereocenter downstream: the bad HA
    contributes almost all of the Cα's angle strain, ``constraints=HBonds``
    fixes the CA-HA length so the minimizer cannot relieve it by moving HA, and
    ff14SB has no improper on Cα — so the only path left is pushing CB through
    the plane. Observed on roughly one prep in six of the 3P8F example.

    For any tetrahedral Cα, CB and HA lie on opposite sides of the N/CA/C
    plane. That holds for L- and D-amino acids alike, so this repair reads the
    correct side off the actual CB position and is handedness-agnostic (safe for
    D-residues). It is a no-op on correct structures.
    """
    import numpy as np
    from openmm import Vec3, unit

    pos = np.array(positions.value_in_unit(unit.nanometer))
    repaired = []
    for res in topology.residues():
        idx = {a.name: a.index for a in res.atoms()}
        if not {"N", "CA", "C", "CB", "HA"} <= set(idx):
            continue  # e.g. GLY (no CB) has no Cα stereocenter
        ca = pos[idx["CA"]]
        n, c, cb, ha = (pos[idx[k]] for k in ("N", "C", "CB", "HA"))
        v_cb = np.dot(n - ca, np.cross(c - ca, cb - ca))
        v_ha = np.dot(n - ca, np.cross(c - ca, ha - ca))
        if (v_cb > 0) != (v_ha > 0):
            continue  # opposite faces — correct
        u = sum((x - ca) / np.linalg.norm(x - ca) for x in (n, c, cb))
        norm = np.linalg.norm(u)
        if norm < _DEGENERATE_TRIPOD_NORM:
            continue  # degenerate planar tripod
        pos[idx["HA"]] = ca - u / norm * _CA_HA_BOND_NM  # ideal 4th tetrahedral vertex
        repaired.append(f"{res.name}{res.id}/{res.chain.id}")

    if repaired and verbose:
        print(f"  Repaired {len(repaired)} wrong-side Cα hydrogen(s): {', '.join(repaired)}")
    # Vec3, not bare tuples: downstream consumers index positions as p.x/p.y/p.z.
    return unit.Quantity([Vec3(*map(float, p)) for p in pos], unit.nanometer)


def prep_structure(
    topology,
    positions,
    ph: float = 7.4,
    keep_water: bool = False,
    canonicalize: bool = False,
    rebuild_zero_coord_atoms: bool = True,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    report: Optional[dict] = None,
) -> tuple:
    """Fix missing residues/atoms and add hydrogens in one PDBFixer pass.

    Args:
        topology: OpenMM Topology
        positions: OpenMM positions
        ph: pH for hydrogen placement (default 7.4)
        keep_water: If True, retain crystallographic water molecules
        canonicalize: If True, replace non-standard residues with their nearest
            standard equivalents (e.g. MSE→MET, SEP→SER, acetyl-LYS→LYS).
            If False (default), non-standard residues and non-canonical amino
            acids are preserved so they can be parameterised downstream with
            GAFF2 (``--small-molecules auto`` in binding-metrics-relax).
        rebuild_zero_coord_atoms: If True (default), detect atoms placed at the
            origin (zero-coordinate placeholders from pipelines that skip atom
            modelling) and let PDBFixer rebuild them via findMissingAtoms().
            Set to False when the input is known to be fully modelled.
        random_seed: Seed for the stochastic steps of prep (hydrogen-placement
            jitter and PDBFixer's atom-rebuild minimization). A fixed int (the
            default) makes prep reproducible; ``None`` opts into fresh
            randomness.
        report: Optional dict filled in place with what prep changed. Lists
            and counts accumulate when one dict is passed to several calls.
            Keys:

            * ``removed_heterogens`` (list[str]): ``"NAME (chain X)"`` for each
              removed non-water heterogen (free ligands, additives, glycans).
            * ``n_removed_waters`` (int): water molecules removed (0 when
              ``keep_water`` is True).
            * ``kept_nonstandard`` (list[str]): non-standard residues and metal
              ions that were kept.
            * ``n_missing_atoms_rebuilt`` (int): heavy atoms PDBFixer added,
              including atoms deleted as origin placeholders and terminal OXT.
            * ``n_missing_residue_gaps`` (int): chain positions where residues
              are unresolved and were left as a gap, not rebuilt.

            Behaviour is identical when ``report`` is None.

    Returns:
        Tuple of (topology, positions) with repaired and protonated structure
    """
    # Capture non-sequential intra-chain bonds (e.g. head-to-tail N→C) before
    # the PDBFixer round-trip drops them (PDBxFile.writeFile only writes SS bonds).
    # SS bonds are NOT captured here — they are re-detected from rebuilt geometry
    # after addMissingAtoms via _rebuild_connect_records.
    custom_bonds = _extract_custom_bonds(topology)

    # Residue numbers and chain IDs do not survive the PDBFixer round trip, so
    # read the numbering gaps and the caller's chain IDs off the input topology.
    input_chain_ids = [chain.id for chain in topology.chains()]
    n_residue_gaps = _count_residue_gaps(topology, positions)

    fixer = _topology_to_fixer(topology, positions)

    if rebuild_zero_coord_atoms:
        fixer = _delete_zero_coord_atoms(fixer)

    fixer.findMissingResidues()
    # PDBFixer's own list is empty on this path (see _count_residue_gaps) but
    # would be authoritative if sequence records existed.
    n_residue_gaps = max(n_residue_gaps, len(fixer.missingResidues))
    fixer.findNonstandardResidues()

    if canonicalize:
        fixer.replaceNonstandardResidues()
        log.info("--canonicalize: non-standard residues replaced with standard equivalents.")
    else:
        nonstandard = getattr(fixer, "nonstandardResidues", [])
        if nonstandard:
            names = ", ".join(f"{r.name}" for r, _ in nonstandard)
            log.info("Non-standard residues detected: %s", names)
            log.info(
                "These will be preserved. Use --small-molecules auto in relax to "
                "parameterise them with GAFF2, or --canonicalize to replace them."
            )

    # Chain-aware heterogen filter — replaces PDBFixer's removeHeterogens():
    #   • Standard AA / nucleotide          → always keep
    #   • Metal ion (by element)            → always keep (ff14SB has parameters)
    #   • Non-standard residue in a protein chain → keep (non-canonical AA)
    #   • Water                             → keep if keep_water, else remove
    #   • Everything else (free ligands, crystallographic additives, glycans
    #     in their own chain …)             → remove
    fixer.findMissingAtoms()
    # Seeded: addMissingAtoms minimizes rebuilt atoms with a stochastic
    # integrator, so an unseeded call makes prep irreproducible for any
    # structure with missing side-chain atoms. addMissingAtoms(seed=None) leaves
    # the integrator unseeded (fresh randomness), matching random_seed=None.
    n_atoms_before_rebuild = fixer.topology.getNumAtoms()
    fixer.addMissingAtoms(seed=random_seed)
    n_atoms_rebuilt = fixer.topology.getNumAtoms() - n_atoms_before_rebuild

    # Detect and register all non-standard bonds from the rebuilt geometry
    # (SS bonds, CYS→CYX rename). Must run after addMissingAtoms so that
    # zero-coord placeholder atoms are in their correct positions.
    _rebuild_connect_records(fixer)

    protein_chains: set = set()
    for chain in fixer.topology.chains():
        for res in chain.residues():
            if res.name in _STANDARD_RESIDUES:
                protein_chains.add(chain.id)
                break

    residues_to_remove = []
    kept_nonstandard: list = []
    removed_heterogens: list = []
    n_removed_waters = 0

    # PDBFixer regenerates chain IDs (A, B, C, ...); name chains by the caller's IDs
    # when the chain count survived the round trip.
    keep_input_ids = fixer.topology.getNumChains() == len(input_chain_ids)

    for chain in fixer.topology.chains():
        chain_label = input_chain_ids[chain.index] if keep_input_ids else chain.id
        for res in chain.residues():
            if res.name in _STANDARD_RESIDUES:
                continue

            elements = {atom.element.symbol for atom in res.atoms() if atom.element is not None}
            if elements & _METAL_ELEMENTS:
                kept_nonstandard.append(f"{res.name} (metal, chain {chain_label})")
                continue

            if res.name in _WATER_NAMES:
                if not keep_water:
                    residues_to_remove.append(res)
                    n_removed_waters += 1
                continue

            if chain.id in protein_chains:
                kept_nonstandard.append(f"{res.name} (chain {chain_label})")
                continue

            removed_heterogens.append(f"{res.name} (chain {chain_label})")
            residues_to_remove.append(res)

    if kept_nonstandard:
        log.info("Kept non-standard residues: %s", ", ".join(kept_nonstandard))
    if removed_heterogens:
        log.info("Removed heterogens: %s", ", ".join(removed_heterogens))

    if report is not None:
        extend_report(report, "removed_heterogens", removed_heterogens)
        add_to_report(report, "n_removed_waters", n_removed_waters)
        extend_report(report, "kept_nonstandard", kept_nonstandard)
        add_to_report(report, "n_missing_atoms_rebuilt", n_atoms_rebuilt)
        add_to_report(report, "n_missing_residue_gaps", n_residue_gaps)

    if residues_to_remove:
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete(residues_to_remove)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

    if custom_bonds:
        # Use cyclic-aware H placement: patch_cyclic_topology (called inside)
        # removes C-terminal OXT that PDBFixer added, adds the N-C closure bond,
        # then uses cyclic FF templates for addHydrogens.  Do NOT restore bonds
        # here — patch_cyclic_topology detects and adds the bond itself.
        result_topo, result_pos = _add_hydrogens_cyclic(
            fixer.topology, fixer.positions, custom_bonds, ph, random_seed=random_seed
        )
    else:
        # Inline of PDBFixer.addMissingHydrogens so we can pin the H-placement
        # minimization to the deterministic Reference platform (the method itself
        # exposes no platform argument).
        _h_modeller = Modeller(fixer.topology, fixer.positions)
        with deterministic_hydrogen_placement(random_seed):
            _h_modeller.addHydrogens(pH=ph, platform=_hydrogen_placement_platform())
        result_topo, result_pos = _h_modeller.topology, _h_modeller.positions

    # addHydrogens jitters new H randomly and relaxes them with the heavy atoms
    # frozen, which intermittently strands a Cα H on the wrong face; left in
    # place it inverts the stereocenter during minimization.
    result_pos = repair_ca_hydrogen_chirality(result_topo, result_pos)

    return result_topo, result_pos


def solvate(
    topology,
    positions,
    forcefield: ForceField | None = None,
    forcefield_name: Literal["amber", "charmm"] = "amber",
    padding: float = 1.0,
    ionic_strength: float = 0.15,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    *,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
) -> Modeller:
    """Add explicit solvent and ions with periodic boundary conditions.

    Ion placement is stochastic: ``Modeller.addSolvent`` replaces randomly
    chosen water molecules with ions, drawing from Python's global ``random``.
    This step is seeded, so the same input and seed give the same ion positions.

    Args:
        topology: OpenMM Topology
        positions: OpenMM positions
        forcefield: Pre-configured ForceField. If None, uses forcefield_name.
        forcefield_name: Force field to use if forcefield is None
        padding: Distance in nm between solute and box edge
        ionic_strength: Salt concentration in M
        positive_ion: Positive ion type
        negative_ion: Negative ion type
        random_seed: Seed for ion placement. A fixed int (the default) makes
            the placement reproducible; ``None`` opts into fresh randomness.

    Returns:
        Modeller with solvated and ionized system
    """
    if forcefield is None:
        forcefield = get_forcefield(forcefield_name)
    modeller = Modeller(topology, positions)
    with deterministic_hydrogen_placement(random_seed):
        modeller.addSolvent(
            forcefield,
            padding=padding * unit.nanometer,
            ionicStrength=ionic_strength * unit.molar,
            positiveIon=positive_ion,
            negativeIon=negative_ion,
        )
    return modeller


def prepare_system(
    pdb: PDBFile,
    forcefield: ForceField | None = None,
    forcefield_name: Literal["amber", "charmm"] = "amber",
    padding: float = 1.0,
    ionic_strength: float = 0.15,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    fix: bool = True,
    ph: float = 7.4,
    *,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
) -> Modeller:
    """Prepare a molecular system for simulation: fix+protonate → solvate.

    Hydrogen placement, atom rebuilding and ion placement are seeded by
    ``random_seed``, so the same input gives the same prepared system.

    Args:
        pdb: Loaded PDB file with the molecular structure
        forcefield: Pre-configured ForceField object. If None, uses forcefield_name.
        forcefield_name: Name of force field to use if forcefield is None
        padding: Distance in nm between solute and box edge
        ionic_strength: Salt concentration in M (molar)
        positive_ion: Positive ion type for neutralization
        negative_ion: Negative ion type for neutralization
        fix: If True and pdbfixer is available, fix missing atoms and protonate
        ph: pH for hydrogen placement when fix=True (default 7.4)
        random_seed: Seed for every stochastic step (hydrogen placement, PDBFixer
            atom rebuild, ion placement). A fixed int (the default) makes the
            preparation reproducible; ``None`` opts into fresh randomness.

    Returns:
        Modeller object with solvated and ionized system
    """
    topology, positions = pdb.topology, pdb.positions

    if fix and HAS_PDBFIXER:
        topology, positions = prep_structure(topology, positions, ph=ph, random_seed=random_seed)
    else:
        ff = forcefield if forcefield is not None else get_forcefield(forcefield_name)
        tmp_modeller = Modeller(topology, positions)
        with deterministic_hydrogen_placement(random_seed):
            tmp_modeller.addHydrogens(ff, platform=_hydrogen_placement_platform())
        topology, positions = tmp_modeller.topology, tmp_modeller.positions

    return solvate(
        topology,
        positions,
        forcefield=forcefield,
        forcefield_name=forcefield_name,
        padding=padding,
        ionic_strength=ionic_strength,
        positive_ion=positive_ion,
        negative_ion=negative_ion,
        random_seed=random_seed,
    )


def get_system_info(modeller: Modeller) -> dict:
    """Get information about the prepared system.

    Args:
        modeller: Prepared Modeller object

    Returns:
        Dictionary with system information
    """
    topology = modeller.topology
    n_atoms = topology.getNumAtoms()
    n_residues = topology.getNumResidues()
    n_chains = topology.getNumChains()

    # Count water molecules and ions
    n_waters = 0
    n_ions = 0
    ion_names = {"NA", "CL", "K", "MG", "CA", "ZN"}

    for residue in topology.residues():
        if residue.name == "HOH" or residue.name == "WAT":
            n_waters += 1
        elif residue.name in ion_names:
            n_ions += 1

    # Get box vectors
    box_vectors = modeller.topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        box_size = [v[i].value_in_unit(unit.nanometer) for i, v in enumerate(box_vectors)]
    else:
        box_size = None

    return {
        "n_atoms": n_atoms,
        "n_residues": n_residues,
        "n_chains": n_chains,
        "n_waters": n_waters,
        "n_ions": n_ions,
        "box_size_nm": box_size,
    }
