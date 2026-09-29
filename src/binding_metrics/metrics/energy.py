"""Peptide-receptor interaction energies.

Two families of functions:

* ``calculate_interaction_energy`` / ``calculate_component_energies``: per-frame
  Coulomb + Lennard-Jones sums over ligand-receptor atom pairs of a trajectory
  (vacuum, no cutoff).
* ``compute_interaction_energy`` (CLI: ``binding-metrics-energy``): end-state
  subsystem decomposition E_complex - E_peptide - E_receptor in implicit solvent
  for a single structure, evaluated at the input geometry (``raw``), after a
  minimisation (``relaxed``) and after a short MD run (``after_md``).

All energies are in kJ/mol. Negative interaction energies are favourable.

References:
    Maier et al., J. Chem. Theory Comput. 11, 3696 (2015): ff14SB (force field).
    Onufriev, Bashford & Case, Proteins 55, 383 (2004): OBC generalized Born.
    Nguyen, Roe & Simmerling, J. Chem. Theory Comput. 9, 2020 (2013): GBn2.
    Kollman et al., Acc. Chem. Res. 33, 889 (2000): end-state decomposition of
        implicit-solvent energies (MM-PBSA/GBSA).
"""

import argparse
import traceback
import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np

try:
    import mdtraj as md
except ImportError:
    md = None

# OpenMM is imported inside the functions that use it, so this module can be
# imported (and its CLI parser built) on installs without OpenMM.
try:
    from binding_metrics.core.system import DEFAULT_RANDOM_SEED
except ImportError:
    # core.system imports OpenMM at module level. Without OpenMM the value is
    # duplicated here; tests/test_l6_import.py checks that the two stay equal.
    DEFAULT_RANDOM_SEED = 1

# 1 / (4 pi eps0) in OpenMM units (kJ nm mol^-1 e^-2).
_COULOMB_K_KJ_NM_MOL_E2 = 138.935456
# Ligand-receptor pairs closer than this (0.1 A) are skipped so that coincident
# atoms cannot cause a division by zero.
_MIN_PAIR_DISTANCE_NM = 0.01
# Peptide-receptor atom pairs counted as "contacts" / "close contacts" in the
# result of compute_interaction_energy.
_CONTACT_CUTOFF_ANGSTROM = 8.0
_CLOSE_CONTACT_CUTOFF_ANGSTROM = 4.0
# Backbone atoms held by a harmonic restraint during the first minimisation
# stage. 100 kJ/mol/nm^2 is soft: a 0.1 nm displacement costs 0.5 kJ/mol.
_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O"})
_BACKBONE_RESTRAINT_K_KJ_MOL_NM2 = 100.0


def __getattr__(name: str):
    """Resolve the OpenMM names this module used to import eagerly (PEP 562)."""
    if name == "openmm":
        import openmm

        return openmm
    if name == "unit":
        import openmm.unit as unit

        return unit
    if name in ("ForceField", "PDBFile"):
        import openmm.app

        return getattr(openmm.app, name)
    if name == "get_forcefield":
        from binding_metrics.core.forcefields import get_forcefield

        return get_forcefield
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def calculate_interaction_energy(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    forcefield_name: Literal["amber", "charmm"] = "amber",
) -> np.ndarray:
    """Calculate ligand-receptor interaction energy for each frame.

    For pairwise-additive non-bonded terms, E_complex - E_ligand - E_receptor
    reduces to the sum over all ligand x receptor atom pairs of

        E_ij = k q_i q_j / r_ij + 4 eps_ij [(sigma_ij / r_ij)^12 - (sigma_ij / r_ij)^6]

    with k = 1 / (4 pi eps0) and Lorentz-Berthelot combining rules
    (sigma_ij = (sigma_i + sigma_j) / 2, eps_ij = sqrt(eps_i eps_j)), which
    is how the pairs are evaluated here. Charges and LJ parameters come from the
    force field's NonbondedForce (vacuum, no cutoff, no solvent).

    Exclusions and 1-4 scaling are not applied to ligand-receptor pairs. Pairs
    bonded through a covalent ligand-receptor link (for example an inter-chain
    disulfide) therefore contribute at full strength.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file (PDB: read with OpenMM's PDBFile)
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor
        forcefield_name: Force field for energy calculation

    Returns:
        Array of shape (n_frames,) with the interaction energy in kJ/mol for
        each frame. Negative values are favourable.

    Raises:
        ImportError: If mdtraj is not installed.
        RuntimeError: If the force field system has no NonbondedForce.
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for energy calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    import openmm
    import openmm.unit as unit
    from openmm.app import PDBFile

    from binding_metrics.core.forcefields import get_forcefield

    traj = md.load(str(trajectory_path), top=str(topology_path))

    pdb = PDBFile(str(topology_path))
    forcefield = get_forcefield(forcefield_name)

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=openmm.app.NoCutoff,
        constraints=None,
    )

    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
            break

    if nonbonded_force is None:
        raise RuntimeError("No NonbondedForce found in system")

    interaction_energies = []

    for frame_idx in range(traj.n_frames):
        positions = traj.xyz[frame_idx]  # nm

        energy = 0.0

        for lig_idx in ligand_indices:
            q1, sig1, eps1 = nonbonded_force.getParticleParameters(lig_idx)
            q1 = q1.value_in_unit(unit.elementary_charge)
            sig1 = sig1.value_in_unit(unit.nanometer)
            eps1 = eps1.value_in_unit(unit.kilojoule_per_mole)

            for rec_idx in receptor_indices:
                q2, sig2, eps2 = nonbonded_force.getParticleParameters(rec_idx)
                q2 = q2.value_in_unit(unit.elementary_charge)
                sig2 = sig2.value_in_unit(unit.nanometer)
                eps2 = eps2.value_in_unit(unit.kilojoule_per_mole)

                r = np.linalg.norm(positions[lig_idx] - positions[rec_idx])

                if r < _MIN_PAIR_DISTANCE_NM:
                    continue

                e_coulomb = _COULOMB_K_KJ_NM_MOL_E2 * q1 * q2 / r

                sigma = (sig1 + sig2) / 2
                epsilon = np.sqrt(eps1 * eps2)
                if epsilon > 0 and sigma > 0:
                    sr6 = (sigma / r) ** 6
                    e_lj = 4 * epsilon * (sr6**2 - sr6)
                else:
                    e_lj = 0.0

                energy += e_coulomb + e_lj

        interaction_energies.append(energy)

    return np.array(interaction_energies)


def calculate_component_energies(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    forcefield_name: Literal["amber", "charmm"] = "amber",
) -> dict[str, np.ndarray]:
    """Calculate separated electrostatic and vdW interaction energies.

    Same pair sum, force-field parameters and caveats as
    ``calculate_interaction_energy``, with the Coulomb and Lennard-Jones terms
    returned separately.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file (PDB: read with OpenMM's PDBFile)
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor
        forcefield_name: Force field for energy calculation

    Returns:
        Dictionary of arrays of shape (n_frames,), all in kJ/mol:
            electrostatic: Coulomb term.
            vdw: Lennard-Jones term.
            total: electrostatic + vdw.

    Raises:
        ImportError: If mdtraj is not installed.
        RuntimeError: If the force field system has no NonbondedForce.
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for energy calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    import openmm
    import openmm.unit as unit
    from openmm.app import PDBFile

    from binding_metrics.core.forcefields import get_forcefield

    traj = md.load(str(trajectory_path), top=str(topology_path))
    pdb = PDBFile(str(topology_path))
    forcefield = get_forcefield(forcefield_name)

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=openmm.app.NoCutoff,
        constraints=None,
    )

    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
            break

    if nonbonded_force is None:
        raise RuntimeError("No NonbondedForce found in system")

    coulomb_energies = []
    lj_energies = []

    for frame_idx in range(traj.n_frames):
        positions = traj.xyz[frame_idx]
        e_coulomb_total = 0.0
        e_lj_total = 0.0

        for lig_idx in ligand_indices:
            q1, sig1, eps1 = nonbonded_force.getParticleParameters(lig_idx)
            q1 = q1.value_in_unit(unit.elementary_charge)
            sig1 = sig1.value_in_unit(unit.nanometer)
            eps1 = eps1.value_in_unit(unit.kilojoule_per_mole)

            for rec_idx in receptor_indices:
                q2, sig2, eps2 = nonbonded_force.getParticleParameters(rec_idx)
                q2 = q2.value_in_unit(unit.elementary_charge)
                sig2 = sig2.value_in_unit(unit.nanometer)
                eps2 = eps2.value_in_unit(unit.kilojoule_per_mole)

                r = np.linalg.norm(positions[lig_idx] - positions[rec_idx])
                if r < _MIN_PAIR_DISTANCE_NM:
                    continue

                e_coulomb_total += _COULOMB_K_KJ_NM_MOL_E2 * q1 * q2 / r

                sigma = (sig1 + sig2) / 2
                epsilon = np.sqrt(eps1 * eps2)
                if epsilon > 0 and sigma > 0:
                    sr6 = (sigma / r) ** 6
                    e_lj_total += 4 * epsilon * (sr6**2 - sr6)

        coulomb_energies.append(e_coulomb_total)
        lj_energies.append(e_lj_total)

    elec = np.array(coulomb_energies)
    vdw = np.array(lj_energies)

    return {
        "electrostatic": elec,
        "vdw": vdw,
        "total": elec + vdw,
    }


# ---------------------------------------------------------------------------
# Subsystem decomposition: E_complex - E_peptide - E_receptor
# ---------------------------------------------------------------------------


def _create_implicit_system(
    topology,
    positions,
    solvent_model: str = "obc2",
    peptide_chain: Optional[str] = None,
    ph: float = 7.4,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
):
    """Create an OpenMM implicit-solvent system after adding hydrogens.

    Force field: AMBER ff14SB (``amber14-all.xml``) plus a generalized Born model,
    OBC2 (Onufriev, Bashford & Case 2004) or GBn2 (Nguyen, Roe & Simmerling 2013),
    from OpenMM's ``implicit/*.xml``. These use OpenMM's defaults of solute
    dielectric 1 and solvent dielectric 78.5, and include its "ACE" nonpolar
    surface-area term. No cutoff; X-H bonds are constrained.

    Cyclic topology is always detected and patched before addHydrogens.
    peptide_chain must be resolved by the caller (auto-detection happens once
    upstream so all subsequent steps receive the same resolved chain ID).

    Args:
        topology: OpenMM Topology, heterogens already stripped.
        positions: Atom positions (Quantity, nm).
        solvent_model: 'obc2' (default) or 'gbn2'.
        peptide_chain: Chain ID of the peptide, used for cyclic and
            non-standard residue handling.
        ph: pH for choosing protonation states in ``Modeller.addHydrogens``.
            If that call raises, it is repeated without ``pH``, which is
            OpenMM's default of 7.0.
        random_seed: Seed for hydrogen placement (OpenMM draws the initial
            hydrogen offsets from Python's global RNG); None leaves it unseeded.

    Returns:
        Tuple of (system, topology_with_h, positions_with_h, bond_info, ncaa_xmls)
        where ncaa_xmls is a list of GAFF2 residue-template XML strings generated
        for non-canonical residues (empty when there are none). Callers that build
        separate subsystem force fields must reload these XMLs.
    """
    import openmm
    from openmm.app import ForceField

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

    gb_file = "implicit/gbn2.xml" if solvent_model == "gbn2" else "implicit/obc2.xml"
    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml", gb_file)

    # Phosphorylated residues use the AMBER phosaa parameters (net -2), not GAFF.
    from binding_metrics.core import phosaa

    phosaa.register(ff)
    phosaa.ensure_hydrogen_definitions()

    # Accumulate every extra residue-template XML used to build the complex so
    # the peptide/receptor subsystems (which build their own force fields) can
    # reload them: curated N-methyl templates + GAFF NCAA templates. (Lactam
    # closure templates ride along in bond_info via load_extra_xmls.)
    extra_xmls: list = []

    # D-amino-acid / N-methyl residues are renamed to their L / parent template
    # names (no-op when prep or relaxation already did it).
    if peptide_chain is not None:
        ns_info = detect_nonstandard(topology, peptide_chain)
        if not ns_info.is_empty:
            topology, positions = patch_nonstandard(topology, positions, peptide_chain, ns_info)
            load_nonstandard_xmls(ff, ns_info)
            extra_xmls.extend(ns_info.extra_ff_xmls)

    topology, positions, bond_info = patch_cyclic_topology(topology, positions, peptide_chain)
    topology, positions = rename_disulfide_cys_to_cyx(topology, positions)
    if bond_info:
        load_extra_xmls(ff, bond_info)

    # GAFF2 ExternalBond templates for exotic NCAAs (BMT/ABA/…). This rebuilds
    # the topology to inject their hydrogens, so it must run before addHydrogens.
    topology, positions, ncaa_xmls = parameterize_ncaa_residues(topology, positions, ff)
    extra_xmls.extend(ncaa_xmls)

    from openmm.app import Modeller

    modeller = Modeller(topology, positions)
    addh_variants = None
    if bond_info:
        addh_variants = get_addh_variants(modeller.topology, bond_info, peptide_chain)
    from binding_metrics.core.system import deterministic_hydrogen_placement

    try:
        with deterministic_hydrogen_placement(random_seed):
            modeller.addHydrogens(ff, pH=ph, variants=addh_variants)
    except (ValueError, openmm.OpenMMException) as e:
        print(f"  Warning: addHydrogens with pH failed ({e}), retrying without pH")
        with deterministic_hydrogen_placement(random_seed):
            modeller.addHydrogens(ff, variants=addh_variants)

    # A wrong-side Cα H (random jitter + frozen heavy atoms during H addition)
    # perturbs the energy decomposition. No-op on already-clean structures.
    from binding_metrics.core.system import repair_ca_hydrogen_chirality

    modeller.positions = repair_ca_hydrogen_chirality(modeller.topology, modeller.positions)

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=openmm.app.NoCutoff,
        constraints=openmm.app.HBonds,
    )
    return system, modeller.topology, modeller.positions, bond_info, extra_xmls


def _repair_orphaned_cys(
    topology, positions, solvent_model: str = "obc2", label: str = ""
) -> tuple:
    """Add HG back to CYS/CYX residues whose disulfide partner was severed.

    When the complex has a peptide–receptor disulfide, addHydrogens treats the
    bonded CYS as CYX (no HG). After _extract_chain drops the cross-chain SS bond,
    the residue has no HG and no SS partner — matching neither CYS nor CYX template.

    Also handles CYX residues (already renamed from CYS) that lost their SS partner
    during chain extraction: CYX with no SS bond needs to be renamed back to CYS and
    have HG added so OpenMM can find the correct template.

    Bypasses addHydrogens (which requires template matching before H placement)
    and directly inserts HG into a rebuilt topology at a geometric position along
    the CB→SG bond direction.

    Args:
        topology: OpenMM Topology of an extracted chain (hydrogens present).
        positions: Positions (Quantity in nm, or an array in nm).
        solvent_model: Unused; kept so callers pass the same arguments as for
            the other subsystem helpers.
        label: Prefix for the printed message.

    Returns:
        (topology, positions) with HG added to orphaned cysteines; the inputs
        are returned unchanged when there is none.
    """
    import openmm.unit as unit
    from openmm.app import Element, Topology

    _SH_BOND_NM = 0.134  # S–H bond length (1.34 Å)

    orphans: dict = {}  # res.index -> (sg_atom_index, cb_atom_index_or_None)
    for res in topology.residues():
        if res.name not in ("CYS", "CYX"):
            continue
        sg = next((a for a in res.atoms() if a.name == "SG"), None)
        if sg is None or any(a.name == "HG" for a in res.atoms()):
            continue
        has_ss = any(
            (b.atom1 is sg or b.atom2 is sg)
            and (b.atom2 if b.atom1 is sg else b.atom1).element.symbol == "S"
            for b in topology.bonds()
        )
        if not has_ss:
            cb = next((a for a in res.atoms() if a.name == "CB"), None)
            orphans[res.index] = (sg.index, cb.index if cb else None)

    if not orphans:
        return topology, positions

    names = [f"{r.name}{r.id}" for r in topology.residues() if r.index in orphans]
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}  Repairing orphaned CYS (cross-chain disulfide severed): {names}")

    H_element = Element.getBySymbol("H")
    try:
        pos_array = np.array([[p.x, p.y, p.z] for p in positions])
    except AttributeError:
        pos_array = np.asarray(positions)  # already a numpy array (from _extract_chain)

    new_topo = Topology()
    old_to_new: dict = {}
    new_pos_list: list = []
    sg_new_for_res: dict = {}  # res.index -> new SG Atom
    hg_new_for_res: dict = {}  # res.index -> new HG Atom

    for chain in topology.chains():
        new_chain = new_topo.addChain(chain.id)
        for res in chain.residues():
            # CYX with no SS bond (orphaned) needs to become CYS once HG is added
            res_name = "CYS" if res.index in orphans and res.name == "CYX" else res.name
            new_res = new_topo.addResidue(res_name, new_chain)
            for atom in res.atoms():
                new_atom = new_topo.addAtom(atom.name, atom.element, new_res)
                old_to_new[atom.index] = new_atom
                new_pos_list.append(pos_array[atom.index])
                if atom.name == "SG":
                    sg_new_for_res[res.index] = new_atom

            if res.index in orphans:
                sg_idx, cb_idx = orphans[res.index]
                hg_atom = new_topo.addAtom("HG", H_element, new_res)
                hg_new_for_res[res.index] = hg_atom
                sg_pos = pos_array[sg_idx]
                if cb_idx is not None:
                    direction = sg_pos - pos_array[cb_idx]
                    norm = float(np.linalg.norm(direction))
                    direction = direction / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
                else:
                    direction = np.array([0.0, 0.0, 1.0])
                new_pos_list.append(sg_pos + _SH_BOND_NM * direction)

    for bond in topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        if a1.index in old_to_new and a2.index in old_to_new:
            new_topo.addBond(old_to_new[a1.index], old_to_new[a2.index])
    for res_idx in orphans:
        new_topo.addBond(sg_new_for_res[res_idx], hg_new_for_res[res_idx])

    new_pos = unit.Quantity(np.array(new_pos_list), unit.nanometers)
    return new_topo, new_pos


def _build_subsystem(topology, solvent_model: str = "obc2", bond_info=None, ncaa_xmls=None):
    """Build an implicit-solvent system for a topology that already has hydrogens.

    Same force field as ``_create_implicit_system``, without hydrogen addition.
    ``bond_info`` (lactam / cyclic closure templates) and ``ncaa_xmls`` (GAFF2
    residue templates) are the values returned by ``_create_implicit_system``;
    a subsystem that contains none of those residues can omit them.
    """
    import openmm
    from openmm.app import ForceField

    gb_file = "implicit/gbn2.xml" if solvent_model == "gbn2" else "implicit/obc2.xml"
    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml", gb_file)
    from binding_metrics.core import phosaa

    phosaa.register(ff)
    if bond_info:
        from binding_metrics.core.cyclic import load_extra_xmls

        load_extra_xmls(ff, bond_info)
    if ncaa_xmls:
        from binding_metrics.core.gaff_ncaa import _load_ffxml

        for xml_str in ncaa_xmls:
            _load_ffxml(ff, xml_str)
    return ff.createSystem(
        topology,
        nonbondedMethod=openmm.app.NoCutoff,
        constraints=openmm.app.HBonds,
    )


def _get_platform(device: str = "cuda"):
    """Get the best available OpenMM platform.

    Returns:
        (platform, properties). CUDA uses "mixed" precision, which is not
        bit-for-bit reproducible between runs. Falls back to CPU when device is
        not "cuda" or the CUDA platform is unavailable.
    """
    import openmm

    if device == "cuda":
        try:
            platform = openmm.Platform.getPlatformByName("CUDA")
            return platform, {"CudaPrecision": "mixed"}
        except openmm.OpenMMException:
            # CUDA plugin not loaded: fall through to the CPU platform.
            pass
    return openmm.Platform.getPlatformByName("CPU"), {}


def _evaluate_potential_energy(
    system, topology, positions, device: str = "cuda", min_iterations: int = 0
) -> float:
    """Evaluate the potential energy of a system at the given positions.

    Args:
        system: OpenMM System.
        topology: Matching OpenMM Topology.
        positions: Positions (Quantity, nm).
        device: 'cuda' or 'cpu' (see ``_get_platform``).
        min_iterations: Optional brief minimization before evaluation (0 = none).

    Returns:
        Potential energy in kJ/mol.
    """
    import openmm
    import openmm.unit as unit
    from openmm.app import Simulation

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform, properties = _get_platform(device)
    sim = Simulation(topology, system, integrator, platform, properties)
    sim.context.setPositions(positions)
    if min_iterations > 0:
        sim.minimizeEnergy(maxIterations=min_iterations)
    state = sim.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


def _extract_chain(topology, positions, chain_id: str):
    """Extract a single chain as a new (topology, positions) pair.

    Bonds to atoms of other chains (such as an inter-chain disulfide) are dropped.

    Returns:
        (topology, positions) with positions as a Quantity of shape (n_atoms, 3) in nm.
    """
    import openmm.unit as unit
    from openmm.app import Topology

    new_topology = Topology()
    new_positions = []
    old_to_new: dict[int, object] = {}

    for chain in topology.chains():
        if chain.id == chain_id:
            new_chain = new_topology.addChain(chain.id)
            for residue in chain.residues():
                new_residue = new_topology.addResidue(residue.name, new_chain)
                for atom in residue.atoms():
                    new_atom = new_topology.addAtom(atom.name, atom.element, new_residue)
                    old_to_new[atom.index] = new_atom
                    new_positions.append(positions[atom.index])

    for bond in topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        if a1.index in old_to_new and a2.index in old_to_new:
            new_topology.addBond(old_to_new[a1.index], old_to_new[a2.index])

    new_positions = unit.Quantity(
        np.array([[p.x, p.y, p.z] for p in new_positions]),
        unit.nanometers,
    )
    return new_topology, new_positions


def _evaluate_subsystem_energies(
    simulation,
    topo_h,
    positions,
    peptide_chain: str,
    receptor_chain: str,
    solvent_model: str,
    device: str,
    bond_info=None,
    ncaa_xmls=None,
    failures: Optional[list] = None,
) -> tuple:
    """Evaluate E_complex, E_peptide, E_receptor at given positions.

    Uses the existing simulation context for complex energy (avoids rebuilding
    the system). Creates fresh subsystem simulations for peptide and receptor.
    The isolated partners are evaluated at their geometry in the complex (single
    trajectory approximation), so no reorganisation energy is included.

    Args:
        simulation: Simulation of the complex.
        topo_h: Topology of the complex with hydrogens.
        positions: Positions at which to evaluate (Quantity, nm).
        peptide_chain: Chain ID of the peptide.
        receptor_chain: Chain ID of the receptor.
        solvent_model: 'obc2' or 'gbn2'.
        device: 'cuda' or 'cpu' for the subsystem evaluations.
        bond_info: Cyclic / lactam closure info for the peptide subsystem.
        ncaa_xmls: GAFF2 residue-template XML strings for non-canonical residues.
        failures: If given, the reason for every failed evaluation is appended
            to this list as a string.

    Returns:
        (e_complex, e_peptide, e_receptor) in kJ/mol. All three are None when
        the complex energy is not finite or an exception occurred; the last two
        are None when a subsystem energy is not finite.
    """
    import openmm.unit as unit

    try:
        simulation.context.setPositions(positions)
        e_c = (
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )
        if not np.isfinite(e_c):
            if failures is not None:
                failures.append("complex energy is not finite")
            return None, None, None

        pep_topo, pep_pos = _extract_chain(topo_h, positions, peptide_chain)
        pep_topo, pep_pos = _repair_orphaned_cys(pep_topo, pep_pos, solvent_model)
        sys_p = _build_subsystem(pep_topo, solvent_model, bond_info=bond_info, ncaa_xmls=ncaa_xmls)
        e_p = _evaluate_potential_energy(sys_p, pep_topo, pep_pos, device)

        rec_topo, rec_pos = _extract_chain(topo_h, positions, receptor_chain)
        rec_topo, rec_pos = _repair_orphaned_cys(rec_topo, rec_pos, solvent_model)
        sys_r = _build_subsystem(rec_topo, solvent_model, ncaa_xmls=ncaa_xmls)
        e_r = _evaluate_potential_energy(sys_r, rec_topo, rec_pos, device)

        if not (np.isfinite(e_p) and np.isfinite(e_r)):
            if failures is not None:
                failures.append("peptide or receptor energy is not finite")
            return e_c, None, None

        return e_c, e_p, e_r

    except Exception as e:
        # Per-mode isolation: a failed evaluation must not discard the other modes.
        print(f"  Warning: subsystem energy evaluation failed: {e}")
        traceback.print_exc()
        if failures is not None:
            failures.append(f"{type(e).__name__}: {e}")
        return None, None, None


def _append_error_message(result: dict, message: str) -> None:
    """Append ``message`` to ``result["error_message"]`` without touching ``success``.

    A failed mode leaves the other modes usable, so it is recorded next to the
    energies that did come out. Messages are joined with "; ".
    """
    previous = result["error_message"]
    result["error_message"] = f"{previous}; {message}" if previous else message


def compute_interaction_energy(
    input_path: str | Path,
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    solvent_model: str = "obc2",
    device: str = "cuda",
    sample_id: Optional[str] = None,
    modes: tuple = ("raw", "relaxed", "after_md"),
    ph: float = 7.4,
    relaxed_min_steps_restrained: int = 500,
    relaxed_min_steps_full: int = 2000,
    after_md_duration_ps: float = 10.0,
    after_md_timestep_fs: float = 2.0,
    after_md_temperature_k: float = 300.0,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
) -> dict:
    """Compute interaction energies via subsystem decomposition for multiple modes.

    For each mode, computes E_interaction = E_complex - E_peptide - E_receptor
    using AMBER ff14SB + implicit solvent (OBC2 or GBn2). System preparation
    (hydrogen addition) is performed once and shared across modes.

    E_int is a single-trajectory end-state estimate in the spirit of MM-GBSA
    (Kollman et al. 2000): the isolated peptide and receptor are evaluated at
    their geometry in the complex, so it contains the force-field interaction
    plus the change in generalized Born polar solvation and in the nonpolar
    surface-area term, but no conformational reorganisation energy and no
    entropy. Compare values between structures of the same system, not with
    experimental binding free energies. The raw and relaxed values depend on
    the added hydrogens and on the minimizer's stopping point.

    Modes:
        raw:      Evaluate at the H-added input geometry. May return None for
                  structures with severe clashes (useful as a clash indicator).
        relaxed:  Backbone-restrained minimization → full unrestrained minimization,
                  then evaluate. Resolves clashes while preserving backbone geometry.
        after_md: Perform an independent backbone-restrained + unrestrained minimization,
                  then run a short MD and evaluate at the final frame. Does not
                  require 'relaxed' to be in modes.

    Args:
        input_path: Path to CIF or PDB structure file
        peptide_chain: Peptide chain ID. Auto-detected if None (smallest chain).
        receptor_chain: Receptor chain ID. Auto-detected if None (largest chain).
        solvent_model: Implicit solvent model ('obc2' or 'gbn2')
        device: Compute device ('cuda' or 'cpu')
        sample_id: Identifier for this computation (defaults to file stem)
        modes: Tuple of modes to compute. Subset of ('raw', 'relaxed', 'after_md').
        relaxed_min_steps_restrained: Backbone-restrained minimization steps.
        relaxed_min_steps_full: Unrestrained minimization steps.
        after_md_duration_ps: Short MD duration in picoseconds.
        after_md_timestep_fs: MD timestep in femtoseconds.
        after_md_temperature_k: MD temperature in Kelvin.
        ph: pH used to choose protonation states when hydrogens are added
            (default 7.4). If the pH-aware call raises, hydrogens are added
            again without a pH, which is OpenMM's default of 7.0.
        random_seed: Seed for hydrogen placement, the Langevin integrator and
            the initial MD velocities (default ``DEFAULT_RANDOM_SEED``). Pass
            ``None`` to draw fresh randomness on every call. CUDA "mixed"
            precision is not bit-for-bit reproducible, so GPU energies from
            the same seed can still differ slightly between runs.

    Returns:
        Flat dictionary with keys:
            sample_id (str), success (bool), error_message (str or None),
            num_contacts (int or None), num_close_contacts (int or None),
            and for each requested mode {mode}_interaction_energy,
            {mode}_e_complex, {mode}_e_peptide, {mode}_e_receptor
            (float in kJ/mol, or None when that mode could not be computed).
            Negative values of {mode}_interaction_energy indicate a favorable
            (stabilizing) interaction between peptide and receptor.
            num_contacts and num_close_contacts count peptide-receptor atom
            pairs closer than 8 and 4 angstrom in the input structure (atoms
            as present in the file, before hydrogens are added); they do not
            depend on the modes.
            ``success`` is True when at least one mode produced an energy.
            ``error_message`` is None when nothing failed; otherwise it holds
            one "<mode>: <reason>" entry per failed mode (joined by "; "), so a
            mode that came back as None can be explained even when ``success``
            is True.
    """
    import openmm
    import openmm.unit as unit

    from binding_metrics.io.structures import detect_chains, load_structure

    input_path = Path(input_path)
    if sample_id is None:
        sample_id = input_path.stem

    result: dict = {
        "sample_id": sample_id,
        "success": False,
        "error_message": None,
        "num_contacts": None,
        "num_close_contacts": None,
    }
    for mode in modes:
        result[f"{mode}_interaction_energy"] = None
        result[f"{mode}_e_complex"] = None
        result[f"{mode}_e_peptide"] = None
        result[f"{mode}_e_receptor"] = None

    try:
        topology, positions = load_structure(input_path)
        # No structure repair here: callers are expected to pass a clean structure
        # (raw input goes through PDBFixer in the relaxation step; MD output is
        # already fully prepared). Running PDBFixer here would mangle cyclic
        # peptides by adding spurious terminus atoms (OXT, H2/H3) after losing
        # the closure bond in the PDB round-trip.

        if peptide_chain is None or receptor_chain is None:
            auto_pep, auto_rec = detect_chains(topology)
            peptide_chain = peptide_chain or auto_pep
            receptor_chain = receptor_chain or auto_rec

        # Explicit waters, ions and ligands do not belong in an implicit-solvent system.
        from binding_metrics.io.structures import strip_heterogens

        topology, positions = strip_heterogens(topology, positions, peptide_chain, receptor_chain)

        if peptide_chain is None or receptor_chain is None:
            raise ValueError("Could not identify two protein chains in structure")

        print(f"[{sample_id}] Chains: peptide={peptide_chain}, receptor={receptor_chain}")

        # Contact counts describe the input geometry (before hydrogens are added),
        # so they are the same for every mode.
        pos_array = np.array([[p.x, p.y, p.z] for p in positions]) * 10  # nm -> Å
        pep_indices = [a.index for a in topology.atoms() if a.residue.chain.id == peptide_chain]
        rec_indices = [a.index for a in topology.atoms() if a.residue.chain.id == receptor_chain]
        if pep_indices and rec_indices:
            distances = np.linalg.norm(
                pos_array[pep_indices][:, np.newaxis, :] - pos_array[rec_indices][np.newaxis, :, :],
                axis=-1,
            )
            result["num_contacts"] = int(np.sum(distances < _CONTACT_CUTOFF_ANGSTROM))
            result["num_close_contacts"] = int(np.sum(distances < _CLOSE_CONTACT_CUTOFF_ANGSTROM))

        # Hydrogens are added once here and the result is shared by all modes.
        sys_complex, topo_h, pos_h, bond_info, extra_xmls = _create_implicit_system(
            topology,
            positions,
            solvent_model,
            peptide_chain=peptide_chain,
            ph=ph,
            random_seed=random_seed,
        )
        platform, props = _get_platform(device)

        # Single simulation object used for all modes (sequential: raw → relaxed → after_md)
        integrator = openmm.LangevinMiddleIntegrator(
            after_md_temperature_k * unit.kelvin,
            1.0 / unit.picosecond,
            after_md_timestep_fs * unit.femtosecond,
        )
        if random_seed is not None:
            integrator.setRandomNumberSeed(random_seed)
        simulation = openmm.app.Simulation(topo_h, sys_complex, integrator, platform, props)
        simulation.context.setPositions(pos_h)

        any_success = False

        # --- RAW mode ---
        if "raw" in modes:
            print(f"[{sample_id}] Raw mode...")
            failures: list[str] = []
            e_c, e_p, e_r = _evaluate_subsystem_energies(
                simulation,
                topo_h,
                pos_h,
                peptide_chain,
                receptor_chain,
                solvent_model,
                device,
                bond_info=bond_info,
                ncaa_xmls=extra_xmls,
                failures=failures,
            )
            if e_c is not None and e_p is not None and e_r is not None:
                result["raw_e_complex"] = e_c
                result["raw_e_peptide"] = e_p
                result["raw_e_receptor"] = e_r
                result["raw_interaction_energy"] = e_c - e_p - e_r
                any_success = True
                print(f"[{sample_id}]   E_int(raw) = {result['raw_interaction_energy']:.1f} kJ/mol")
            else:
                print(f"[{sample_id}]   Raw evaluation returned NaN (likely clashes)")
                _append_error_message(result, "raw: " + "; ".join(failures))

        # --- RELAXED mode (also prepares state for after_md) ---
        pos_relaxed = pos_h
        if "relaxed" in modes or "after_md" in modes:
            print(f"[{sample_id}] Minimizing (backbone-restrained + unrestrained)...")
            try:
                restraint = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
                restraint.addGlobalParameter(
                    "k",
                    _BACKBONE_RESTRAINT_K_KJ_MOL_NM2 * unit.kilojoules_per_mole / unit.nanometer**2,
                )
                restraint.addPerParticleParameter("x0")
                restraint.addPerParticleParameter("y0")
                restraint.addPerParticleParameter("z0")
                restrained_residues: set = set()
                for atom in topo_h.atoms():
                    if atom.name in _BACKBONE_ATOM_NAMES:
                        pos = pos_h[atom.index]
                        restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
                        restrained_residues.add(atom.residue.index)
                for residue in topo_h.residues():
                    if residue.index not in restrained_residues:
                        warnings.warn(
                            f"compute_interaction_energy: no backbone atoms "
                            f"(N/CA/C/O) found for residue {residue.name}{residue.id} "
                            f"(chain {residue.chain.id}); backbone restraint skipped "
                            "for this residue.",
                            stacklevel=2,
                        )
                restraint_index = sys_complex.getNumForces()
                sys_complex.addForce(restraint)
                simulation.context.reinitialize(preserveState=True)

                simulation.minimizeEnergy(maxIterations=relaxed_min_steps_restrained)
                simulation.context.setParameter("k", 0.0)
                simulation.minimizeEnergy(maxIterations=relaxed_min_steps_full)

                state = simulation.context.getState(getPositions=True)
                pos_relaxed = state.getPositions()

                sys_complex.removeForce(restraint_index)
                simulation.context.reinitialize(preserveState=True)

                if "relaxed" in modes:
                    failures = []
                    e_c, e_p, e_r = _evaluate_subsystem_energies(
                        simulation,
                        topo_h,
                        pos_relaxed,
                        peptide_chain,
                        receptor_chain,
                        solvent_model,
                        device,
                        bond_info=bond_info,
                        ncaa_xmls=extra_xmls,
                        failures=failures,
                    )
                    if e_c is not None and e_p is not None and e_r is not None:
                        result["relaxed_e_complex"] = e_c
                        result["relaxed_e_peptide"] = e_p
                        result["relaxed_e_receptor"] = e_r
                        result["relaxed_interaction_energy"] = e_c - e_p - e_r
                        any_success = True
                        print(
                            f"[{sample_id}]   E_int(relaxed) = "
                            f"{result['relaxed_interaction_energy']:.1f} kJ/mol"
                        )
                    else:
                        _append_error_message(result, "relaxed: " + "; ".join(failures))
            except Exception as e:
                # Per-step isolation: the other modes stay usable; the reason is recorded.
                print(f"[{sample_id}] Warning: relaxed/minimization failed: {e}")
                step = "relaxed" if "relaxed" in modes else "after_md"
                _append_error_message(
                    result, f"{step}: minimization failed: {type(e).__name__}: {e}"
                )

        # --- AFTER_MD mode ---
        if "after_md" in modes:
            print(f"[{sample_id}] Running MD ({after_md_duration_ps} ps)...")
            try:
                simulation.context.setPositions(pos_relaxed)
                if random_seed is not None:
                    simulation.context.setVelocitiesToTemperature(
                        after_md_temperature_k * unit.kelvin, random_seed
                    )
                else:
                    simulation.context.setVelocitiesToTemperature(
                        after_md_temperature_k * unit.kelvin
                    )
                n_steps = int(after_md_duration_ps * 1000 / after_md_timestep_fs)
                simulation.step(n_steps)

                state = simulation.context.getState(getPositions=True)
                pos_md = state.getPositions()

                failures = []
                e_c, e_p, e_r = _evaluate_subsystem_energies(
                    simulation,
                    topo_h,
                    pos_md,
                    peptide_chain,
                    receptor_chain,
                    solvent_model,
                    device,
                    bond_info=bond_info,
                    ncaa_xmls=extra_xmls,
                    failures=failures,
                )
                if e_c is not None and e_p is not None and e_r is not None:
                    result["after_md_e_complex"] = e_c
                    result["after_md_e_peptide"] = e_p
                    result["after_md_e_receptor"] = e_r
                    result["after_md_interaction_energy"] = e_c - e_p - e_r
                    any_success = True
                    print(
                        f"[{sample_id}]   E_int(after_md) = "
                        f"{result['after_md_interaction_energy']:.1f} kJ/mol"
                    )
                else:
                    _append_error_message(result, "after_md: " + "; ".join(failures))
            except Exception as e:
                print(f"[{sample_id}] Warning: after_md failed: {e}")
                _append_error_message(result, f"after_md: {type(e).__name__}: {e}")

        result["success"] = any_success

    except Exception as e:
        _append_error_message(result, f"{type(e).__name__}: {e}")
        print(f"[{sample_id}] ERROR: {result['error_message']}")
        traceback.print_exc()

    return result


def _seed_arg(value: str) -> Optional[int]:
    """Parse ``--random-seed``: an integer, or 'none'/'random'/'off' to disable seeding."""
    if value.strip().lower() in ("none", "random", "off"):
        return None
    return int(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute interaction energy via subsystem decomposition (implicit solvent)"
    )
    parser.add_argument("--input", "-i", type=Path, help="Single input structure file")
    parser.add_argument("--input-dir", type=Path, help="Directory of structure files")
    parser.add_argument("--glob-pattern", default="*.cif", help="Glob pattern for --input-dir")
    parser.add_argument("--output", "-o", type=Path, help="Output CSV path")
    parser.add_argument("--peptide-chain", type=str, default=None)
    parser.add_argument("--receptor-chain", type=str, default=None)
    parser.add_argument("--solvent-model", choices=["obc2", "gbn2"], default="obc2")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["raw", "relaxed", "after_md"],
        choices=["raw", "relaxed", "after_md"],
        help="Which modes to compute (default: all three)",
    )
    parser.add_argument("--relaxed-min-steps-restrained", type=int, default=500)
    parser.add_argument("--relaxed-min-steps-full", type=int, default=2000)
    parser.add_argument("--after-md-duration-ps", type=float, default=10.0)
    parser.add_argument("--after-md-timestep-fs", type=float, default=2.0)
    parser.add_argument("--after-md-temperature-k", type=float, default=300.0)
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH used to choose protonation states when hydrogens are added (default: 7.4)",
    )
    parser.add_argument(
        "--random-seed",
        type=_seed_arg,
        default=DEFAULT_RANDOM_SEED,
        metavar="INT|none",
        help=(
            "Seed for hydrogen placement, the thermostat and the initial MD velocities "
            f"(default: {DEFAULT_RANDOM_SEED}); pass 'none' for fresh randomness on each run."
        ),
    )
    from binding_metrics.cli import add_log_file_arg

    add_log_file_arg(parser)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    from binding_metrics.cli import log_to_file

    with log_to_file(args.log_file):
        import pandas as pd

        if args.input:
            input_files = [Path(args.input)]
        elif args.input_dir:
            input_files = sorted(Path(args.input_dir).glob(args.glob_pattern))
        else:
            parser.error("Must specify --input or --input-dir")

        results = []
        for path in input_files:
            r = compute_interaction_energy(
                path,
                peptide_chain=args.peptide_chain,
                receptor_chain=args.receptor_chain,
                solvent_model=args.solvent_model,
                device=args.device,
                modes=tuple(args.modes),
                relaxed_min_steps_restrained=args.relaxed_min_steps_restrained,
                relaxed_min_steps_full=args.relaxed_min_steps_full,
                after_md_duration_ps=args.after_md_duration_ps,
                after_md_timestep_fs=args.after_md_timestep_fs,
                after_md_temperature_k=args.after_md_temperature_k,
                ph=args.ph,
                random_seed=args.random_seed,
            )
            results.append(r)

        df = pd.DataFrame(results)
        if args.output:
            df.to_csv(args.output, index=False, float_format="%.4f")
            print(f"\nSaved to: {args.output}")
        else:
            print("\nResults:")
            print(df.to_string())


if __name__ == "__main__":
    main()
