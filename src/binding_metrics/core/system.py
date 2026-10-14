"""System preparation utilities for MD simulations."""

import contextlib
import logging
import os
import tempfile
from typing import Literal, Optional

import openmm.unit as unit
from openmm.app import ForceField, Modeller, PDBFile, Topology

from binding_metrics._constants import (
    DEFAULT_PH,
    DEFAULT_RANDOM_SEED,  # re-exported: callers import it from here
)
from binding_metrics.core import phosaa
from binding_metrics.core.forcefields import get_forcefield
from binding_metrics.core.residues import (
    AMBER_PROTONATION_VARIANTS,
    AMBER_STANDARD_RESIDUES,
    ION_NAMES_COMMON,
    METAL_ELEMENTS,
    STANDARD_AMINO_ACIDS,
    TERMINAL_CAP_NAMES,
    WATER_NAMES_ALL,
    WATER_NAMES_PDB_AMBER,
    residue_label,
    residue_number,
)
from binding_metrics.io.structures import (
    _amino_acid_names,
    author_chain_ids,
    copy_author_chain_ids,
    has_alpha_carbon,
    sequence_records,
)
from binding_metrics.utils import add_to_report, extend_report

log = logging.getLogger(__name__)

# Optional pdbfixer for structure repair
try:
    from pdbfixer import PDBFixer

    HAS_PDBFIXER = True
except ImportError:
    HAS_PDBFIXER = False

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


def _drop_duplicate_bonds(topology) -> tuple:
    """Return ``topology`` with every bond listed once, and the number of bonds dropped.

    ``PDBFixer.addMissingAtoms`` builds the standard bonds of the rebuilt topology and
    then adds again each bond of the old one that touches a residue the standard table
    does not know, so the peptide bond on either side of a phosphoserine, an
    N-methylated or any other non-standard residue comes out twice. OpenMM turns the
    second copy into one more harmonic bond term, and the GAFF2 cap logic into a second
    cap on the same carbon.

    A bond is identified by its two atoms, in either order. The first listing is kept
    with its type and order, and the others keep their relative order, so the result
    does not depend on anything but the input. OpenMM has no call that removes a bond:
    when there is a duplicate the topology is rebuilt through the public interface,
    with the same chains, residues (numbers and insertion codes), atoms (indices and
    IDs) and box. A topology without a duplicate is returned as it is.
    """
    seen: set = set()
    unique_bonds = []
    for bond in topology.bonds():
        pair = frozenset((bond.atom1.index, bond.atom2.index))
        if pair not in seen:
            seen.add(pair)
            unique_bonds.append(bond)
    n_dropped = topology.getNumBonds() - len(unique_bonds)
    if not n_dropped:
        return topology, 0

    cleaned = Topology()
    cleaned.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    new_atom = {}
    for chain in topology.chains():
        new_chain = cleaned.addChain(chain.id)
        for residue in chain.residues():
            new_residue = cleaned.addResidue(
                residue.name, new_chain, residue.id, residue.insertionCode
            )
            for atom in residue.atoms():
                new_atom[atom] = cleaned.addAtom(atom.name, atom.element, new_residue, atom.id)
    for bond in unique_bonds:
        cleaned.addBond(new_atom[bond.atom1], new_atom[bond.atom2], bond.type, bond.order)
    return cleaned, n_dropped


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

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cif", delete=False, encoding="utf-8"
    ) as tmp:
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

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cif", delete=False, encoding="utf-8"
    ) as tmp:
        PDBxFile.writeFile(topology, positions, tmp)
        tmp_path = tmp.name
    try:
        fixer = PDBFixer(filename=tmp_path)
    finally:
        os.unlink(tmp_path)
    return fixer


#: Longest C(i)-N(i+1) distance still read as a peptide bond. A real amide bond
#: is about 1.33 A (Engh and Huber, Acta Cryst. A47, 392-400, 1991); one missing
#: residue puts the neighbours at 3.8 A or more, so 2.0 A separates the two cases
#: with room for poor geometry.
_PEPTIDE_BOND_MAX_ANGSTROM = 2.0


def _numbering_gaps(topology, positions):
    """Yield ``(chain, residue before, residue after, n_missing)`` for each numbering gap.

    A gap is a pair of consecutive amino-acid residues in one chain whose residue
    numbers jump by more than 1 and whose C and N atoms are not bonded; ``n_missing``
    is the number of residue numbers skipped. A residue with an insertion code shares
    the number of its predecessor (60, 60A, 60B, 61), so it never makes a jump. The
    distance test keeps a chain that is merely renumbered from being counted. When the
    C or N atom is absent, or sits at the origin as a placeholder, the numbering jump
    alone decides.
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

    for chain in topology.chains():
        previous = None
        for res in chain.residues():
            if not has_alpha_carbon(res):
                continue  # water, ion, ligand or nucleotide
            atoms = {a.name: a.index for a in res.atoms()}
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
                        yield chain, previous[0], res, jump - 1
            previous = (res, atoms)


def _count_residue_gaps(topology, positions) -> int:
    """Count places where a chain skips residues (unresolved loops).

    See :func:`_numbering_gaps` for what counts. Call it on the topology as loaded
    from the file. PDBFixer's own ``missingResidues`` cannot serve: it reads the
    sequence records of the input file, which the topology-to-CIF round trip in
    :func:`_topology_to_fixer` does not carry, and that round trip also renumbers
    residues from 1 in each chain.
    """
    return sum(1 for _ in _numbering_gaps(topology, positions))


def _chain_breaks(topology, positions):
    """Yield ``(chain, residue before, residue after, distance in A)`` for each chain break."""
    import numpy as np

    coords_nm = np.array(positions.value_in_unit(unit.nanometer))

    def _atom_angstrom(atoms: dict, name: str):
        index = atoms.get(name)
        if index is None or np.abs(coords_nm[index]).max() < _ORIGIN_PLACEHOLDER_TOL_NM:
            return None
        return coords_nm[index] * 10.0

    for chain in topology.chains():
        previous = None
        for res in chain.residues():
            atoms = {a.name: a.index for a in res.atoms()}
            if "CA" not in atoms and res.name not in TERMINAL_CAP_NAMES | {"FOR"}:
                continue  # water, ion, ligand or nucleotide
            if previous is not None:
                c_xyz = _atom_angstrom(previous[1], "C")
                n_xyz = _atom_angstrom(atoms, "N")
                if c_xyz is not None and n_xyz is not None:
                    distance = float(np.linalg.norm(c_xyz - n_xyz))
                    if distance > _PEPTIDE_BOND_MAX_ANGSTROM:
                        yield chain, previous[0], res, distance
            previous = (res, atoms)


def find_chain_breaks(topology, positions) -> list:
    """Find consecutive residues of one chain that are too far apart to be bonded.

    OpenMM bonds every residue to the next one in the chain by name
    (``createStandardBonds``), whatever the distance, and so does the relaxation,
    which then closes the gap by stretching the two ends together. A chain break
    caused by an unresolved loop or a segment that a model placed elsewhere thus
    disappears from the result without a message. This lists them: residues i and
    i+1 of a chain (amino acids or capping groups, waters and ligands in between
    skipped) whose C(i) and N(i+1) atoms are more than
    ``_PEPTIDE_BOND_MAX_ANGSTROM`` (2.0 A, against 1.33 A for an amide bond) apart.
    A pair where either atom is missing or sits at the origin as a placeholder cannot
    be judged and is not listed.

    Returns:
        A list with one dict per break: ``chain`` (the author chain ID of a topology read
        from an mmCIF, see ``io.structures.author_chain_ids``, otherwise the chain ID),
        ``residue_before`` and ``residue_after`` (residue numbers as strings, with the insertion
        code appended when there is one: ``"60B"``) and
        ``c_n_distance_angstrom``.
    """
    author_ids = author_chain_ids(topology)
    return [
        {
            "chain": author_ids[chain.index],
            "residue_before": residue_number(before),
            "residue_after": residue_number(after),
            "c_n_distance_angstrom": round(distance, 2),
        }
        for chain, before, after, distance in _chain_breaks(topology, positions)
    ]


#: ``detected_by`` of a ``find_missing_residues`` entry that the sequence records of the
#: input file gave: the SEQRES of a PDB file and the ``_entity_poly_seq`` of an mmCIF.
_SEQUENCE_RECORD_SOURCES = ("SEQRES", "_entity_poly_seq")


def _missing_entry(
    chain: str,
    kind: str,
    detected_by: str,
    n_residues: Optional[int],
    before=None,
    after=None,
    terminus: Optional[str] = None,
    sequence_positions: Optional[list] = None,
) -> dict:
    return {
        "chain": chain,
        "kind": kind,
        "terminus": terminus,
        "n_residues": n_residues,
        "residue_before": None if before is None else residue_number(before),
        "residue_after": None if after is None else residue_number(after),
        "sequence_positions": sequence_positions,
        "detected_by": detected_by,
    }


def _missing_from_records(chain, author_id: str, record) -> Optional[list]:
    """The entries of one chain whose sequence record is known, or None when it cannot serve.

    The positions of the modelled residues in the sequence must increase along the chain;
    a chain whose numbering runs backwards (a circular permutation) is left to the
    numbering.
    """
    placed = [
        (residue, position)
        for residue, position in zip(chain.residues(), record.positions)
        if position is not None
    ]
    if not placed:
        return []
    positions = [position for _, position in placed]
    if any(later <= earlier for earlier, later in zip(positions, positions[1:])):
        return None

    source = record.source
    entries = []
    if positions[0] > 1:
        entries.append(
            _missing_entry(
                author_id,
                "terminal_tail",
                source,
                positions[0] - 1,
                after=placed[0][0],
                terminus="N",
                sequence_positions=[1, positions[0] - 1],
            )
        )
    for (before, first), (after, last) in zip(placed, placed[1:]):
        if last - first > 1:
            entries.append(
                _missing_entry(
                    author_id,
                    "gap",
                    source,
                    last - first - 1,
                    before=before,
                    after=after,
                    sequence_positions=[first + 1, last - 1],
                )
            )
    n_sequence = len(record.residues)
    if positions[-1] < n_sequence:
        entries.append(
            _missing_entry(
                author_id,
                "terminal_tail",
                source,
                n_sequence - positions[-1],
                before=placed[-1][0],
                terminus="C",
                sequence_positions=[positions[-1] + 1, n_sequence],
            )
        )
    return entries


def find_missing_residues(topology, positions) -> list:
    """List the residues a model is missing: unresolved loops and truncated chain ends.

    Nothing is rebuilt. PDBFixer could do it from the sequence records of the input
    file, but ``prep_structure`` hands it a topology rewritten as a CIF that has none,
    and a rebuilt loop or tail would be invented coordinates that enter the interaction
    energy, the shape complementarity and the surface areas. This only reports.

    Where ``io.structures.load_structure`` recorded the sequence of a chain (SEQRES of a
    PDB file, ``_entity_poly_seq`` of an mmCIF) the chain is compared with it: the
    positions of the sequence that no residue occupies are missing, at the ends of the
    chain (terminal tails) and between two modelled residues (gaps). Without a record
    only gaps can be found, from the residue numbers and the geometry of the model:

    * the numbers jump by more than 1 and the C and N atoms of the two neighbours are not
      bonded (``"numbering"``). Residues with insertion codes share a number (60, 60A,
      60B, 61), so they never make a jump, and a renumbered chain whose atoms are bonded
      is not a gap;
    * the numbers are consecutive but the neighbours are more than 2.0 A apart
      (``"chain_break"``): a break of unknown size, which :func:`find_chain_breaks` lists
      as well. A chain with a sequence record is not searched for it, because its
      numbering is complete.

    Only chains of amino acids are searched.

    Returns:
        A list of dicts, chains in order and each chain from its N- to its C-terminus.
        Each has the keys ``chain`` (author chain ID), ``kind`` (``"gap"`` or
        ``"terminal_tail"``), ``terminus`` (``"N"`` or ``"C"`` for a tail, else None),
        ``n_residues`` (None for a chain break of unknown size), ``residue_before`` and
        ``residue_after`` (the modelled neighbours as residue numbers with their insertion
        code; None on the side of a tail that has no neighbour), ``sequence_positions``
        (``[first, last]`` in the sequence record, else None) and ``detected_by``
        (``"SEQRES"``, ``"_entity_poly_seq"``, ``"numbering"`` or ``"chain_break"``).
    """
    author_ids = author_chain_ids(topology)
    records = sequence_records(topology)

    # Per chain, the gaps by the residue after them, in chain order. A numbering gap
    # replaces the chain break at the same place: it knows how many residues are missing.
    found: dict[int, dict[int, dict]] = {}
    for chain, before, after, _ in _chain_breaks(topology, positions):
        found.setdefault(chain.index, {})[after.index] = _missing_entry(
            author_ids[chain.index], "gap", "chain_break", None, before=before, after=after
        )
    for chain, before, after, n_skipped in _numbering_gaps(topology, positions):
        found.setdefault(chain.index, {})[after.index] = _missing_entry(
            author_ids[chain.index], "gap", "numbering", n_skipped, before=before, after=after
        )

    missing = []
    for chain in topology.chains():
        record = records[chain.index]
        from_records = (
            _missing_from_records(chain, author_ids[chain.index], record)
            if record is not None
            else None
        )
        if from_records is not None:
            missing.extend(from_records)
        else:
            missing.extend(entry for _, entry in sorted(found.get(chain.index, {}).items()))
    return missing


def _describe_missing(entry: dict) -> str:
    n_residues = entry["n_residues"]
    if entry["kind"] == "terminal_tail":
        first, last = entry["sequence_positions"]
        return (
            f"{n_residues} at the {entry['terminus']}-terminus (sequence positions {first}-{last})"
        )
    return f"{n_residues} between residues {entry['residue_before']} and {entry['residue_after']}"


def _warn_missing_residues(missing: list) -> None:
    """One warning per chain for the entries ``find_missing_residues`` found.

    A chain break of unknown size has its own warning (``prep_structure`` logs every break
    it finds), so it is not repeated here.
    """
    by_chain: dict[str, list] = {}
    for entry in missing:
        if entry["detected_by"] != "chain_break":
            by_chain.setdefault(entry["chain"], []).append(entry)
    for chain_id, entries in by_chain.items():
        sources = sorted({e["detected_by"] for e in entries})
        log.warning(
            "Missing residues in chain %s: %s. Found from %s. The residues are not in the "
            "model and prep does not rebuild them.",
            chain_id,
            "; ".join(_describe_missing(e) for e in entries),
            " and ".join("the residue numbers" if s == "numbering" else s for s in sources),
        )


def find_open_c_termini(topology) -> list:
    """Find chains whose last residue is a standard amino acid without its terminal oxygen.

    ``Modeller.addHydrogens`` adds hydrogens and never heavy atoms, and the force field has
    no template for a standard residue at the end of a chain that lacks OXT: it stops with
    "No template found for residue ... the bonds are different". PDBFixer adds the OXT in
    prep. A chain counts when its last residue is a standard amino acid or an AMBER variant
    (CYX, HID, ...), has a carbonyl carbon, has no OXT and its carbon has no bond to
    another residue, which a head-to-tail or C-terminal lactam closure and a capping
    group give it. Call it after ``patch_cyclic_topology``, which adds those bonds.

    Returns:
        A list with one ``(chain ID, residue name, residue number)`` per such chain, IDs as in
        ``topology``.
    """
    amino_acids = STANDARD_AMINO_ACIDS | AMBER_PROTONATION_VARIANTS
    found = []
    for chain in topology.chains():
        residues = list(chain.residues())
        if not residues or residues[-1].name not in amino_acids:
            continue
        last = residues[-1]
        atoms = {atom.name: atom for atom in last.atoms()}
        carbon = atoms.get("C")
        if carbon is None or "OXT" in atoms:
            continue
        bonded_outside = any(
            (bond.atom1 is carbon and bond.atom2.residue is not last)
            or (bond.atom2 is carbon and bond.atom1.residue is not last)
            for bond in topology.bonds()
        )
        if not bonded_outside:
            found.append((chain.id, last.name, str(last.id)))
    return found


def require_closed_c_termini(topology, chain_names: Optional[dict] = None) -> None:
    """Raise ``ValueError`` when a chain ends in a standard residue without its terminal oxygen.

    The relaxation and the interaction energy expect a prepared structure and would stop
    later, inside the force field, with a message that does not say why. See
    :func:`find_open_c_termini` for what counts.

    Args:
        topology: OpenMM Topology.
        chain_names: Optional ``{ID in the topology: name to show}``, for the author IDs of
            a topology that no longer carries them. Defaults to ``author_chain_ids``.

    Raises:
        ValueError: Naming each chain and residue, and saying that the structure must be
            prepared (``binding-metrics-prep``, or ``binding-metrics-run`` without
            ``--skip-prep``) or have a terminal oxygen or a cap on every chain.
    """
    open_ends = find_open_c_termini(topology)
    if not open_ends:
        return
    if chain_names is None:
        chain_names = {}
        for chain, author_id in zip(topology.chains(), author_chain_ids(topology)):
            chain_names.setdefault(chain.id, author_id)
    ends = ", ".join(
        f"chain {chain_names.get(chain_id, chain_id)} ends in {name}{number}"
        for chain_id, name, number in open_ends
    )
    raise ValueError(
        f"{ends} without its terminal oxygen (OXT): the structure is not prepared, and the "
        "force field has no template for the residue. Prepare it first (binding-metrics-prep, "
        "or binding-metrics-run without --skip-prep), or give a structure whose chains end in "
        "OXT or a cap (ACE, NME, NH2)."
    )


def _add_hydrogens_cyclic(
    topology,
    positions,
    custom_bonds: list,
    ph: float,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    report: Optional[dict] = None,
) -> tuple:
    """Add hydrogens to a topology that contains non-sequential cyclic bonds.

    PDBFixer's addMissingHydrogens cannot handle cyclic peptides — it applies
    standard N-terminal templates that try to place H2/H3 on the N atom, which
    fails when the N is already bonded to the C-terminus carbon.  This function
    uses the cyclic-aware ForceField templates instead.

    ``report``, when given, receives ``ncaa_bond_order_source`` (see
    ``prep_structure``) for the residues that needed a GAFF template.
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
        _detect_nonstandard,
        _patch_nonstandard,
        load_nonstandard_xmls,
        restore_nonstandard_names,
    )

    # custom bonds are intra-chain, so both ends share the same chain ID.
    cyclic_chain = custom_bonds[0][0]

    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    # Phosphorylated residues → AMBER phosaa params (net −2), not GAFF.
    phosaa.register(ff)

    # D-amino-acid / N-methyl rename first (e.g. DAL→ALA, SAR→NMG) so their
    # standard/curated templates match; must precede patch_cyclic_topology.
    ns_info = _detect_nonstandard(topology, cyclic_chain)
    if not ns_info.is_empty:
        topology, positions = _patch_nonstandard(topology, positions, cyclic_chain, ns_info)
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
    topology, positions, ncaa_xmls = parameterize_ncaa_residues(
        topology, positions, ff, random_seed=random_seed
    )
    sources = getattr(ncaa_xmls, "bond_order_source_by_residue", {})
    if report is not None and sources:
        report.setdefault("ncaa_bond_order_source", {}).update(sources)

    modeller = Modeller(topology, positions)
    addh_variants = (
        get_addh_variants(modeller.topology, bond_info, cyclic_chain) if bond_info else None
    )
    with deterministic_hydrogen_placement(random_seed), phosaa.hydrogen_definitions():
        modeller.addHydrogens(
            ff,
            pH=ph,
            variants=addh_variants,
            platform=_hydrogen_placement_platform(),
        )
    # The force field needed the L / template names above; the topology that
    # leaves prep (and is written to the prepped file) keeps the input names, so
    # the relaxation step and later metrics see DAL and SAR rather than ALA and NMG.
    restore_nonstandard_names(modeller.topology, ns_info)
    return modeller.topology, modeller.positions


def _rename_d_residues_to_l(topology) -> list:
    """Rename the D-amino-acid residues of ``topology`` to their L templates, in place.

    ``Modeller.addHydrogens`` places hydrogens by residue name and its definitions have no
    D-amino acid (DAL, DPN, ...), so the linear route of :func:`prep_structure` gave such
    residues no hydrogen. The cyclic route renames them (``core.nonstandard``, DAL to ALA)
    and so does this, with the same table. The hydrogens are placed from the heavy-atom
    geometry, which is the D geometry whatever the name, and the force field of the short
    minimization has no energy term that tells L from D at the alpha carbon.

    A residue read under its D name never got the bonds of its L template
    (``createStandardBonds`` knows the L names only), which ``addHydrogens`` needs to find
    the neighbours of a parent atom, so they are added.

    Returns:
        ``[(residue index, D name), ...]`` for :func:`_restore_d_residue_names`.
    """
    from binding_metrics.core.nonstandard import D_AA_MAP, _add_standard_intraresidue_bonds

    renamed = []
    for residue in topology.residues():
        l_name = D_AA_MAP.get(residue.name)
        if l_name is None:
            continue
        renamed.append((residue.index, residue.name))
        residue.name = l_name
        _add_standard_intraresidue_bonds(topology, residue)
    return renamed


def _restore_d_residue_names(topology, renamed: list) -> None:
    """Give the residues listed by :func:`_rename_d_residues_to_l` their D names back.

    ``addHydrogens`` adds hydrogens to residues and no residue, so the indices still fit; it
    may rename a residue to a protonation variant of its L name (HIS to HIE), which is
    accepted. Anything else means the topology was edited in between, and the residue keeps
    its name with a warning.
    """
    from binding_metrics.core.nonstandard import _PATCHED_NAME_VARIANTS, D_AA_MAP

    residues = list(topology.residues())
    for index, d_name in renamed:
        l_name = D_AA_MAP[d_name]
        allowed = {l_name} | _PATCHED_NAME_VARIANTS.get(l_name, frozenset())
        if index >= len(residues) or residues[index].name not in allowed:
            log.warning(
                "Residue index %d: expected %s, found %s; the name %s was not restored.",
                index,
                l_name,
                "no residue" if index >= len(residues) else residues[index].name,
                d_name,
            )
            continue
        residues[index].name = d_name


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
        repaired.append(f"{residue_label(res)}/{res.chain.id}")

    if repaired and verbose:
        log.info("  Repaired %d wrong-side Cα hydrogen(s): %s", len(repaired), ", ".join(repaired))
    # Vec3, not bare tuples: downstream consumers index positions as p.x/p.y/p.z.
    return unit.Quantity([Vec3(*map(float, p)) for p in pos], unit.nanometer)


def prep_structure(
    topology,
    positions,
    ph: float = DEFAULT_PH,
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
              removed non-water heterogen (free ligands, additives, glycans). Here
              and in ``kept_nonstandard`` and ``chain_breaks``, X is the author chain
              ID of a topology read from an mmCIF (``load_structure`` records it; see
              ``io.structures.author_chain_ids``), where OpenMM's own ID is the label
              ID for a file with more label IDs than author IDs: the peptide of 1CWA
              is chain C, not B.
            * ``n_removed_waters`` (int): water molecules removed (0 when
              ``keep_water`` is True).
            * ``kept_nonstandard`` (list[str]): non-standard residues and metal
              ions that were kept.
            * ``n_missing_atoms_rebuilt`` (int): heavy atoms PDBFixer added,
              including atoms deleted as origin placeholders and terminal OXT.
            * ``n_missing_residue_gaps`` (int): chain positions where residues
              are unresolved and were left as a gap, not rebuilt. Terminal tails
              are not counted.
            * ``missing_residues`` (list[dict]): the residues the input lacks, each as
              ``{"chain", "kind", "terminus", "n_residues", "residue_before",
              "residue_after", "sequence_positions", "detected_by"}`` (see
              :func:`find_missing_residues`): a gap or a terminal tail, found from the
              SEQRES or ``_entity_poly_seq`` records of the input file when it has them
              and from the residue numbers and chain breaks otherwise. Prep logs a
              warning for each chain and rebuilds nothing.
            * ``chain_breaks`` (list[dict]): consecutive residues of one chain whose C
              and N atoms are more than 2.0 A apart in the input, each as ``{"chain",
              "residue_before", "residue_after", "c_n_distance_angstrom"}`` (see
              :func:`find_chain_breaks`). Prep leaves them as they are; the
              relaxation bonds the two residues anyway.
            * ``ncaa_bond_order_source`` (dict[str, str]): for a cyclic peptide
              with non-canonical residues that needed a GAFF template, the
              source of each residue's bond orders, ``"ccd"`` (Chemical
              Component Dictionary) or ``"single_bonds"`` (the residue is not in
              the dictionary or disagrees with it, so double bonds, aromatic
              rings and the hydrogen count are unreliable). Absent otherwise.

            Behaviour is identical when ``report`` is None.

    Returns:
        Tuple of (topology, positions) with repaired and protonated structure. The
        chains keep the IDs they have in ``topology``, so a chain ID the caller
        passes to a later step still names the same chain (PDBFixer would
        otherwise relabel them A, B, C, ...), and carry its author IDs. Residue
        numbers of a PDB input restart at 1 in each chain; a CIF input gets its own
        back in ``save_cif``.
    """
    # Capture non-sequential intra-chain bonds (e.g. head-to-tail N→C) before
    # the PDBFixer round-trip drops them (PDBxFile.writeFile only writes SS bonds).
    # SS bonds are NOT captured here — they are re-detected from rebuilt geometry
    # after addMissingAtoms via _rebuild_connect_records.
    custom_bonds = _extract_custom_bonds(topology)

    # Residue numbers and chain IDs do not survive the PDBFixer round trip, so
    # read the numbering gaps and the caller's chain IDs off the input topology.
    input_chain_ids = [chain.id for chain in topology.chains()]
    input_author_ids = author_chain_ids(topology)  # the names a user knows the chains by
    n_residue_gaps = _count_residue_gaps(topology, positions)
    chain_breaks = find_chain_breaks(topology, positions)
    missing_residues = find_missing_residues(topology, positions)
    _warn_missing_residues(missing_residues)
    # A gap that the sequence records show and the numbering does not (a chain renumbered
    # after its loop was cut out) is unresolved all the same.
    n_residue_gaps = max(
        n_residue_gaps,
        sum(
            1
            for entry in missing_residues
            if entry["kind"] == "gap" and entry["detected_by"] in _SEQUENCE_RECORD_SOURCES
        ),
    )
    for gap in chain_breaks:
        log.warning(
            "Chain break in chain %s: residues %s and %s are %.2f A apart (C to N). "
            "Relaxation bonds them, which closes the gap by pulling the two segments together.",
            gap["chain"],
            gap["residue_before"],
            gap["residue_after"],
            gap["c_n_distance_angstrom"],
        )

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
    # PDBFixer lists the bonds next to a non-standard residue twice; the cyclic and
    # GAFF2 steps below read the bond list, so it is cleaned before they run.
    fixer.topology, n_duplicate_bonds = _drop_duplicate_bonds(fixer.topology)
    if n_duplicate_bonds:
        log.debug("Dropped %d bond(s) that PDBFixer listed twice", n_duplicate_bonds)

    # Detect and register all non-standard bonds from the rebuilt geometry
    # (SS bonds, CYS→CYX rename). Must run after addMissingAtoms so that
    # zero-coord placeholder atoms are in their correct positions.
    _rebuild_connect_records(fixer)

    # A chain is a protein chain when it holds a residue of the AMBER standard set or an
    # amino acid by the definition every chain detector uses, which has the D-amino
    # acids: a peptide made only of D residues has no standard one.
    amino_acids = _amino_acid_names()
    protein_chains: set = set()
    for chain in fixer.topology.chains():
        for res in chain.residues():
            if res.name in AMBER_STANDARD_RESIDUES or res.name in amino_acids:
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
        chain_label = input_author_ids[chain.index] if keep_input_ids else chain.id
        for res in chain.residues():
            if res.name in AMBER_STANDARD_RESIDUES:
                continue

            elements = {atom.element.symbol for atom in res.atoms() if atom.element is not None}
            if elements & METAL_ELEMENTS:
                kept_nonstandard.append(f"{res.name} (metal, chain {chain_label})")
                continue

            if res.name in WATER_NAMES_ALL:
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
        extend_report(report, "chain_breaks", chain_breaks)
        extend_report(report, "missing_residues", missing_residues)

    # Chains that keep at least one residue, by the caller's ID. Modeller.delete drops
    # a chain that ends up empty (the water chains of a PDB file) and keeps the order.
    removed = set(residues_to_remove)
    surviving_ids = (
        [
            input_chain_ids[chain.index]
            for chain in fixer.topology.chains()
            if any(res not in removed for res in chain.residues())
        ]
        if keep_input_ids
        else []
    )

    if residues_to_remove:
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete(residues_to_remove)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

    if surviving_ids and len(set(surviving_ids)) == len(surviving_ids):
        # Name the chains as the caller did: the returned topology, the file written
        # from it and the chain IDs the caller passes on must agree. Chains that
        # still share an ID (waters kept after the polymer of a PDB file) keep
        # PDBFixer's letters, which tell them apart.
        for chain, chain_id in zip(fixer.topology.chains(), surviving_ids):
            chain.id = chain_id

    if custom_bonds:
        # Use cyclic-aware H placement: patch_cyclic_topology (called inside)
        # removes C-terminal OXT that PDBFixer added, adds the N-C closure bond,
        # then uses cyclic FF templates for addHydrogens.  Do NOT restore bonds
        # here — patch_cyclic_topology detects and adds the bond itself.
        result_topo, result_pos = _add_hydrogens_cyclic(
            fixer.topology,
            fixer.positions,
            custom_bonds,
            ph,
            random_seed=random_seed,
            report=report,
        )
    else:
        # Inline of PDBFixer.addMissingHydrogens so we can pin the H-placement
        # minimization to the deterministic Reference platform (the method itself
        # exposes no platform argument).
        d_residues = _rename_d_residues_to_l(fixer.topology)
        _h_modeller = Modeller(fixer.topology, fixer.positions)
        with deterministic_hydrogen_placement(random_seed), phosaa.hydrogen_definitions():
            _h_modeller.addHydrogens(pH=ph, platform=_hydrogen_placement_platform())
        # The topology that leaves prep keeps the input names, as in the cyclic route.
        _restore_d_residue_names(_h_modeller.topology, d_residues)
        result_topo, result_pos = _h_modeller.topology, _h_modeller.positions

    # addHydrogens jitters new H randomly and relaxes them with the heavy atoms
    # frozen, which intermittently strands a Cα H on the wrong face; left in
    # place it inverts the stereocenter during minimization.
    result_pos = repair_ca_hydrogen_chirality(result_topo, result_pos)

    copy_author_chain_ids(topology, result_topo)
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
    ph: float = DEFAULT_PH,
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

    for residue in topology.residues():
        if residue.name in WATER_NAMES_PDB_AMBER:
            n_waters += 1
        elif residue.name in ION_NAMES_COMMON:
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
