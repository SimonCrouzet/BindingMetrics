"""Automatic GAFF2 parameterisation of backbone-embedded non-canonical amino acids.

Motivation
----------
``openmmforcefields.GAFFTemplateGenerator`` cannot parameterise a residue whose
backbone N/C carry peptide (external) bonds: its ``_match_residue`` only matches a
topology residue to a GAFF small-molecule template when *every* atom's
``number_of_external_bonds`` equals the template's (always 0).  A backbone residue's
N and C always have external bonds, so GAFF-via-openmmforcefields can never match
them.

This module works around that by generating a full OpenMM ``ForceField`` residue
template that carries ``<ExternalBond>`` tags (the automatic analogue of the curated
NMe / lactam templates in :mod:`binding_metrics.core.nonstandard` and
:mod:`binding_metrics.core.cyclic`).  OpenMM's *native* template matcher understands
``<ExternalBond>``, so such a residue matches and ``createSystem`` succeeds.

Per non-canonical residue the generator:

1. Builds an RDKit heavy-atom molecule from the residue's intra-residue topology
   bonds, adding a carbon *cap* atom at each external bond (a bond with exactly one
   endpoint inside the residue — backbone N/C and any cyclic-closure atom),
   positioned at the external partner's coordinate.
2. Perceives bond orders from the 3D geometry
   (``rdDetermineBonds.DetermineBondOrders``), then sanitises and adds explicit H.
3. Runs ``GAFFTemplateGenerator.generate_residue_template`` to obtain GAFF atom
   types, bonded parameters and AM1-BCC charges.
4. Rewrites the ``<Residue>`` block: renames it to the real residue name, gives the
   residue heavy atoms their topology names, drops the cap atoms (and any H bonded
   only to a cap), adds an ``<ExternalBond>`` for every capped residue atom, and
   redistributes the removed cap charge so the template is net-neutral (integer).
5. Injects the residue's kept hydrogens into the OpenMM topology (rebuilding it) so
   the topology residue matches the generated template.

The GAFF force blocks (``<AtomTypes>``/``<HarmonicBondForce>``/…) are kept unchanged.

GAFF2 is a *general* small-molecule force field — this is a pragmatic approximation
for exotic building blocks (e.g. BMT/ABA in cyclosporin A), not a substitute for
purpose-built RESP-fitted parameters.  Residues already covered by curated templates
(NMG/NMA/MVA/MLE, the ASPL/GLUL/LYSL lactams, CYX) are skipped here.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np


# Residue names handled by ff14SB directly or by curated XML templates elsewhere.
# Anything NOT in this set (and with >1 heavy atom, non-metal) is treated as an
# exotic NCAA and parameterised with GAFF2.
GAFF_SKIP_RESIDUES = frozenset({
    # Canonical amino acids + protonation variants
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
    "CYX", "HID", "HIE", "HIP", "HIN", "LYN", "ASH", "GLH",
    # Curated non-standard templates (nonstandard.py / cyclic.py)
    "NMG", "NMA", "MVA", "MLE", "ASPL", "GLUL", "LYSL",
    # Capping groups
    "ACE", "NME", "FOR",
    # Nucleotides
    "DA", "DC", "DG", "DT", "A", "C", "G", "T", "U",
    # Water / ions
    "HOH", "WAT", "H2O", "SOL", "TIP", "TIP3",
    "NA", "CL", "K", "MG", "CA", "ZN", "LI", "RB", "CS", "FE", "MN", "CU",
})

_METAL_SYMBOLS = frozenset({
    "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
})


def _pos_to_angstrom(positions) -> np.ndarray:
    try:
        return np.array([[p.x, p.y, p.z] for p in positions]) * 10.0
    except (AttributeError, TypeError):
        return np.asarray(positions) * 10.0


def _heavy_atoms(res) -> list:
    return [a for a in res.atoms()
            if a.element is not None and a.element.atomic_number > 1]


def _is_ncaa(res) -> bool:
    """True if this residue should be GAFF-parameterised."""
    if res.name in GAFF_SKIP_RESIDUES:
        return False
    heavy = _heavy_atoms(res)
    if len(heavy) < 2:
        return False  # lone ions / monoatomics
    # Pure metal clusters are not GAFF small molecules.
    if all(a.element.symbol in _METAL_SYMBOLS for a in heavy):
        return False
    return True


def _perceive_bond_orders(mol):
    """Return a sanitised copy of ``mol`` with bond orders perceived from geometry.

    Tries charged-fragment perception first (best chemistry), then radical-free
    neutral perception, then a plain single-bond sanitisation as a last resort.
    ``mol`` must already carry a 3D conformer and single-bond connectivity.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    for kwargs in ({}, {"allowChargedFragments": False}):
        candidate = Chem.Mol(mol)
        try:
            rdDetermineBonds.DetermineBondOrders(
                candidate, charge=0, embedChiral=False, **kwargs
            )
            Chem.SanitizeMol(candidate)
            return candidate
        except Exception:
            continue
    candidate = Chem.Mol(mol)
    Chem.SanitizeMol(candidate)
    return candidate


def _build_capped_molecule(res, topology, pos_A):
    """Build an RDKit molecule for ``res`` with carbon caps at every external bond.

    Returns ``(mol, rd_res_names, cap_indices, ext_atom_names)`` where:
        mol           : RWMol (single bonds, 3D conformer set, no H yet)
        rd_res_names  : {rdkit_idx: topology_atom_name} for the residue heavy atoms
        cap_indices   : set of rdkit indices that are cap atoms
        ext_atom_names: ordered list of residue atom names that carry an external bond
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    heavy = _heavy_atoms(res)
    # Residue-boundary test uses ALL residue atoms (incl. any existing H) so a
    # H–heavy bond in an already-protonated residue is NOT mistaken for an
    # external bond. Inter-residue bonds are always heavy–heavy (peptide C–N,
    # disulfide, cyclic closure), so caps are only ever added at heavy atoms.
    res_atom_indices = {a.index for a in res.atoms()}

    rw = Chem.RWMol()
    rd_idx: dict = {}
    rd_res_names: dict = {}
    coords: list = []
    for atom in heavy:
        j = rw.AddAtom(Chem.Atom(atom.element.atomic_number))
        rd_idx[atom.index] = j
        rd_res_names[j] = atom.name
        coords.append(pos_A[atom.index])

    for bond in topology.bonds():
        i1, i2 = bond.atom1.index, bond.atom2.index
        if i1 in rd_idx and i2 in rd_idx:
            rw.AddBond(rd_idx[i1], rd_idx[i2], Chem.BondType.SINGLE)

    cap_indices: set = set()
    ext_atom_names: list = []
    for bond in topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        in1, in2 = a1.index in res_atom_indices, a2.index in res_atom_indices
        if in1 == in2:
            continue  # both inside or both outside the residue
        inner = a1 if in1 else a2
        outer = a2 if in1 else a1
        if inner.index not in rd_idx:
            continue  # boundary atom is not a heavy atom we model (defensive)
        cap = rw.AddAtom(Chem.Atom(6))  # carbon cap → clean sp2/sp3 perception
        rw.AddBond(rd_idx[inner.index], cap, Chem.BondType.SINGLE)
        coords.append(pos_A[outer.index])
        cap_indices.add(cap)
        if inner.name not in ext_atom_names:
            ext_atom_names.append(inner.name)

    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf)
    return mol, rd_res_names, cap_indices, ext_atom_names


def _hydrogen_names(keep_h, rd_res_names) -> dict:
    """Assign PDB-ish unique names to kept hydrogens: parent 'CB' → 'HB', 'HB2'…"""
    names: dict = {}
    per_parent: dict = {}
    for h_idx, parent in keep_h:
        pname = rd_res_names.get(parent, "X")
        stem = pname[1:] if len(pname) > 1 else pname
        per_parent[parent] = per_parent.get(parent, 0) + 1
        k = per_parent[parent]
        names[h_idx] = f"H{stem}" if k == 1 else f"H{stem}{k}"
    # Enforce global uniqueness against each other.
    seen: dict = {}
    for h_idx in list(names):
        nm = names[h_idx]
        if nm in seen:
            seen[nm] += 1
            names[h_idx] = f"{nm}x{seen[nm]}"
        else:
            seen[nm] = 1
    return names


def _generate_residue_template(res, topology, pos_A, gaff_version: str):
    """Build one GAFF ExternalBond template for ``res``.

    Returns ``(ffxml_string, h_inject)`` where ``h_inject`` is a list of
    ``(h_name, parent_atom_name, position_nm_ndarray)`` for the hydrogens that must
    be injected into the topology residue, or ``None`` on failure.
    """
    from rdkit import Chem
    from openff.toolkit import Molecule
    from openmmforcefields.generators import GAFFTemplateGenerator

    mol, rd_res_names, cap_indices, ext_atom_names = _build_capped_molecule(
        res, topology, pos_A
    )
    mol = _perceive_bond_orders(mol)
    mh = Chem.AddHs(mol, addCoords=True)
    hconf = mh.GetConformer()

    # Classify hydrogens: keep those bonded to a residue heavy atom; drop cap-H.
    keep_h: list = []
    drop_h: set = set()
    for atom in mh.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if not nbrs or nbrs[0] in cap_indices:
            drop_h.add(atom.GetIdx())
        else:
            keep_h.append((atom.GetIdx(), nbrs[0]))

    off = Molecule.from_rdkit(mh, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    gaff = GAFFTemplateGenerator(molecules=[off], forcefield=gaff_version)
    ffxml = gaff.generate_residue_template(off)

    root = ET.fromstring(ffxml)
    resel = root.find(".//Residue")
    if resel is None:
        return None
    tatoms = resel.findall("Atom")
    if len(tatoms) != mh.GetNumAtoms():
        # Template atom order must equal the RDKit atom order for the index map.
        return None

    # Name every template atom by its RDKit index.
    h_names = _hydrogen_names(keep_h, rd_res_names)
    new_name: dict = {}
    for j, nm in rd_res_names.items():
        new_name[j] = nm
    for h_idx, _parent in keep_h:
        new_name[h_idx] = h_names[h_idx]

    dropped = set(cap_indices) | drop_h
    keep_idx = [i for i in range(len(tatoms)) if i not in dropped]

    # Redistribute dropped cap/cap-H charge so the residue template is net-neutral.
    kept_charge = {i: float(tatoms[i].get("charge")) for i in keep_idx}
    total = sum(kept_charge.values())
    target = round(total)  # neutral NCAAs → 0
    delta = (target - total) / len(keep_idx)
    for i in keep_idx:
        kept_charge[i] += delta

    # Rewrite the <Residue> block.
    for el in list(resel):
        resel.remove(el)
    for i in keep_idx:
        a = ET.SubElement(resel, "Atom")
        a.set("name", new_name[i])
        a.set("type", tatoms[i].get("type"))
        a.set("charge", f"{kept_charge[i]:.6f}")
    keep_set = set(keep_idx)
    for bond in mh.GetBonds():
        i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i1 in keep_set and i2 in keep_set:
            b = ET.SubElement(resel, "Bond")
            b.set("atomName1", new_name[i1])
            b.set("atomName2", new_name[i2])
    for nm in ext_atom_names:
        eb = ET.SubElement(resel, "ExternalBond")
        eb.set("atomName", nm)
    resel.set("name", res.name)

    ffxml_out = ET.tostring(root, encoding="unicode")

    h_inject = []
    for h_idx, parent in keep_h:
        p = hconf.GetAtomPosition(h_idx)
        h_inject.append(
            (new_name[h_idx], rd_res_names[parent],
             np.array([p.x, p.y, p.z]) / 10.0)  # Å → nm
        )
    return ffxml_out, h_inject


def _load_ffxml(ff, ffxml_string: str) -> None:
    fd, path = tempfile.mkstemp(suffix=".xml")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(ffxml_string)
        ff.loadFile(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _rebuild_topology_with_injected_h(topology, pos_nm, h_by_res: dict):
    """Return a new (topology, positions) with the given H injected per residue.

    Existing hydrogens on the injected residues are dropped first (so repeated
    passes stay consistent); every other atom is copied unchanged.  ``h_by_res``
    maps ``res.index`` → list of ``(h_name, parent_name, pos_nm)``.
    """
    from openmm.app import Topology
    from openmm.app import element as elem
    from openmm import unit

    new_top = Topology()
    old_to_new: dict = {}
    new_pos: list = []
    for chain in topology.chains():
        new_chain = new_top.addChain(chain.id)
        for res in chain.residues():
            new_res = new_top.addResidue(res.name, new_chain, id=res.id)
            drop_h = res.index in h_by_res
            for atom in res.atoms():
                if drop_h and atom.element is not None and atom.element.atomic_number == 1:
                    continue  # drop existing H on injected residues
                na = new_top.addAtom(atom.name, atom.element, new_res)
                old_to_new[atom.index] = na
                new_pos.append(pos_nm[atom.index])
            if drop_h:
                by_name = {a.name: a for a in new_res.atoms()}
                for h_name, parent, hpos in h_by_res[res.index]:
                    ha = new_top.addAtom(h_name, elem.hydrogen, new_res)
                    new_pos.append(np.asarray(hpos))
                    parent_atom = by_name.get(parent)
                    if parent_atom is not None:
                        new_top.addBond(ha, parent_atom)

    for bond in topology.bonds():
        na1 = old_to_new.get(bond.atom1.index)
        na2 = old_to_new.get(bond.atom2.index)
        if na1 is not None and na2 is not None:
            new_top.addBond(na1, na2)

    new_positions = unit.Quantity(np.array(new_pos), unit.nanometers)
    return new_top, new_positions


def parameterize_ncaa_residues(topology, positions, ff, *,
                               gaff_version: str = "gaff-2.2.20",
                               verbose: bool = True):
    """Generate + load GAFF ExternalBond templates for exotic NCAA residues.

    Must run AFTER the D-amino-acid / N-methyl rename, ``patch_cyclic_topology``
    (which reconstructs intra-residue bonds and adds closure bonds) and
    ``rename_disulfide_cys_to_cyx``; and BEFORE ``addHydrogens`` / ``createSystem``.

    For every residue not covered by ff14SB or a curated template, a residue
    template with ``<ExternalBond>`` tags is generated from the topology geometry,
    loaded into ``ff``, and the residue's hydrogens are injected into a rebuilt
    topology so the residue matches the template.

    Args:
        topology: OpenMM Topology (heavy atoms; may already carry some H).
        positions: OpenMM positions (nm).
        ff: OpenMM ForceField to load the generated templates into.
        gaff_version: GAFF2 version string for ``GAFFTemplateGenerator``.
        verbose: Print a one-line summary per generated template.

    Returns:
        ``(topology, positions, ncaa_ffxmls)`` — the (possibly rebuilt) topology
        and positions, and the list of generated force-field XML strings (one per
        unique NCAA residue name; empty if there were none).  The XMLs are already
        loaded into ``ff``; callers that build *separate* force fields (e.g. the
        peptide/receptor subsystems in the energy decomposition) must reload them.
    """
    ncaa_residues = [res for res in topology.residues() if _is_ncaa(res)]
    if not ncaa_residues:
        return topology, positions, []

    try:
        import openmmforcefields  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "openmmforcefields is required to parameterise non-canonical residues. "
            "Install with: conda install -c conda-forge openmmforcefields openff-toolkit"
        ) from exc

    pos_A = _pos_to_angstrom(positions)
    pos_nm = pos_A / 10.0

    loaded_names: set = set()
    ncaa_ffxmls: list = []
    h_by_res: dict = {}
    expected_h: dict = {}  # name -> tuple of kept H names (consistency guard)

    for res in ncaa_residues:
        try:
            result = _generate_residue_template(res, topology, pos_A, gaff_version)
        except Exception as exc:
            if verbose:
                print(f"  [warning] GAFF NCAA template failed for '{res.name}': {exc}")
            result = None
        if result is None:
            continue
        ffxml, h_inject = result
        h_by_res[res.index] = h_inject
        h_names = tuple(h[0] for h in h_inject)

        if res.name not in loaded_names:
            _load_ffxml(ff, ffxml)
            loaded_names.add(res.name)
            expected_h[res.name] = h_names
            ncaa_ffxmls.append(ffxml)
            if verbose:
                net = _template_net_charge(ffxml)
                print(f"  Auto-GAFF2: '{res.name}' template generated "
                      f"({len(h_inject)} H, net charge {net:+.4f})")
        elif h_names != expected_h.get(res.name):
            # A second instance perceived differently — reuse the first template
            # but warn; createSystem will surface a mismatch if truly incompatible.
            if verbose:
                print(f"  [warning] '{res.name}' instance differs from first "
                      f"template; reusing first (H {len(h_names)} vs "
                      f"{len(expected_h.get(res.name, ()))}).")

    if not h_by_res:
        return topology, positions, ncaa_ffxmls

    topology, positions = _rebuild_topology_with_injected_h(topology, pos_nm, h_by_res)
    return topology, positions, ncaa_ffxmls


def _template_net_charge(ffxml_string: str) -> float:
    root = ET.fromstring(ffxml_string)
    resel = root.find(".//Residue")
    if resel is None:
        return 0.0
    return sum(float(a.get("charge", 0.0)) for a in resel.findall("Atom"))
