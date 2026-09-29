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
2. Takes bond orders from the wwPDB Chemical Component Dictionary (bundled with
   biotite, so no network access) by residue and atom name, neutralises the
   ionisable groups, sanitises and adds explicit H. The RDKit molecule has no
   hydrogens, so bond orders cannot be recovered from its connectivity or its
   geometry: valence-based perception (``rdDetermineBonds``) leaves every bond
   single there and would turn an alkene into an alkane. A residue that is not in
   the dictionary, or whose atoms disagree with its entry, falls back to that
   single-bond graph and is reported (``NcaaTemplateList.bond_order_source_by_residue``).
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

References
----------
* GAFF: Wang, J., Wolf, R.M., Caldwell, J.W., Kollman, P.A., Case, D.A.
  Development and testing of a general amber force field. J. Comput. Chem. 2004,
  25, 1157-1174 (GAFF2 is the AmberTools revision of it, shipped as
  ``gaff-2.2.20`` in ``openmmforcefields``).
* AM1-BCC charges: Jakalian, A., Bush, B.L., Jack, D.B., Bayly, C.I. J. Comput.
  Chem. 2000, 21, 132-146 (method) and Jakalian, A., Jack, D.B., Bayly, C.I.
  J. Comput. Chem. 2002, 23, 1623-1641 (parameterisation and validation).
* ff14SB backbone types kept on the NCAA backbone: Maier, J.A. et al. J. Chem.
  Theory Comput. 2015, 11, 3696-3713.
* Chemical Component Dictionary (bond orders and formal charges): Westbrook, J.D.
  et al. Bioinformatics 2015, 31, 1274-1278.
"""

from __future__ import annotations

import functools
import logging
import os
import tempfile
import warnings
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

from binding_metrics._constants import DEFAULT_RANDOM_SEED
from binding_metrics.core.residues import (
    AMBER_STANDARD_VARIANTS,
    BACKBONE_HEAVY_ATOM_NAMES,
    FORCE_FIELD_CAP_NAMES,
    ION_NAMES_COMMON,
    LACTAM_TEMPLATE_RESIDUES,
    METAL_ELEMENTS,
    N_METHYLATED_RESIDUES,
    NUCLEOTIDE_RESIDUES,
    PHOSPHO_RESIDUES,
    STANDARD_AMINO_ACIDS,
    WATER_NAMES_ALL,
)

logger = logging.getLogger(__name__)

# Residue names handled by ff14SB directly or by curated XML templates elsewhere.
# Anything NOT in this set (and with >1 heavy atom, non-metal) is treated as an
# exotic NCAA and parameterised with GAFF2.
GAFF_SKIP_RESIDUES = frozenset(
    # Canonical amino acids + protonation variants (CYM included)
    STANDARD_AMINO_ACIDS
    | AMBER_STANDARD_VARIANTS
    # Curated non-standard templates (nonstandard.py / cyclic.py)
    | N_METHYLATED_RESIDUES
    | LACTAM_TEMPLATE_RESIDUES
    # Phosphorylated residues — handled by core.phosaa (AMBER phosaa
    # params). GAFF perceives these at neutral charge and would protonate
    # the phosphate to net 0, destroying the −2 charge; they MUST skip
    # the GAFF path.
    | PHOSPHO_RESIDUES
    | {"S1P", "T1P", "Y1P", "H1D", "H2D", "H1E", "H2E"}
    | FORCE_FIELD_CAP_NAMES
    | NUCLEOTIDE_RESIDUES
    # Water / ions
    | WATER_NAMES_ALL
    | ION_NAMES_COMMON
    | {"LI", "RB", "CS", "FE", "MN", "CU"}
)


def _pos_to_angstrom(positions) -> np.ndarray:
    try:
        return np.array([[p.x, p.y, p.z] for p in positions]) * 10.0
    except (AttributeError, TypeError):
        return np.asarray(positions) * 10.0


def _heavy_atoms(res) -> list:
    return [a for a in res.atoms() if a.element is not None and a.element.atomic_number > 1]


def _is_ncaa(res) -> bool:
    """True if this residue should be GAFF-parameterised."""
    if res.name in GAFF_SKIP_RESIDUES:
        return False
    heavy = _heavy_atoms(res)
    if len(heavy) < 2:
        return False  # lone ions / monoatomics
    # Pure metal clusters are not GAFF small molecules.
    if all(a.element.symbol in METAL_ELEMENTS for a in heavy):
        return False
    return True


#: Bond-order source recorded per residue in ``NcaaTemplateList.bond_order_source_by_residue``.
BOND_ORDER_SOURCE_CCD = "ccd"
BOND_ORDER_SOURCE_SINGLE_BONDS = "single_bonds"

_CCD_BOND_ORDER = {
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    # The CCD stores aromatic rings as one Kekule structure; RDKit re-aromatises it.
    "AROMATIC_SINGLE": 1,
    "AROMATIC_DOUBLE": 2,
    "AROMATIC_TRIPLE": 3,
}


@functools.lru_cache(maxsize=None)
def _ccd_heavy_atom_chemistry(res_name: str):
    """Heavy-atom chemistry of a wwPDB Chemical Component Dictionary entry.

    Reads the dictionary that ships with biotite (``biotite.structure.info``), so
    no network access is needed.

    Returns ``(atoms, bonds)`` with ``atoms = {atom_name: (element_symbol,
    formal_charge)}`` and ``bonds = {frozenset({name1, name2}): order}`` (order 1,
    2 or 3), both restricted to non-hydrogen atoms. Returns ``None`` if the name is
    not in the dictionary, if biotite is unavailable, or if the entry has a bond
    type without a plain order (quadruple, unspecified).
    """
    try:
        from biotite.structure import info

        ccd_bonds = info.bonds_in_residue(res_name)
        with warnings.catch_warnings():
            # Entries without ideal coordinates make biotite warn; only names,
            # elements and charges are read here.
            warnings.simplefilter("ignore", UserWarning)
            component = info.residue(res_name) if ccd_bonds else None
    except Exception as exc:  # noqa: BLE001 - biotite missing or too old; None is the fallback
        logger.debug("Chemical Component Dictionary lookup failed for '%s': %s", res_name, exc)
        return None
    if not ccd_bonds:
        return None  # biotite returns an empty dict for a name it does not know

    atoms = {
        str(name): (str(element).upper(), int(charge))
        for name, element, charge in zip(component.atom_name, component.element, component.charge)
        if str(element).upper() not in ("H", "D")
    }
    bonds: dict = {}
    for (name1, name2), bond_type in ccd_bonds.items():
        if name1 not in atoms or name2 not in atoms:
            continue  # a bond to a hydrogen
        order = _CCD_BOND_ORDER.get(bond_type.name)
        if order is None:
            return None
        bonds[frozenset((name1, name2))] = order
    return atoms, bonds


def _ccd_bond_orders(mol, atom_names: dict, res_name: str):
    """Set bond orders and formal charges of a residue's heavy atoms from the CCD.

    ``mol`` is the capped, hydrogen-free molecule from :func:`_build_capped_molecule`
    and ``atom_names`` maps its residue atom indices to topology atom names; cap
    atoms are not in ``atom_names`` and keep their single bonds.

    The dictionary entry is used only if it describes the same molecule: every atom
    name must exist with the same element, and the bonds between named atoms must
    be the same set. That guards against a residue code reused for a different
    compound (``UNL``, or a code chosen by a design tool).

    The dictionary lists ionisable groups in their deposited protonation state.
    They are neutralised (:class:`rdkit.Chem.MolStandardize.rdMolStandardize.Uncharger`)
    so the template keeps the neutral convention of the generator, where
    :func:`_neutral_ionizable_groups` reports the groups that ionise at pH 7.4. What
    cannot be neutralised (quaternary ammonium, nitro) keeps its charge.

    Returns ``(mol, "")`` on success, or ``(None, reason)``.
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    chemistry = _ccd_heavy_atom_chemistry(res_name)
    if chemistry is None:
        return None, f"'{res_name}' is not in the Chemical Component Dictionary"
    ccd_atoms, ccd_bonds = chemistry

    rw = Chem.RWMol(mol)
    for idx, name in atom_names.items():
        entry = ccd_atoms.get(name)
        if entry is None:
            return None, f"atom '{name}' is not in the dictionary entry for '{res_name}'"
        element, charge = entry
        if rw.GetAtomWithIdx(idx).GetSymbol().upper() != element:
            return None, f"atom '{name}' is not a {element} in the entry for '{res_name}'"
        rw.GetAtomWithIdx(idx).SetFormalCharge(charge)

    bond_orders = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    seen: set = set()
    for bond in rw.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in atom_names or j not in atom_names:
            continue  # bond to a cap
        key = frozenset((atom_names[i], atom_names[j]))
        order = ccd_bonds.get(key)
        if order is None:
            return None, f"bond {'-'.join(sorted(key))} is not in the entry for '{res_name}'"
        bond.SetBondType(bond_orders[order])
        seen.add(key)
    present = set(atom_names.values())
    absent = [key for key in ccd_bonds if key <= present and key not in seen]
    if absent:
        return None, f"bond {'-'.join(sorted(absent[0]))} of the entry for '{res_name}' is missing"

    try:
        perceived = rw.GetMol()
        Chem.SanitizeMol(perceived)
        neutral = rdMolStandardize.Uncharger().uncharge(perceived)
        Chem.SanitizeMol(neutral)
    except Exception as exc:  # noqa: BLE001 - RDKit raises several types for a bad valence
        return None, f"the dictionary bond orders do not sanitise in RDKit ({exc})"
    return neutral, ""


def _perceive_residue_bond_orders(mol, atom_names: dict, res_name: str):
    """Sanitised copy of ``mol`` with the residue's bond orders and charges.

    Uses the Chemical Component Dictionary (:func:`_ccd_bond_orders`). If that
    cannot describe the residue, falls back to :func:`_perceive_bond_orders`, which
    keeps every bond single on this hydrogen-free molecule.

    Returns ``(mol, fallback_reason)``; ``fallback_reason`` is empty when the
    dictionary was used, else it says why it could not be.
    """
    perceived, reason = _ccd_bond_orders(mol, atom_names, res_name)
    if perceived is not None:
        return perceived, ""
    return _perceive_bond_orders(mol), reason


def _perceive_bond_orders(mol):
    """Return a sanitised copy of ``mol`` with bond orders perceived from valence.

    Tries charged-fragment perception first, then radical-free neutral perception,
    then a plain single-bond sanitisation as a last resort. ``mol`` must carry a
    3D conformer and single-bond connectivity. Only valence rules are applied (the
    conformer is not read), so on a molecule without hydrogens every atom can
    absorb its missing valence as hydrogens and every bond stays single.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    for kwargs in ({}, {"allowChargedFragments": False}):
        candidate = Chem.Mol(mol)
        try:
            rdDetermineBonds.DetermineBondOrders(candidate, charge=0, embedChiral=False, **kwargs)
            Chem.SanitizeMol(candidate)
            return candidate
        except Exception as exc:  # noqa: BLE001 - RDKit raises several types; next attempt follows
            logger.debug("Bond-order perception failed with %s: %s", kwargs or "defaults", exc)
            continue
    candidate = Chem.Mol(mol)
    Chem.SanitizeMol(candidate)
    return candidate


#: Groups that are charged at pH 7.4 but that template generation perceives in
#: their neutral form (bond-order perception runs at total charge 0). Each entry is
#: (label, SMARTS with the ionisable heavy atom first, net charge at pH 7.4). The
#: charges are the textbook majority state; phosphate monoesters are taken as -2
#: (as in the AMBER phosaa residues, see :mod:`binding_metrics.core.phosaa`).
#: Order matters: a group claims its atoms, so the two-OH phosphate is matched
#: before the one-OH pattern and guanidine before primary amine.
#:
#: The patterns describe the neutral group with its real bond orders (C(=O)OH,
#: C(=NH)(NH2)2). A residue built with single bonds only (see
#: ``BOND_ORDER_SOURCE_SINGLE_BONDS``) has C(OH)2 and C(NH2)3 instead, which none of
#: them match; that residue is reported by its own warning.
_GUANIDINE_NITROGEN = "N+0;!$(N-[#6]=O);!$(N-a);!$(N-C#N)"
_IONIZABLE_GROUPS = (
    ("phosphate (two acidic OH)", "[#15](=[OX1])([OX2H1])([OX2H1])[#8]", -2),
    ("phosphate (one acidic OH)", "[#15](=[OX1])([#8])([#8])[OX2H1]", -1),
    ("sulfate or sulfonic acid", "[#16](=[OX1])(=[OX1])[OX2H1]", -1),
    ("carboxylic acid", "[#6X3]([#6])(=[OX1])[OX2H1]", -1),
    (
        "guanidine",
        f"[#6X3;!a](=[{_GUANIDINE_NITROGEN}])(-[{_GUANIDINE_NITROGEN}])-[{_GUANIDINE_NITROGEN}]",
        +1,
    ),
    ("primary amine", "[NX3;H2;!$(N-[#6]=[O,S,N]);!$(N-a);!$(N-[#7,#8,#15,#16])][CX4]", +1),
)


def _neutral_ionizable_groups(mol, rd_res_names, cap_indices) -> list:
    """List the side-chain groups of ``mol`` that are perceived neutral but ionise at pH 7.4.

    ``mol`` is the capped, hydrogen-complete RDKit molecule of one residue. A group
    already perceived in its charged form (``C(=O)[O-]``, ``[NH3+]``) does not
    match, so only groups the template treats as neutral are reported. The
    backbone atoms and the carbon caps are skipped: the cap hides the backbone
    carbonyl, which the perception can turn into a spurious acid.

    Returns:
        List of ``(label, charge)`` pairs, one per group found.
    """
    from rdkit import Chem

    found: list = []
    claimed: set = set()
    for label, smarts, charge in _IONIZABLE_GROUPS:
        pattern = Chem.MolFromSmarts(smarts)
        for match in mol.GetSubstructMatches(pattern):
            key_atom = match[0]
            if key_atom in cap_indices or rd_res_names.get(key_atom) in _BACKBONE_HEAVY:
                continue
            if claimed & set(match):
                continue
            claimed.update(match)
            found.append((label, charge))
    return found


def _build_capped_molecule(res, topology, pos_A):
    """Build an RDKit molecule for ``res`` with carbon caps at every external bond.

    Returns ``(mol, rd_res_names, cap_indices, ext_atom_names, cap_partner)`` where:
        mol           : RWMol (single bonds, 3D conformer set, no H yet)
        rd_res_names  : {rdkit_idx: topology_atom_name} for the residue heavy atoms
        cap_indices   : set of rdkit indices that are cap atoms
        ext_atom_names: ordered list of residue atom names that carry an external bond
        cap_partner   : {cap_rdkit_idx: external_partner_atom_name} — the *real*
                        neighbouring-residue atom the cap stands in for (e.g. the
                        previous residue's ``C`` or the next residue's ``N``), used
                        to give the junction its ff14SB atom class.
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
    cap_partner: dict = {}
    capped_pairs: set = set()
    for bond in topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        in1, in2 = a1.index in res_atom_indices, a2.index in res_atom_indices
        if in1 == in2:
            continue  # both inside or both outside the residue
        inner = a1 if in1 else a2
        outer = a2 if in1 else a1
        if inner.index not in rd_idx:
            continue  # boundary atom is not a heavy atom we model (defensive)
        # PDBFixer can list one bond twice (the 1XY4 IAM-THR peptide bond). A
        # second cap on the same atom gave the backbone C a valence of 5.
        if (inner.index, outer.index) in capped_pairs:
            continue
        capped_pairs.add((inner.index, outer.index))
        cap = rw.AddAtom(Chem.Atom(6))  # carbon cap → clean sp2/sp3 perception
        rw.AddBond(rd_idx[inner.index], cap, Chem.BondType.SINGLE)
        coords.append(pos_A[outer.index])
        cap_indices.add(cap)
        cap_partner[cap] = outer.name
        if inner.name not in ext_atom_names:
            ext_atom_names.append(inner.name)

    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf)
    return mol, rd_res_names, cap_indices, ext_atom_names, cap_partner


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


# Protein-backbone atom names whose GAFF types we override with ff14SB types.
_BACKBONE_HEAVY = BACKBONE_HEAVY_ATOM_NAMES
# Map an external-partner atom name to its ff14SB atom *class*, so a junction that
# crosses into the neighbouring residue (peptide C(i)–N(i+1), head-to-tail closure)
# is keyed to amber classes on both sides.  Extendable for exotic linkages.
_PARTNER_CLASS = {"C": "C", "N": "N", "CA": "CX", "O": "O", "SG": "S"}


def _amber_backbone_types(ff) -> Optional[dict]:
    """Read the canonical ff14SB backbone ``<Type>`` names/classes from ALA.

    Returns ``{atom_name: (type_name, class_name)}`` for N, H, CA, HA, C, O by
    parsing a loaded standard amino-acid residue template (rather than hardcoding
    the ff14SB type strings), or ``None`` if no template is available.
    """
    template = None
    for name in ("ALA", "LEU", "VAL", "SER"):
        template = getattr(ff, "_templates", {}).get(name)
        if template is not None:
            break
    if template is None:
        return None
    name_to_type = {a.name: a.type for a in template.atoms}
    result: dict = {}
    for nm in ("N", "H", "CA", "HA", "C", "O"):
        t = name_to_type.get(nm)
        if t is None:
            continue
        atype = ff._atomTypes.get(t)
        cls = atype.atomClass if atype is not None else None
        result[nm] = (t, cls)
    # Require the full protein backbone set for a usable retyping.
    if not _BACKBONE_HEAVY <= set(result) or "H" not in result or "HA" not in result:
        return None
    return result


def _parse_gaff_forces(root):
    """Index the GAFF-generated bonded parameters by atom class.

    Returns ``(bonds, angles, propers)``:
        bonds   : {frozenset({c1, c2}): {'length':…, 'k':…}}
        angles  : list of (c1, c2, c3, attrib_dict)  [c2 is the vertex]
        propers : list of (('c1','c2','c3','c4'), attrib_dict)
    """
    bonds: dict = {}
    bf = root.find("HarmonicBondForce")
    if bf is not None:
        for b in bf.findall("Bond"):
            key = frozenset({b.get("class1"), b.get("class2")})
            bonds[key] = {"length": b.get("length"), "k": b.get("k")}
    angles: list = []
    af = root.find("HarmonicAngleForce")
    if af is not None:
        for a in af.findall("Angle"):
            angles.append((a.get("class1"), a.get("class2"), a.get("class3"), a.attrib))
    propers: list = []
    tf = root.find("PeriodicTorsionForce")
    if tf is not None:
        for p in tf.findall("Proper"):
            propers.append(
                ((p.get("class1"), p.get("class2"), p.get("class3"), p.get("class4")), p.attrib)
            )
    return bonds, angles, propers


def _lookup_gaff_angle(angles, middle, ends):
    """First GAFF angle whose vertex class is ``middle`` and end classes == ``ends``."""
    ends = frozenset(ends)
    for c1, c2, c3, attrib in angles:
        if c2 == middle and frozenset({c1, c3}) == ends:
            return attrib
    return None


def _lookup_gaff_proper(propers, classes):
    """First GAFF proper matching the ordered class quartet (either direction)."""
    rev = tuple(reversed(classes))
    for cls, attrib in propers:
        if cls == classes or cls == rev:
            return attrib
    return None


#: sqm keywords that ``antechamber -c bcc`` writes for AM1-BCC (AmberTools 24), without
#: ``qmcharge``, which antechamber adds from ``-nc``. ``-ek`` replaces the whole set, so
#: the defaults are repeated here and only ``diag_routine`` is added.
_SQM_AM1BCC_KEYWORDS = "qm_theory='AM1', grms_tol=0.0005, scfconv=1.d-10, ndiis_attempts=700"

#: ``diag_routine=1`` selects sqm's own diagonaliser. With the default (0) sqm times
#: seven diagonalisation routines at start-up and keeps the fastest, so the routine, and
#: with it the rounding of every SCF step, depends on the load of the machine.
_SQM_DIAG_ROUTINE_INTERNAL = 1

#: RDKit's ETKDG prune threshold (Å) that the OpenFF toolkit uses for the AM1-BCC conformer.
_AM1BCC_CONFORMER_RMS_CUTOFF_ANGSTROM = 0.25


def _run_antechamber(args: list, workdir: str) -> None:
    """Run ``antechamber`` in ``workdir`` on one thread; raise ``RuntimeError`` on failure.

    sqm spreads a 40-atom molecule over every core the BLAS library offers and gains
    nothing: one thread takes as long, without the busy-waiting of the others.
    """
    import shutil
    import subprocess

    if shutil.which("antechamber") is None:
        raise RuntimeError(
            "antechamber (AmberTools) is not on PATH; it is needed for the AM1-BCC charges "
            "of non-canonical residues. Install with: conda install -c conda-forge ambertools"
        )
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = subprocess.run(
        ["antechamber", *args], cwd=workdir, env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
        raise RuntimeError(f"antechamber failed ({' '.join(args[:6])} ...): " + " | ".join(tail))


def _am1bcc_charges(molecule, random_seed: Optional[int] = DEFAULT_RANDOM_SEED) -> np.ndarray:
    """AM1-BCC partial charges (in e) of an OpenFF molecule, identical from run to run.

    Follows the protocol of ``AmberToolsToolkitWrapper.assign_partial_charges("am1bcc")``
    (Jakalian et al. 2000, 2002): one RDKit ETKDG conformer, pruned at 0.25 Å, then
    ``antechamber -c bcc``, which minimises the geometry with AM1 in sqm and adds the bond
    charge corrections. Two things differ from the toolkit call, both to make the result
    reproducible:

    * The conformer is embedded with ``random_seed`` (the toolkit fixes it at 1, which is
      the default here, so the default conformer is the toolkit's). ``None`` draws a new
      conformer per call. AM1-BCC charges depend on the conformer: those of MeBmt differ by
      up to 0.065 e between seeds 1 and 2, so keep the seed fixed to compare runs. The
      stereochemistry of the embedded conformer is that of ``molecule``; a molecule without
      stereochemistry gets a random stereoisomer for each seed.
    * sqm's diagonaliser is fixed (see ``_SQM_DIAG_ROUTINE_INTERNAL``). With the toolkit
      call sqm chose it by timing, and one of the choices ended the AM1 minimisation of a
      flexible molecule in another geometry: the AM1-BCC charges of MeBmt (cyclosporin A)
      differed by up to 0.013 e from one build to the next (up to 9e-4 e in the template,
      once the charge of the removed caps is spread over the residue).

    The charges are shifted evenly to sum to the total formal charge, as the toolkit does.

    Raises:
        RuntimeError: RDKit cannot embed the molecule, antechamber is missing or fails, or
            it returns a different number of charges than there are atoms.
    """
    import tempfile
    from pathlib import Path

    from openff.units import Quantity, unit
    from rdkit.Chem import AllChem

    total_charge = int(round(molecule.total_charge.m_as(unit.elementary_charge)))
    rdmol = molecule.to_rdkit()
    seed = -1 if random_seed is None else int(random_seed)  # RDKit: a negative seed is random
    embed_args = {
        "numConfs": 1,
        "pruneRmsThresh": _AM1BCC_CONFORMER_RMS_CUTOFF_ANGSTROM,
        "randomSeed": seed,
    }
    conformer_ids = AllChem.EmbedMultipleConfs(rdmol, **embed_args)
    if not len(conformer_ids):
        # The toolkit's fallback for molecules that fail with the default embedding.
        conformer_ids = AllChem.EmbedMultipleConfs(rdmol, useRandomCoords=True, **embed_args)
    if not len(conformer_ids):
        raise RuntimeError("RDKit could not embed a conformer for the AM1-BCC charge calculation")
    coordinates = rdmol.GetConformer(int(conformer_ids[0])).GetPositions()

    with_conformer = type(molecule)(molecule)
    with_conformer.clear_conformers()
    with_conformer.add_conformer(Quantity(coordinates, unit.angstrom))

    with tempfile.TemporaryDirectory() as workdir:
        with_conformer.to_file(str(Path(workdir, "molecule.sdf")), file_format="sdf")
        sqm_keywords = f"{_SQM_AM1BCC_KEYWORDS}, diag_routine={_SQM_DIAG_ROUTINE_INTERNAL}"
        _run_antechamber(
            ["-i", "molecule.sdf", "-fi", "sdf", "-o", "charged.mol2", "-fo", "mol2"]
            + ["-pf", "yes", "-dr", "n", "-c", "bcc", "-nc", str(total_charge)]
            + ["-ek", sqm_keywords],
            workdir,
        )
        # Second call only reads the charges out of the mol2 file.
        _run_antechamber(
            ["-dr", "n", "-i", "charged.mol2", "-fi", "mol2", "-o", "charges2.mol2"]
            + ["-fo", "mol2", "-c", "wc", "-cf", "charges.txt", "-pf", "yes"],
            workdir,
        )
        charges = np.array(Path(workdir, "charges.txt").read_text().split(), dtype=float)

    if charges.shape != (molecule.n_atoms,):
        raise RuntimeError(
            f"antechamber returned {charges.size} AM1-BCC charges for {molecule.n_atoms} atoms"
        )
    return charges + (total_charge - charges.sum()) / molecule.n_atoms


def _generate_residue_template(
    res,
    topology,
    pos_A,
    gaff_version: str,
    backbone_amber: Optional[dict] = None,
    *,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
):
    """Build one hybrid amber-backbone / GAFF-sidechain ExternalBond template.

    The protein backbone atoms (N, H, CA, HA, C, O) are typed with standard
    ff14SB protein atom types so every inter-residue junction (the peptide C–N
    bond, its angles and the ω/φ/ψ torsions) matches ff14SB natively; the
    sidechain keeps its GAFF2 types.  The boundary terms that straddle the two
    (the CA–CB bond, N-methyl N–C bond, and every angle/torsion mixing an amber
    backbone class with a GAFF sidechain class) are emitted explicitly with the
    GAFF force constants so nothing is silently dropped by ``createSystem``.

    Returns ``(ffxml_string, h_inject, neutral_groups, single_bond_reason)`` where
    ``h_inject`` is a list of ``(h_name, parent_atom_name, position_nm_ndarray)``
    for the hydrogens that must be injected into the topology residue,
    ``neutral_groups`` lists the ``(label, charge)`` of side-chain groups that
    ionise at pH 7.4 but were built neutral (see
    :func:`_neutral_ionizable_groups`), and ``single_bond_reason`` is empty when the
    bond orders come from the Chemical Component Dictionary, else the reason the
    residue was built with single bonds only. Returns ``None`` on failure.

    ``random_seed`` seeds the conformer of the AM1-BCC charge calculation (see
    :func:`_am1bcc_charges`); the same residue graph and seed give the same template.
    """
    from openff.toolkit import Molecule
    from openff.units import Quantity, unit
    from openmmforcefields.generators import GAFFTemplateGenerator
    from rdkit import Chem

    mol, rd_res_names, cap_indices, ext_atom_names, cap_partner = _build_capped_molecule(
        res, topology, pos_A
    )
    mol, single_bond_reason = _perceive_residue_bond_orders(mol, rd_res_names, res.name)
    mh = Chem.AddHs(mol, addCoords=True)
    hconf = mh.GetConformer()
    # Read the stereocentres and the E/Z double bonds from the geometry. The graph carries no
    # stereochemistry, and without it the conformer that sqm minimises is a random
    # stereoisomer: MeBmt (1CWA) was embedded as a diastereomer with a Z double bond.
    Chem.AssignStereochemistryFrom3D(mh)
    neutral_groups = _neutral_ionizable_groups(mh, rd_res_names, cap_indices)

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
    # Charges from here, not from GAFFTemplateGenerator: its own AM1-BCC call lets sqm pick
    # the diagonaliser by timing (see _am1bcc_charges). The generator keeps charges that are
    # already set on the molecule and only adds the GAFF2 types and bonded parameters.
    off.partial_charges = Quantity(_am1bcc_charges(off, random_seed), unit.elementary_charge)
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

    # --- Identify backbone atoms and drop spurious backbone-carbonyl hydrogens ---
    # With bond orders from the dictionary the backbone C=O is a carbonyl and none
    # of this fires. On the single-bond fallback the backbone carbonyl is an sp3
    # alcohol and gets a spurious H on C and an -OH on O. Those hydrogens have no
    # place on a real peptide carbonyl: drop them and let the amber C/O types (and
    # the ff14SB C–O bond + carbonyl improper) describe it.
    retype = backbone_amber or {}
    bb_heavy = {j for j, nm in rd_res_names.items() if nm in _BACKBONE_HEAVY and nm in retype}
    amide_h: set = set()
    alpha_h: set = set()
    spurious_h: set = set()
    for h_idx, parent in keep_h:
        pname = rd_res_names.get(parent)
        if parent not in bb_heavy:
            continue
        if pname == "N":
            amide_h.add(h_idx)
        elif pname == "CA":
            alpha_h.add(h_idx)
        elif pname in ("C", "O"):
            spurious_h.add(h_idx)
    keep_h = [(h, p) for (h, p) in keep_h if h not in spurious_h]

    # Name every template atom by its RDKit index.
    h_names = _hydrogen_names(keep_h, rd_res_names)
    new_name: dict = {}
    for j, nm in rd_res_names.items():
        new_name[j] = nm
    for h_idx, _parent in keep_h:
        new_name[h_idx] = h_names[h_idx]

    dropped = set(cap_indices) | drop_h | spurious_h
    keep_idx = [i for i in range(len(tatoms)) if i not in dropped]

    # --- New atom type per kept atom: amber for backbone, GAFF for sidechain ---
    orig_class = {i: tatoms[i].get("type") for i in range(len(tatoms))}
    new_type: dict = {}
    new_class: dict = {}
    is_amber: dict = {}  # atom carries an ff14SB (protein) class
    for i in keep_idx:
        nm = rd_res_names.get(i)
        if i in bb_heavy:
            t, c = retype[nm]
            new_type[i], new_class[i], is_amber[i] = t, c, True
        elif i in amide_h and "H" in retype:
            t, c = retype["H"]
            new_type[i], new_class[i], is_amber[i] = t, c, True
        elif i in alpha_h and "HA" in retype:
            t, c = retype["HA"]
            new_type[i], new_class[i], is_amber[i] = t, c, True
        else:
            new_type[i] = orig_class[i]
            new_class[i] = orig_class[i]
            is_amber[i] = False
    # Cap atoms stand in for the real neighbouring-residue atom: give them the
    # partner's ff14SB class so every junction term is keyed to amber on that side.
    for cap in cap_indices:
        pc = _PARTNER_CLASS.get(cap_partner.get(cap))
        if pc is not None:
            new_class[cap] = pc
            is_amber[cap] = True
        else:
            new_class[cap] = orig_class[cap]
            is_amber[cap] = False
    changed = {i: new_class[i] != orig_class[i] for i in new_class}

    # Redistribute dropped cap/cap-H/spurious-H charge so the template is neutral.
    # The integer target is taken from the set *before* dropping the spurious
    # carbonyl hydrogens (i.e. the true residue charge, normally 0); the dropped
    # H charge is then absorbed into the redistribution so the net stays integer.
    pre_drop = [i for i in range(len(tatoms)) if i not in (set(cap_indices) | drop_h)]
    target = round(sum(float(tatoms[i].get("charge")) for i in pre_drop))
    kept_charge = {i: float(tatoms[i].get("charge")) for i in keep_idx}
    total = sum(kept_charge.values())
    delta = (target - total) / len(keep_idx)
    for i in keep_idx:
        kept_charge[i] += delta

    # --- Rewrite the <Residue> block with the mixed types ---
    for el in list(resel):
        resel.remove(el)
    for i in keep_idx:
        a = ET.SubElement(resel, "Atom")
        a.set("name", new_name[i])
        a.set("type", new_type[i])
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

    # --- Emit explicit boundary parameters (amber-class ⟷ GAFF-class terms) ---
    _inject_boundary_terms(
        root, mh, keep_set, cap_indices, orig_class, new_class, is_amber, changed
    )

    ffxml_out = ET.tostring(root, encoding="unicode")

    h_inject = []
    for h_idx, parent in keep_h:
        p = hconf.GetAtomPosition(h_idx)
        h_inject.append(
            (new_name[h_idx], rd_res_names[parent], np.array([p.x, p.y, p.z]) / 10.0)  # Å → nm
        )
    return ffxml_out, h_inject, neutral_groups, single_bond_reason


def _inject_boundary_terms(
    root, mh, keep_set, cap_indices, orig_class, new_class, is_amber, changed
):
    """Add explicit Bond/Angle/Proper entries for every backbone↔sidechain term.

    After retyping the backbone to amber and keeping the sidechain on GAFF, any
    bonded term that mixes an amber protein class with a GAFF class matches
    neither ff14SB nor the GAFF template and would be *silently omitted* by
    ``createSystem``.  We therefore re-key the GAFF force constant for each such
    term to the new mixed classes.  Terms that are purely amber (handled by
    ff14SB, incl. its wildcard backbone torsions) or purely GAFF sidechain
    (already in the template) are left untouched — injecting them would
    double-count.

    Rules per term:
        bond  (i, j):        skip if both amber (ff14SB) or neither changed
                             (pure sidechain); else inject GAFF value.
        angle (i, j, k):     skip if all three amber, or none changed; else inject.
        proper(a, b, c, d):  skip if the central bond (b, c) is amber–amber
                             (ff14SB wildcard torsions cover it) or no atom
                             changed; else inject the GAFF value if one exists.
    """
    bonds, angles, propers = _parse_gaff_forces(root)

    node = keep_set | set(cap_indices)
    adj: dict = {i: [] for i in node}
    for bond in mh.GetBonds():
        i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i1 in node and i2 in node:
            adj[i1].append(i2)
            adj[i2].append(i1)

    bf = root.find("HarmonicBondForce")
    if bf is None:
        bf = ET.SubElement(root, "HarmonicBondForce")
    af = root.find("HarmonicAngleForce")
    if af is None:
        af = ET.SubElement(root, "HarmonicAngleForce")
    tf = root.find("PeriodicTorsionForce")
    if tf is None:
        tf = ET.SubElement(root, "PeriodicTorsionForce")

    seen_b: set = set()
    seen_a: set = set()
    seen_p: set = set()

    # Bonds ----------------------------------------------------------------
    for i in node:
        for j in adj[i]:
            if j <= i:
                continue
            if is_amber[i] and is_amber[j]:
                continue
            if not (changed[i] or changed[j]):
                continue
            val = bonds.get(frozenset({orig_class[i], orig_class[j]}))
            if val is None:
                continue
            key = frozenset({new_class[i], new_class[j]})
            if key in seen_b:
                continue
            seen_b.add(key)
            e = ET.SubElement(bf, "Bond")
            e.set("class1", new_class[i])
            e.set("class2", new_class[j])
            e.set("length", val["length"])
            e.set("k", val["k"])

    # Angles ---------------------------------------------------------------
    for j in node:
        nb = adj[j]
        for m in range(len(nb)):
            for n in range(m + 1, len(nb)):
                i, k = nb[m], nb[n]
                if is_amber[i] and is_amber[j] and is_amber[k]:
                    continue
                if not (changed[i] or changed[j] or changed[k]):
                    continue
                attrib = _lookup_gaff_angle(angles, orig_class[j], (orig_class[i], orig_class[k]))
                if attrib is None:
                    continue
                key = (new_class[j], frozenset({new_class[i], new_class[k]}))
                if key in seen_a:
                    continue
                seen_a.add(key)
                e = ET.SubElement(af, "Angle")
                e.set("class1", new_class[i])
                e.set("class2", new_class[j])
                e.set("class3", new_class[k])
                e.set("angle", attrib["angle"])
                e.set("k", attrib["k"])

    # Propers --------------------------------------------------------------
    for b in node:
        for c in adj[b]:
            if c <= b:
                continue
            if is_amber[b] and is_amber[c]:
                continue  # amber central bond → ff14SB (wildcard) torsions cover it
            for a in adj[b]:
                if a == c:
                    continue
                for d in adj[c]:
                    if d == b or d == a:
                        continue
                    if not (changed[a] or changed[b] or changed[c] or changed[d]):
                        continue
                    attrib = _lookup_gaff_proper(
                        propers,
                        (orig_class[a], orig_class[b], orig_class[c], orig_class[d]),
                    )
                    if attrib is None:
                        continue  # GAFF has no torsion here (k=0) → nothing to add
                    quartet = (new_class[a], new_class[b], new_class[c], new_class[d])
                    canon = min(quartet, tuple(reversed(quartet)))
                    if canon in seen_p:
                        continue
                    seen_p.add(canon)
                    e = ET.SubElement(tf, "Proper")
                    e.set("class1", new_class[a])
                    e.set("class2", new_class[b])
                    e.set("class3", new_class[c])
                    e.set("class4", new_class[d])
                    for ak, av in attrib.items():
                        if ak.startswith(("periodicity", "phase", "k")):
                            e.set(ak, av)


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
    from openmm import unit
    from openmm.app import Topology
    from openmm.app import element as elem

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


class NcaaTemplateList(list):
    """The generated force-field XML strings, plus per-residue charge bookkeeping.

    Behaves as the plain ``list`` of XML strings (one per unique residue name) that
    :func:`parameterize_ncaa_residues` has always returned.

    Attributes:
        net_charge_by_residue: ``{residue name: net charge in e of its template}``.
            The template is an integer: bond-order perception runs at total
            charge 0, so this is 0 unless the perceived graph carries formal
            charges.
        neutral_ionizable_groups: ``{residue name: [group label, ...]}`` for
            residues that carry a side-chain group charged at pH 7.4 (carboxylic
            acid, phosphate, sulfate, primary amine, guanidine) yet were
            parameterised in the neutral form. Residues without such a group are
            absent.
        bond_order_source_by_residue: ``{residue name: "ccd" or "single_bonds"}``.
            ``"ccd"`` means the bond orders come from the Chemical Component
            Dictionary. ``"single_bonds"`` means the residue was not found there
            (or disagrees with its entry) and was built with every bond single, so
            its unsaturation, aromaticity and hydrogen count are unreliable; a
            warning is logged for each of those.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.net_charge_by_residue: dict = {}
        self.neutral_ionizable_groups: dict = {}
        self.bond_order_source_by_residue: dict = {}


def parameterize_ncaa_residues(
    topology,
    positions,
    ff,
    *,
    gaff_version: str = "gaff-2.2.20",
    verbose: bool = True,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
):
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
        verbose: Log a one-line summary per generated template, and the warnings
            of the template step, through the module logger.
        random_seed: Seed of the conformer that the AM1-BCC charge calculation
            starts from. The same residue graph and seed give the same template in
            every process and at every pipeline step; the default matches the
            conformer the OpenFF toolkit has always used. ``None`` draws a new
            conformer per call, so the charges can differ between calls.

    Returns:
        ``(topology, positions, ncaa_ffxmls)`` — the (possibly rebuilt) topology
        and positions, and the list of generated force-field XML strings (one per
        unique NCAA residue name; empty if there were none).  The XMLs are already
        loaded into ``ff``; callers that build *separate* force fields (e.g. the
        peptide/receptor subsystems in the energy decomposition) must reload them.
        The list is an :class:`NcaaTemplateList`, which also carries the net
        charge of each template and the residues built neutral although they hold
        a group charged at pH 7.4; a warning is logged for each of those.
    """
    ncaa_ffxmls = NcaaTemplateList()
    ncaa_residues = [res for res in topology.residues() if _is_ncaa(res)]
    if not ncaa_residues:
        return topology, positions, ncaa_ffxmls

    try:
        import openmmforcefields  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "openmmforcefields is required to parameterise non-canonical residues. "
            "Install with: conda install -c conda-forge openmmforcefields openff-toolkit"
        ) from exc

    pos_A = _pos_to_angstrom(positions)
    pos_nm = pos_A / 10.0

    # Canonical ff14SB backbone types (read from a standard residue template, not
    # hardcoded) used to retype the NCAA protein backbone so junctions match ff14SB.
    backbone_amber = _amber_backbone_types(ff)
    if backbone_amber is None and verbose:
        logger.warning(
            "  [warning] could not read ff14SB backbone types; "
            "NCAA backbones stay on GAFF (junctions may be under-parameterised)."
        )

    loaded_names: set = set()
    h_by_res: dict = {}
    expected_h: dict = {}  # name -> tuple of kept H names (consistency guard)

    for res in ncaa_residues:
        try:
            result = _generate_residue_template(
                res, topology, pos_A, gaff_version, backbone_amber, random_seed=random_seed
            )
        except Exception as exc:  # noqa: BLE001 - per-residue isolation; the residue is skipped
            if verbose:
                logger.warning("  [warning] GAFF NCAA template failed for '%s': %s", res.name, exc)
            else:
                logger.debug("GAFF NCAA template failed for '%s': %s", res.name, exc)
            result = None
        if result is None:
            continue
        ffxml, h_inject, neutral_groups, single_bond_reason = result
        h_by_res[res.index] = h_inject
        h_names = tuple(h[0] for h in h_inject)

        if res.name not in loaded_names:
            _load_ffxml(ff, ffxml)
            loaded_names.add(res.name)
            expected_h[res.name] = h_names
            ncaa_ffxmls.append(ffxml)
            net = _template_net_charge(ffxml)
            ncaa_ffxmls.net_charge_by_residue[res.name] = round(net, 4)  # drop XML rounding noise
            ncaa_ffxmls.bond_order_source_by_residue[res.name] = (
                BOND_ORDER_SOURCE_SINGLE_BONDS if single_bond_reason else BOND_ORDER_SOURCE_CCD
            )
            if single_bond_reason:
                logger.warning(
                    "GAFF NCAA '%s' is built with single bonds only (%s), so double "
                    "bonds, aromatic rings and the hydrogen count are unreliable. Use "
                    "the Chemical Component Dictionary residue code and atom names to "
                    "get the right chemistry.",
                    res.name,
                    single_bond_reason,
                )
            if neutral_groups:
                ncaa_ffxmls.neutral_ionizable_groups[res.name] = [g for g, _ in neutral_groups]
                logger.warning(
                    "GAFF NCAA '%s' is assumed neutral (template net charge %+.2f) but "
                    "carries group(s) charged at pH 7.4: %s. Expected net charge is about "
                    "%+d, so electrostatics and energies involving this residue are "
                    "unreliable unless its charge is set by hand.",
                    res.name,
                    net,
                    ", ".join(g for g, _ in neutral_groups),
                    round(net) + sum(c for _, c in neutral_groups),
                )
            if verbose:
                logger.info(
                    "  Auto-GAFF2: '%s' template generated (%d H, net charge %+.4f)",
                    res.name,
                    len(h_inject),
                    net,
                )
        elif h_names != expected_h.get(res.name):
            # A second instance perceived differently — reuse the first template
            # but warn; createSystem will surface a mismatch if truly incompatible.
            if verbose:
                logger.warning(
                    "  [warning] '%s' instance differs from first "
                    "template; reusing first (H %d vs %d).",
                    res.name,
                    len(h_names),
                    len(expected_h.get(res.name, ())),
                )

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
