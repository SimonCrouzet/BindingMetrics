"""Cyclic peptide topology detection and patching for OpenMM.

Supported cyclization types (detected by inter-atom distance on heavy atoms):
    head_to_tail  : backbone C(last) — N(first) amide bond
    disulfide     : CYS SG — SG (residues renamed to CYX)
    lactam_n_asp  : ASP sidechain CG — backbone N-terminus amide
    lactam_n_glu  : GLU sidechain CD — backbone N-terminus amide
    lactam_c_lys  : LYS sidechain NZ — backbone C-terminus amide

Unsupported cyclization types (raises CyclizationError with guidance):
    - Hydrocarbon staples (all-carbon bridges)
    - Thioether bridges (non-disulfide S-C)
    - Macrolactone / macrolactam via Ser/Thr/Tyr
    - Biaryl ethers (vancomycin-type)
    - Any other non-standard crosslink

Lactam residue templates (ASPL, GLUL, LYSL) use approximate AMBER-analog
charges derived from the nearest ASN/GLN ff14SB analogs. For high-accuracy
MD (production runs, free-energy calculations), recompute RESP charges with
a quantum chemistry program and supply custom XML via custom_bond_handler.

Usage (via RelaxationConfig):
    >>> config = RelaxationConfig(is_cyclic=True)
    >>> relaxer = ImplicitRelaxation(config)
    >>> result = relaxer.run(Path("cyclic_peptide_complex.cif"), Path("out/"))
"""

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Threshold constants (nm)
# ---------------------------------------------------------------------------
_AMIDE_BOND_THRESH = 0.20   # N–C amide bond detection
_DISULFIDE_THRESH  = 0.26   # S–S disulfide detection
_SUSPECT_THRESH    = 0.22   # any short inter-residue contact (potential cyclic)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class CyclizationError(ValueError):
    """Raised for unsupported or unrecognised cyclization types."""


_UNSUPPORTED_MSG = """
Unsupported cyclization type detected in the peptide chain.

BindingMetrics supports automatic topology patching for:
  • Head-to-tail amide   (backbone C-terminus → N-terminus)
  • Disulfide bridge     (CYS SG — SG, residues renamed CYX)
  • Lactam N-terminal    (ASP CG or GLU CD → backbone N-terminus)
  • Lactam C-terminal    (LYS NZ → backbone C-terminus)

Your structure appears to contain a different cyclization:
  • Hydrocarbon staple   (all-carbon bridge, e.g. Aileron-type)
  • Thioether bridge     (S–C, e.g. lanthipeptide)
  • Macrolactone/ester   (Ser/Thr/Tyr O → carbonyl C)
  • Biaryl ether         (vancomycin-type)

To handle these, pass a custom_bond_handler to RelaxationConfig that:
  1. Adds the closure bond to the OpenMM Topology.
  2. Returns extra_ff_xmls with GAFF2 or SMIRNOFF parameters.

Example using openmmforcefields:
    from openmmforcefields.generators import GAFFTemplateGenerator
    ...
"""


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CyclicBondInfo:
    """Metadata about the detected cyclic bond, used downstream for restraints.

    Attributes:
        cyclic_type: One of 'head_to_tail', 'disulfide', 'lactam_n_asp',
            'lactam_n_glu', 'lactam_c_lys'.
        atom1_id: (chain_id, residue_index_in_chain, atom_name) for atom 1.
        atom2_id: Same for atom 2.
        omega_ids: 4-tuple of (chain_id, res_idx, atom_name) for the ω
            dihedral across the closure bond (C–N amide or C–S–S–C for
            disulfide). Used for dihedral restraints during MD warmup.
            None for disulfide (no ω dihedral).
        extra_ff_xmls: List of AMBER-format XML strings to load into the
            ForceField for lactam residue templates (ASPL, GLUL, LYSL).
    """
    cyclic_type: str
    atom1_id: tuple   # (chain_id, res_idx_in_chain, atom_name)
    atom2_id: tuple
    omega_ids: Optional[tuple] = None   # 4 × (chain_id, res_idx, atom_name)
    extra_ff_xmls: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lactam residue XML templates
# Charges: approximate AMBER-analog values derived from ASN/GLN/neutral-LYS
# ff14SB; adjusted so each residue is exactly neutral. Suitable for
# scoring and exploratory MD. Recompute RESP charges for production work.
# ---------------------------------------------------------------------------

# ASPL – ASP with sidechain CG acting as amide carbonyl, OD1 as amide O.
# OD2 is absent. CG has an ExternalBond for the ring-closure amide.
# Charges derived from ASN ff14SB (CG=0.7257 adjusted for neutrality).
_XML_ASPL = """\
<ForceField>
 <Residues>
  <Residue name="ASPL">
   <Atom name="N"   type="protein-N"   charge="-0.4157"/>
   <Atom name="H"   type="protein-H"   charge=" 0.2719"/>
   <Atom name="CA"  type="protein-CT"  charge="-0.0966"/>
   <Atom name="HA"  type="protein-H1"  charge=" 0.1231"/>
   <Atom name="CB"  type="protein-CT"  charge="-0.2041"/>
   <Atom name="HB2" type="protein-HC"  charge=" 0.0797"/>
   <Atom name="HB3" type="protein-HC"  charge=" 0.0797"/>
   <Atom name="CG"  type="protein-C"   charge=" 0.7257"/>
   <Atom name="OD1" type="protein-O"   charge="-0.5931"/>
   <Atom name="C"   type="protein-C"   charge=" 0.5973"/>
   <Atom name="O"   type="protein-O"   charge="-0.5679"/>
   <Bond atomName1="N"   atomName2="CA"/>
   <Bond atomName1="CA"  atomName2="CB"/>
   <Bond atomName1="CA"  atomName2="C"/>
   <Bond atomName1="CB"  atomName2="CG"/>
   <Bond atomName1="CG"  atomName2="OD1"/>
   <Bond atomName1="C"   atomName2="O"/>
   <ExternalBond atomName="N"/>
   <ExternalBond atomName="C"/>
   <ExternalBond atomName="CG"/>
  </Residue>
 </Residues>
</ForceField>
"""

# GLUL – GLU with sidechain CD acting as amide carbonyl, OE1 as amide O.
# OE2 is absent. CD has an ExternalBond for the ring-closure amide.
# Charges derived from GLN ff14SB (CD=0.6231 adjusted for neutrality).
_XML_GLUL = """\
<ForceField>
 <Residues>
  <Residue name="GLUL">
   <Atom name="N"   type="protein-N"   charge="-0.4157"/>
   <Atom name="H"   type="protein-H"   charge=" 0.2719"/>
   <Atom name="CA"  type="protein-CT"  charge="-0.0597"/>
   <Atom name="HA"  type="protein-H1"  charge=" 0.1231"/>
   <Atom name="CB"  type="protein-CT"  charge="-0.0036"/>
   <Atom name="HB2" type="protein-HC"  charge=" 0.0171"/>
   <Atom name="HB3" type="protein-HC"  charge=" 0.0171"/>
   <Atom name="CG"  type="protein-CT"  charge="-0.0645"/>
   <Atom name="HG2" type="protein-HC"  charge=" 0.0352"/>
   <Atom name="HG3" type="protein-HC"  charge=" 0.0352"/>
   <Atom name="CD"  type="protein-C"   charge=" 0.6231"/>
   <Atom name="OE1" type="protein-O"   charge="-0.6086"/>
   <Atom name="C"   type="protein-C"   charge=" 0.5973"/>
   <Atom name="O"   type="protein-O"   charge="-0.5679"/>
   <Bond atomName1="N"   atomName2="CA"/>
   <Bond atomName1="CA"  atomName2="CB"/>
   <Bond atomName1="CA"  atomName2="C"/>
   <Bond atomName1="CB"  atomName2="CG"/>
   <Bond atomName1="CG"  atomName2="CD"/>
   <Bond atomName1="CD"  atomName2="OE1"/>
   <Bond atomName1="C"   atomName2="O"/>
   <ExternalBond atomName="N"/>
   <ExternalBond atomName="C"/>
   <ExternalBond atomName="CD"/>
  </Residue>
 </Residues>
</ForceField>
"""

# LYSL – LYS with NZ acting as amide N, one HZ as amide proton.
# NZ has an ExternalBond for the ring-closure amide to C-terminus C.
# Charges: backbone from LYS ff14SB; NZ/HZ approximate amide N/H;
# CE=-0.1032 adjusted for neutrality.
_XML_LYSL = """\
<ForceField>
 <Residues>
  <Residue name="LYSL">
   <Atom name="N"   type="protein-N"   charge="-0.4157"/>
   <Atom name="H"   type="protein-H"   charge=" 0.2719"/>
   <Atom name="CA"  type="protein-CT"  charge="-0.0597"/>
   <Atom name="HA"  type="protein-H1"  charge=" 0.1231"/>
   <Atom name="CB"  type="protein-CT"  charge="-0.0894"/>
   <Atom name="HB2" type="protein-HC"  charge=" 0.0621"/>
   <Atom name="HB3" type="protein-HC"  charge=" 0.0621"/>
   <Atom name="CG"  type="protein-CT"  charge="-0.0416"/>
   <Atom name="HG2" type="protein-HC"  charge=" 0.0620"/>
   <Atom name="HG3" type="protein-HC"  charge=" 0.0620"/>
   <Atom name="CD"  type="protein-CT"  charge="-0.0238"/>
   <Atom name="HD2" type="protein-HC"  charge=" 0.0523"/>
   <Atom name="HD3" type="protein-HC"  charge=" 0.0523"/>
   <Atom name="CE"  type="protein-CT"  charge="-0.1032"/>
   <Atom name="HE2" type="protein-HP"  charge=" 0.0500"/>
   <Atom name="HE3" type="protein-HP"  charge=" 0.0500"/>
   <Atom name="NZ"  type="protein-N"   charge="-0.4157"/>
   <Atom name="HZ1" type="protein-H"   charge=" 0.2719"/>
   <Atom name="C"   type="protein-C"   charge=" 0.5973"/>
   <Atom name="O"   type="protein-O"   charge="-0.5679"/>
   <Bond atomName1="N"   atomName2="CA"/>
   <Bond atomName1="CA"  atomName2="CB"/>
   <Bond atomName1="CA"  atomName2="C"/>
   <Bond atomName1="CB"  atomName2="CG"/>
   <Bond atomName1="CG"  atomName2="CD"/>
   <Bond atomName1="CD"  atomName2="CE"/>
   <Bond atomName1="CE"  atomName2="NZ"/>
   <Bond atomName1="NZ"  atomName2="HZ1"/>
   <Bond atomName1="C"   atomName2="O"/>
   <ExternalBond atomName="N"/>
   <ExternalBond atomName="C"/>
   <ExternalBond atomName="NZ"/>
  </Residue>
 </Residues>
</ForceField>
"""

_LACTAM_XMLS = {
    "lactam_n_asp": _XML_ASPL,
    "lactam_n_glu": _XML_GLUL,
    "lactam_c_lys": _XML_LYSL,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pos_nm(positions) -> np.ndarray:
    """Return positions as (N, 3) float array in nm."""
    try:
        # OpenMM Quantity
        return np.array([[p.x, p.y, p.z] for p in positions])
    except AttributeError:
        return np.asarray(positions)


def _dist(pos: np.ndarray, i: int, j: int) -> float:
    """Euclidean distance in nm between atoms i and j."""
    d = pos[i] - pos[j]
    return float(np.sqrt(d @ d))


def _peptide_residues(topology, chain_id: str):
    """Return list of Residue objects for the given chain, ordered by index."""
    for chain in topology.chains():
        if chain.id == chain_id:
            return list(chain.residues())
    raise ValueError(f"Chain '{chain_id}' not found in topology.")


def _find_atom(residue, name: str):
    """Return the Atom with the given name in residue, or None."""
    for atom in residue.atoms():
        if atom.name == name:
            return atom
    return None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_cyclization(topology, positions, chain_id: str) -> list:
    """Detect all cyclizations in the peptide chain by inter-atom distance.

    Exhaustively scans for all supported cyclization patterns (a single peptide
    can have more than one, e.g. SFTI-1 is bicyclic: head-to-tail + disulfide).

    Args:
        topology: OpenMM Topology (heavy atoms only, post-PDBFixer repair).
        positions: Corresponding atom positions (OpenMM Quantity or ndarray, nm).
        chain_id: Chain ID of the peptide to examine.

    Returns:
        List of CyclicBondInfo (one entry per detected bond). Empty list if the
        peptide is linear.

    Raises:
        CyclizationError: If a short inter-residue contact is found that does
            not match any supported pattern (unsupported cyclization).
    """
    pos = _pos_nm(positions)
    residues = _peptide_residues(topology, chain_id)
    if len(residues) < 2:
        return []

    results: list = []
    # Track detected atom-index pairs so the unsupported scan ignores them.
    detected_pairs: set = set()

    first, last = residues[0], residues[-1]
    n_first = _find_atom(first, "N")
    c_last  = _find_atom(last,  "C")

    if n_first is None:
        warnings.warn(
            f"detect_cyclization: atom N not found on first residue "
            f"{first.name}{first.id} (chain {chain_id}); "
            "head-to-tail amide and N-terminal lactam bonds may not be detected.",
            stacklevel=2,
        )
    if c_last is None:
        warnings.warn(
            f"detect_cyclization: atom C not found on last residue "
            f"{last.name}{last.id} (chain {chain_id}); "
            "head-to-tail amide and C-terminal lactam bonds may not be detected.",
            stacklevel=2,
        )

    # ---- 1. Head-to-tail amide: C(last) — N(first) ----
    if n_first is not None and c_last is not None:
        if _dist(pos, n_first.index, c_last.index) < _AMIDE_BOND_THRESH:
            ca_last  = _find_atom(last,  "CA")
            ca_first = _find_atom(first, "CA")
            omega = None
            if ca_last and ca_first:
                omega = (
                    (chain_id, len(residues) - 1, "CA"),
                    (chain_id, len(residues) - 1, "C"),
                    (chain_id, 0,                 "N"),
                    (chain_id, 0,                 "CA"),
                )
            results.append(CyclicBondInfo(
                cyclic_type="head_to_tail",
                atom1_id=(chain_id, len(residues) - 1, "C"),
                atom2_id=(chain_id, 0,                 "N"),
                omega_ids=omega,
            ))
            detected_pairs.add((c_last.index, n_first.index))
            detected_pairs.add((n_first.index, c_last.index))

    # ---- 2. Disulfide: SG — SG (all CYS pairs, supports multiple disulfides) ----
    cys_residues = [(i, r) for i, r in enumerate(residues) if r.name == "CYS"]
    for i, (ri, res_i) in enumerate(cys_residues):
        sg_i = _find_atom(res_i, "SG")
        if sg_i is None:
            warnings.warn(
                f"detect_cyclization: atom SG not found on CYS residue "
                f"{res_i.name}{res_i.id} (chain {chain_id}, index {ri}); "
                "disulfide bond may not be detected.",
                stacklevel=2,
            )
            continue
        for rj, res_j in cys_residues[i + 1:]:
            sg_j = _find_atom(res_j, "SG")
            if sg_j is None:
                warnings.warn(
                    f"detect_cyclization: atom SG not found on CYS residue "
                    f"{res_j.name}{res_j.id} (chain {chain_id}, index {rj}); "
                    "disulfide bond may not be detected.",
                    stacklevel=2,
                )
                continue
            if _dist(pos, sg_i.index, sg_j.index) < _DISULFIDE_THRESH:
                results.append(CyclicBondInfo(
                    cyclic_type="disulfide",
                    atom1_id=(chain_id, ri, "SG"),
                    atom2_id=(chain_id, rj, "SG"),
                    omega_ids=None,
                ))
                detected_pairs.add((sg_i.index, sg_j.index))
                detected_pairs.add((sg_j.index, sg_i.index))

    # ---- 3. Lactam ASP/GLU sidechain — N-terminus ----
    if n_first is not None:
        for i, res in enumerate(residues):
            if res.name == "ASP":
                cg = _find_atom(res, "CG")
                if cg is None:
                    warnings.warn(
                        f"detect_cyclization: atom CG not found on ASP residue "
                        f"{res.name}{res.id} (chain {chain_id}, index {i}); "
                        "lactam_n_asp bond may not be detected.",
                        stacklevel=2,
                    )
                if cg and _dist(pos, cg.index, n_first.index) < _AMIDE_BOND_THRESH:
                    ca_res   = _find_atom(res,   "CA")
                    ca_first = _find_atom(first, "CA")
                    omega = None
                    if ca_res and ca_first:
                        omega = (
                            (chain_id, i, "CA"),
                            (chain_id, i, "CG"),
                            (chain_id, 0, "N"),
                            (chain_id, 0, "CA"),
                        )
                    results.append(CyclicBondInfo(
                        cyclic_type="lactam_n_asp",
                        atom1_id=(chain_id, i, "CG"),
                        atom2_id=(chain_id, 0, "N"),
                        omega_ids=omega,
                        extra_ff_xmls=[_XML_ASPL],
                    ))
                    detected_pairs.add((cg.index, n_first.index))
                    detected_pairs.add((n_first.index, cg.index))
            elif res.name == "GLU":
                cd = _find_atom(res, "CD")
                if cd is None:
                    warnings.warn(
                        f"detect_cyclization: atom CD not found on GLU residue "
                        f"{res.name}{res.id} (chain {chain_id}, index {i}); "
                        "lactam_n_glu bond may not be detected.",
                        stacklevel=2,
                    )
                if cd and _dist(pos, cd.index, n_first.index) < _AMIDE_BOND_THRESH:
                    ca_res   = _find_atom(res,   "CA")
                    ca_first = _find_atom(first, "CA")
                    omega = None
                    if ca_res and ca_first:
                        omega = (
                            (chain_id, i, "CA"),
                            (chain_id, i, "CD"),
                            (chain_id, 0, "N"),
                            (chain_id, 0, "CA"),
                        )
                    results.append(CyclicBondInfo(
                        cyclic_type="lactam_n_glu",
                        atom1_id=(chain_id, i, "CD"),
                        atom2_id=(chain_id, 0, "N"),
                        omega_ids=omega,
                        extra_ff_xmls=[_XML_GLUL],
                    ))
                    detected_pairs.add((cd.index, n_first.index))
                    detected_pairs.add((n_first.index, cd.index))

    # ---- 4. Lactam LYS sidechain — C-terminus ----
    if c_last is not None:
        for i, res in enumerate(residues):
            if res.name == "LYS":
                nz = _find_atom(res, "NZ")
                if nz is None:
                    warnings.warn(
                        f"detect_cyclization: atom NZ not found on LYS residue "
                        f"{res.name}{res.id} (chain {chain_id}, index {i}); "
                        "lactam_c_lys bond may not be detected.",
                        stacklevel=2,
                    )
                if nz and _dist(pos, nz.index, c_last.index) < _AMIDE_BOND_THRESH:
                    ca_last = _find_atom(last, "CA")
                    ca_res  = _find_atom(res,  "CA")
                    omega = None
                    if ca_last and ca_res:
                        omega = (
                            (chain_id, len(residues) - 1, "CA"),
                            (chain_id, len(residues) - 1, "C"),
                            (chain_id, i,                 "NZ"),
                            (chain_id, i,                 "CE"),
                        )
                    results.append(CyclicBondInfo(
                        cyclic_type="lactam_c_lys",
                        atom1_id=(chain_id, len(residues) - 1, "C"),
                        atom2_id=(chain_id, i,                 "NZ"),
                        omega_ids=omega,
                        extra_ff_xmls=[_XML_LYSL],
                    ))
                    detected_pairs.add((c_last.index, nz.index))
                    detected_pairs.add((nz.index, c_last.index))

    # ---- 5. Catch unsupported cyclization: suspicious short contact not already detected ----
    # Only scan heavy atoms — H atoms are within H-bond distance of acceptors and
    # would produce false-positive "unsupported cyclization" errors.
    atom_list = [a for r in residues for a in r.atoms()
                 if a.element is not None and a.element.symbol != "H"]

    # Build 1-2 (directly bonded) and 1-3 (angle) exclusion sets.
    # 1-3 contacts arise naturally at amide junctions (e.g. N···O=C), and must
    # not be mistaken for unsupported cyclizations.
    # Also include detected closure bonds in the neighbor graph: PDBFixer may
    # have dropped them from the topology, but they are real covalent bonds and
    # their 1-3 contacts (e.g. GLY1.N — ASP14.O through the head-to-tail amide)
    # must be excluded from the unsupported scan.
    bonded_12: set = set()
    neighbors: dict = {}
    for bond in topology.bonds():
        i, j = bond.atom1.index, bond.atom2.index
        bonded_12.add((i, j))
        bonded_12.add((j, i))
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)
    # Inject detected closure bonds into the graph
    for (i, j) in detected_pairs:
        bonded_12.add((i, j))
        bonded_12.add((j, i))
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)

    bonded_13: set = set()
    for atom in atom_list:
        for nb1 in neighbors.get(atom.index, set()):
            for nb2 in neighbors.get(nb1, set()):
                if nb2 != atom.index:
                    bonded_13.add((atom.index, nb2))
                    bonded_13.add((nb2, atom.index))

    # 1-4 contacts (e.g. O···CA across a peptide bond: O=C–N–CA) are normal
    # backbone geometry and must not be flagged as unsupported cyclizations.
    bonded_14: set = set()
    for atom in atom_list:
        for nb1 in neighbors.get(atom.index, set()):
            for nb2 in neighbors.get(nb1, set()):
                for nb3 in neighbors.get(nb2, set()):
                    if nb3 != atom.index:
                        bonded_14.add((atom.index, nb3))
                        bonded_14.add((nb3, atom.index))

    excluded = bonded_12 | bonded_13 | bonded_14 | detected_pairs

    res_of = {a.index: a.residue for a in atom_list}
    for ii, ai in enumerate(atom_list):
        for aj in atom_list[ii + 1:]:
            if res_of[ai.index] is res_of[aj.index]:
                continue
            if (ai.index, aj.index) in excluded:
                continue
            if _dist(pos, ai.index, aj.index) < _SUSPECT_THRESH:
                raise CyclizationError(
                    f"Potential unsupported cyclization detected: "
                    f"{ai.residue.name}{ai.residue.id}.{ai.name} — "
                    f"{aj.residue.name}{aj.residue.id}.{aj.name} "
                    f"({_dist(pos, ai.index, aj.index)*10:.2f} Å).\n"
                    + _UNSUPPORTED_MSG
                )

    return results


# ---------------------------------------------------------------------------
# Topology patching
# ---------------------------------------------------------------------------

def patch_cyclic_topology(topology, positions, chain_id: str):
    """Detect and patch all cyclizations in the peptide topology.

    Must be called AFTER PDBFixer heavy-atom repair and BEFORE
    ``modeller.addHydrogens()``.  Handles monocyclic and bicyclic (or higher)
    peptides by applying every detected cyclization in sequence.

    Per cyclization type, the function:
    - Removes spurious terminal atoms added by PDBFixer (OXT, HG on SG).
    - Renames CYS → CYX for disulfide partners.
    - Renames ASP/GLU/LYS → ASPL/GLUL/LYSL for lactam partners.
    - Adds the closure covalent bond if not already present.

    Args:
        topology: OpenMM Topology (heavy atoms only).
        positions: Atom positions (OpenMM Quantity, nm).
        chain_id: Peptide chain ID.

    Returns:
        Tuple ``(topology, positions, bond_info_list)`` where bond_info_list is
        a list of CyclicBondInfo (one per detected bond), or ``[]`` if linear.
    """
    try:
        from openmm import app
    except ImportError as e:
        raise ImportError("OpenMM is required for cyclic peptide patching.") from e

    info_list = detect_cyclization(topology, positions, chain_id)
    if not info_list:
        return topology, positions, []

    for info in info_list:
        # Residue list must be re-fetched after each patch (atom indices shift)
        residues = _peptide_residues(topology, chain_id)

        if info.cyclic_type == "head_to_tail":
            topology, positions = _patch_head_to_tail(topology, positions, residues, app)

        elif info.cyclic_type == "disulfide":
            ri = info.atom1_id[1]
            rj = info.atom2_id[1]
            topology, positions = _patch_disulfide(
                topology, positions, residues, ri, rj, app
            )

        elif info.cyclic_type in ("lactam_n_asp", "lactam_n_glu"):
            sidechain_res_idx = info.atom1_id[1]
            sidechain_atom    = info.atom1_id[2]
            new_name = "ASPL" if info.cyclic_type == "lactam_n_asp" else "GLUL"
            od_name  = "OD2"  if info.cyclic_type == "lactam_n_asp" else "OE2"
            topology, positions = _patch_lactam_n(
                topology, positions, residues,
                sidechain_res_idx, sidechain_atom, od_name, new_name, app,
            )

        elif info.cyclic_type == "lactam_c_lys":
            lys_res_idx = info.atom2_id[1]
            topology, positions = _patch_lactam_c_lys(
                topology, positions, residues, lys_res_idx, app
            )

    # Verify all closure atoms are still findable after patches
    residues = _peptide_residues(topology, chain_id)
    for info in info_list:
        _refresh_indices(info, residues, chain_id)

    return topology, positions, info_list


def _patch_head_to_tail(topology, positions, residues, app):
    """Remove OXT / extra terminal H from linear-peptide artefacts, add N–C bond."""
    first, last = residues[0], residues[-1]

    # Atoms to remove: OXT on last residue (added by PDBFixer treating it as linear)
    # and any pre-existing terminal H atoms (H1/H2/H3) on first residue's N.
    to_remove = []
    for atom in last.atoms():
        if atom.name == "OXT":
            to_remove.append(atom)
    for atom in first.atoms():
        if atom.name in ("H1", "H2", "H3"):
            to_remove.append(atom)

    if to_remove:
        modeller = app.Modeller(topology, positions)
        modeller.delete(to_remove)
        topology, positions = modeller.topology, modeller.positions

    # Re-find atoms after potential deletion
    residues = _peptide_residues(topology, residues[0].chain.id)
    first, last = residues[0], residues[-1]
    n_first = _find_atom(first, "N")
    c_last  = _find_atom(last,  "C")

    # Add bond only if not already present (e.g. read directly from CIF CONECT)
    if n_first and c_last:
        already_bonded = any(
            (b.atom1.index == n_first.index and b.atom2.index == c_last.index) or
            (b.atom2.index == n_first.index and b.atom1.index == c_last.index)
            for b in topology.bonds()
        )
        if not already_bonded:
            topology.addBond(n_first, c_last)

    return topology, positions


def _patch_disulfide(topology, positions, residues, ri: int, rj: int, app):
    """Rename CYS → CYX and add SG–SG bond."""
    res_i = residues[ri]
    res_j = residues[rj]

    # Remove HG (thiol proton) if present — PDBFixer may have added it
    to_remove = []
    for res in (res_i, res_j):
        for atom in res.atoms():
            if atom.name == "HG":
                to_remove.append(atom)
    if to_remove:
        modeller = app.Modeller(topology, positions)
        modeller.delete(to_remove)
        topology, positions = modeller.topology, modeller.positions

    # Rename CYS → CYX (ff14SB has a CYX template for disulfide Cys)
    chain_id = residues[0].chain.id
    residues = _peptide_residues(topology, chain_id)
    for idx in (ri, rj):
        residues[idx]._name = "CYX"

    # Add SG–SG bond (only if not already present)
    sg_i = _find_atom(residues[ri], "SG")
    sg_j = _find_atom(residues[rj], "SG")
    if sg_i and sg_j:
        already = any(
            (b.atom1.index == sg_i.index and b.atom2.index == sg_j.index) or
            (b.atom2.index == sg_i.index and b.atom1.index == sg_j.index)
            for b in topology.bonds()
        )
        if not already:
            topology.addBond(sg_i, sg_j)

    return topology, positions


def _patch_lactam_n(topology, positions, residues,
                    sc_res_idx: int, sc_atom_name: str, od_name: str,
                    new_res_name: str, app):
    """Rename ASP/GLU → ASPL/GLUL, remove free carboxylate O, add CG/CD–N bond."""
    sc_res  = residues[sc_res_idx]
    first   = residues[0]

    # Remove OD2/OE2 (leaves only the amide O)
    to_remove = [a for a in sc_res.atoms() if a.name == od_name]
    if to_remove:
        modeller = app.Modeller(topology, positions)
        modeller.delete(to_remove)
        topology, positions = modeller.topology, modeller.positions

    # Rename residue
    chain_id = residues[0].chain.id
    residues = _peptide_residues(topology, chain_id)
    residues[sc_res_idx]._name = new_res_name

    # Add sidechain-C — N(first) bond (only if not already present)
    sc_atom = _find_atom(residues[sc_res_idx], sc_atom_name)
    n_first = _find_atom(residues[0], "N")
    if sc_atom and n_first:
        already = any(
            (b.atom1.index == sc_atom.index and b.atom2.index == n_first.index) or
            (b.atom2.index == sc_atom.index and b.atom1.index == n_first.index)
            for b in topology.bonds()
        )
        if not already:
            topology.addBond(sc_atom, n_first)

    return topology, positions


def _patch_lactam_c_lys(topology, positions, residues, lys_idx: int, app):
    """Rename LYS → LYSL, remove OXT and extra HZ, add NZ–C(last) bond."""
    lys_res = residues[lys_idx]
    last    = residues[-1]

    # Remove OXT from C-terminal residue and extra amine protons (HZ2, HZ3)
    to_remove = []
    for atom in last.atoms():
        if atom.name == "OXT":
            to_remove.append(atom)
    for atom in lys_res.atoms():
        if atom.name in ("HZ2", "HZ3"):
            to_remove.append(atom)

    if to_remove:
        modeller = app.Modeller(topology, positions)
        modeller.delete(to_remove)
        topology, positions = modeller.topology, modeller.positions

    chain_id = residues[0].chain.id
    residues = _peptide_residues(topology, chain_id)
    residues[lys_idx]._name = "LYSL"

    # Add NZ — C(last) bond (only if not already present)
    nz     = _find_atom(residues[lys_idx], "NZ")
    c_last = _find_atom(residues[-1], "C")
    if nz and c_last:
        already = any(
            (b.atom1.index == nz.index and b.atom2.index == c_last.index) or
            (b.atom2.index == nz.index and b.atom1.index == c_last.index)
            for b in topology.bonds()
        )
        if not already:
            topology.addBond(nz, c_last)

    return topology, positions


# ---------------------------------------------------------------------------
# Post-patch helpers
# ---------------------------------------------------------------------------

def _refresh_indices(info: CyclicBondInfo, residues: list, chain_id: str) -> CyclicBondInfo:
    """Return a new CyclicBondInfo with atom_ids still valid (indices unchanged)."""
    # atom1_id and atom2_id store (chain_id, res_idx_in_chain, atom_name)
    # residue indices are stable after deletion (we re-fetch the list each time).
    # Just verify atoms are still present; no-op if so.
    ri1, name1 = info.atom1_id[1], info.atom1_id[2]
    ri2, name2 = info.atom2_id[1], info.atom2_id[2]
    if _find_atom(residues[ri1], name1) is None or _find_atom(residues[ri2], name2) is None:
        raise CyclizationError(
            f"After patching, could not find closure atoms "
            f"({name1} on res {ri1}, {name2} on res {ri2}). "
            "This is a bug — please report it."
        )
    return info


def resolve_closure_atoms(topology, bond_info: CyclicBondInfo, chain_id: str) -> tuple:
    """Return (atom1_index, atom2_index) for the closure bond in current topology.

    Call this AFTER addHydrogens, when final atom indices are known.

    Args:
        topology: Final OpenMM Topology (with hydrogens).
        bond_info: CyclicBondInfo from patch_cyclic_topology.
        chain_id: Peptide chain ID.

    Returns:
        Tuple (idx1, idx2) of integer atom indices.
    """
    residues = _peptide_residues(topology, chain_id)
    ri1, name1 = bond_info.atom1_id[1], bond_info.atom1_id[2]
    ri2, name2 = bond_info.atom2_id[1], bond_info.atom2_id[2]
    a1 = _find_atom(residues[ri1], name1)
    a2 = _find_atom(residues[ri2], name2)
    if a1 is None or a2 is None:
        raise CyclizationError(
            f"Could not resolve closure atoms after hydrogen addition: "
            f"res[{ri1}].{name1}, res[{ri2}].{name2}"
        )
    return a1.index, a2.index


def resolve_omega_atoms(topology, bond_info: CyclicBondInfo, chain_id: str) -> Optional[tuple]:
    """Return (i1, i2, i3, i4) indices for the ω dihedral, or None.

    Args:
        topology: Final OpenMM Topology (with hydrogens).
        bond_info: CyclicBondInfo from patch_cyclic_topology.
        chain_id: Peptide chain ID.

    Returns:
        4-tuple of integer atom indices, or None if omega_ids is None.
    """
    if bond_info.omega_ids is None:
        return None
    residues = _peptide_residues(topology, chain_id)
    indices = []
    for (_, ri, aname) in bond_info.omega_ids:
        atom = _find_atom(residues[ri], aname)
        if atom is None:
            return None  # graceful fallback
        indices.append(atom.index)
    return tuple(indices)


def _internal_h_list(res_name: str):
    """Return [(h_name, parent_name), ...] for an *internal* residue (non-terminal).

    OpenMM's addHydrogens uses the ``terminal`` field on each H definition to
    decide whether to add that H in N-terminal/C-terminal/internal contexts.
    The list-format variant bypasses position-based N/C-terminal detection and
    lets us specify exactly which H atoms to add.

    Returns None if the residue has no entry in OpenMM's hydrogen definitions
    (addHydrogens won't touch it anyway in that case).
    """
    from openmm.app.modeller import Modeller
    Modeller._loadStandardHydrogenDefinitions()
    if res_name not in Modeller._residueHydrogens:
        return None
    spec = Modeller._residueHydrogens[res_name]
    # Include H atoms that are applicable to internal residues:
    #   terminal is None  → always added (applies to all forms)
    #   terminal contains '-' → specifically for internal
    # Exclude terminal='N' (N-terminal only) and terminal='C' (C-terminal only).
    return [
        (h.name, h.parent)
        for h in spec.hydrogens
        if h.terminal is None or "-" in h.terminal
    ]


def get_addh_variants(topology, bond_info_list: list, chain_id: str) -> list:
    """Build the ``variants`` list for ``modeller.addHydrogens()`` for cyclic topologies.

    When a cyclic peptide closure bond is present, the first and/or last residue
    in the chain must be treated as *internal* (no N/C terminal protons) by
    ``addHydrogens``. OpenMM's default heuristic uses chain position (first/last)
    to assign N/C-terminal protons regardless of the actual bond connectivity.

    We override this by passing an explicit list of ``(atom_name, parent_name)``
    tuples for each closure residue, so ``addHydrogens`` adds only the H atoms
    that belong on an internal residue (amide H on N; no H1/H2/H3).

    Args:
        topology: OpenMM Topology (after ``patch_cyclic_topology``; before H addition).
        bond_info_list: List of CyclicBondInfo from ``patch_cyclic_topology``.
        chain_id: Peptide chain ID.

    Returns:
        List of length ``topology.getNumResidues()`` where each element is either
        ``None`` (auto-detect) or an explicit H-spec list for closure residues.
    """
    variants: list = [None] * topology.getNumResidues()
    chain_residues = _peptide_residues(topology, chain_id)

    for bond_info in bond_info_list:
        if bond_info.cyclic_type == "head_to_tail":
            first = chain_residues[0]
            last  = chain_residues[-1]
            h_first = _internal_h_list(first.name)
            if h_first is not None:
                variants[first.index] = h_first
            h_last = _internal_h_list(last.name)
            if h_last is not None:
                variants[last.index] = h_last
        elif bond_info.cyclic_type in ("lactam_n_asp", "lactam_n_glu"):
            # N-terminal residue is now internal (its N is part of the ring closure)
            first = chain_residues[0]
            h_first = _internal_h_list(first.name)
            if h_first is not None:
                variants[first.index] = h_first
        elif bond_info.cyclic_type == "lactam_c_lys":
            # C-terminal residue is now internal (its C is part of the ring closure)
            last = chain_residues[-1]
            h_last = _internal_h_list(last.name)
            if h_last is not None:
                variants[last.index] = h_last
        # disulfide: CYX template is handled by the CYS→CYX rename + bond;
        # addHydrogens auto-detects CYX via the disulfide bond check (line 146-147).

    return variants


def load_extra_xmls(ff, bond_info_list: list) -> None:
    """Load lactam XML residue templates into the ForceField object.

    Args:
        ff: OpenMM ForceField instance (before createSystem).
        bond_info_list: List of CyclicBondInfo; extra_ff_xmls from each are loaded.
    """
    seen: set = set()
    for bond_info in bond_info_list:
        for xml_str in bond_info.extra_ff_xmls:
            if xml_str in seen:
                continue
            seen.add(xml_str)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xml")
            try:
                with os.fdopen(tmp_fd, "w") as fh:
                    fh.write(xml_str)
                ff.loadFile(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
