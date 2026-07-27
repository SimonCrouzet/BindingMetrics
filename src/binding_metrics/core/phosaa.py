"""AMBER phosaa parameterisation of phosphorylated residues, adapted to amber14-all.

Why this exists
---------------
Phosphorylated residues (SEP/TPO/PTR and the phospho-His / mono-protonated
variants) carry a net −2 (or −1) phosphate. The generic GAFF NCAA path
(:mod:`binding_metrics.core.gaff_ncaa`) perceives every exotic residue at
*neutral* charge, so on a phosphate it adds O–H / P–H protons and returns a
**net-zero** residue — silently destroying the modification's defining negative
charge (verified: 1QJB SEP came out at net 0 with four spurious H instead of
−2). ``phosaa`` is AMBER's purpose-built, RESP-fitted parameter set for exactly
these residues, and is the correct source.

``openmmforcefields`` ships ``phosaa14SB.xml``, but two things stop it loading
against this codebase's ``amber14-all.xml`` base:

1. Its residue templates reference **bare** ff14SB atom-type *names*
   (``N``/``CX``/``2C``/…) that ``amber14-all`` exposes only under the
   ``protein-`` prefix.
2. It references the phosphorus type ``P`` without defining it (``P`` normally
   comes from the nucleic-acid parameters, absent from ``amber14-all``).

Crucially, its bonded and nonbonded blocks are keyed by atom **class**, and
``amber14-all``'s protein types already use the bare class names (``protein-N``
has class ``N``), so those parameters are directly compatible. This module
therefore adapts the shipped file rather than re-deriving anything:

* remap each phospho template's base-type references ``X`` → ``protein-X``
  (leaving phosaa's own phosphate types ``OX``/``OZ``/… and ``P`` untouched),
* inject the missing ``P`` atom type + its Lennard-Jones parameters, copied
  verbatim from the bare ``ff14SB.xml``.

All charges, bonded terms and phosphate LJ values are phosaa's own, unchanged —
only atom-type *names* are rewritten, which does not affect the physics.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache

# Base ff14SB atom-type names → the corresponding amber14-all prefixed type.
# Applied to every phospho residue-template atom whose type is neither a
# phosaa-defined phosphate type nor phosphorus (P).
_PROTEIN_PREFIX = "protein-"


def _ffxml_dir() -> str:
    import openmmforcefields

    return os.path.join(os.path.dirname(openmmforcefields.__file__), "ffxml", "amber")


def _phosaa_source() -> str:
    return os.path.join(_ffxml_dir(), "phosaa14SB.xml")


def _ff14_source() -> str:
    return os.path.join(_ffxml_dir(), "ff14SB.xml")


@lru_cache(maxsize=1)
def phospho_residue_names() -> frozenset:
    """Residue names provided by phosaa14SB.xml (SEP/TPO/PTR + variants)."""
    root = ET.parse(_phosaa_source()).getroot()
    return frozenset(r.get("name") for r in root.find("Residues").findall("Residue"))


def _is_hydrogen_type(type_name: str) -> bool:
    """True for an AMBER hydrogen atom type (names start with H; phosaa's own
    phosphate types are all O/C, so this is unambiguous within a phospho template)."""
    return type_name.startswith("H")


@lru_cache(maxsize=1)
def build_adapter_ffxml() -> str:
    """Return a single OpenMM ForceField XML adapting phosaa onto amber14-all.

    Contains phosaa's phosphate atom types + the injected ``P`` type, phosaa's
    (class-keyed) nonbonded and bonded force blocks, and the phospho residue
    templates with their base-type references remapped to ``protein-*``.
    """
    phosaa = ET.parse(_phosaa_source()).getroot()

    phosphate_types = {t.get("name") for t in phosaa.find("AtomTypes").findall("Type")}

    # 1. Remap base-type references in every phospho residue template.
    for res in phosaa.find("Residues").findall("Residue"):
        for atom in res.findall("Atom"):
            t = atom.get("type")
            if t in phosphate_types or t == "P":
                continue  # phosaa's own phosphate type / phosphorus: keep as-is
            atom.set("type", _PROTEIN_PREFIX + t)

    # 2. Inject the phosphorus type + LJ from the bare ff14SB.xml.
    ff14 = ET.parse(_ff14_source()).getroot()
    p_type = ff14.find('.//AtomTypes/Type[@name="P"]')
    if p_type is None:
        raise RuntimeError("phosaa adapter: atom type 'P' not found in ff14SB.xml")
    p_lj = next(
        (a for a in ff14.find("NonbondedForce").findall("Atom") if a.get("class") == "P"),
        None,
    )
    if p_lj is None:
        raise RuntimeError("phosaa adapter: Lennard-Jones for class 'P' not found in ff14SB.xml")

    phosaa.find("AtomTypes").append(ET.fromstring(ET.tostring(p_type)))
    nb = phosaa.find("NonbondedForce")
    nb.append(
        ET.fromstring(
            f'<Atom class="P" sigma="{p_lj.get("sigma")}" epsilon="{p_lj.get("epsilon")}"/>'
        )
    )

    return ET.tostring(phosaa, encoding="unicode")


@lru_cache(maxsize=1)
def build_hydrogen_definitions_xml() -> str:
    """Return an OpenMM hydrogen-definitions XML for the phospho residues.

    Derived directly from the phospho templates: each hydrogen (type name starts
    with 'H') is emitted with the heavy atom it is bonded to. Lets
    ``Modeller.addHydrogens`` place H on heavy-atom-only phospho inputs, which it
    cannot otherwise do (the residues are not in its standard definitions).
    """
    phosaa = ET.parse(_phosaa_source()).getroot()
    out = ET.Element("Residues")
    for res in phosaa.find("Residues").findall("Residue"):
        atoms = {a.get("name"): a.get("type") for a in res.findall("Atom")}
        bonds = [(b.get("atomName1"), b.get("atomName2")) for b in res.findall("Bond")]
        rd = ET.SubElement(out, "Residue", {"name": res.get("name")})
        for hname, htype in atoms.items():
            if not _is_hydrogen_type(htype):
                continue
            parent = next(
                (b for a, b in ((x, y) for x, y in bonds) if a == hname),
                None,
            ) or next((a for a, b in bonds if b == hname), None)
            if parent is not None:
                ET.SubElement(rd, "H", {"name": hname, "parent": parent})
    return ET.tostring(out, encoding="unicode")


def _write_temp(xml_string: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write(xml_string)
    return path


def register(ff) -> None:
    """Load the phosaa adapter templates/parameters into a ForceField.

    Must be called AFTER ``amber14-all.xml`` is loaded into ``ff`` (the phospho
    templates reference its ``protein-*`` types).
    """
    path = _write_temp(build_adapter_ffxml(), ".xml")
    try:
        ff.loadFile(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


_hydrogen_definitions_loaded = False


def ensure_hydrogen_definitions() -> None:
    """Register phospho-residue hydrogen definitions with Modeller (idempotent)."""
    global _hydrogen_definitions_loaded
    if _hydrogen_definitions_loaded:
        return
    from openmm.app import Modeller

    path = _write_temp(build_hydrogen_definitions_xml(), ".xml")
    try:
        Modeller.loadHydrogenDefinitions(path)
        _hydrogen_definitions_loaded = True
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def topology_has_phospho(topology) -> bool:
    """True if the topology contains any phosphorylated residue."""
    names = phospho_residue_names()
    return any(r.name in names for r in topology.residues())
