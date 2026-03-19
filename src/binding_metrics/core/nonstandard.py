"""Non-standard residue handling: D-amino acids and N-methylated amino acids.

D-amino acids
-------------
D-amino acids use the same AMBER ff14SB bonded parameters as their L counterparts
(bond lengths, angles, torsion terms are chirality-independent in AMBER). They are
detected by PDB CCD residue name and renamed to their L counterpart in the OpenMM
topology so that ``ForceField.createSystem`` can match the standard template.
The original name is preserved in ``NonstandardInfo`` for metric reporting.

Ramachandran validation is corrected automatically: φ/ψ angles for D-residues are
negated before region classification so that a D-α-helix (φ ≈ +57°, ψ ≈ +47°)
maps to the same "favoured" basin as an L-α-helix (φ ≈ −57°, ψ ≈ −47°).

N-methylated amino acids
------------------------
N-methylation replaces the backbone amide NH with an N-methyl group (−N(CH₃)−).
This eliminates the hydrogen-bond donor, increases cis-amide probability (similar
to proline), and requires a custom AMBER XML residue template.

Custom templates are provided for the four most common residues in cyclic peptide
drugs (cyclosporin A, macrocycles):

    NMG  N-methyl-glycine (sarcosine / SAR)
    NMA  N-methyl-alanine
    MVA  N-methyl-valine
    MLE  N-methyl-leucine

Backbone N atom type: ``N`` (same as standard amide N and proline N in ff14SB).
Methyl group atom types: CN ``CT`` (sp3 carbon), HN1/HN2/HN3 ``HC``.

Charges: RESP-fitted values from ForceField_NCAA (Khoury et al., ACS Synth. Biol.
2014, PMC4277759, ffncaa.zip supporting data). Charges were derived using the ff03
RESP protocol (HF/6-31G*, condensed-phase dielectric). This introduces a minor
charge-model inconsistency with ff14SB (gas-phase RESP), but is the best available
published set and is widely used in cyclic peptide MD studies (e.g. cyclosporin A).
Atom types for CA use ``CX`` (ff14SB alpha-carbon type) to preserve correct ff14SB
backbone torsion parameters. HN1/HN2/HN3 use ``H1`` (H adjacent to N, one EWG).

Supported NMe PDB codes (input → canonical template name):
    SAR, NMG  →  NMG
    NMA, MAA  →  NMA
    MVA       →  MVA
    MLE       →  MLE
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# D-amino acid registry  (PDB CCD 3-letter code → L counterpart residue name)
# ---------------------------------------------------------------------------

#: All standard D-amino acid PDB CCD codes and their L counterparts.
#: Glycine is achiral and has no D form.
D_AA_MAP: dict[str, str] = {
    "DAL": "ALA",   # D-alanine
    "DAS": "ASP",   # D-aspartic acid
    "DSG": "ASN",   # D-asparagine
    "DCY": "CYS",   # D-cysteine
    "DGL": "GLU",   # D-glutamic acid
    "DGN": "GLN",   # D-glutamine
    "DHI": "HIS",   # D-histidine
    "DIL": "ILE",   # D-isoleucine
    "DLE": "LEU",   # D-leucine
    "DLY": "LYS",   # D-lysine
    "DME": "MET",   # D-methionine
    "DPN": "PHE",   # D-phenylalanine
    "DPR": "PRO",   # D-proline
    "DSN": "SER",   # D-serine
    "DTH": "THR",   # D-threonine
    "DTR": "TRP",   # D-tryptophan
    "DTY": "TYR",   # D-tyrosine
    "DVA": "VAL",   # D-valine
    "DAR": "ARG",   # D-arginine
}


# ---------------------------------------------------------------------------
# N-methylated amino acid registry (input code → canonical template name)
# ---------------------------------------------------------------------------

#: Maps known PDB CCD codes for N-methylated amino acids to the canonical
#: template name used in our custom XML definitions below.
NME_AA_MAP: dict[str, str] = {
    "SAR": "NMG",   # sarcosine (most common code in PDB)
    "NMG": "NMG",   # explicit N-methyl-glycine
    "NMA": "NMA",   # N-methyl-alanine
    "MAA": "NMA",   # alternate code
    "MVA": "MVA",   # N-methyl-valine
    "MLE": "MLE",   # N-methyl-leucine
}

#: Atom names of the N-methyl group atoms to *remove* if PDBFixer added them
#: incorrectly (it may add a backbone H to a tertiary N).
_NME_N_BAD_H = {"H", "HN"}


# ---------------------------------------------------------------------------
# XML residue templates — N-methylated amino acids
#
# Charge convention (ensuring residue neutrality):
#   N-methyl group:  N = −0.2548 (proline-N model)
#                    CN = −0.0690 (methyl C)
#                    HN1/HN2/HN3 = +0.0600 each
#   Group sum: −0.2548 − 0.0690 + 3(0.0600) = −0.1438  [= standard N + H sum]
#   All other atoms: standard ff14SB partial charges.
#   Each template verified: Σq = 0.
# ---------------------------------------------------------------------------

# NMG – N-methyl-glycine (sarcosine)
# Charges: ForceField_NCAA RESP (Khoury et al. ACS Synth. Biol. 2014, PMC4277759).
# Atom type for HN1/HN2/HN3: H1 (H on C bonded to N, one EWG neighbour).
# CA type: CX (ff14SB alpha-carbon type, for correct backbone phi/psi torsions).
_XML_NMG = """\
<ForceField>
 <Residues>
  <Residue name="NMG">
   <Atom name="N"   type="N"   charge="-0.058784"/>
   <Atom name="CN"  type="CT"  charge="-0.311738"/>
   <Atom name="HN1" type="H1"  charge=" 0.124265"/>
   <Atom name="HN2" type="H1"  charge=" 0.124265"/>
   <Atom name="HN3" type="H1"  charge=" 0.124265"/>
   <Atom name="CA"  type="CX"  charge="-0.240556"/>
   <Atom name="HA2" type="H1"  charge=" 0.117262"/>
   <Atom name="HA3" type="H1"  charge=" 0.117262"/>
   <Atom name="C"   type="C"   charge=" 0.576907"/>
   <Atom name="O"   type="O"   charge="-0.573149"/>
   <Bond atomName1="N"   atomName2="CN"/>
   <Bond atomName1="N"   atomName2="CA"/>
   <Bond atomName1="CN"  atomName2="HN1"/>
   <Bond atomName1="CN"  atomName2="HN2"/>
   <Bond atomName1="CN"  atomName2="HN3"/>
   <Bond atomName1="CA"  atomName2="HA2"/>
   <Bond atomName1="CA"  atomName2="HA3"/>
   <Bond atomName1="CA"  atomName2="C"/>
   <Bond atomName1="C"   atomName2="O"/>
   <ExternalBond atomName="N"/>
   <ExternalBond atomName="C"/>
  </Residue>
 </Residues>
</ForceField>
"""

# NMA – N-methyl-alanine
# Charges: ForceField_NCAA RESP (Khoury et al. ACS Synth. Biol. 2014, PMC4277759).
_XML_NMA = """\
<ForceField>
 <Residues>
  <Residue name="NMA">
   <Atom name="N"   type="N"   charge="-0.036802"/>
   <Atom name="CN"  type="CT"  charge="-0.353242"/>
   <Atom name="HN1" type="H1"  charge=" 0.129448"/>
   <Atom name="HN2" type="H1"  charge=" 0.129448"/>
   <Atom name="HN3" type="H1"  charge=" 0.129448"/>
   <Atom name="CA"  type="CX"  charge=" 0.020348"/>
   <Atom name="HA"  type="H1"  charge=" 0.047291"/>
   <Atom name="CB"  type="CT"  charge="-0.182508"/>
   <Atom name="HB1" type="HC"  charge=" 0.067317"/>
   <Atom name="HB2" type="HC"  charge=" 0.067317"/>
   <Atom name="HB3" type="HC"  charge=" 0.067317"/>
   <Atom name="C"   type="C"   charge=" 0.437383"/>
   <Atom name="O"   type="O"   charge="-0.522764"/>
   <Bond atomName1="N"   atomName2="CN"/>
   <Bond atomName1="N"   atomName2="CA"/>
   <Bond atomName1="CN"  atomName2="HN1"/>
   <Bond atomName1="CN"  atomName2="HN2"/>
   <Bond atomName1="CN"  atomName2="HN3"/>
   <Bond atomName1="CA"  atomName2="HA"/>
   <Bond atomName1="CA"  atomName2="CB"/>
   <Bond atomName1="CA"  atomName2="C"/>
   <Bond atomName1="CB"  atomName2="HB1"/>
   <Bond atomName1="CB"  atomName2="HB2"/>
   <Bond atomName1="CB"  atomName2="HB3"/>
   <Bond atomName1="C"   atomName2="O"/>
   <ExternalBond atomName="N"/>
   <ExternalBond atomName="C"/>
  </Residue>
 </Residues>
</ForceField>
"""

# MVA – N-methyl-valine
# Charges: ForceField_NCAA RESP (Khoury et al. ACS Synth. Biol. 2014, PMC4277759).
_XML_MVA = """\
<ForceField>
 <Residues>
  <Residue name="MVA">
   <Atom name="N"   type="N"   charge="-0.077388"/>
   <Atom name="CN"  type="CT"  charge="-0.228891"/>
   <Atom name="HN1" type="H1"  charge=" 0.095543"/>
   <Atom name="HN2" type="H1"  charge=" 0.095543"/>
   <Atom name="HN3" type="H1"  charge=" 0.095543"/>
   <Atom name="CA"  type="CX"  charge="-0.370509"/>
   <Atom name="HA"  type="H1"  charge=" 0.145778"/>
   <Atom name="CB"  type="3C"  charge=" 0.353677"/>
   <Atom name="HB"  type="HC"  charge=" 0.053477"/>
   <Atom name="CG1" type="CT"  charge="-0.484027"/>
   <Atom name="HG11" type="HC" charge=" 0.122836"/>
   <Atom name="HG12" type="HC" charge=" 0.122836"/>
   <Atom name="HG13" type="HC" charge=" 0.122836"/>
   <Atom name="CG2" type="CT"  charge="-0.484027"/>
   <Atom name="HG21" type="HC" charge=" 0.122836"/>
   <Atom name="HG22" type="HC" charge=" 0.122836"/>
   <Atom name="HG23" type="HC" charge=" 0.122836"/>
   <Atom name="C"   type="C"   charge=" 0.602902"/>
   <Atom name="O"   type="O"   charge="-0.534639"/>
   <Bond atomName1="N"   atomName2="CN"/>
   <Bond atomName1="N"   atomName2="CA"/>
   <Bond atomName1="CN"  atomName2="HN1"/>
   <Bond atomName1="CN"  atomName2="HN2"/>
   <Bond atomName1="CN"  atomName2="HN3"/>
   <Bond atomName1="CA"  atomName2="HA"/>
   <Bond atomName1="CA"  atomName2="CB"/>
   <Bond atomName1="CA"  atomName2="C"/>
   <Bond atomName1="CB"  atomName2="HB"/>
   <Bond atomName1="CB"  atomName2="CG1"/>
   <Bond atomName1="CB"  atomName2="CG2"/>
   <Bond atomName1="CG1" atomName2="HG11"/>
   <Bond atomName1="CG1" atomName2="HG12"/>
   <Bond atomName1="CG1" atomName2="HG13"/>
   <Bond atomName1="CG2" atomName2="HG21"/>
   <Bond atomName1="CG2" atomName2="HG22"/>
   <Bond atomName1="CG2" atomName2="HG23"/>
   <Bond atomName1="C"   atomName2="O"/>
   <ExternalBond atomName="N"/>
   <ExternalBond atomName="C"/>
  </Residue>
 </Residues>
</ForceField>
"""

# MLE – N-methyl-leucine
# Charges: ForceField_NCAA RESP (Khoury et al. ACS Synth. Biol. 2014, PMC4277759).
_XML_MLE = """\
<ForceField>
 <Residues>
  <Residue name="MLE">
   <Atom name="N"   type="N"   charge="-0.073133"/>
   <Atom name="CN"  type="CT"  charge="-0.351941"/>
   <Atom name="HN1" type="H1"  charge=" 0.129060"/>
   <Atom name="HN2" type="H1"  charge=" 0.129060"/>
   <Atom name="HN3" type="H1"  charge=" 0.129060"/>
   <Atom name="CA"  type="CX"  charge="-0.095313"/>
   <Atom name="HA"  type="H1"  charge=" 0.050991"/>
   <Atom name="CB"  type="2C"  charge="-0.139229"/>
   <Atom name="HB2" type="HC"  charge=" 0.051454"/>
   <Atom name="HB3" type="HC"  charge=" 0.051454"/>
   <Atom name="CG"  type="3C"  charge=" 0.418706"/>
   <Atom name="HG"  type="HC"  charge="-0.026087"/>
   <Atom name="CD1" type="CT"  charge="-0.439742"/>
   <Atom name="HD11" type="HC" charge=" 0.100252"/>
   <Atom name="HD12" type="HC" charge=" 0.100252"/>
   <Atom name="HD13" type="HC" charge=" 0.100252"/>
   <Atom name="CD2" type="CT"  charge="-0.439742"/>
   <Atom name="HD21" type="HC" charge=" 0.100252"/>
   <Atom name="HD22" type="HC" charge=" 0.100252"/>
   <Atom name="HD23" type="HC" charge=" 0.100252"/>
   <Atom name="C"   type="C"   charge=" 0.542163"/>
   <Atom name="O"   type="O"   charge="-0.538274"/>
   <Bond atomName1="N"   atomName2="CN"/>
   <Bond atomName1="N"   atomName2="CA"/>
   <Bond atomName1="CN"  atomName2="HN1"/>
   <Bond atomName1="CN"  atomName2="HN2"/>
   <Bond atomName1="CN"  atomName2="HN3"/>
   <Bond atomName1="CA"  atomName2="HA"/>
   <Bond atomName1="CA"  atomName2="CB"/>
   <Bond atomName1="CA"  atomName2="C"/>
   <Bond atomName1="CB"  atomName2="HB2"/>
   <Bond atomName1="CB"  atomName2="HB3"/>
   <Bond atomName1="CB"  atomName2="CG"/>
   <Bond atomName1="CG"  atomName2="HG"/>
   <Bond atomName1="CG"  atomName2="CD1"/>
   <Bond atomName1="CG"  atomName2="CD2"/>
   <Bond atomName1="CD1" atomName2="HD11"/>
   <Bond atomName1="CD1" atomName2="HD12"/>
   <Bond atomName1="CD1" atomName2="HD13"/>
   <Bond atomName1="CD2" atomName2="HD21"/>
   <Bond atomName1="CD2" atomName2="HD22"/>
   <Bond atomName1="CD2" atomName2="HD23"/>
   <Bond atomName1="C"   atomName2="O"/>
   <ExternalBond atomName="N"/>
   <ExternalBond atomName="C"/>
  </Residue>
 </Residues>
</ForceField>
"""

_NME_XMLS: dict[str, str] = {
    "NMG": _XML_NMG,
    "NMA": _XML_NMA,
    "MVA": _XML_MVA,
    "MLE": _XML_MLE,
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class NonstandardInfo:
    """Records non-standard residues found in a peptide chain.

    Attributes:
        d_residues: List of D-amino acid entries, each a dict with keys:
            ``res_idx`` (int, 0-based index in chain),
            ``original_name`` (str, e.g. "DAL"),
            ``l_name`` (str, e.g. "ALA").
            The residue has been renamed to ``l_name`` in the topology.
        nmethyl_residues: List of N-methylated residue entries, each a dict
            with keys: ``res_idx``, ``original_name`` (e.g. "SAR"),
            ``template_name`` (e.g. "NMG"). Renamed in topology.
        extra_ff_xmls: AMBER XML strings to load into ForceField before
            ``createSystem`` — one per unique NMe template in use.
    """
    d_residues: list = field(default_factory=list)
    nmethyl_residues: list = field(default_factory=list)
    extra_ff_xmls: list = field(default_factory=list)

    @property
    def has_d_residues(self) -> bool:
        return bool(self.d_residues)

    @property
    def has_nmethyl(self) -> bool:
        return bool(self.nmethyl_residues)

    @property
    def is_empty(self) -> bool:
        return not self.d_residues and not self.nmethyl_residues


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _peptide_residues(topology, chain_id: str) -> list:
    for chain in topology.chains():
        if chain.id == chain_id:
            return list(chain.residues())
    raise ValueError(f"Chain '{chain_id}' not found in topology.")


def _find_atom(residue, name: str):
    for atom in residue.atoms():
        if atom.name == name:
            return atom
    return None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_nonstandard(topology, chain_id: str) -> NonstandardInfo:
    """Scan a chain for D-amino acids and N-methylated residue codes.

    Identification is purely by residue name — no coordinate analysis.
    Call this on the raw topology (before any renaming).

    Args:
        topology: OpenMM Topology.
        chain_id: Chain ID to scan.

    Returns:
        NonstandardInfo describing all detected non-standard residues.
    """
    residues = _peptide_residues(topology, chain_id)
    d_list: list = []
    nme_list: list = []
    seen_templates: set[str] = set()
    xmls: list[str] = []

    for idx, res in enumerate(residues):
        name = res.name
        if name in D_AA_MAP:
            d_list.append({
                "res_idx": idx,
                "original_name": name,
                "l_name": D_AA_MAP[name],
            })
        elif name in NME_AA_MAP:
            tmpl = NME_AA_MAP[name]
            nme_list.append({
                "res_idx": idx,
                "original_name": name,
                "template_name": tmpl,
            })
            if tmpl not in seen_templates:
                seen_templates.add(tmpl)
                xmls.append(_NME_XMLS[tmpl])

    return NonstandardInfo(
        d_residues=d_list,
        nmethyl_residues=nme_list,
        extra_ff_xmls=xmls,
    )


# ---------------------------------------------------------------------------
# Topology patching
# ---------------------------------------------------------------------------

def patch_nonstandard(topology, positions, chain_id: str, info: NonstandardInfo):
    """Rename non-standard residues in the OpenMM topology for FF compatibility.

    Must be called AFTER PDBFixer and BEFORE ``addHydrogens()``.

    D-amino acids
        Renamed to their L counterpart (e.g. DAL → ALA). Coordinates are
        untouched — AMBER ff14SB has no chirality-sensitive energy terms.
        ``addHydrogens`` will then add H in the correct positions for the
        existing heavy-atom geometry. PDBFixer artefacts (spurious backbone
        H on Cα, etc.) should be absent if the input structure is complete.

    N-methylated amino acids
        Renamed to the canonical template name (e.g. SAR → NMG). Any
        backbone amide H erroneously added by PDBFixer is removed.
        The N-methyl heavy atoms (CN, HN1/HN2/HN3) must already be present
        in the input structure — this function does not add missing heavy
        atoms. ``addHydrogens`` will correctly add HN1/HN2/HN3 after seeing
        the NMG/NMA/MVA/MLE template (which has no H on N).

    Args:
        topology: OpenMM Topology (heavy atoms only, post-PDBFixer).
        positions: Atom positions (OpenMM Quantity, nm).
        chain_id: Chain ID of the peptide to patch.
        info: NonstandardInfo from :func:`detect_nonstandard`.

    Returns:
        Tuple ``(topology, positions)`` with renames applied and spurious H
        removed from NMe-N atoms.
    """
    if info.is_empty:
        return topology, positions

    try:
        from openmm import app
    except ImportError as e:
        raise ImportError("OpenMM is required for topology patching.") from e

    residues = _peptide_residues(topology, chain_id)

    # --- Rename D-amino acids → L counterpart ---
    for entry in info.d_residues:
        residues[entry["res_idx"]]._name = entry["l_name"]

    # --- Rename NMe-AAs and remove spurious backbone H on N ---
    atoms_to_remove = []
    for entry in info.nmethyl_residues:
        res = residues[entry["res_idx"]]
        res._name = entry["template_name"]
        # Remove any H that PDBFixer added to backbone N (should be absent in NMe)
        for atom in res.atoms():
            if atom.name in _NME_N_BAD_H:
                # Confirm it's actually on the backbone N by checking it's bonded to N
                n_atom = _find_atom(res, "N")
                if n_atom is not None:
                    bonded_to_n = any(
                        (b.atom1.index == n_atom.index and b.atom2.index == atom.index) or
                        (b.atom2.index == n_atom.index and b.atom1.index == atom.index)
                        for b in topology.bonds()
                    )
                    if bonded_to_n:
                        atoms_to_remove.append(atom)

    if atoms_to_remove:
        modeller = app.Modeller(topology, positions)
        modeller.delete(atoms_to_remove)
        topology, positions = modeller.topology, modeller.positions

    return topology, positions


def load_nonstandard_xmls(ff, info: NonstandardInfo) -> None:
    """Load NMe-AA XML residue templates into a ForceField object.

    Args:
        ff: OpenMM ForceField instance (before createSystem).
        info: NonstandardInfo with extra_ff_xmls populated.
    """
    import os
    import tempfile

    for xml_str in info.extra_ff_xmls:
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


# ---------------------------------------------------------------------------
# Ramachandran helper
# ---------------------------------------------------------------------------

def is_d_residue(res_name: str) -> bool:
    """Return True if ``res_name`` is a D-amino acid PDB CCD code."""
    return res_name in D_AA_MAP
