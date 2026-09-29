"""Residue and water name sets shared by the package.

Pure Python with no third-party imports, so the module loads without OpenMM,
biotite or numpy. The sets below look alike on purpose in places (a protonation
variant list and its subset that biotite does not know); each constant says what
it is for, and two sets that differ only in a few names stay separate so that
neither consumer changes behaviour.
"""

#: The 20 canonical L-amino acids (three-letter PDB codes).
STANDARD_AMINO_ACIDS: frozenset[str] = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)

#: AMBER names of protonation and disulfide variants of standard residues
#: (HID/HIE/HIN/HIP histidines, CYX/CYM cysteines, ASH, GLH, LYN).
AMBER_PROTONATION_VARIANTS: frozenset[str] = frozenset(
    "HID HIE HIN HIP CYX CYM ASH GLH LYN".split()
)

#: AMBER protonation variants that biotite's CCD-based ``filter_amino_acids``
#: does not treat as amino acids, so a polymer selection has to add them.
#: Smaller than ``AMBER_PROTONATION_VARIANTS`` because the CCD lists HIP, CYM,
#: GLH and LYN as peptide-linking components, so biotite already keeps those.
AMBER_VARIANTS_OUTSIDE_CCD: frozenset[str] = frozenset({"HID", "HIE", "HIN", "CYX", "ASH"})

#: Terminal capping groups of a peptide chain (acetyl, N-methylamide, amide).
TERMINAL_CAP_NAMES: frozenset[str] = frozenset({"ACE", "NME", "NH2"})

#: Phosphorylated residues of the AMBER phosaa set (phosphoserine,
#: phosphothreonine, phosphotyrosine).
PHOSPHO_RESIDUES: frozenset[str] = frozenset({"SEP", "TPO", "PTR"})

#: Residues whose ionisation state the Coulomb metric models or knows to be
#: neutral; an amino acid outside this set may carry a charge it cannot see.
IONISATION_MODELLED_RESIDUES: frozenset[str] = (
    STANDARD_AMINO_ACIDS | AMBER_PROTONATION_VARIANTS | PHOSPHO_RESIDUES | {"MSE"}
)

#: Cysteine names that can take part in a disulfide: neutral CYS and the
#: bonded AMBER form CYX.
CYSTEINE_NAMES: frozenset[str] = frozenset({"CYS", "CYX"})

#: Protonation-state and disulfide variants (AMBER and CHARMM names) mapped to
#: the amino acid they name; two structures that differ only by these names
#: hold the same residue.
VARIANT_TO_PARENT_RESIDUE: dict[str, str] = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HIN": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "CYX": "CYS",
    "CYM": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
}

#: Water residue names of PDB files (HOH) and AMBER-prepared files (WAT). The
#: gemmi-based structure comparison skips exactly these two; the wider
#: solvent-name sets used by the structure preparation (TIP3, SOL, H2O, ...)
#: are separate because adding a name here would change which atoms the
#: comparison RMSD uses.
WATER_NAMES_PDB_AMBER: frozenset[str] = frozenset({"HOH", "WAT"})

#: Lactam-bridge residues whose templates ``core.cyclic`` loads (ASPL, GLUL and
#: LYSL: the aspartate, glutamate and lysine that form the side-chain amide).
LACTAM_TEMPLATE_RESIDUES: frozenset[str] = frozenset({"ASPL", "GLUL", "LYSL"})

#: N-methylated residues whose templates ``core.nonstandard`` supplies:
#: sarcosine (NMG) and N-methyl alanine, valine and leucine.
N_METHYLATED_RESIDUES: frozenset[str] = frozenset({"NMG", "NMA", "MVA", "MLE"})

#: Residue names that count as protein when chains are ranked by size and when
#: heterogens are stripped. Only four AMBER variants are listed: HIN, ASH, GLH,
#: LYN and CYM are left out.
PROTEIN_RESIDUES: frozenset[str] = (
    STANDARD_AMINO_ACIDS
    | {"HID", "HIE", "HIP", "CYX"}
    | LACTAM_TEMPLATE_RESIDUES
    | N_METHYLATED_RESIDUES
)
