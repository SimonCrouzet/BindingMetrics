"""Atom-type-aware Eisenberg-McLachlan solvation parameters in the interface metrics."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402
import biotite.structure.io.pdbx as pdbx  # noqa: E402
from biotite.structure.info import vdw_radius_single  # noqa: E402

from binding_metrics.metrics import interface  # noqa: E402
from binding_metrics.metrics.interface import (  # noqa: E402
    _SOLVATION_PARAMS,
    _gamma_array,
    _solvation_types,
    compute_interface_metrics,
    load_biotite_structure,
)

DATA = Path(__file__).resolve().parents[1] / "data"
P53_MDM2 = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI1_TRYPSIN = DATA / "example_bicyclic_sfti1_3P8F.cif"

# Eisenberg & McLachlan, Nature 319:199 (1986), cal/mol/Å² of accessible area,
# as tabulated in Table 1 (column A) of Krissinel & Henrick, JMB 372:774 (2007).
EISENBERG_MCLACHLAN_CAL = {"C": 16, "S": 21, "N/O": -6, "O-": -24, "N+": -50}

# Side-chain atoms that carry a formal charge at pH 7 (by residue name and atom name).
CHARGED_ATOMS = {
    "N+": {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")},
    "O-": {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")},
}

# Shrake-Rupley sampling of the reference area calculation: water probe, 960 points per atom.
PROBE_ANGSTROM = 1.4
POINTS_PER_ATOM = 960
# Areas are float32 sums added in a different order than in the package: they agree to better
# than 1e-7 relative, and the tests compare them to this.
RELATIVE = 1e-5


def read_atoms(path):
    """AtomArray of the first model of a PDB or mmCIF file, read with biotite alone."""
    if Path(path).suffix == ".cif":
        return pdbx.get_structure(pdbx.CIFFile.read(str(path)), model=1)
    return pdb_io.get_structure(pdb_io.PDBFile.read(str(path)), model=1)


def polymer_atoms(atoms):
    """Heavy atoms of the amino-acid residues: waters, ions and ligands are left out."""
    return atoms[struc.filter_amino_acids(atoms) & ~np.isin(atoms.element, ["H", "D"])]


def sasa_per_atom(atoms):
    """Shrake-Rupley area per atom with biotite's element radii (1.8 A for an unlisted element)."""
    radii = np.array([vdw_radius_single(str(e).strip()) or 1.8 for e in atoms.element])
    values = struc.sasa(
        atoms, probe_radius=PROBE_ANGSTROM, vdw_radii=radii, point_number=POINTS_PER_ATOM
    )
    return np.where(np.isfinite(values), values, 0.0).astype(float)


def solvation_parameter_per_atom(atoms):
    """Solvation parameter of each atom in kcal/mol per A^2 of buried area.

    The published value (cal/mol/A^2 of accessible area) enters with the opposite sign: burying
    an atom removes its exposure.
    """
    gamma = np.zeros(len(atoms))
    for i, (res_name, atom_name, element) in enumerate(
        zip(atoms.res_name, atoms.atom_name, atoms.element)
    ):
        if (res_name, atom_name) in CHARGED_ATOMS["N+"]:
            kind = "N+"
        elif (res_name, atom_name) in CHARGED_ATOMS["O-"]:
            kind = "O-"
        elif element in ("N", "O"):
            kind = "N/O"
        elif element in ("C", "S"):
            kind = element
        else:
            continue
        gamma[i] = -EISENBERG_MCLACHLAN_CAL[kind] / 1000.0
    return gamma


def expected_interface(atoms, design_chain, receptor_chain):
    """delta_sasa (A^2) and delta_g_int (kcal/mol) of two chains from biotite alone.

    The buried area of an atom is its area in its chain alone minus its area in the complex;
    summed over the atoms it is sasa(design) + sasa(receptor) - sasa(complex).
    """
    design = atoms[atoms.chain_id == design_chain]
    receptor = atoms[atoms.chain_id == receptor_chain]
    together = atoms[(atoms.chain_id == design_chain) | (atoms.chain_id == receptor_chain)]
    in_complex = sasa_per_atom(together)
    buried_design = sasa_per_atom(design) - in_complex[together.chain_id == design_chain]
    buried_receptor = sasa_per_atom(receptor) - in_complex[together.chain_id == receptor_chain]
    return {
        "delta_sasa": float(buried_design.sum() + buried_receptor.sum()),
        "delta_g_int": float(
            solvation_parameter_per_atom(design) @ buried_design
            + solvation_parameter_per_atom(receptor) @ buried_receptor
        ),
    }


def _atoms(records):
    """AtomArray from (res_name, atom_name, element) tuples."""
    arr = struc.AtomArray(len(records))
    arr.res_name = np.array([r[0] for r in records])
    arr.atom_name = np.array([r[1] for r in records])
    arr.element = np.array([r[2] for r in records])
    return arr


class TestParameterTable:
    @pytest.mark.parametrize("atom_type", list(EISENBERG_MCLACHLAN_CAL))
    def test_value_is_minus_the_published_parameter_per_buried_area(self, atom_type):
        published_kcal = EISENBERG_MCLACHLAN_CAL[atom_type] / 1000.0
        assert _SOLVATION_PARAMS[atom_type] == pytest.approx(-published_kcal)

    def test_neutral_polar_atoms_are_far_cheaper_than_charged_ones(self):
        assert abs(_SOLVATION_PARAMS["N/O"]) < abs(_SOLVATION_PARAMS["O-"])
        assert abs(_SOLVATION_PARAMS["N/O"]) < abs(_SOLVATION_PARAMS["N+"])

    def test_burying_apolar_atoms_is_favorable_and_polar_atoms_is_not(self):
        assert _SOLVATION_PARAMS["C"] < 0.0
        assert _SOLVATION_PARAMS["S"] < 0.0
        assert all(_SOLVATION_PARAMS[t] > 0.0 for t in ("N/O", "O-", "N+"))


class TestAtomTyping:
    def test_types_follow_element_and_charge_state(self):
        atoms = _atoms(
            [
                ("ALA", "CA", "C"),
                ("ALA", "N", "N"),  # backbone amide: neutral
                ("ALA", "O", "O"),  # backbone carbonyl: neutral
                ("SER", "OG", "O"),
                ("TRP", "NE1", "N"),
                ("HIS", "NE2", "N"),  # plain histidine is neutral
                ("LYS", "NZ", "N"),
                ("ARG", "NH1", "N"),
                ("ARG", "NE", "N"),
                ("ASP", "OD1", "O"),
                ("GLU", "OE2", "O"),
                ("CYS", "SG", "S"),
                ("MET", "SD", "S"),
                ("ALA", "HA", "H"),
                ("SEP", "P", "P"),
            ]
        )
        assert list(_solvation_types(atoms)) == [
            "C",
            "N/O",
            "N/O",
            "N/O",
            "N/O",
            "N/O",
            "N+",
            "N+",
            "N+",
            "O-",
            "O-",
            "S",
            "S",
            "",
            "",
        ]

    def test_gamma_array_matches_the_table(self):
        atoms = _atoms([("LYS", "NZ", "N"), ("ALA", "N", "N"), ("ALA", "H", "H")])
        np.testing.assert_allclose(
            _gamma_array(atoms),
            [_SOLVATION_PARAMS["N+"], _SOLVATION_PARAMS["N/O"], 0.0],
        )

    def test_real_structure_has_all_five_types(self):
        atoms = load_biotite_structure(P53_MDM2)
        assert set(_solvation_types(atoms)) >= {"C", "S", "N/O", "O-", "N+"}


class TestDeltaGInt:
    # The expected values are recomputed from the file with biotite and the published parameters
    # (the helpers above).
    def test_p53_mdm2_is_favorable(self):
        result = compute_interface_metrics(P53_MDM2)
        expected = expected_interface(polymer_atoms(read_atoms(P53_MDM2)), "B", "A")
        assert (result["peptide_chain"], result["receptor_chain"]) == ("B", "A")
        assert result["delta_g_int"] == pytest.approx(expected["delta_g_int"], rel=RELATIVE)
        # Sign convention of delta_g_int: negative = burial is favorable.
        assert result["delta_g_int"] < 0.0

    def test_cyclosporin_cyclophilin_with_waters_is_favorable(self):
        result = compute_interface_metrics(CYCLOSPORIN)
        expected = expected_interface(polymer_atoms(read_atoms(CYCLOSPORIN)), "C", "A")
        assert (result["peptide_chain"], result["receptor_chain"]) == ("C", "A")
        assert result["delta_g_int"] == pytest.approx(expected["delta_g_int"], rel=RELATIVE)
        assert result["delta_sasa"] == pytest.approx(expected["delta_sasa"], rel=RELATIVE)
        assert result["delta_g_int"] < 0.0

    def test_sfti1_trypsin_is_favorable(self):
        result = compute_interface_metrics(SFTI1_TRYPSIN)
        expected = expected_interface(polymer_atoms(read_atoms(SFTI1_TRYPSIN)), "I", "A")
        assert (result["peptide_chain"], result["receptor_chain"]) == ("I", "A")
        assert result["delta_g_int"] == pytest.approx(expected["delta_g_int"], rel=RELATIVE)
        assert result["delta_g_int"] < 0.0

    def test_kj_value_is_the_kcal_value_converted(self):
        result = compute_interface_metrics(P53_MDM2)
        assert result["delta_g_int_kJ"] == pytest.approx(result["delta_g_int"] * 4.184, rel=1e-9)

    def test_per_residue_terms_add_up_to_the_total(self):
        result = compute_interface_metrics(P53_MDM2, interface_threshold=0.0)
        per_residue_total = sum(r["delta_g_res"] for r in result["per_residue"])
        assert per_residue_total == pytest.approx(result["delta_g_int"], abs=1e-6)

    def test_areas_are_independent_of_the_parameters(self, monkeypatch):
        reference = compute_interface_metrics(P53_MDM2)
        monkeypatch.setattr(interface, "_SOLVATION_PARAMS", {k: 0.0 for k in _SOLVATION_PARAMS})
        zeroed = compute_interface_metrics(P53_MDM2)
        assert zeroed["delta_g_int"] == 0.0
        assert zeroed["delta_sasa"] == reference["delta_sasa"]
        assert zeroed["polar_area"] == reference["polar_area"]
