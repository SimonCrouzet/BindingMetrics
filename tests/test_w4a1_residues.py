"""Shared residue and water name sets (``binding_metrics.core.residues``).

The reference literals below are the hand-written sets the metrics modules carried
before they imported the shared constants. Each test pins the new constant to its old
literal, so the refactor is documented and no residue is silently added or dropped.
"""

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.core import residues
from binding_metrics.metrics import electrostatics, evobind, interface
from binding_metrics.metrics.comparison import _get_coords

# --- old literals -----------------------------------------------------------------

# electrostatics._RECOGNISED_RESIDUES
OLD_RECOGNISED_RESIDUES = frozenset(
    (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
        "HID HIE HIN HIP CYX CYM ASH GLH LYN MSE SEP TPO PTR"
    ).split()
)

# interface._AMBER_VARIANT_NAMES and interface._CAP_NAMES
OLD_AMBER_VARIANT_NAMES = frozenset({"HID", "HIE", "HIN", "CYX", "ASH"})
OLD_CAP_NAMES = frozenset({"ACE", "NME", "NH2"})

# evobind._RESNAME_EQUIVALENTS
OLD_RESNAME_EQUIVALENTS = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "CYX": "CYS",
    "CYM": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
}

# comparison._get_coords and compare_structures: inline {"HOH", "WAT"}
OLD_COMPARISON_WATERS = frozenset({"HOH", "WAT"})

# energy._repair_orphaned_cys: inline ("CYS", "CYX")
OLD_CYSTEINE_NAMES = frozenset(("CYS", "CYX"))


class TestConstantsEqualTheOldLiterals:
    def test_recognised_residues(self):
        assert residues.IONISATION_MODELLED_RESIDUES == OLD_RECOGNISED_RESIDUES
        assert electrostatics._RECOGNISED_RESIDUES == OLD_RECOGNISED_RESIDUES

    def test_amber_variants_outside_the_ccd(self):
        assert residues.AMBER_VARIANTS_OUTSIDE_CCD == OLD_AMBER_VARIANT_NAMES

    def test_terminal_caps(self):
        assert residues.TERMINAL_CAP_NAMES == OLD_CAP_NAMES

    def test_variant_to_parent_residue(self):
        # The old EvoBind table lacked HIN, which core.nonstandard already treats as
        # a histidine variant; HIN is the one deliberate addition.
        assert residues.VARIANT_TO_PARENT_RESIDUE == {**OLD_RESNAME_EQUIVALENTS, "HIN": "HIS"}

    def test_hin_is_a_histidine_variant(self):
        assert residues.VARIANT_TO_PARENT_RESIDUE["HIN"] == "HIS"

    def test_comparison_waters(self):
        assert residues.WATER_NAMES_PDB_AMBER == OLD_COMPARISON_WATERS

    def test_cysteine_names(self):
        assert residues.CYSTEINE_NAMES == OLD_CYSTEINE_NAMES

    def test_standard_amino_acids_are_the_twenty_canonical_ones(self):
        assert len(residues.STANDARD_AMINO_ACIDS) == 20
        assert {"ALA", "GLY", "TRP", "VAL"} <= residues.STANDARD_AMINO_ACIDS
        assert "MSE" not in residues.STANDARD_AMINO_ACIDS


class TestSetRelations:
    def test_outside_ccd_is_a_subset_of_the_amber_variants(self):
        assert residues.AMBER_VARIANTS_OUTSIDE_CCD < residues.AMBER_PROTONATION_VARIANTS

    def test_variants_map_onto_standard_amino_acids(self):
        assert set(residues.VARIANT_TO_PARENT_RESIDUE.values()) <= residues.STANDARD_AMINO_ACIDS
        assert residues.CYSTEINE_NAMES - {"CYX"} <= residues.STANDARD_AMINO_ACIDS

    def test_the_ccd_split_matches_biotite(self):
        """The variants kept in the CCD-gap set are the ones biotite does not list."""
        import biotite.structure.info as info

        ccd_amino_acids = set(info.amino_acid_names())
        outside = residues.AMBER_VARIANTS_OUTSIDE_CCD
        inside = residues.AMBER_PROTONATION_VARIANTS - outside
        assert not outside & ccd_amino_acids
        assert inside <= ccd_amino_acids

    def test_water_names_of_the_comparison_exclude_wider_solvent_names(self):
        assert not {"TIP3", "SOL", "H2O"} & residues.WATER_NAMES_PDB_AMBER


class TestModulesUseTheSharedSets:
    def test_interface_polymer_filter_keeps_variants_and_caps_and_drops_solvent(self):
        import biotite.structure as struc

        names = ["ALA", "HID", "CYX", "ACE", "NME", "HOH", "ZN"]
        atoms = struc.AtomArray(len(names))
        atoms.coord[:] = np.arange(len(names))[:, None] * np.ones(3)
        atoms.res_name[:] = names
        atoms.atom_name[:] = ["CA", "CA", "SG", "C", "N", "O", "ZN"]
        atoms.element[:] = ["C", "C", "S", "C", "N", "O", "ZN"]
        atoms.chain_id[:] = "A"
        atoms.res_id[:] = np.arange(1, len(names) + 1)

        kept = interface.filter_hetero_atoms(atoms, "ignore")

        assert list(kept.res_name) == ["ALA", "HID", "CYX", "ACE", "NME"]

    @pytest.mark.parametrize(
        ("name_a", "name_b", "expected"),
        [
            ("HID", "HIE", 0.0),
            ("CYX", "CYS", 0.0),
            ("HSD", "HIS", 0.0),
            ("LYN", "LYS", 0.0),
            ("HIS", "ASP", 1.0),
            ("ALA", "GLY", 1.0),
        ],
    )
    def test_evobind_treats_protonation_variants_as_the_same_residue(
        self, name_a, name_b, expected
    ):
        import biotite.structure as struc

        def _one_atom(name):
            atoms = struc.AtomArray(1)
            atoms.res_name[:] = name
            return atoms

        assert evobind._resname_mismatch_fraction(_one_atom(name_a), _one_atom(name_b)) == expected

    def test_comparison_skips_hoh_and_wat_but_not_other_solvent_names(self):
        import gemmi

        structure = gemmi.Structure()
        model = gemmi.Model("1")
        chain = gemmi.Chain("A")
        for number, name in enumerate(["ALA", "HOH", "WAT", "TIP3", "SOL"], start=1):
            residue = gemmi.Residue()
            residue.name = name
            residue.seqid = gemmi.SeqId(number, " ")
            atom = gemmi.Atom()
            atom.name = "CA" if name == "ALA" else "O"
            atom.element = gemmi.Element("C" if name == "ALA" else "O")
            atom.pos = gemmi.Position(float(number), 0.0, 0.0)
            residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)

        _, keys = _get_coords(structure)

        assert [key[1] for key in keys] == [1, 4, 5]


class TestPurePython:
    def test_module_imports_only_the_standard_library(self):
        source = Path(residues.__file__).read_text(encoding="utf-8")
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert imported <= set(sys.stdlib_module_names)

    def test_loads_with_the_scientific_packages_blocked(self):
        script = textwrap.dedent(
            """
            import importlib.abc
            import json
            import sys

            BLOCKED = ("numpy", "biotite", "openmm", "simtk", "gemmi", "scipy", "mdtraj")


            class _Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    if name.split(".")[0] in BLOCKED:
                        raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                    return None


            sys.meta_path.insert(0, _Block())
            from binding_metrics.core import residues

            print(json.dumps(sorted(residues.WATER_NAMES_PDB_AMBER)))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr[-1500:]
        assert json.loads(completed.stdout) == ["HOH", "WAT"]
