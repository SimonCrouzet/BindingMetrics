"""The prepared structure does not depend on what was prepared before it (#163).

``Modeller.addHydrogens`` places the hydrogens of the phosphorylated residues only when
their hydrogen definitions are loaded into ``Modeller``, which is process-global state.
The cyclic route of ``prep_structure`` loaded them and the linear route did not, so
1QJB (SEP, linear) came out with 120 atoms in a fresh process, SEP without a single
hydrogen, and with 124 atoms once a cyclic peptide had been prepared before it in the
same process. phosaa14SB gives SEP its hydrogens H, HA, HB2 and HB3 whatever the pH: the
protonation state of the phosphate is chosen by the residue name (SEP is the dianion,
S1P the monoanion), not by ``pH``. The four hydrogens are part of the residue, and both
routes now load the definitions for the duration of the hydrogen addition and restore
the earlier state afterwards.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openmm")

from conftest import DATA_DIR
from openmm.app import Modeller

PHOSPHO = DATA_DIR / "example_phospho_1QJB.pdb"
LACTAM = DATA_DIR / "example_lactam_somatostatin_1XY4.cif"

#: Runs in a fresh interpreter: prep each file named on the command line, in that order,
#: and write for each the atoms (chain, residue name, residue number, atom name) and the
#: coordinates to a JSON file.
_PREP_SCRIPT = """
import json, sys, warnings
warnings.simplefilter("ignore")
import openmm.unit as unit
from binding_metrics.core.system import prep_structure
from binding_metrics.io.structures import load_structure

out = {}
for path in sys.argv[2:]:
    topology, positions = load_structure(path)
    prepped, prepped_positions = prep_structure(topology, positions)
    out[path] = {
        "atoms": [
            [a.residue.chain.id, a.residue.name, a.residue.id, a.name] for a in prepped.atoms()
        ],
        "positions_nm": [
            [v.x, v.y, v.z] for v in prepped_positions.value_in_unit(unit.nanometer)
        ],
    }
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(out, handle)
"""


def _prep_in_fresh_process(tmp_path: Path, name: str, *paths: Path) -> dict:
    output = tmp_path / f"{name}.json"
    subprocess.run(
        [sys.executable, "-c", _PREP_SCRIPT, str(output), *map(str, paths)],
        check=True,
        capture_output=True,
        timeout=600,
    )
    return json.loads(output.read_text(encoding="utf-8"))


class TestHydrogenDefinitionsAreScoped:
    def test_the_definitions_are_loaded_inside_the_block_only(self, monkeypatch):
        from binding_metrics.core import phosaa

        for name in phosaa.phospho_residue_names():
            monkeypatch.delitem(Modeller._residueHydrogens, name, raising=False)

        with phosaa.hydrogen_definitions():
            inside = {n: Modeller._residueHydrogens.get(n) for n in phosaa.phospho_residue_names()}
        after = [n for n in phosaa.phospho_residue_names() if n in Modeller._residueHydrogens]

        assert all(entry is not None for entry in inside.values())
        assert {h.name for h in inside["SEP"].hydrogens} == {"H", "HA", "HB2", "HB3"}
        assert after == []

    def test_an_earlier_registration_is_back_after_the_block(self, monkeypatch):
        from binding_metrics.core import phosaa

        earlier = object()
        monkeypatch.setitem(Modeller._residueHydrogens, "SEP", earlier)

        with phosaa.hydrogen_definitions():
            assert Modeller._residueHydrogens["SEP"] is not earlier

        assert Modeller._residueHydrogens["SEP"] is earlier

    def test_the_state_is_restored_when_the_block_raises(self, monkeypatch):
        from binding_metrics.core import phosaa

        monkeypatch.delitem(Modeller._residueHydrogens, "SEP", raising=False)
        with pytest.raises(RuntimeError, match="boom"):
            with phosaa.hydrogen_definitions():
                raise RuntimeError("boom")

        assert "SEP" not in Modeller._residueHydrogens


class TestPrepOfAPhosphorylatedPeptide:
    def test_sep_gets_its_four_hydrogens_and_none_on_the_phosphate(self, monkeypatch):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core import phosaa
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        for name in phosaa.phospho_residue_names():
            monkeypatch.delitem(Modeller._residueHydrogens, name, raising=False)
        topology, positions = load_structure(PHOSPHO)
        prepped, _ = prep_structure(topology, positions)

        sep = next(r for r in prepped.residues() if r.name == "SEP")
        hydrogens = sorted(a.name for a in sep.atoms() if a.element.symbol == "H")
        assert hydrogens == ["H", "HA", "HB2", "HB3"]  # the phosphate of SEP is the dianion
        assert prepped.getNumAtoms() == 124

    def test_prep_leaves_no_definition_behind(self, monkeypatch):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core import phosaa
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        for name in phosaa.phospho_residue_names():
            monkeypatch.delitem(Modeller._residueHydrogens, name, raising=False)
        topology, positions = load_structure(PHOSPHO)
        prep_structure(topology, positions)

        assert [n for n in phosaa.phospho_residue_names() if n in Modeller._residueHydrogens] == []


@pytest.fixture(scope="module")
def prepped_in_three_orders(tmp_path_factory):
    """1QJB alone, 1XY4 then 1QJB, and 1QJB then 1XY4, each in a fresh process."""
    pytest.importorskip("pdbfixer")
    pytest.importorskip("rdkit")
    if shutil.which("antechamber") is None:
        pytest.skip("antechamber is not on PATH (the cyclic peptide needs GAFF2)")
    directory = tmp_path_factory.mktemp("orders")
    return {
        "alone": _prep_in_fresh_process(directory, "alone", PHOSPHO),
        "cyclic_first": _prep_in_fresh_process(directory, "cyclic_first", LACTAM, PHOSPHO),
        "linear_first": _prep_in_fresh_process(directory, "linear_first", PHOSPHO, LACTAM),
    }


@pytest.mark.slow
class TestOrderOfPreps:
    def test_1qjb_is_the_same_alone_and_after_a_cyclic_peptide(self, prepped_in_three_orders):
        alone = prepped_in_three_orders["alone"][str(PHOSPHO)]
        after = prepped_in_three_orders["cyclic_first"][str(PHOSPHO)]

        assert len(alone["atoms"]) == 124
        assert alone["atoms"] == after["atoms"]
        assert alone["positions_nm"] == after["positions_nm"]

    def test_a_cyclic_peptide_is_the_same_before_and_after_1qjb(self, prepped_in_three_orders):
        first = prepped_in_three_orders["cyclic_first"]
        second = prepped_in_three_orders["linear_first"]

        assert first[str(LACTAM)]["atoms"] == second[str(LACTAM)]["atoms"]
        assert first[str(LACTAM)]["positions_nm"] == second[str(LACTAM)]["positions_nm"]
        assert first[str(PHOSPHO)] == second[str(PHOSPHO)]
