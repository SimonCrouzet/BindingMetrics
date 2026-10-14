"""The relaxation setup does not change what Modeller does for the rest of the process (#163).

``Modeller.addHydrogens`` places the hydrogens of a phosphorylated residue only when its
definitions are loaded into ``Modeller``, which is state of the whole process. Prep and the
energy step load them for the duration of their hydrogen addition
(``phosaa.hydrogen_definitions``); the relaxation setup loaded them for good, so everything
that added hydrogens after a relaxation in the same process ran with them and the same
call before a relaxation ran without. The relaxation setup now uses the scoped form.

The receptor energy (``receptor_quality._receptor_energy``) adds hydrogens without a
force field of its own, but it never meets a phosphorylated residue: PDBFixer turns SEP,
TPO and PTR into SER, THR and TYR first, and without PDBFixer the force field has no
template for the residue. Its result is therefore the same in either order (the subprocess
tests below).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("pdbfixer")

from conftest import DATA_DIR
from openmm.app import Modeller

PHOSPHO = DATA_DIR / "example_phospho_1QJB.pdb"

pytestmark = pytest.mark.slow


@pytest.fixture
def clean_definitions(monkeypatch):
    """Run with no phospho definition in Modeller and put the earlier ones back afterwards.

    ``phosaa.ensure_hydrogen_definitions`` remembers in a module flag that it has loaded them,
    so the flag is reset too: a test must not depend on what ran before it in the process.
    """
    from binding_metrics.core import phosaa

    monkeypatch.setattr(phosaa, "_hydrogen_definitions_loaded", False)
    names = phosaa.phospho_residue_names()
    earlier = {n: Modeller._residueHydrogens[n] for n in names if n in Modeller._residueHydrogens}
    for name in names:
        Modeller._residueHydrogens.pop(name, None)
    yield names
    for name in names:
        Modeller._residueHydrogens.pop(name, None)
    Modeller._residueHydrogens.update(earlier)


def _left_in_modeller(names) -> list:
    return sorted(name for name in names if name in Modeller._residueHydrogens)


class TestRelaxationSetup:
    def test_no_definition_stays_loaded(self, prepped_example_cif, clean_definitions):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        ImplicitRelaxation(RelaxationConfig())._setup_system(prepped_example_cif)

        assert _left_in_modeller(clean_definitions) == []

    def test_an_earlier_definition_is_back_afterwards(self, prepped_example_cif, clean_definitions):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        earlier = object()
        Modeller._residueHydrogens["SEP"] = earlier

        ImplicitRelaxation(RelaxationConfig())._setup_system(prepped_example_cif)

        assert Modeller._residueHydrogens["SEP"] is earlier

    def test_the_state_is_restored_when_the_hydrogen_addition_fails(
        self, prepped_example_cif, clean_definitions, monkeypatch
    ):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        def fail(self, *args, **kwargs):
            raise ValueError("boom")

        monkeypatch.setattr(Modeller, "addHydrogens", fail)
        with pytest.raises(RuntimeError, match="addHydrogens failed"):
            ImplicitRelaxation(RelaxationConfig())._setup_system(prepped_example_cif)

        assert _left_in_modeller(clean_definitions) == []

    def test_the_phosphoserine_still_gets_its_hydrogens(self, tmp_path, clean_definitions):
        """The definitions are loaded where the hydrogens are added: SEP5 of 1QJB keeps all four."""
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure, save_structure
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        topology, positions = load_structure(PHOSPHO)
        topology, positions = prep_structure(topology, positions)
        path = tmp_path / "1QJB_prepped.pdb"
        save_structure(topology, positions, path)
        # Heavy atoms only, so that the relaxation setup has to place the hydrogens itself.
        stripped = tmp_path / "1QJB_heavy.pdb"
        stripped.write_text(
            "".join(
                line
                for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
                if not (line.startswith(("ATOM", "HETATM")) and line[76:78].strip() == "H")
            ),
            encoding="utf-8",
        )

        _, setup_topology, _, _ = ImplicitRelaxation(RelaxationConfig())._setup_system(stripped)

        sep = next(r for r in setup_topology.residues() if r.name == "SEP")
        assert sorted(a.name for a in sep.atoms() if a.element.symbol == "H") == [
            "H",
            "HA",
            "HB2",
            "HB3",
        ]


#: Runs in a fresh interpreter: do the steps named on the command line in that order and write
#: what the 1QJB steps produced to a JSON file.
_STEPS = """
import json, sys, warnings
warnings.simplefilter("ignore")
from pathlib import Path
import biotite.structure as struc
import biotite.structure.io as sio
import openmm.unit as unit
from binding_metrics.core.system import prep_structure
from binding_metrics.io.structures import load_structure, save_structure
from binding_metrics.metrics.receptor_quality import _receptor_energy
from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

out_path, phospho, other_prepped = sys.argv[1:4]
out = {}
for step in sys.argv[4:]:
    if step == "relax_other":
        ImplicitRelaxation(RelaxationConfig())._setup_system(Path(other_prepped))
    elif step == "prep_phospho":
        topology, positions = prep_structure(*load_structure(phospho))
        out[step] = {
            "atoms": [[a.residue.name, a.residue.id, a.name] for a in topology.atoms()],
            "positions_nm": [[v.x, v.y, v.z] for v in positions.value_in_unit(unit.nanometer)],
        }
    elif step == "energy_phospho":
        atoms = sio.load_structure(phospho)
        atoms = atoms[0] if isinstance(atoms, struc.AtomArrayStack) else atoms
        out[step] = _receptor_energy(atoms[struc.filter_amino_acids(atoms)], device="cpu")
Path(out_path).write_text(json.dumps(out), encoding="utf-8")
"""


def _run(tmp_path: Path, name: str, other_prepped: Path, *steps: str) -> dict:
    output = tmp_path / f"{name}.json"
    subprocess.run(
        [sys.executable, "-c", _STEPS, str(output), str(PHOSPHO), str(other_prepped), *steps],
        check=True,
        capture_output=True,
        timeout=900,
    )
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory, prepped_example_cif):
    """The 1QJB steps alone, after a relaxation setup of 1YCR, and before one."""
    directory = tmp_path_factory.mktemp("order")
    other = Path(prepped_example_cif)
    return {
        "alone": _run(directory, "alone", other, "prep_phospho", "energy_phospho"),
        "relax_first": _run(
            directory, "relax_first", other, "relax_other", "prep_phospho", "energy_phospho"
        ),
        "relax_last": _run(
            directory, "relax_last", other, "prep_phospho", "energy_phospho", "relax_other"
        ),
    }


class TestOrderOfSteps:
    def test_prep_gives_the_same_atoms_and_coordinates_after_a_relaxation(self, runs):
        assert runs["alone"]["prep_phospho"] == runs["relax_first"]["prep_phospho"]
        assert runs["alone"]["prep_phospho"] == runs["relax_last"]["prep_phospho"]

    def test_the_receptor_energy_is_the_same_after_a_relaxation(self, runs):
        """Same atoms; the energy only to the summation order of the multithreaded CPU platform."""
        alone = runs["alone"]["energy_phospho"]
        assert alone["error"] is None
        for other in ("relax_first", "relax_last"):
            energy = runs[other]["energy_phospho"]
            assert energy["n_atoms_with_h"] == alone["n_atoms_with_h"]
            assert energy["energy_kJ_mol"] == pytest.approx(alone["energy_kJ_mol"], rel=1e-6)
