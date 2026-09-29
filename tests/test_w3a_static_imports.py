"""The relaxation module and ``io.structures`` import without OpenMM.

The checks run in a subprocess whose ``sys.meta_path`` refuses OpenMM, so the fake
"not installed" state cannot leak into the test session (the same approach as
``test_w2a_import_without_openmm``). OpenMM is still required where it does the work:
running a relaxation or reading a structure into an OpenMM topology.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

EXAMPLE_PDB = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import json
    import sys
    import traceback
    from pathlib import Path

    BLOCKED = ("openmm", "simtk")


    class _BlockOpenMM(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
            return None


    sys.meta_path.insert(0, _BlockOpenMM())

    EXAMPLE = {example!r}
    results = {{}}


    def check(function):
        try:
            function()
        except BaseException:
            results[function.__name__] = traceback.format_exc()[-1500:]
        else:
            results[function.__name__] = "ok"


    @check
    def structures_module_imports():
        import binding_metrics.io.structures as structures

        for name in ("detect_chains", "detect_chains_from_file", "detect_models",
                     "extract_model_to_tempfile", "merge_cif_models", "save_cif",
                     "load_structure", "strip_heterogens"):
            assert callable(getattr(structures, name)), name


    @check
    def chain_detection_from_file_needs_no_openmm():
        from binding_metrics.io.structures import detect_chains_from_file

        detected = detect_chains_from_file(EXAMPLE, verbose=False)
        # 1YCR: MDM2 (chain A, 85 residues) bound to a 13-residue p53 peptide (chain B).
        assert (detected["peptide_chain"], detected["receptor_chain"]) == ("B", "A"), detected


    @check
    def openmm_names_are_resolved_on_demand():
        import binding_metrics.io.structures as structures

        for name in ("app", "PDBFile"):
            try:
                getattr(structures, name)
            except ModuleNotFoundError as error:
                assert error.name == "openmm.app" or error.name == "openmm", error.name
            else:
                raise AssertionError(f"structures.{{name}} needs OpenMM")
        try:
            structures.no_such_name
        except AttributeError:
            pass
        else:
            raise AssertionError("unknown attribute must raise AttributeError")


    @check
    def load_structure_still_needs_openmm():
        from binding_metrics.io.structures import load_structure

        try:
            load_structure(EXAMPLE)
        except ModuleNotFoundError as error:
            assert "openmm" in str(error), str(error)
        else:
            raise AssertionError("load_structure builds an OpenMM topology")


    @check
    def relaxation_module_imports():
        from binding_metrics.protocols.relaxation import (
            ImplicitRelaxation, RelaxationConfig, RelaxationResult,
        )
        from binding_metrics import RelaxationConfig as exported_config

        assert exported_config is RelaxationConfig
        config = RelaxationConfig()
        assert config.random_seed == 1
        assert ImplicitRelaxation(config).config is config
        assert RelaxationResult(sample_id="x", success=False).success is False


    @check
    def relaxer_contract_imports_and_can_be_subclassed():
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig
        from binding_metrics.protocols.relaxer import Relaxer

        class Stub(Relaxer):
            def run(self, input_path, output_dir, sample_id=None):
                raise NotImplementedError

        assert isinstance(Stub(), Relaxer)
        assert issubclass(ImplicitRelaxation, Relaxer)
        assert isinstance(ImplicitRelaxation(RelaxationConfig()), Relaxer)


    @check
    def relaxation_run_names_the_missing_package():
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        try:
            ImplicitRelaxation(RelaxationConfig()).run(Path(EXAMPLE), Path("out"))
        except ImportError as error:
            assert "OpenMM is required" in str(error), str(error)
        else:
            raise AssertionError("run needs OpenMM")


    @check
    def no_openmm_module_was_imported():
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
        assert not loaded, loaded


    print("RESULTS=" + json.dumps(results))
    """
)

CHECKS = [
    "structures_module_imports",
    "chain_detection_from_file_needs_no_openmm",
    "openmm_names_are_resolved_on_demand",
    "load_structure_still_needs_openmm",
    "relaxation_module_imports",
    "relaxer_contract_imports_and_can_be_subclassed",
    "relaxation_run_names_the_missing_package",
    "no_openmm_module_was_imported",
]


@pytest.fixture(scope="module")
def blocked_results(tmp_path_factory):
    script = _SCRIPT.format(example=str(EXAMPLE_PDB))
    run = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path_factory.mktemp("no_openmm"),
        timeout=600,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(x for x in run.stdout.splitlines() if x.startswith("RESULTS="))
    return json.loads(line[len("RESULTS=") :])


@pytest.mark.parametrize("check", CHECKS)
def test_without_openmm(blocked_results, check):
    assert blocked_results[check] == "ok", blocked_results[check]


def test_relaxation_seed_is_the_shared_constant():
    from binding_metrics import _constants
    from binding_metrics.protocols import relaxation

    assert relaxation.DEFAULT_RANDOM_SEED is _constants.DEFAULT_RANDOM_SEED


def test_structures_openmm_names_still_resolve_with_openmm():
    openmm_app = pytest.importorskip("openmm.app")
    import binding_metrics.io.structures as structures

    assert structures.PDBFile is openmm_app.PDBFile
    assert structures.app is openmm_app
    topology, positions = structures.load_structure(EXAMPLE_PDB)
    # 1YCR is a heavy-atom model: 818 atoms for the 85-residue MDM2 fragment and p53 peptide.
    assert topology.getNumAtoms() == len(positions)
    assert 700 < topology.getNumAtoms() < 1000
