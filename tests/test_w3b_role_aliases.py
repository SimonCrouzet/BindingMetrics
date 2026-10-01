"""Role aliases ``binder_chain`` / ``target_chain`` on the metric functions.

Every metric function that names the binder ``peptide_chain``, ``design_chain``
or ``chain`` accepts ``binder_chain``, and every one that names the target
``receptor_chain`` accepts ``target_chain``. The alias fills the old parameter;
the two spellings with different IDs are a ValueError.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")

from binding_metrics.metrics._common import resolve_chain_role  # noqa: E402
from binding_metrics.metrics.comparison import compute_structure_rmsd  # noqa: E402
from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain  # noqa: E402
from binding_metrics.metrics.evobind import (  # noqa: E402
    compute_evobind_adversarial_check,
    compute_evobind_score,
)
from binding_metrics.metrics.geometry import (  # noqa: E402
    compute_buried_void_volume,
    compute_omega_planarity,
    compute_ramachandran,
    compute_shape_complementarity,
)
from binding_metrics.metrics.interface import (  # noqa: E402
    compute_interface_metrics,
    load_biotite_structure,
)
from binding_metrics.metrics.polar_contacts import compute_hbonds, compute_saltbridges  # noqa: E402
from binding_metrics.metrics.sasa import compute_delta_sasa_static  # noqa: E402

P53 = (
    Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
)  # A: MDM2, B: p53 peptide

# Roles swapped relative to auto-detection (which picks B as binder), so an ignored
# alias changes the result.
BINDER_ID, TARGET_ID = "A", "B"

pytestmark = pytest.mark.skipif(not P53.exists(), reason="1YCR example not bundled")


def _assert_same(a, b, path="result"):
    """Deep equality that treats NaN as equal to NaN and compares arrays exactly."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a:
            _assert_same(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b), path
    else:
        assert a == b, path


class Case:
    """One metric function and the names of its chain parameters.

    ``binder`` and ``target`` are the old parameter names (None when the
    function has no such role); ``required`` marks functions whose chains
    are mandatory. ``binder_id`` and ``target_id`` are the chain IDs the test
    passes.
    """

    def __init__(
        self,
        fn,
        fixed,
        binder=None,
        target=None,
        required=False,
        binder_id=BINDER_ID,
        target_id=TARGET_ID,
    ):
        self.fn, self.fixed = fn, fixed
        self.binder, self.target, self.required = binder, target, required
        self.binder_id, self.target_id = binder_id, target_id

    def call(self, **chains):
        return self.fn(**self.fixed, **chains)


def _atoms():
    return load_biotite_structure(P53)


CASES = {
    "interface": lambda: Case(
        compute_interface_metrics, {"cif_path": P53}, "design_chain", "receptor_chain"
    ),
    "coulomb": lambda: Case(
        compute_coulomb_cross_chain, {"cif_path": P53}, "peptide_chain", "receptor_chain"
    ),
    "shape_complementarity": lambda: Case(
        compute_shape_complementarity, {"cif_path": P53}, "peptide_chain", "receptor_chain"
    ),
    "void_volume": lambda: Case(
        compute_buried_void_volume,
        {"cif_path": P53, "grid_spacing": 1.0},
        "peptide_chain",
        "receptor_chain",
    ),
    "ramachandran": lambda: Case(compute_ramachandran, {"cif_path": P53}, binder="chain"),
    "omega": lambda: Case(compute_omega_planarity, {"cif_path": P53}, binder="chain"),
    "structure_rmsd": lambda: Case(
        compute_structure_rmsd, {"initial_path": P53, "processed_path": P53}, binder="design_chain"
    ),
    "delta_sasa_static": lambda: Case(
        compute_delta_sasa_static,
        {"cif_path": P53},
        "peptide_chain",
        "receptor_chain",
        required=True,
    ),
    "hbonds": lambda: Case(
        compute_hbonds, {"atoms": _atoms()}, "peptide_chain", "receptor_chain", required=True
    ),
    "saltbridges": lambda: Case(
        compute_saltbridges, {"atoms": _atoms()}, "peptide_chain", "receptor_chain", required=True
    ),
    "evobind_score": lambda: Case(
        compute_evobind_score,
        {"structure_path": P53, "plddt_per_atom": None, "binder_chain": "B"},
        binder=None,
        target="receptor_chain",
        required=True,
        target_id="A",
    ),
    "evobind_adversarial": lambda: Case(
        compute_evobind_adversarial_check,
        {"design_structure_path": P53, "afm_structure_path": P53, "binder_chain": "B"},
        binder=None,
        target="receptor_chain",
        required=True,
        target_id="A",
    ),
}


def _names_where(attribute):
    """Case names whose function has the given role (parametrize only what applies)."""
    return sorted(name for name, make in CASES.items() if getattr(make(), attribute))


@pytest.fixture(params=sorted(CASES))
def case(request):
    return CASES[request.param]()


@pytest.fixture(params=_names_where("binder"))
def binder_case(request):
    return CASES[request.param]()


@pytest.fixture(params=_names_where("target"))
def target_case(request):
    return CASES[request.param]()


@pytest.fixture(params=_names_where("required"))
def required_case(request):
    return CASES[request.param]()


def _legacy(case):
    chains = {}
    if case.binder:
        chains[case.binder] = case.binder_id
    if case.target:
        chains[case.target] = case.target_id
    return chains


def _aliased(case):
    chains = {}
    if case.binder:
        chains["binder_chain"] = case.binder_id
    if case.target:
        chains["target_chain"] = case.target_id
    return chains


def test_alias_gives_the_same_result_as_the_old_names(case):
    _assert_same(case.call(**_aliased(case)), case.call(**_legacy(case)))


def test_alias_and_old_name_with_the_same_id_are_accepted(case):
    both = {**_legacy(case), **_aliased(case)}
    _assert_same(case.call(**both), case.call(**_legacy(case)))


def test_conflicting_binder_ids_are_a_value_error(binder_case):
    chains = {**_legacy(binder_case), "binder_chain": "Z"}
    with pytest.raises(ValueError, match="binder_chain"):
        binder_case.call(**chains)


def test_conflicting_target_ids_are_a_value_error(target_case):
    chains = {**_legacy(target_case), "target_chain": "Z"}
    with pytest.raises(ValueError, match="target_chain"):
        target_case.call(**chains)


def test_missing_chains_raise_type_error_where_they_are_mandatory(required_case):
    with pytest.raises(TypeError, match="missing required"):
        required_case.call()


def test_aliases_are_keyword_only(case):
    params = inspect.signature(case.fn).parameters
    for name in ("binder_chain", "target_chain"):
        if name in params and not (name == "binder_chain" and case.binder is None):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert params[name].default is None


class TestResolveChainRole:
    def test_neither_given(self):
        assert resolve_chain_role("peptide_chain", None, "binder_chain", None) is None

    def test_legacy_only(self):
        assert resolve_chain_role("peptide_chain", "B", "binder_chain", None) == "B"

    def test_alias_only(self):
        assert resolve_chain_role("peptide_chain", None, "binder_chain", "B") == "B"

    def test_conflict_names_both_spellings(self):
        with pytest.raises(ValueError, match=r"peptide_chain='B'.*binder_chain='C'"):
            resolve_chain_role("peptide_chain", "B", "binder_chain", "C")

    def test_required_missing(self):
        with pytest.raises(TypeError, match="peptide_chain.*binder_chain"):
            resolve_chain_role("peptide_chain", None, "binder_chain", None, required=True)


class TestMetricsThatCannotRunHere:
    """Functions that need OpenMM, a trajectory or model output: check what reaches the core."""

    def test_interaction_energy_passes_the_resolved_chains_on(self, monkeypatch):
        pytest.importorskip("openmm")
        import openmm.app

        import binding_metrics.io.structures as structures
        from binding_metrics.metrics.energy import compute_interaction_energy

        seen = []

        def fake_strip(topology, positions, peptide_chain, receptor_chain):
            seen.append((peptide_chain, receptor_chain))
            raise RuntimeError("stop after chain resolution")

        # an empty topology: the chain IDs are looked up as author IDs first, then taken as given
        monkeypatch.setattr(structures, "load_structure", lambda path: (openmm.app.Topology(), []))
        monkeypatch.setattr(structures, "strip_heterogens", fake_strip)
        compute_interaction_energy(P53, binder_chain="B", target_chain="A", modes=("raw",))
        compute_interaction_energy(P53, peptide_chain="B", receptor_chain="A", modes=("raw",))
        assert seen == [("B", "A"), ("B", "A")]

    def test_interaction_energy_conflict_is_raised_before_any_work(self):
        from binding_metrics.metrics.energy import compute_interaction_energy

        with pytest.raises(ValueError, match="binder_chain"):
            compute_interaction_energy("missing.pdb", peptide_chain="B", binder_chain="C")
        with pytest.raises(ValueError, match="target_chain"):
            compute_interaction_energy("missing.pdb", receptor_chain="A", target_chain="C")

    def test_receptor_quality_passes_the_resolved_chain_on(self, monkeypatch):
        from binding_metrics.metrics import receptor_quality

        class StopHereError(Exception):
            pass

        def fake_score(atoms, receptor_chain, *args, **kwargs):
            raise StopHereError(receptor_chain)

        monkeypatch.setattr(receptor_quality, "_score_model", fake_score)
        with pytest.raises(StopHereError, match="A"):
            receptor_quality.compute_receptor_quality(P53, target_chain="A")
        with pytest.raises(ValueError, match="target_chain"):
            receptor_quality.compute_receptor_quality(P53, receptor_chain="A", target_chain="B")

    def test_receptor_drift_alias(self, tmp_path):
        md = pytest.importorskip("mdtraj")
        from binding_metrics.metrics.rmsd import compute_receptor_drift

        traj = md.load(str(P53))
        dcd = tmp_path / "one_frame.dcd"
        traj.save_dcd(str(dcd))
        via_alias = compute_receptor_drift(dcd, P53, target_chain="A")
        via_old_name = compute_receptor_drift(dcd, P53, "A")
        assert via_alias["n_receptor_ca"] == via_old_name["n_receptor_ca"] > 0
        with pytest.raises(ValueError, match="target_chain"):
            compute_receptor_drift(dcd, P53, "A", target_chain="B")
        with pytest.raises(TypeError, match="missing required"):
            compute_receptor_drift(dcd, P53)

    def test_interface_pae_alias(self, monkeypatch):
        from binding_metrics.metrics import openfold

        monkeypatch.setattr(openfold, "_parse_confidences", lambda path: {"pae": object()})
        monkeypatch.setattr(openfold, "_load_atoms", lambda path: object())
        monkeypatch.setattr(
            openfold,
            "_interface_pae_stats",
            lambda pae, atoms, binder, receptor, token_ranges=None: (binder, receptor),
        )
        assert openfold.compute_interface_pae("c.json", "s.cif", "B", target_chain="A") == (
            "B",
            "A",
        )
        assert openfold.compute_interface_pae("c.json", "s.cif", "B", "A") == ("B", "A")
        with pytest.raises(ValueError, match="target_chain"):
            openfold.compute_interface_pae("c.json", "s.cif", "B", "A", target_chain="C")
        with pytest.raises(TypeError, match="missing required"):
            openfold.compute_interface_pae("c.json", "s.cif", "B")

    def test_openfold_metrics_alias(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        result = compute_openfold_metrics(tmp_path, "q", binder_chain="B", target_chain="A")
        assert "no confidence files" in result["reason"]
        with pytest.raises(ValueError, match="target_chain"):
            compute_openfold_metrics(tmp_path, "q", receptor_chain="A", target_chain="C")
