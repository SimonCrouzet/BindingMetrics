"""Tests for the consumer metadata on ``MetricSpec``.

``direction``, ``unit``, ``cost_class``, ``requires_extras``, ``requires_gpu`` and
``headline_key`` let a generic caller rank and schedule metrics without knowing
each function. Three kinds of check keep them honest:

* every value comes from the documented sets, and the fields agree with each
  other and with the fields that existed before (input type, chain mode);
* every ``requires_extras`` name is an extra that ``pyproject.toml`` defines;
* every ``headline_key`` is a key the function really returns, and the value
  has the range its unit implies. These run the cheap metrics on the bundled
  1YCR complex (MDM2 chain A, p53 peptide chain B, waters removed) and on a
  two-frame trajectory built from it.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics.registry import (
    COST_CLASSES,
    DIRECTIONS,
    KNOWN_UNITS,
    METRICS,
    MetricSpec,
    get_metric,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_1YCR = REPO_ROOT / "data" / "example_linear_p53_1YCR.pdb"


def _resolve(result, dotted_key: str):
    """Follow a dotted headline key (``"summary.molprobity_score"``) into a result dict."""
    value = result
    for part in dotted_key.split("."):
        value = value[part]
    return value


# ---------------------------------------------------------------------------
# Field values and cross-field consistency
# ---------------------------------------------------------------------------


class TestMetadataValues:
    def test_defaults_leave_a_minimal_spec_unchanged(self):
        spec = MetricSpec(
            name="x", import_path="m:f", description="d", input_type="static_structure"
        )
        assert spec.headline_key is None
        assert spec.direction is None
        assert spec.unit is None
        assert spec.cost_class is None
        assert spec.requires_extras == ()
        assert spec.requires_gpu is False

    def test_old_positional_construction_still_works(self):
        """The eleven original fields keep their positions; new fields come after."""
        spec = MetricSpec(
            "x",
            "m:f",
            "d",
            "trajectory",
            "interface",
            ("pdb",),
            "trajectory_path",
            None,
            None,
            "ligand_indices",
            "receptor_indices",
        )
        assert spec.peptide_chain_arg == "ligand_indices"
        assert spec.receptor_chain_arg == "receptor_indices"
        assert spec.cost_class is None

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_direction_is_from_the_allowed_set(self, spec):
        assert spec.direction is None or spec.direction in DIRECTIONS

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_unit_is_from_the_allowed_set(self, spec):
        assert spec.unit is None or spec.unit in KNOWN_UNITS

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_every_registered_metric_declares_a_cost_class(self, spec):
        assert spec.cost_class in COST_CLASSES

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_requires_extras_is_a_tuple_of_names(self, spec):
        assert isinstance(spec.requires_extras, tuple)
        assert all(isinstance(e, str) and e for e in spec.requires_extras)
        assert len(set(spec.requires_extras)) == len(spec.requires_extras)

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_requires_gpu_is_a_bool(self, spec):
        assert isinstance(spec.requires_gpu, bool)

    def test_allowed_sets_are_ascii(self):
        for value in (*DIRECTIONS, *COST_CLASSES, *KNOWN_UNITS):
            assert value.isascii(), value


class TestMetadataConsistency:
    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_headline_key_comes_with_a_direction_or_a_unit(self, spec):
        if spec.headline_key is not None:
            assert spec.headline_key.strip()
            assert spec.direction or spec.unit, f"{spec.name}: headline_key without direction/unit"

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_gpu_metrics_are_not_static(self, spec):
        """A metric that runs on a CUDA device by default builds a force-field system."""
        if spec.requires_gpu:
            assert spec.cost_class in ("structural", "md")

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_trajectory_and_simulation_inputs_are_md_cost(self, spec):
        if spec.input_type in ("trajectory", "md_simulation"):
            assert spec.cost_class == "md"

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_chain_mode_matches_declared_chain_args(self, spec):
        chain_args = [spec.chain_arg, spec.peptide_chain_arg, spec.receptor_chain_arg]
        if spec.chain_mode == "none":
            assert not any(chain_args)
        elif spec.chain_mode == "single":
            assert spec.chain_arg
            assert not (spec.peptide_chain_arg or spec.receptor_chain_arg)
        elif spec.chain_mode == "interface":
            assert spec.peptide_chain_arg or spec.receptor_chain_arg
            assert not spec.chain_arg
        else:
            assert spec.chain_mode == "interface_2paths"
            assert not spec.chain_arg

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_secondary_path_only_for_two_path_modes(self, spec):
        assert bool(spec.secondary_path_arg) == (spec.chain_mode == "interface_2paths")

    def test_energy_style_headlines_prefer_lower(self):
        """Interaction and Coulomb energies are negative when attractive."""
        for name in (
            "coulomb",
            "interaction_energy",
            "component_energies",
            "structure_interaction_energy",
            "hbonds",
            "saltbridges",
        ):
            assert get_metric(name).direction == "lower_is_better", name

    def test_buried_area_and_complementarity_prefer_higher(self):
        for name in ("delta_sasa_static", "buried_sasa", "shape_complementarity", "dockq"):
            assert get_metric(name).direction == "higher_is_better", name

    def test_bundles_declare_no_headline(self):
        """Results that mix directions and units carry no single-value metadata."""
        for name in ("interface", "openfold", "md_implicit", "contact_residues"):
            spec = get_metric(name)
            assert spec.headline_key is None
            assert spec.direction is None
            assert spec.unit is None

    def test_unclear_direction_is_left_unset(self):
        """More interface contacts is not better or worse without a question attached."""
        spec = get_metric("contacts")
        assert spec.direction is None
        assert spec.unit == "count"


class TestRequiresExtras:
    @staticmethod
    def _defined_extras() -> set[str]:
        pyproject = REPO_ROOT / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found")
        data = tomllib.loads(pyproject.read_text())
        return set(data["project"]["optional-dependencies"])

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_extras_are_defined_in_pyproject(self, spec):
        defined = self._defined_extras()
        unknown = [e for e in spec.requires_extras if e not in defined]
        assert not unknown, f"{spec.name}: not in [project.optional-dependencies]: {unknown}"

    def test_dockq_metric_needs_the_dockq_extra(self):
        assert "dockq" in get_metric("dockq").requires_extras

    def test_mdtraj_metrics_need_the_analysis_extra(self):
        for name in ("rmsd", "rmsf", "buried_sasa", "contacts", "interface_sasa"):
            assert "analysis" in get_metric(name).requires_extras, name


# ---------------------------------------------------------------------------
# Headline keys are real keys of real results
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def complex_pdb(tmp_path_factory) -> Path:
    """1YCR without heteroatoms: chain A = MDM2 (receptor), chain B = p53 peptide."""
    if not EXAMPLE_1YCR.exists():
        pytest.skip(f"bundled example not found: {EXAMPLE_1YCR}")
    lines = [ln for ln in EXAMPLE_1YCR.read_text().splitlines(True) if ln.startswith("ATOM")]
    out = tmp_path_factory.mktemp("l9_meta") / "complex.pdb"
    out.write_text("".join(lines) + "END\n")
    return out


@pytest.fixture(scope="module")
def two_frame_pdb(complex_pdb) -> Path:
    """The complex written as two identical models: a trajectory and its topology in one file."""
    lines = complex_pdb.read_text().splitlines(True)
    atoms = [ln for ln in lines if ln.startswith("ATOM")]
    text = "".join(f"MODEL     {i:>4}\n" + "".join(atoms) + "ENDMDL\n" for i in (1, 2))
    out = complex_pdb.parent / "two_frames.pdb"
    out.write_text(text + "END\n")
    return out


@pytest.fixture(scope="module")
def atoms(complex_pdb):
    pytest.importorskip("biotite")
    import biotite.structure.io.pdb as pdb_io

    return pdb_io.get_structure(pdb_io.PDBFile.read(str(complex_pdb)), model=1)


def _call(name: str, **kwargs):
    try:
        return get_metric(name).call(**kwargs)
    except ImportError as e:
        pytest.skip(f"{name}: optional dependency not installed — {e}")


@pytest.fixture(scope="module")
def static_results(complex_pdb, atoms) -> dict:
    """Results of the cheap dict-returning metrics on the 1YCR complex."""
    path = str(complex_pdb)
    plddt = np.full(len(atoms), 80.0)
    return {
        "coulomb": _call("coulomb", cif_path=path, peptide_chain="B", receptor_chain="A"),
        "ramachandran": _call("ramachandran", cif_path=path, chain="B"),
        "omega": _call("omega", cif_path=path, chain="B"),
        "shape_complementarity": _call(
            "shape_complementarity", cif_path=path, peptide_chain="B", receptor_chain="A"
        ),
        "void_volume": _call("void_volume", cif_path=path, peptide_chain="B", receptor_chain="A"),
        "structure_rmsd": _call(
            "structure_rmsd", initial_path=path, processed_path=path, design_chain="B"
        ),
        "delta_sasa_static": _call(
            "delta_sasa_static", cif_path=path, peptide_chain="B", receptor_chain="A"
        ),
        "hbonds": _call("hbonds", atoms=atoms.copy(), peptide_chain="B", receptor_chain="A"),
        "saltbridges": _call(
            "saltbridges", atoms=atoms.copy(), peptide_chain="B", receptor_chain="A"
        ),
        "evobind_score": _call(
            "evobind_score",
            structure_path=path,
            plddt_per_atom=plddt,
            binder_chain="B",
            receptor_chain="A",
        ),
        "evobind_adversarial": _call(
            "evobind_adversarial",
            design_structure_path=path,
            afm_structure_path=path,
            binder_chain="B",
            receptor_chain="A",
            afm_plddt_per_atom=plddt,
        ),
    }


@pytest.fixture(scope="module")
def trajectory_results(two_frame_pdb) -> dict:
    pytest.importorskip("mdtraj")
    import mdtraj as md

    path = str(two_frame_pdb)
    top = md.load(path, top=path).topology
    ligand = [a.index for a in top.atoms if a.residue.chain.chain_id == "B"]
    receptor = [a.index for a in top.atoms if a.residue.chain.chain_id == "A"]
    common = {"trajectory_path": path, "topology_path": path}
    pair = {"ligand_indices": ligand, "receptor_indices": receptor}
    return {
        "ligand_rmsd": _call("ligand_rmsd", **common, **pair),
        "interface_sasa": _call("interface_sasa", **common, **pair),
        "receptor_drift": _call("receptor_drift", **common, receptor_chain="A"),
        "rmsd": _call("rmsd", **common),
        "rmsf": _call("rmsf", **common),
        "buried_sasa": _call("buried_sasa", **common, **pair),
        "contacts": _call("contacts", **common, **pair),
    }


class TestHeadlineKeysAreReal:
    _STATIC_WITH_HEADLINE = (
        "coulomb",
        "ramachandran",
        "omega",
        "shape_complementarity",
        "void_volume",
        "structure_rmsd",
        "delta_sasa_static",
        "hbonds",
        "saltbridges",
        "evobind_score",
        "evobind_adversarial",
    )

    @pytest.mark.parametrize("name", _STATIC_WITH_HEADLINE)
    def test_static_headline_is_a_finite_number(self, static_results, name):
        spec = get_metric(name)
        value = _resolve(static_results[name], spec.headline_key)
        assert isinstance(value, (int, float)) and np.isfinite(value), f"{name}: {value!r}"

    @pytest.mark.parametrize("name", _STATIC_WITH_HEADLINE)
    def test_static_headline_has_the_range_its_unit_implies(self, static_results, name):
        spec = get_metric(name)
        value = float(_resolve(static_results[name], spec.headline_key))
        if spec.unit == "percent":
            assert 0.0 <= value <= 100.0
        elif spec.unit == "fraction":
            assert 0.0 <= value <= 1.0
        elif spec.unit in ("angstrom", "angstrom^2", "angstrom^3"):
            assert value >= 0.0
        elif spec.unit == "kcal/mol" and name in ("hbonds", "saltbridges"):
            # Both are sums of attractive pair terms.
            assert value <= 0.0

    def test_interface_bundle_returns_the_keys_the_scorecard_reads(self, complex_pdb):
        """The bundle has no headline; the descriptors it holds keep their own keys."""
        result = _call("interface", cif_path=str(complex_pdb), design_chain="B", receptor_chain="A")
        for key in ("delta_sasa", "delta_g_int", "hbonds", "saltbridges"):
            assert key in result

    def test_dockq_headline(self):
        from binding_metrics.metrics.dockq import _parse_dockq_json

        spec = get_metric("dockq")
        parsed = _parse_dockq_json({"GlobalDockQ": 0.81, "best_result": {}})
        assert _resolve(parsed, spec.headline_key) == pytest.approx(0.81)

    def test_receptor_quality_headline_is_in_the_summary(self):
        pytest.importorskip("biotite")
        from binding_metrics.metrics.receptor_quality import _aggregate_summary

        spec = get_metric("receptor_quality")
        model = {"model_index": 1, "molprobity_score": 1.7}
        summary = {"summary": _aggregate_summary([model])}
        assert _resolve(summary, spec.headline_key) == pytest.approx(1.7)

    def test_interface_pae_headline_is_the_mean_of_the_slice(self, complex_pdb, atoms, tmp_path):
        """A uniform 2.5 A error matrix gives a mean interface PAE of 2.5 A."""
        spec = get_metric("interface_pae")
        n_tokens = sum(
            int(np.unique(atoms.res_id[atoms.chain_id == c]).size)
            for c in np.unique(atoms.chain_id)
        )
        confidences = tmp_path / "sample_confidences.json"
        confidences.write_text(json.dumps({"pae": np.full((n_tokens, n_tokens), 2.5).tolist()}))
        result = _call(
            "interface_pae",
            confidences_path=str(confidences),
            structure_path=str(complex_pdb),
            binder_chain="B",
            receptor_chain="A",
        )
        assert _resolve(result, spec.headline_key) == pytest.approx(2.5)
        assert spec.unit == "angstrom"

    @pytest.mark.parametrize("name", ["ligand_rmsd", "interface_sasa", "receptor_drift"])
    def test_trajectory_dict_headline_is_a_key(self, trajectory_results, name):
        spec = get_metric(name)
        assert spec.headline_key in trajectory_results[name]

    @pytest.mark.parametrize("name", ["rmsd", "rmsf", "buried_sasa", "contacts"])
    def test_array_metrics_have_no_headline_key(self, trajectory_results, name):
        """No key: the returned array is the quantity direction and unit describe."""
        assert get_metric(name).headline_key is None
        assert isinstance(trajectory_results[name], np.ndarray)

    def test_trajectory_units_match_the_computed_magnitudes(self, trajectory_results):
        """MDTraj returns nm^2: ~14.6 nm^2 is the same 1466 A^2 the static metric gives."""
        buried_nm2 = float(trajectory_results["buried_sasa"][0])
        assert get_metric("buried_sasa").unit == "nm^2"
        assert buried_nm2 == pytest.approx(14.66, abs=1.0)
        # Two identical frames do not drift.
        assert trajectory_results["receptor_drift"]["drift_aligned_mean"] == pytest.approx(0.0)
