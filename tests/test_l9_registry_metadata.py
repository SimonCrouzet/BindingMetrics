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
ANGSTROM2_PER_NM2 = 100.0


def _resolve(result, dotted_key: str):
    """Follow a dotted headline key (``"summary.molprobity_score"``) into a result dict."""
    value = result
    for part in dotted_key.split("."):
        value = value[part]
    return value


def _defined_extras() -> set[str] | None:
    """The extras that ``pyproject.toml`` defines; None when the file is not next to the tests."""
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return None
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return set(data["project"]["optional-dependencies"])


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
    def test_the_metadata_values_are_well_formed(self, spec):
        """One case per metric; a failure lists every field that is wrong.

        ``direction`` and ``unit`` are unset or from the documented sets, every metric
        declares a ``cost_class``, ``requires_extras`` is a tuple of distinct non-empty
        names that ``pyproject.toml`` defines as extras, and ``requires_gpu`` is a bool.
        """
        problems = []
        if spec.direction is not None and spec.direction not in DIRECTIONS:
            problems.append(f"direction {spec.direction!r} is not in DIRECTIONS")
        if spec.unit is not None and spec.unit not in KNOWN_UNITS:
            problems.append(f"unit {spec.unit!r} is not in KNOWN_UNITS")
        if spec.cost_class not in COST_CLASSES:
            problems.append(f"cost_class {spec.cost_class!r} is not in COST_CLASSES")
        extras = spec.requires_extras
        if not isinstance(extras, tuple):
            problems.append(f"requires_extras has type {type(extras).__name__}, not tuple")
        else:
            if not all(isinstance(e, str) and e for e in extras):
                problems.append(f"requires_extras holds a non-name: {extras!r}")
            if len(set(extras)) != len(extras):
                problems.append(f"requires_extras repeats a name: {extras!r}")
            defined = _defined_extras()
            unknown = [e for e in extras if defined is not None and e not in defined]
            if unknown:
                problems.append(f"not in [project.optional-dependencies]: {unknown}")
        if not isinstance(spec.requires_gpu, bool):
            problems.append(f"requires_gpu has type {type(spec.requires_gpu).__name__}, not bool")
        assert not problems, f"{spec.name}: " + "; ".join(problems)

    def test_allowed_sets_are_ascii(self):
        for value in (*DIRECTIONS, *COST_CLASSES, *KNOWN_UNITS):
            assert value.isascii(), value


class TestMetadataConsistency:
    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_the_fields_agree_with_each_other(self, spec):
        """One case per metric; a failure lists every disagreement.

        * A ``headline_key`` comes with a ``direction`` or a ``unit``.
        * A metric that runs on a CUDA device by default builds a force-field system, so its
          ``cost_class`` is ``structural`` or ``md``.
        * Trajectory and simulation inputs cost ``md``.
        * ``chain_mode`` matches the chain arguments the spec declares.
        * A ``secondary_path_arg`` exists only for the two-path mode.
        """
        problems = []
        if spec.headline_key is not None:
            if not spec.headline_key.strip():
                problems.append("headline_key is blank")
            if not (spec.direction or spec.unit):
                problems.append("headline_key without direction or unit")
        if spec.requires_gpu and spec.cost_class not in ("structural", "md"):
            problems.append(f"requires_gpu with cost_class {spec.cost_class!r}")
        if spec.input_type in ("trajectory", "md_simulation") and spec.cost_class != "md":
            problems.append(f"{spec.input_type} input with cost_class {spec.cost_class!r}")

        chain_args = [spec.chain_arg, spec.peptide_chain_arg, spec.receptor_chain_arg]
        if spec.chain_mode == "none":
            consistent = not any(chain_args)
        elif spec.chain_mode == "single":
            consistent = bool(spec.chain_arg) and not (
                spec.peptide_chain_arg or spec.receptor_chain_arg
            )
        elif spec.chain_mode == "interface":
            consistent = bool(spec.peptide_chain_arg or spec.receptor_chain_arg) and not (
                spec.chain_arg
            )
        else:
            consistent = spec.chain_mode == "interface_2paths" and not spec.chain_arg
        if not consistent:
            problems.append(f"chain_mode {spec.chain_mode!r} does not match {chain_args}")

        if bool(spec.secondary_path_arg) != (spec.chain_mode == "interface_2paths"):
            problems.append(
                f"secondary_path_arg {spec.secondary_path_arg!r} with chain_mode "
                f"{spec.chain_mode!r}"
            )
        assert not problems, f"{spec.name}: " + "; ".join(problems)

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
    lines = [
        ln
        for ln in EXAMPLE_1YCR.read_text(encoding="utf-8").splitlines(True)
        if ln.startswith("ATOM")
    ]
    out = tmp_path_factory.mktemp("l9_meta") / "complex.pdb"
    out.write_text("".join(lines) + "END\n", encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def two_frame_pdb(complex_pdb) -> Path:
    """The complex written as two identical models: a trajectory and its topology in one file."""
    lines = complex_pdb.read_text(encoding="utf-8").splitlines(True)
    atoms = [ln for ln in lines if ln.startswith("ATOM")]
    text = "".join(f"MODEL     {i:>4}\n" + "".join(atoms) + "ENDMDL\n" for i in (1, 2))
    out = complex_pdb.parent / "two_frames.pdb"
    out.write_text(text + "END\n", encoding="utf-8")
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
        confidences.write_text(
            json.dumps({"pae": np.full((n_tokens, n_tokens), 2.5).tolist()}), encoding="utf-8"
        )
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

    def test_trajectory_units_match_the_computed_magnitudes(
        self, trajectory_results, static_results
    ):
        """MDTraj returns nm^2: the buried area is that of the static metric (A^2) over 100."""
        buried_nm2 = float(trajectory_results["buried_sasa"][0])
        assert get_metric("buried_sasa").unit == "nm^2"
        static_angstrom2 = static_results["delta_sasa_static"]["delta_sasa"]
        # the two engines use different radii tables: 1 nm^2 is 100 A^2 and the tolerance only
        # has to stop a missing or doubled unit factor
        assert buried_nm2 == pytest.approx(static_angstrom2 / ANGSTROM2_PER_NM2, rel=0.1)
        # Two identical frames do not drift.
        assert trajectory_results["receptor_drift"]["drift_aligned_mean"] == pytest.approx(0.0)
