"""Contract tests for the metric registry.

Two things are pinned here. First, the fields that downstream code reads from
every ``MetricSpec`` (``name``, ``input_type``, ``chain_mode``, ``path_arg``,
``peptide_chain_arg``, ``receptor_chain_arg``, ``chain_arg`` and the rest of the
original fields) keep the values they had before the registry grew: a renamed
kwarg or a changed input type would break an adapter that calls
``spec.call(**kwargs)``. Second, the newly registered metrics can be driven the
way a generic runner drives them, from the spec fields alone.

Structures are derived from the bundled 1YCR example (MDM2 chain A, p53 peptide
chain B) with the waters removed, so the expected values do not depend on how
heteroatoms are handled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from binding_metrics.metrics.registry import METRICS, MetricSpec, get_metric

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_1YCR = REPO_ROOT / "data" / "example_linear_p53_1YCR.pdb"

# ---------------------------------------------------------------------------
# The entries that existed before the registry was extended, field by field.
# Changing any of these values is an interface change and must be deliberate.
# ---------------------------------------------------------------------------

_ORIGINAL_FIELDS = (
    "import_path",
    "input_type",
    "chain_mode",
    "formats",
    "path_arg",
    "secondary_path_arg",
    "chain_arg",
    "peptide_chain_arg",
    "receptor_chain_arg",
)

_ORIGINAL_ENTRIES: dict[str, tuple] = {
    "interface": (
        "binding_metrics.metrics.interface:compute_interface_metrics",
        "static_structure",
        "interface",
        ("pdb", "cif"),
        "cif_path",
        None,
        None,
        "design_chain",
        "receptor_chain",
    ),
    "coulomb": (
        "binding_metrics.metrics.electrostatics:compute_coulomb_cross_chain",
        "static_structure",
        "interface",
        ("pdb", "cif"),
        "cif_path",
        None,
        None,
        "peptide_chain",
        "receptor_chain",
    ),
    "ramachandran": (
        "binding_metrics.metrics.geometry:compute_ramachandran",
        "static_structure",
        "single",
        ("pdb", "cif"),
        "cif_path",
        None,
        "chain",
        None,
        None,
    ),
    "omega": (
        "binding_metrics.metrics.geometry:compute_omega_planarity",
        "static_structure",
        "single",
        ("pdb", "cif"),
        "cif_path",
        None,
        "chain",
        None,
        None,
    ),
    "shape_complementarity": (
        "binding_metrics.metrics.geometry:compute_shape_complementarity",
        "static_structure",
        "interface",
        ("pdb", "cif"),
        "cif_path",
        None,
        None,
        "peptide_chain",
        "receptor_chain",
    ),
    "void_volume": (
        "binding_metrics.metrics.geometry:compute_buried_void_volume",
        "static_structure",
        "interface",
        ("pdb", "cif"),
        "cif_path",
        None,
        None,
        "peptide_chain",
        "receptor_chain",
    ),
    "structure_rmsd": (
        "binding_metrics.metrics.comparison:compute_structure_rmsd",
        "static_structure",
        "interface_2paths",
        ("pdb", "cif"),
        "initial_path",
        "processed_path",
        None,
        "design_chain",
        None,
    ),
    "dockq": (
        "binding_metrics.metrics.dockq:compute_dockq_metrics",
        "static_structure",
        "interface_2paths",
        ("pdb", "cif"),
        "model_path",
        "reference_path",
        None,
        None,
        None,
    ),
    "interaction_energy": (
        "binding_metrics.metrics.energy:calculate_interaction_energy",
        "trajectory",
        "interface",
        ("pdb",),
        "trajectory_path",
        None,
        None,
        "ligand_indices",
        "receptor_indices",
    ),
    "component_energies": (
        "binding_metrics.metrics.energy:calculate_component_energies",
        "trajectory",
        "interface",
        ("pdb",),
        "trajectory_path",
        None,
        None,
        "ligand_indices",
        "receptor_indices",
    ),
    "rmsd": (
        "binding_metrics.metrics.rmsd:calculate_rmsd",
        "trajectory",
        "none",
        ("pdb", "cif"),
        "trajectory_path",
        None,
        None,
        None,
        None,
    ),
    "rmsf": (
        "binding_metrics.metrics.rmsd:calculate_rmsf",
        "trajectory",
        "none",
        ("pdb", "cif"),
        "trajectory_path",
        None,
        None,
        None,
        None,
    ),
    "ligand_rmsd": (
        "binding_metrics.metrics.rmsd:calculate_ligand_rmsd",
        "trajectory",
        "interface",
        ("pdb", "cif"),
        "trajectory_path",
        None,
        None,
        "ligand_indices",
        "receptor_indices",
    ),
    "receptor_drift": (
        "binding_metrics.metrics.rmsd:compute_receptor_drift",
        "trajectory",
        "single",
        ("pdb", "cif"),
        "trajectory_path",
        None,
        "receptor_chain",
        None,
        None,
    ),
    "buried_sasa": (
        "binding_metrics.metrics.sasa:calculate_buried_sasa",
        "trajectory",
        "interface",
        ("pdb", "cif"),
        "trajectory_path",
        None,
        None,
        "ligand_indices",
        "receptor_indices",
    ),
    "contacts": (
        "binding_metrics.metrics.contacts:calculate_contacts",
        "trajectory",
        "interface",
        ("pdb", "cif"),
        "trajectory_path",
        None,
        None,
        "ligand_indices",
        "receptor_indices",
    ),
    "md_implicit": (
        "binding_metrics.protocols.relaxation:run_implicit_relaxation",
        "md_simulation",
        "none",
        ("pdb", "cif"),
        "input_path",
        None,
        None,
        None,
        None,
    ),
    "openfold": (
        "binding_metrics.metrics.openfold:compute_openfold_metrics",
        "openfold_json",
        "none",
        (),
        "output_dir",
        None,
        None,
        None,
        None,
    ),
}


class TestOriginalEntriesUnchanged:
    def test_dataclass_keeps_the_original_field_names(self):
        import dataclasses

        names = [f.name for f in dataclasses.fields(MetricSpec)]
        for original in ("name", "description", *_ORIGINAL_FIELDS):
            assert original in names, f"MetricSpec lost the field {original!r}"

    @pytest.mark.parametrize("name", sorted(_ORIGINAL_ENTRIES))
    def test_entry_fields_unchanged(self, name):
        spec = get_metric(name)
        actual = tuple(getattr(spec, f) for f in _ORIGINAL_FIELDS)
        assert actual == _ORIGINAL_ENTRIES[name], f"{name}: contract fields changed"

    def test_no_original_entry_was_dropped(self):
        assert set(_ORIGINAL_ENTRIES) <= {m.name for m in METRICS}

    def test_original_entries_keep_their_order(self):
        """Iteration order of METRICS is visible to runners that print a table."""
        names = [m.name for m in METRICS if m.name in _ORIGINAL_ENTRIES]
        assert names == list(_ORIGINAL_ENTRIES)


# ---------------------------------------------------------------------------
# Newly registered metrics, driven from the spec fields only
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def complex_pdb(tmp_path_factory) -> Path:
    """1YCR with heteroatoms removed: chain A = MDM2 (receptor), chain B = p53 peptide."""
    if not EXAMPLE_1YCR.exists():
        pytest.skip(f"bundled example not found: {EXAMPLE_1YCR}")
    lines = [ln for ln in EXAMPLE_1YCR.read_text().splitlines(True) if ln.startswith("ATOM")]
    out = tmp_path_factory.mktemp("l9") / "1ycr_atoms.pdb"
    out.write_text("".join(lines) + "END\n")
    return out


def _kwargs_from_spec(spec: MetricSpec, primary, peptide: str, receptor: str) -> dict:
    """Build call kwargs the way a generic runner does: from the spec fields only."""
    kwargs = {spec.path_arg: primary}
    if spec.secondary_path_arg:
        kwargs[spec.secondary_path_arg] = primary
    if spec.chain_mode == "single" and spec.chain_arg:
        kwargs[spec.chain_arg] = peptide
    elif spec.chain_mode in ("interface", "interface_2paths"):
        if spec.peptide_chain_arg:
            kwargs[spec.peptide_chain_arg] = peptide
        if spec.receptor_chain_arg:
            kwargs[spec.receptor_chain_arg] = receptor
    return kwargs


def _call(name: str, **kwargs):
    try:
        return get_metric(name).call(**kwargs)
    except ImportError as e:
        pytest.skip(f"{name}: optional dependency not installed — {e}")


class TestNewMetricsCallThroughTheSpec:
    def test_delta_sasa_static_buries_surface_on_binding(self, complex_pdb):
        pytest.importorskip("biotite")
        spec = get_metric("delta_sasa_static")
        result = _call(spec.name, **_kwargs_from_spec(spec, complex_pdb, "B", "A"))
        # A 13-residue peptide in the MDM2 cleft buries well over 1000 A^2.
        assert result["delta_sasa"] > 1000.0
        assert result["delta_sasa"] == pytest.approx(
            result["sasa_peptide"] + result["sasa_receptor"] - result["sasa_complex"], rel=1e-4
        )

    def test_evobind_adversarial_of_a_structure_with_itself_has_no_displacement(self, complex_pdb):
        pytest.importorskip("biotite")
        spec = get_metric("evobind_adversarial")
        result = _call(spec.name, **_kwargs_from_spec(spec, complex_pdb, "B", "A"))
        assert result["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-3)
        assert result["afm_mean_if_dist"] > 0.0

    def test_evobind_score_needs_only_the_path_and_the_confidence_array(self, complex_pdb):
        pytest.importorskip("biotite")
        spec = get_metric("evobind_score")
        assert spec.input_type == "predicted_structure"
        kwargs = _kwargs_from_spec(spec, complex_pdb, "B", "A")
        result = _call(spec.name, plddt_per_atom=None, **kwargs)
        # Without pLDDT only the distance terms exist; the peptide sits in the cleft.
        assert 0.0 < result["if_dist_pep_to_rec"] < 8.0
        assert result["evobind_score"] is None

    @pytest.mark.parametrize(
        "name, keys",
        [
            ("hbonds", {"hbonds", "hbond_energy"}),
            ("saltbridges", {"saltbridges", "saltbridges_bidentate", "saltbridge_energy"}),
        ],
    )
    def test_polar_contacts_take_an_atom_array(self, complex_pdb, name, keys):
        pytest.importorskip("biotite")
        import biotite.structure.io.pdb as pdb_io

        spec = get_metric(name)
        assert spec.input_type == "atom_array"
        atoms = pdb_io.get_structure(pdb_io.PDBFile.read(str(complex_pdb)), model=1)
        result = _call(name, **_kwargs_from_spec(spec, atoms, "B", "A"))
        assert keys <= set(result)
        energy_key = "hbond_energy" if name == "hbonds" else "saltbridge_energy"
        # Attractive interactions only: energies are <= 0 by construction.
        assert result[energy_key] <= 0.0
        assert result[next(k for k in keys if k in ("hbonds", "saltbridges"))] >= 1


# ---------------------------------------------------------------------------
# Mandatory fields and call semantics
# ---------------------------------------------------------------------------


class TestMandatoryFields:
    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_required_string_fields_are_nonempty(self, spec):
        for field in ("name", "import_path", "description", "input_type", "chain_mode", "path_arg"):
            value = getattr(spec, field)
            assert isinstance(value, str) and value.strip(), f"{spec.name}: empty {field}"

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_optional_arg_names_are_none_or_identifiers(self, spec):
        for field in (
            "secondary_path_arg",
            "chain_arg",
            "peptide_chain_arg",
            "receptor_chain_arg",
        ):
            value = getattr(spec, field)
            assert value is None or value.isidentifier(), f"{spec.name}: {field}={value!r}"

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_formats_is_a_tuple_of_lowercase_strings(self, spec):
        assert isinstance(spec.formats, tuple)
        assert all(f == f.lower() for f in spec.formats)

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_kwarg_names_are_distinct(self, spec):
        """One kwarg cannot receive two roles."""
        names = [
            n
            for n in (
                spec.path_arg,
                spec.secondary_path_arg,
                spec.chain_arg,
                spec.peptide_chain_arg,
                spec.receptor_chain_arg,
            )
            if n
        ]
        assert len(names) == len(set(names)), f"{spec.name}: duplicated kwarg names {names}"

    def test_spec_is_frozen(self):
        spec = get_metric("interface")
        with pytest.raises(AttributeError):
            spec.name = "renamed"  # type: ignore[misc]


class TestCallSemantics:
    """``spec.call(**kwargs)`` forwards every keyword to the loaded function, unchanged."""

    def test_call_forwards_keywords_and_returns_the_result(self):
        spec = MetricSpec(
            name="probe",
            import_path="builtins:dict",
            description="d",
            input_type="static_structure",
        )
        assert spec.call(a=1, b=[2]) == {"a": 1, "b": [2]}

    def test_call_passes_no_positional_arguments(self):
        spec = MetricSpec(
            name="probe",
            import_path="builtins:dict",
            description="d",
            input_type="static_structure",
        )
        assert spec.call() == {}

    def test_load_returns_the_function_object(self):
        import json

        spec = MetricSpec(
            name="probe", import_path="json:dumps", description="d", input_type="static_structure"
        )
        assert spec.load() is json.dumps

    def test_construction_does_not_import_the_target(self):
        """Loading is lazy: a spec for a missing module is fine until it is loaded."""
        spec = MetricSpec(
            name="ghost",
            import_path="binding_metrics_no_such_module:fn",
            description="d",
            input_type="static_structure",
        )
        with pytest.raises(ImportError):
            spec.load()

    def test_call_propagates_errors_of_the_metric(self):
        spec = MetricSpec(
            name="probe", import_path="json:dumps", description="d", input_type="static_structure"
        )
        with pytest.raises(TypeError):
            spec.call(not_a_parameter=1)

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_declared_kwargs_bind_to_the_function_signature(self, spec):
        """The kwargs a runner builds from the spec fields bind without a TypeError."""
        import inspect

        try:
            fn = spec.load()
        except ImportError as e:
            pytest.skip(f"{spec.name}: optional dependency not installed — {e}")
        if inspect.isclass(fn):
            pytest.skip(f"{spec.name}: class")
        kwargs = {
            n: object()
            for n in (
                spec.path_arg,
                spec.secondary_path_arg,
                spec.chain_arg,
                spec.peptide_chain_arg,
                spec.receptor_chain_arg,
            )
            if n
        }
        # bind_partial: required parameters the spec does not declare (a pLDDT array,
        # a query name) are the caller's business; unknown names are the registry's bug.
        inspect.signature(fn).bind_partial(**kwargs)
