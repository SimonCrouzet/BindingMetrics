"""Golden tests of the OpenFold3 parse path: the result dict, its CSV row and its report text.

They pin what ``compute_openfold_metrics`` returns for one hand-built run directory, and
how ``report._flatten``, ``report._md_openfold`` and ``binding-metrics-openfold parse``
present it. Every number below follows from the fixture (a 7-token PAE and PDE built from
a linear formula, a 14-atom structure), so each can be checked by hand:

    pde[i, j] = 0.5 + 0.25 i + 0.125 j        pae[i, j] = 1 + 0.5 i + 0.25 j
    tokens: chain A = 0..3, chain B (the binder) = 4..6
    interface PDE, rows B, columns A:   mean 0.5 + 0.25 * 5 + 0.125 * 1.5 = 1.9375, max 2.375
    interface PAE, B x A block mean 3.875, A x B block mean 3.0; their average is 3.4375,
    the larger of the two block maxima 4.75

The tests must pass unchanged after every refactor of the OpenFold3 path (moving helpers,
the adapter layer, ``summarize_prediction``). A failure here means the OpenFold3 output
changed, not that the test is out of date. Two things are left out on purpose because a
fix is expected to change them: the array-valued CSV columns (see the note in
``_flatten_row``) and the file layouts OpenFold3 0.5.0 may add.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests.test_openfold import _default_agg, _make_seed_dir

struc = pytest.importorskip("biotite.structure")
pdbx = pytest.importorskip("biotite.structure.io.pdbx")

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

_QUERY = "gold"
_N_ATOMS = 14  # chain A: 4 residues x (CA, CB); chain B: 3 residues x (CA, CB)
_PLDDT = np.array([92, 94, 90, 88, 86, 84, 80, 78, 60, 64, 88, 90, 75, 79], dtype=float)
_TIMING = {"inference": 45.2, "msa": 12.3}
_I = np.arange(7)[:, None]
_J = np.arange(7)[None, :]
_PDE = 0.5 + 0.25 * _I + 0.125 * _J
_PAE = 1.0 + 0.5 * _I + 0.25 * _J

# The order of the keys is part of the contract: ``report._flatten`` writes the CSV
# columns in this order. ``seed_value`` is the one key added since the capture of this list: a
# deliberate, additive change (the seed behind the sample, from the name of the seed directory),
# appended after ``timing`` so that no older key moves; ``reason`` still comes last.
_KEYS = [
    "query_name",
    "seed",
    "sample",
    "structure_path",
    "avg_plddt",
    "gpde",
    "ptm",
    "iptm",
    "disorder",
    "has_clash",
    "sample_ranking_score",
    "chain_ptm",
    "chain_pair_iptm",
    "bespoke_iptm",
    "plddt_per_atom",
    "n_atoms",
    "pde",
    "max_pde",
    "pae",
    "max_pae",
    "binder_plddt_per_residue",
    "binder_avg_plddt",
    "mean_interface_pde",
    "max_interface_pde",
    "pde_interface",
    "mean_interface_pae",
    "max_interface_pae",
    "pae_interface",
    "binder_ca_rmsd",
    "timing",
    "seed_value",
]


def _atoms(chains):
    """Atoms of ``(chain_id, n_residues, y_offset)`` chains: CA and CB per ALA residue."""
    atoms = []
    for chain_id, n_res, y in chains:
        for i in range(n_res):
            for name, dy in (("CA", 0.0), ("CB", 1.5)):
                atoms.append(
                    struc.Atom(
                        [3.8 * i, y + dy, 0.0],
                        chain_id=chain_id,
                        res_id=i + 1,
                        res_name="ALA",
                        atom_name=name,
                        element="C",
                    )
                )
    return struc.array(atoms)


def _write_cif(atoms, path):
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atoms)
    cif.write(str(path))


def _dimer():
    return _atoms([("A", 4, 0.0), ("B", 3, 6.0)])


def _write_run(tmp_path, *, plddt=_PLDDT, pde=_PDE, pae=_PAE, agg=True, atoms=None, seed=1):
    """One run directory: aggregated and full confidences, timing and a real model CIF."""
    root = _make_seed_dir(
        tmp_path,
        _QUERY,
        seed=seed,
        agg=_default_agg(n_chains=2) if agg else None,
        conf={"plddt": plddt, "gpde": 1.23, "pde": pde, "pae": pae},
        timing=_TIMING,
    )
    model = root / _QUERY / f"seed_{seed}" / f"{_QUERY}_seed_{seed}_sample_1_model.cif"
    _write_cif(_dimer() if atoms is None else atoms, model)
    return root


def _write_reference(tmp_path, binder_shift_z=1.0):
    """The dimer with the binder moved along z, so the binder CA RMSD equals the shift."""
    reference = _dimer()
    reference.coord[reference.chain_id == "B", 2] += binder_shift_z
    path = tmp_path / "reference.cif"
    _write_cif(reference, path)
    return path


def _full(tmp_path, **kwargs):
    from binding_metrics.metrics.openfold import compute_openfold_metrics

    root = _write_run(tmp_path / "run")
    reference = _write_reference(tmp_path)
    return root, compute_openfold_metrics(
        root,
        _QUERY,
        binder_chain="B",
        receptor_chain="A",
        reference_structure_path=reference,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The result dict
# ---------------------------------------------------------------------------


class TestResultDict:
    def test_keys_and_their_order(self, tmp_path):
        _, metrics = _full(tmp_path)
        assert list(metrics) == _KEYS  # a complete run has no "reason"

    def test_identity_and_scalar_values(self, tmp_path):
        root, metrics = _full(tmp_path)
        assert metrics["query_name"] == _QUERY
        assert metrics["seed"] == 1
        assert metrics["seed_value"] == 1  # the directory is seed_1
        assert metrics["sample"] == 1
        assert metrics["structure_path"] == str(
            root / _QUERY / "seed_1" / "gold_seed_1_sample_1_model.cif"
        )
        expected = {
            "avg_plddt": 87.5,
            "gpde": 1.23,
            "ptm": 0.88,
            "iptm": 0.76,
            "disorder": 0.12,
            "has_clash": 0.0,
            "sample_ranking_score": 0.82,
            "max_pde": 2.75,
            "max_pae": 5.5,
            "binder_avg_plddt": 76.0,
            "mean_interface_pde": 1.9375,
            "max_interface_pde": 2.375,
            "mean_interface_pae": 3.4375,
            "max_interface_pae": 4.75,
            "binder_ca_rmsd": 1.0,
        }
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value, abs=1e-9), key
            assert type(metrics[key]) is float, key  # Python floats, not numpy scalars
        assert metrics["n_atoms"] == _N_ATOMS
        assert type(metrics["n_atoms"]) is int

    def test_dictionary_values(self, tmp_path):
        _, metrics = _full(tmp_path)
        # keys are what OpenFold3 writes: chain IDs and the string "(A, B)", not tuples
        assert metrics["chain_ptm"] == {"1": 0.88, "2": 0.80}
        assert metrics["chain_pair_iptm"] == {"(1, 2)": 0.76}
        assert metrics["bespoke_iptm"] == {"(1, 2)": 0.74}
        assert metrics["timing"] == _TIMING

    def test_arrays_with_matrices_requested(self, tmp_path):
        _, metrics = _full(tmp_path, include_matrices=True)
        np.testing.assert_allclose(metrics["plddt_per_atom"], _PLDDT)
        np.testing.assert_allclose(metrics["pde"], _PDE)
        np.testing.assert_allclose(metrics["pae"], _PAE)
        # residue means of the binder atoms (60, 64), (88, 90), (75, 79)
        np.testing.assert_allclose(metrics["binder_plddt_per_residue"], [62.0, 89.0, 77.0])
        # rows = binder tokens 4..6, columns = receptor tokens 0..3
        np.testing.assert_allclose(metrics["pde_interface"], _PDE[4:7, 0:4])
        np.testing.assert_allclose(metrics["pae_interface"], _PAE[4:7, 0:4])
        assert metrics["pde_interface"].shape == (3, 4)
        for key in ("plddt_per_atom", "pde", "pae", "binder_plddt_per_residue"):
            assert metrics[key].dtype == np.float64, key

    def test_matrices_are_withheld_by_default_but_their_statistics_are_not(self, tmp_path):
        _, metrics = _full(tmp_path)
        assert list(metrics) == _KEYS
        assert metrics["pde"] is None
        assert metrics["pae"] is None
        assert metrics["pde_interface"] is None
        assert metrics["pae_interface"] is None
        assert metrics["plddt_per_atom"].shape == (_N_ATOMS,)  # per-atom pLDDT is always kept
        assert metrics["max_pde"] == pytest.approx(2.75)
        assert metrics["mean_interface_pae"] == pytest.approx(3.4375)

    def test_confidence_values_without_chains(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_run(tmp_path)
        metrics = compute_openfold_metrics(root, _QUERY)
        assert list(metrics) == _KEYS
        assert metrics["avg_plddt"] == pytest.approx(87.5)
        assert metrics["max_pae"] == pytest.approx(5.5)
        assert metrics["binder_plddt_per_residue"] is None
        for key in (
            "binder_avg_plddt",
            "mean_interface_pde",
            "max_interface_pde",
            "mean_interface_pae",
            "max_interface_pae",
            "binder_ca_rmsd",
        ):
            assert np.isnan(metrics[key]), key
        assert "reason" not in metrics

    def test_avg_plddt_falls_back_to_the_per_atom_mean_but_gpde_does_not(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_run(tmp_path, agg=False)
        metrics = compute_openfold_metrics(root, _QUERY)
        assert metrics["avg_plddt"] == pytest.approx(float(_PLDDT.mean()))
        assert np.isnan(metrics["gpde"])  # the "gpde" of the confidences file is not read
        assert np.isnan(metrics["ptm"])
        assert metrics["reason"] == "aggregated confidences file not found"

    def test_npz_confidences_give_the_same_values_as_json(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_run(tmp_path / "json")
        from_json = compute_openfold_metrics(root, _QUERY, binder_chain="B", receptor_chain="A")

        npz_root = _write_run(tmp_path / "npz")
        seed_dir = npz_root / _QUERY / "seed_1"
        (seed_dir / "gold_seed_1_sample_1_confidences.json").unlink()
        np.savez(
            seed_dir / "gold_seed_1_sample_1_confidences.npz",
            plddt=_PLDDT,
            gpde=np.float64(1.23),
            pde=_PDE,
            pae=_PAE,
        )
        from_npz = compute_openfold_metrics(npz_root, _QUERY, binder_chain="B", receptor_chain="A")

        assert list(from_npz) == _KEYS
        for key in _KEYS:
            if key in ("structure_path", "pde", "pae"):
                continue
            np.testing.assert_equal(from_npz[key], from_json[key], err_msg=key)

    def test_the_result_can_be_written_as_json(self, tmp_path):
        from binding_metrics.protocols.report import write_report

        _, metrics = _full(tmp_path, include_matrices=True)
        out = write_report({"sample_id": "s", "openfold": metrics}, tmp_path / "rep", "s")
        written = json.loads(out.read_text(encoding="utf-8"))["openfold"]
        assert list(written) == _KEYS
        assert written["plddt_per_atom"] == _PLDDT.tolist()  # arrays become lists
        assert written["pae_interface"] == _PAE[4:7, 0:4].tolist()
        assert written["chain_pair_iptm"] == {"(1, 2)": 0.76}


# ---------------------------------------------------------------------------
# Missing and mismatching inputs: sentinels and "reason"
# ---------------------------------------------------------------------------


class TestReasonsAndSentinels:
    def test_missing_run_directory(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        missing = tmp_path / "nowhere"
        metrics = compute_openfold_metrics(missing, _QUERY)
        assert list(metrics) == [*_KEYS, "reason"]
        assert metrics["reason"] == (
            f"no confidence files found for query '{_QUERY}' (seed index 1, sample 1) in {missing}"
        )
        assert metrics["structure_path"] is None
        assert metrics["chain_ptm"] == {}
        assert metrics["timing"] == {}
        assert metrics["n_atoms"] == 0
        for key in ("plddt_per_atom", "pde", "pae", "binder_plddt_per_residue"):
            assert metrics[key] is None, key
        for key in ("avg_plddt", "gpde", "ptm", "iptm", "max_pde", "max_pae", "binder_ca_rmsd"):
            assert np.isnan(metrics[key]), key

    def test_missing_structure_with_a_binder_chain(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_run(tmp_path)
        (root / _QUERY / "seed_1" / "gold_seed_1_sample_1_model.cif").unlink()
        metrics = compute_openfold_metrics(root, _QUERY, binder_chain="B", receptor_chain="A")
        assert metrics["structure_path"] is None
        assert metrics["reason"] == "structure file not found; per-chain values not computed"
        assert metrics["avg_plddt"] == pytest.approx(87.5)

    def test_missing_matrices(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_run(tmp_path)
        seed_dir = root / _QUERY / "seed_1"
        conf_path = seed_dir / "gold_seed_1_sample_1_confidences.json"
        conf_path.write_text(json.dumps({"plddt": _PLDDT.tolist(), "gpde": 1.23}), encoding="utf-8")
        metrics = compute_openfold_metrics(root, _QUERY, binder_chain="B", receptor_chain="A")
        assert metrics["reason"] == (
            "interface PDE: no PDE matrix in the confidences file; "
            "interface PAE: no PAE matrix in the confidences file"
        )
        assert np.isnan(metrics["max_pde"])
        assert metrics["binder_avg_plddt"] == pytest.approx(76.0)

    def test_ligand_tokens_make_the_interface_values_nan_with_a_reason(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        # 4 + 3 residue tokens and a 5-atom ligand that AlphaFold3-style models tokenise per atom
        ligand = struc.array(
            [
                struc.Atom(
                    [3.8 * k, 20.0, 0.0],
                    chain_id="L",
                    res_id=1,
                    res_name="LIG",
                    atom_name=f"C{k + 1}",
                    element="C",
                    hetero=True,
                )
                for k in range(5)
            ]
        )
        atoms = struc.concatenate([_dimer(), ligand])
        n_tokens = 12
        root = _write_run(
            tmp_path,
            plddt=np.full(atoms.array_length(), 90.0),
            pde=np.full((n_tokens, n_tokens), 2.0),
            pae=np.full((n_tokens, n_tokens), 3.0),
            atoms=atoms,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = compute_openfold_metrics(root, _QUERY, binder_chain="B", receptor_chain="A")

        tail = (
            "(A: 4, B: 3, L: 1). Residue-based chain offsets do not apply, probably because "
            "a ligand, ion or modified residue is tokenised per atom."
        )
        assert metrics["reason"] == (
            f"interface PDE: PDE matrix has 12 tokens but the structure has 8 residues {tail}; "
            f"interface PAE: PAE matrix has 12 tokens but the structure has 8 residues {tail}"
        )
        assert np.isnan(metrics["mean_interface_pde"])
        assert np.isnan(metrics["mean_interface_pae"])
        assert metrics["pde_interface"] is None
        assert metrics["binder_avg_plddt"] == pytest.approx(90.0)  # independent of the tokens
        assert metrics["max_pae"] == pytest.approx(3.0)
        messages = [str(w.message) for w in caught]
        assert len(messages) == 2
        assert messages[0].startswith("compute_openfold_metrics: interface PDE skipped: ")
        assert messages[1].startswith("compute_openfold_metrics: interface PAE skipped: ")

    def test_plddt_that_does_not_fit_the_structure(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_run(tmp_path, plddt=np.full(10, 90.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = compute_openfold_metrics(root, _QUERY, binder_chain="B", receptor_chain="A")
        assert metrics["reason"] == (
            "binder pLDDT: plddt_per_atom length (10) != atom count in structure (14). "
            "The pLDDT array and structure file must be from the same prediction."
        )
        assert metrics["binder_plddt_per_residue"] is None
        assert np.isnan(metrics["binder_avg_plddt"])
        assert metrics["mean_interface_pde"] == pytest.approx(1.9375)  # blocks do not need pLDDT
        assert [str(w.message) for w in caught] == [
            "compute_openfold_metrics: per-residue binder pLDDT skipped: plddt_per_atom length "
            "(10) != atom count in structure (14). The pLDDT array and structure file must be "
            "from the same prediction."
        ]

    def test_reference_with_a_different_binder_length(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        atoms = _dimer()
        short = atoms[(atoms.chain_id == "A") | ((atoms.chain_id == "B") & (atoms.res_id < 3))]
        reference = tmp_path / "short_reference.cif"
        _write_cif(short, reference)
        root = _write_run(tmp_path / "run")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = compute_openfold_metrics(
                root,
                _QUERY,
                binder_chain="B",
                receptor_chain="A",
                reference_structure_path=reference,
            )
        assert metrics["reason"] == (
            "binder RMSD: Binder Cα count mismatch (chain 'B'): predicted 3, reference 2. "
            "Structures may have different sequence lengths."
        )
        assert np.isnan(metrics["binder_ca_rmsd"])
        assert len(caught) == 1

    def test_unreadable_reference_file(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_run(tmp_path)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            metrics = compute_openfold_metrics(
                root,
                _QUERY,
                binder_chain="B",
                reference_structure_path=tmp_path / "absent.cif",
            )
        assert metrics["reason"].startswith("binder RMSD: ")
        assert np.isnan(metrics["binder_ca_rmsd"])

    def test_a_structure_that_cannot_be_parsed_keeps_the_scalar_values(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_run(tmp_path)
        (root / _QUERY / "seed_1" / "gold_seed_1_sample_1_model.cif").write_text(
            "# stub CIF\n", encoding="utf-8"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = compute_openfold_metrics(root, _QUERY, binder_chain="B", receptor_chain="A")
        assert metrics["reason"].startswith("structural analysis failed: ")
        assert metrics["avg_plddt"] == pytest.approx(87.5)
        assert len(caught) == 1
        assert str(caught[0].message).startswith(
            "compute_openfold_metrics: structural analysis failed: "
        )


class TestWarningsNameTheCaller:
    """``stacklevel=2``: the warning points at the code that called the metric function."""

    def _caller_of(self, caught):
        return {Path(w.filename).name for w in caught}

    def test_every_warning_of_compute_openfold_metrics_names_the_calling_file(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        this_file = Path(__file__).name
        bad_plddt = _write_run(tmp_path / "a", plddt=np.full(10, 90.0))
        bad_matrix = _write_run(tmp_path / "b", pde=np.zeros((9, 9)), pae=np.zeros((9, 9)))
        cases = [
            dict(output_dir=bad_plddt),
            dict(output_dir=bad_matrix),
            dict(output_dir=bad_plddt, reference_structure_path=tmp_path / "absent.cif"),
        ]
        seen = set()
        for kwargs in cases:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compute_openfold_metrics(
                    query_name=_QUERY, binder_chain="B", receptor_chain="A", **kwargs
                )
            assert caught, kwargs
            seen |= self._caller_of(caught)
        assert seen == {this_file}

        stub_root = _write_run(tmp_path / "c")
        (stub_root / _QUERY / "seed_1" / "gold_seed_1_sample_1_model.cif").write_text(
            "# stub CIF\n", encoding="utf-8"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_openfold_metrics(stub_root, _QUERY, binder_chain="B", receptor_chain="A")
        assert self._caller_of(caught) == {this_file}
        assert os.path.samefile(caught[0].filename, __file__)


# ---------------------------------------------------------------------------
# Seed index and sample
# ---------------------------------------------------------------------------


class TestSeedIndexAndSample:
    def _two_seed_dirs(self, tmp_path):
        # OpenFold3 names the directories after its own seeds; sorted by name, 42 < 777
        _make_seed_dir(tmp_path, "q", seed=777, agg={"avg_plddt": 77.7})
        _make_seed_dir(tmp_path, "q", seed=42, agg={"avg_plddt": 42.4})
        _make_seed_dir(tmp_path, "q", seed=42, sample=2, agg={"avg_plddt": 42.5})
        return tmp_path

    def test_seed_is_a_position_in_the_sorted_seed_directories(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = self._two_seed_dirs(tmp_path)
        first = compute_openfold_metrics(root, "q", seed=1)
        second = compute_openfold_metrics(root, "q", seed=2)
        assert (first["seed"], first["avg_plddt"]) == (1, pytest.approx(42.4))
        assert (second["seed"], second["avg_plddt"]) == (2, pytest.approx(77.7))
        assert first["structure_path"].endswith("seed_42/q_seed_42_sample_1_model.cif")
        assert second["structure_path"].endswith("seed_777/q_seed_777_sample_1_model.cif")

    def test_sample_selects_the_file_inside_the_seed_directory(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = self._two_seed_dirs(tmp_path)
        sample_2 = compute_openfold_metrics(root, "q", seed=1, sample=2)
        assert sample_2["sample"] == 2
        assert sample_2["avg_plddt"] == pytest.approx(42.5)

    def test_seed_index_wins_over_seed_and_is_reported_as_seed(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = self._two_seed_dirs(tmp_path)
        metrics = compute_openfold_metrics(root, "q", seed=1, seed_index=2)
        assert metrics["seed"] == 2
        assert metrics["avg_plddt"] == pytest.approx(77.7)

    def test_seed_value_is_the_seed_in_the_directory_name(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = self._two_seed_dirs(tmp_path)
        by_position = [compute_openfold_metrics(root, "q", seed=i)["seed_value"] for i in (1, 2)]
        assert by_position == [42, 777]  # numeric order, and the value, not the position

    def test_seed_value_is_none_when_nothing_was_found(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = self._two_seed_dirs(tmp_path)
        assert compute_openfold_metrics(root, "q", seed=3)["seed_value"] is None
        assert compute_openfold_metrics(tmp_path / "nowhere", "q")["seed_value"] is None

    def test_a_position_beyond_the_last_directory_finds_nothing(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = self._two_seed_dirs(tmp_path)
        metrics = compute_openfold_metrics(root, "q", seed=3)
        assert metrics["seed"] == 3
        assert metrics["structure_path"] is None
        assert metrics["reason"] == (
            f"no confidence files found for query 'q' (seed index 3, sample 1) in {root}"
        )


# ---------------------------------------------------------------------------
# The CSV row
# ---------------------------------------------------------------------------

_CSV_SCALAR_COLUMNS = [
    ("sample_id", "gold"),
    ("input", "complex.pdb"),
    ("total_elapsed_s", 1.5),
    ("openfold_query_name", "gold"),
    ("openfold_seed", 1),
    ("openfold_sample", 1),
    ("openfold_structure_path", "<model.cif>"),
    ("openfold_avg_plddt", 87.5),
    ("openfold_gpde", 1.23),
    ("openfold_ptm", 0.88),
    ("openfold_iptm", 0.76),
    ("openfold_disorder", 0.12),
    ("openfold_has_clash", 0.0),
    ("openfold_sample_ranking_score", 0.82),
    ("openfold_chain_ptm_1", 0.88),
    ("openfold_chain_ptm_2", 0.80),
    ("openfold_chain_pair_iptm_(1, 2)", 0.76),
    ("openfold_bespoke_iptm_(1, 2)", 0.74),
    ("openfold_n_atoms", 14),
    ("openfold_max_pde", 2.75),
    ("openfold_max_pae", 5.5),
    ("openfold_binder_avg_plddt", 76.0),
    ("openfold_mean_interface_pde", 1.9375),
    ("openfold_max_interface_pde", 2.375),
    ("openfold_mean_interface_pae", 3.4375),
    ("openfold_max_interface_pae", 4.75),
    ("openfold_binder_ca_rmsd", 1.0),
    ("openfold_timing_inference", 45.2),
    ("openfold_timing_msa", 12.3),
    ("openfold_seed_value", 1),
]


def _flatten_row(results):
    """``report._flatten`` without its array-valued entries.

    ``_flatten`` skips lists but not numpy arrays, so ``openfold_plddt_per_atom``,
    ``openfold_binder_plddt_per_residue`` and the matrix columns hold the numpy print form
    of the array today. That is a known defect (requests/FEAT_A_bugs.md, entry 1) whose fix
    removes or re-encodes those columns, so this golden test compares everything else.
    """
    from binding_metrics.protocols.report import _flatten

    return {k: v for k, v in _flatten(results).items() if not isinstance(v, np.ndarray)}


class TestCsvRow:
    def _results(self, tmp_path):
        root, metrics = _full(tmp_path, include_matrices=True)
        return {
            "sample_id": "gold",
            "input": "complex.pdb",
            "total_elapsed_s": 1.5,
            "openfold": metrics,
        }, Path(metrics["structure_path"])

    def test_columns_and_values(self, tmp_path):
        results, model = self._results(tmp_path)
        row = _flatten_row(results)
        expected = [(k, str(model) if v == "<model.cif>" else v) for k, v in _CSV_SCALAR_COLUMNS]
        assert list(row) == [k for k, _ in expected]
        for (column, value), actual in zip(expected, row.values()):
            if isinstance(value, float):
                assert actual == pytest.approx(value, abs=1e-9), column
            else:
                assert actual == value, column

    def test_nested_dictionaries_become_prefixed_columns(self, tmp_path):
        results, _ = self._results(tmp_path)
        row = _flatten_row(results)
        assert "openfold_chain_ptm" not in row
        assert row["openfold_chain_ptm_2"] == pytest.approx(0.80)
        assert row["openfold_timing_msa"] == pytest.approx(12.3)

    def test_a_run_without_files_keeps_its_nan_and_none_sentinels_and_the_reason(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        metrics = compute_openfold_metrics(tmp_path / "nowhere", _QUERY)
        row = _flatten_row({"openfold": metrics})
        assert row["openfold_query_name"] == _QUERY
        assert row["openfold_structure_path"] is None
        assert np.isnan(row["openfold_avg_plddt"])
        assert row["openfold_n_atoms"] == 0
        assert row["openfold_reason"] == metrics["reason"]
        assert list(row)[-1] == "openfold_reason"

    def test_the_csv_file_has_the_scalar_columns_in_order(self, tmp_path):
        import csv

        from binding_metrics.protocols.report import write_report

        results, _ = self._results(tmp_path)
        out = write_report(results, tmp_path / "rep", "gold", fmt="csv")
        with open(out, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        scalar_columns = [c for c, _ in _CSV_SCALAR_COLUMNS]
        assert [c for c in reader.fieldnames if c in scalar_columns] == scalar_columns
        assert row["openfold_avg_plddt"] == "87.5"
        assert row["openfold_mean_interface_pde"] == "1.9375"
        assert row["openfold_chain_pair_iptm_(1, 2)"] == "0.76"


# ---------------------------------------------------------------------------
# The report text
# ---------------------------------------------------------------------------

_MD_FULL = """\
## OpenFold

| Metric         | Value  |
| -------------- | ------ |
| avg pLDDT      | 87.50  |
| pTM            | 0.880  |
| ipTM           | 0.760  |
| gPDE           | 1.23 Å |
| Refolding RMSD | 1.00 Å |

⚠️ **Low binder pLDDT (< 70):** res1 (62.0)
"""

_MD_NO_FILES = """\
## OpenFold

| Metric    | Value |
| --------- | ----- |
| avg pLDDT | —     |
| pTM       | —     |
| ipTM      | —     |
| gPDE      | — Å   |
"""


class TestReportText:
    def test_section_of_a_full_run(self, tmp_path):
        from binding_metrics.protocols.report import _md_openfold

        _, metrics = _full(tmp_path)
        assert _md_openfold(metrics) == _MD_FULL

    def test_section_without_reference_or_low_residues(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics
        from binding_metrics.protocols.report import _md_openfold

        root = _write_run(tmp_path, plddt=np.full(_N_ATOMS, 95.0))
        metrics = compute_openfold_metrics(root, _QUERY, binder_chain="B", receptor_chain="A")
        assert _md_openfold(metrics) == (
            "## OpenFold\n\n"
            "| Metric    | Value  |\n"
            "| --------- | ------ |\n"
            "| avg pLDDT | 87.50  |\n"
            "| pTM       | 0.880  |\n"
            "| ipTM      | 0.760  |\n"
            "| gPDE      | 1.23 Å |\n"
        )

    def test_section_of_a_run_without_files(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics
        from binding_metrics.protocols.report import _md_openfold

        metrics = compute_openfold_metrics(tmp_path / "nowhere", _QUERY)
        assert _md_openfold(metrics) == _MD_NO_FILES

    def test_skipped_and_absent_sections(self):
        from binding_metrics.protocols.report import _md_openfold

        assert _md_openfold({"skipped": True}) == "## OpenFold\n_Skipped._\n"
        assert _md_openfold(None) == "## OpenFold\n_Absent._\n"
        assert _md_openfold({}) == "## OpenFold\n_Absent._\n"

    def test_the_summary_contains_the_section(self, tmp_path):
        from binding_metrics.protocols.report import _build_summary

        _, metrics = _full(tmp_path)
        summary = _build_summary({"sample_id": "gold", "input": "complex.pdb", "openfold": metrics})
        assert _MD_FULL in summary


# ---------------------------------------------------------------------------
# The command line: binding-metrics-openfold parse
# ---------------------------------------------------------------------------

_PARSE_STDOUT = """
Parsing OpenFold3 metrics for: gold

OpenFold3 confidence metrics (seed=1, sample=1):
  Structure:            <MODEL>
  Atoms:                14
  avg_pLDDT [0–100]:         87.5000
  gPDE (Å):                  1.2300
  pTM [0–1]:                 0.8800
  ipTM [0–1]:                0.7600
  Disorder:                  0.1200
  has_clash:                 0.0000
  Ranking score:             0.8200
  Max PDE (Å):               2.7500
  Binder avg pLDDT:          76.0000
  Interface PDE mean (Å):    1.9375
  Interface PDE max (Å):     2.3750
  Binder Cα RMSD (Å):        1.0000

  Per-chain pTM:
    chain 1: 0.8800
    chain 2: 0.8000

  Chain-pair ipTM:
    (1, 2): 0.7600

  Binder per-residue pLDDT (3 residues):
    Min: 62.0  Median: 77.0  Max: 89.0
    ≥90: 0  70–89: 2  50–69: 1  <50: 0

Timing:
  inference: 45.20s
  msa: 12.30s

Per-atom pLDDT summary (14 atoms):
  Min: 60.0  Median: 85.0  Max: 94.0
  ≥90: 4  70–89: 8  50–69: 2  <50: 0
"""


class TestParseCommand:
    def test_stdout_of_a_full_run(self, tmp_path, monkeypatch, capsys):
        from binding_metrics.metrics import openfold

        root = _write_run(tmp_path / "run")
        reference = _write_reference(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["binding-metrics-openfold", "parse", "--output-dir", str(root),
             "--query-name", _QUERY, "--binder-chain", "B", "--receptor-chain", "A",
             "--reference", str(reference)],
        )  # fmt: skip
        openfold.main()
        model = root / _QUERY / "seed_1" / "gold_seed_1_sample_1_model.cif"
        assert capsys.readouterr().out == _PARSE_STDOUT.replace("<MODEL>", str(model))
