"""The model-agnostic confidence helpers in ``binding_metrics.predictors._confidence``.

They were moved out of ``metrics/openfold.py`` without a change of behaviour; the golden
tests in ``test_openfold_golden.py`` and ``tests/test_openfold.py`` cover the numbers through
the old import path. These tests cover the new location and the rules the record relies on.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

struc = pytest.importorskip("biotite.structure")

MOVED = [
    "_binder_ca_rmsd",
    "_binder_plddt_per_residue",
    "_chain_token_offsets",
    "_check_token_offsets",
    "_import_biotite_struc",
    "_interface_pae_stats",
    "_interface_pde_stats",
    "_load_atoms",
]


def _chain(chain_id, n_res, y, atoms_per_residue=1):
    names = ["CA", "CB"][:atoms_per_residue]
    return [
        struc.Atom(
            [3.8 * i, y + 1.5 * k, 0.0],
            chain_id=chain_id,
            res_id=i + 1,
            res_name="ALA",
            atom_name=name,
            element="C",
        )
        for i in range(n_res)
        for k, name in enumerate(names)
    ]


def _dimer(atoms_per_residue=1):
    """Chain A (4 residues) followed by chain B (3 residues)."""
    return struc.array(
        _chain("A", 4, 0.0, atoms_per_residue) + _chain("B", 3, 6.0, atoms_per_residue)
    )


class TestOldNamesStillResolve:
    @pytest.mark.parametrize("name", MOVED)
    def test_openfold_re_exports_the_same_object(self, name):
        from binding_metrics.metrics import openfold
        from binding_metrics.predictors import _confidence

        assert getattr(openfold, name) is getattr(_confidence, name)

    @pytest.mark.parametrize(
        "name",
        [
            "_binder_plddt_per_residue",
            "_interface_pde_stats",
            "_interface_pae_stats",
            "_binder_ca_rmsd",
        ],
    )
    def test_compute_openfold_metrics_looks_the_helpers_up_on_the_prediction_module(
        self, name, tmp_path, monkeypatch
    ):
        """The analysis moved to ``metrics.prediction``, which is where a test patches a helper."""
        from binding_metrics.metrics import openfold, prediction
        from tests.predictors.test_openfold_golden import _QUERY, _write_reference, _write_run

        calls = []

        def _spy(*a, **k):
            calls.append(name)
            raise RuntimeError("spy")

        monkeypatch.setattr(prediction, name, _spy)
        root = _write_run(tmp_path / "run")
        reference = _write_reference(tmp_path)
        with pytest.warns(UserWarning):
            metrics = openfold.compute_openfold_metrics(
                root,
                _QUERY,
                binder_chain="B",
                receptor_chain="A",
                reference_structure_path=reference,
            )
        assert calls == [name]
        assert "RuntimeError: spy" in metrics["reason"]

    def test_importing_the_helpers_does_not_import_biotite(self):
        code = textwrap.dedent(
            """
            import sys
            import binding_metrics.predictors._confidence
            heavy = [m for m in ("biotite", "scipy", "openmm", "gemmi") if m in sys.modules]
            print(heavy)
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
        assert out.strip() == "[]"


class TestChainTokenOffsets:
    def test_chains_are_ordered_by_first_appearance(self):
        from binding_metrics.predictors._confidence import _chain_token_offsets

        atoms = _dimer(atoms_per_residue=2)  # two atoms per residue count once
        assert _chain_token_offsets(atoms) == {"A": (0, 4), "B": (4, 7)}
        swapped = struc.concatenate([atoms[atoms.chain_id == "B"], atoms[atoms.chain_id == "A"]])
        assert _chain_token_offsets(swapped) == {"B": (0, 3), "A": (3, 7)}

    def test_a_matrix_of_the_residue_count_is_accepted_any_other_size_is_not(self):
        from binding_metrics.predictors._confidence import (
            _chain_token_offsets,
            _check_token_offsets,
        )

        offsets = _chain_token_offsets(_dimer())
        _check_token_offsets(offsets, np.zeros((7, 7)), "PAE")
        with pytest.raises(ValueError, match="PAE matrix has 9 tokens but the structure has 7"):
            _check_token_offsets(offsets, np.zeros((9, 9)), "PAE")
        with pytest.raises(ValueError, match="PDE matrix must be square"):
            _check_token_offsets(offsets, np.zeros((7, 6)), "PDE")


class TestBinderPlddtPerResidue:
    def test_residue_value_is_the_mean_of_its_atoms(self):
        from binding_metrics.predictors._confidence import _binder_plddt_per_residue

        atoms = _dimer(atoms_per_residue=2)
        plddt = np.arange(atoms.array_length(), dtype=float)  # 0..13, two atoms per residue
        per_residue = _binder_plddt_per_residue(plddt, atoms, "B")
        np.testing.assert_allclose(per_residue, [8.5, 10.5, 12.5])

    def test_a_length_mismatch_is_a_value_error_with_both_counts(self):
        from binding_metrics.predictors._confidence import _binder_plddt_per_residue

        atoms = _dimer()
        for n in (6, 8):
            with pytest.raises(ValueError, match=rf"plddt_per_atom length \({n}\) != atom count"):
                _binder_plddt_per_residue(np.full(n, 90.0), atoms, "B")

    def test_a_chain_that_is_absent_gives_an_empty_array(self):
        from binding_metrics.predictors._confidence import _binder_plddt_per_residue

        atoms = _dimer()
        assert _binder_plddt_per_residue(np.full(7, 90.0), atoms, "Z").shape == (0,)


class TestInterfaceStatistics:
    """PAE rows are the alignment frame: ``pae[i, j]`` scores token j when aligned on token i."""

    def _matrices(self):
        i = np.arange(7)[:, None]
        j = np.arange(7)[None, :]
        return 1.0 + 0.5 * i + 0.25 * j

    def test_the_raw_slice_has_binder_rows_and_receptor_columns(self):
        from binding_metrics.predictors._confidence import _interface_pae_stats

        pae = self._matrices()
        stats = _interface_pae_stats(pae, _dimer(), binder_chain="B", receptor_chain="A")
        np.testing.assert_allclose(stats["pae_interface"], pae[4:7, 0:4])
        assert (stats["n_binder_tokens"], stats["n_receptor_tokens"]) == (3, 4)

    def test_mean_and_max_do_not_depend_on_which_chain_is_called_the_binder(self):
        from binding_metrics.predictors._confidence import _interface_pae_stats

        pae = self._matrices()
        as_binder = _interface_pae_stats(pae, _dimer(), binder_chain="B", receptor_chain="A")
        swapped = _interface_pae_stats(pae, _dimer(), binder_chain="A", receptor_chain="B")
        assert as_binder["mean_interface_pae"] == pytest.approx(3.4375)
        assert swapped["mean_interface_pae"] == pytest.approx(as_binder["mean_interface_pae"])
        assert swapped["max_interface_pae"] == pytest.approx(as_binder["max_interface_pae"])
        # the raw slice is the other block: rows of A (the alignment frame), columns of B
        np.testing.assert_allclose(swapped["pae_interface"], pae[0:4, 4:7])

    def test_pde_slice_is_taken_as_stored_without_averaging_directions(self):
        from binding_metrics.predictors._confidence import _interface_pde_stats

        pde = self._matrices()
        stats = _interface_pde_stats(pde, _dimer(), binder_chain="B", receptor_chain="A")
        assert stats["mean_interface_pde"] == pytest.approx(float(pde[4:7, 0:4].mean()))
        assert stats["max_interface_pde"] == pytest.approx(float(pde[4:7, 0:4].max()))


class TestBinderCaRmsd:
    def test_rigid_motion_of_the_whole_complex_gives_zero_in_the_receptor_frame(self):
        from binding_metrics.predictors._confidence import _binder_ca_rmsd

        reference = _dimer()
        moved = struc.translate(struc.rotate(reference, [0.0, 0.0, np.deg2rad(40)]), [10, -5, 2])
        assert _binder_ca_rmsd(moved, reference, "B", "A") == pytest.approx(0.0, abs=1e-4)

    def test_a_binder_shift_is_measured_after_superposing_the_receptor(self):
        from binding_metrics.predictors._confidence import _binder_ca_rmsd

        reference = _dimer()
        predicted = reference.copy()
        predicted.coord[predicted.chain_id == "B", 2] += 1.5
        moved = struc.translate(struc.rotate(predicted, [0.0, 0.0, np.deg2rad(-25)]), [3, 4, 5])
        assert _binder_ca_rmsd(moved, reference, "B", "A") == pytest.approx(1.5, abs=1e-4)

    def test_without_a_receptor_the_binder_is_superposed_on_itself(self):
        from binding_metrics.predictors._confidence import _binder_ca_rmsd

        reference = _dimer()
        moved = struc.translate(struc.rotate(reference, [0.0, 0.0, np.deg2rad(40)]), [10, -5, 2])
        assert _binder_ca_rmsd(moved, reference, "B") == pytest.approx(0.0, abs=1e-4)
        # a binder bent at the middle residue differs in shape, whatever the frame
        bent = moved.copy()
        middle = (bent.chain_id == "B") & (bent.res_id == 2)
        bent.coord[middle] += [0.0, 0.0, 1.5]
        rmsd = _binder_ca_rmsd(bent, reference, "B")
        assert 0.3 < rmsd < 1.0

    def test_receptor_frame_and_binder_frame_differ_for_a_rigid_binder_shift(self):
        from binding_metrics.predictors._confidence import _binder_ca_rmsd

        reference = _dimer()
        shifted = reference.copy()
        shifted.coord[shifted.chain_id == "B", 2] += 2.0
        assert _binder_ca_rmsd(shifted, reference, "B", "A") == pytest.approx(2.0, abs=1e-4)
        assert _binder_ca_rmsd(shifted, reference, "B") == pytest.approx(0.0, abs=1e-4)

    def test_receptors_of_different_length_are_refused_not_measured_in_another_frame(self):
        from binding_metrics.predictors._confidence import _binder_ca_rmsd

        reference = _dimer()
        short = reference[~((reference.chain_id == "A") & (reference.res_id == 4))]
        moved = struc.translate(reference, [10.0, 0.0, 0.0])
        with pytest.raises(
            ValueError, match=r"Receptor Cα count mismatch .*predicted 4, reference 3"
        ):
            _binder_ca_rmsd(moved, short, "B", "A")

    def test_a_receptor_with_fewer_than_three_calphas_or_an_absent_one_is_refused(self):
        from binding_metrics.predictors._confidence import _binder_ca_rmsd

        reference = _dimer()
        with pytest.raises(ValueError, match="'Z' has 0 Cα atoms"):
            _binder_ca_rmsd(reference, reference, "B", "Z")
        two = reference[~((reference.chain_id == "A") & (reference.res_id > 2))]
        with pytest.raises(ValueError, match="'A' has 2 Cα atoms; at least 3"):
            _binder_ca_rmsd(two, two, "B", "A")

    def test_a_binder_of_fewer_than_three_calphas_needs_a_receptor(self):
        from binding_metrics.predictors._confidence import _binder_ca_rmsd

        reference = _dimer()
        short = reference[~((reference.chain_id == "B") & (reference.res_id > 2))]
        assert _binder_ca_rmsd(short, short, "B", "A") == pytest.approx(0.0, abs=1e-4)
        with pytest.raises(ValueError, match="Binder chain 'B' has 2 Cα atoms"):
            _binder_ca_rmsd(short, short, "B")

    def test_different_binder_lengths_raise_and_a_missing_chain_gives_nan(self):
        from binding_metrics.predictors._confidence import _binder_ca_rmsd

        reference = _dimer()
        short = reference[~((reference.chain_id == "B") & (reference.res_id == 3))]
        with pytest.raises(ValueError, match="Binder Cα count mismatch"):
            _binder_ca_rmsd(reference, short, "B", "A")
        assert np.isnan(_binder_ca_rmsd(reference, reference, "Z", "A"))
