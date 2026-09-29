"""Tests for OpenFold3 metrics parsing."""

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: synthetic OpenFold3 output fixtures
# ---------------------------------------------------------------------------


def _make_seed_dir(
    tmp_path: Path,
    query_name: str,
    seed: int = 1,
    sample: int = 1,
    agg: dict | None = None,
    conf: dict | None = None,
    timing: dict | None = None,
) -> Path:
    """Create a minimal OpenFold3 output directory structure.

    {tmp_path}/{query_name}/seed_{seed}/{prefix}_confidences_aggregated.json
                                       /{prefix}_confidences.json
                                       /{prefix}_model.cif
                                       /timing.json
    """
    prefix = f"{query_name}_seed_{seed}_sample_{sample}"
    seed_dir = tmp_path / query_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    if agg is not None:
        (seed_dir / f"{prefix}_confidences_aggregated.json").write_text(json.dumps(agg))

    if conf is not None:
        # Serialise numpy arrays as lists for JSON
        serialisable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in conf.items()}
        (seed_dir / f"{prefix}_confidences.json").write_text(json.dumps(serialisable))

    # Minimal stub structure file
    (seed_dir / f"{prefix}_model.cif").write_text("# stub CIF\n")

    if timing is not None:
        (seed_dir / "timing.json").write_text(json.dumps(timing))

    return tmp_path


def _default_agg(n_chains=1) -> dict:
    agg = {
        "avg_plddt": 87.5,
        "gpde": 1.23,
        "ptm": 0.88,
        "iptm": 0.76,
        "disorder": 0.12,
        "has_clash": 0.0,
        "sample_ranking_score": 0.82,
        "chain_ptm": {"1": 0.88},
        "chain_pair_iptm": {},
        "bespoke_iptm": {},
    }
    if n_chains == 2:
        agg["chain_ptm"] = {"1": 0.88, "2": 0.80}
        agg["chain_pair_iptm"] = {"(1, 2)": 0.76}
        agg["bespoke_iptm"] = {"(1, 2)": 0.74}
    return agg


def _default_conf(n_atoms=10, n_tokens=5, with_pde=True) -> dict:
    # OpenFold3 (>= 0.4.1) persists the full per-token error matrices — both
    # PDE (predicted distance error) and PAE (predicted aligned error) — to the
    # confidences file alongside per-atom pLDDT. with_pde toggles both matrices
    # (a monomer / no-error-head run writes neither).
    conf = {
        "plddt": np.random.uniform(70, 100, n_atoms),
        "gpde": 1.23,
    }
    if with_pde:
        conf["pde"] = np.random.uniform(0, 3, (n_tokens, n_tokens))
        conf["pae"] = np.random.uniform(0, 5, (n_tokens, n_tokens))
    return conf


# ---------------------------------------------------------------------------
# Tests: _find_prediction_files
# ---------------------------------------------------------------------------


class TestFindPredictionFiles:
    def test_finds_all_files(self, tmp_path):
        from binding_metrics.metrics.openfold import _find_prediction_files

        out = _make_seed_dir(
            tmp_path,
            "myq",
            seed=1,
            sample=1,
            agg=_default_agg(),
            conf=_default_conf(),
            timing={"inference": 10.0},
        )
        files = _find_prediction_files(out, "myq", seed=1, sample=1)

        assert files["structure"] is not None
        assert files["confidences"] is not None
        assert files["confidences_aggregated"] is not None
        assert files["timing"] is not None

    def test_missing_files_are_none(self, tmp_path):
        from binding_metrics.metrics.openfold import _find_prediction_files

        files = _find_prediction_files(tmp_path, "nonexistent", seed=1, sample=1)
        assert all(v is None for v in files.values())

    def test_respects_seed_and_sample(self, tmp_path):
        from binding_metrics.metrics.openfold import _find_prediction_files

        _make_seed_dir(tmp_path, "q", seed=2, sample=3, agg=_default_agg())
        files = _find_prediction_files(tmp_path, "q", seed=2, sample=3)
        assert files["confidences_aggregated"] is not None

        # Wrong seed/sample → not found
        files_wrong = _find_prediction_files(tmp_path, "q", seed=1, sample=1)
        assert files_wrong["confidences_aggregated"] is None


# ---------------------------------------------------------------------------
# Tests: _parse_confidences_aggregated
# ---------------------------------------------------------------------------


class TestParseConfidencesAggregated:
    def test_full_with_pae(self, tmp_path):
        from binding_metrics.metrics.openfold import _parse_confidences_aggregated

        agg = _default_agg(n_chains=2)
        path = tmp_path / "agg.json"
        path.write_text(json.dumps(agg))
        result = _parse_confidences_aggregated(path)

        assert result["avg_plddt"] == pytest.approx(87.5)
        assert result["gpde"] == pytest.approx(1.23)
        assert result["ptm"] == pytest.approx(0.88)
        assert result["iptm"] == pytest.approx(0.76)
        assert result["has_clash"] == pytest.approx(0.0)
        assert result["sample_ranking_score"] == pytest.approx(0.82)
        assert "1" in result["chain_ptm"]
        assert "(1, 2)" in result["chain_pair_iptm"]

    def test_without_pae_keys(self, tmp_path):
        from binding_metrics.metrics.openfold import _parse_confidences_aggregated

        path = tmp_path / "agg_nopae.json"
        path.write_text(json.dumps({"avg_plddt": 72.0, "gpde": 2.1}))
        result = _parse_confidences_aggregated(path)

        assert result["avg_plddt"] == pytest.approx(72.0)
        assert np.isnan(result["ptm"])
        assert np.isnan(result["iptm"])
        assert result["chain_ptm"] == {}


# ---------------------------------------------------------------------------
# Tests: _parse_confidences
# ---------------------------------------------------------------------------


class TestParseConfidences:
    def test_json_with_pde(self, tmp_path):
        from binding_metrics.metrics.openfold import _parse_confidences

        conf = _default_conf(n_atoms=15, n_tokens=5, with_pde=True)
        path = tmp_path / "conf.json"
        path.write_text(
            json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in conf.items()})
        )
        result = _parse_confidences(path)

        assert result["plddt_per_atom"] is not None
        assert result["plddt_per_atom"].shape == (15,)
        assert result["pde"] is not None
        assert result["pde"].shape == (5, 5)
        assert result["pae"] is not None
        assert result["pae"].shape == (5, 5)

    def test_json_without_pde(self, tmp_path):
        from binding_metrics.metrics.openfold import _parse_confidences

        conf = _default_conf(with_pde=False)
        path = tmp_path / "conf_nopde.json"
        path.write_text(
            json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in conf.items()})
        )
        result = _parse_confidences(path)

        assert result["pde"] is None
        assert result["pae"] is None
        assert result["plddt_per_atom"] is not None

    def test_npz_format(self, tmp_path):
        from binding_metrics.metrics.openfold import _parse_confidences

        plddt = np.random.uniform(75, 100, 20)
        path = tmp_path / "conf.npz"
        np.savez(path, plddt=plddt, gpde=np.float32(1.5))
        result = _parse_confidences(path.with_suffix(".npz"))

        assert result["plddt_per_atom"].shape == (20,)
        assert result["gpde"] == pytest.approx(1.5, abs=1e-4)


# ---------------------------------------------------------------------------
# Tests: compute_openfold_metrics
# ---------------------------------------------------------------------------


class TestComputeOpenfoldMetrics:
    def test_full_parse(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        n_atoms, n_tokens = 20, 8
        agg = _default_agg(n_chains=2)
        conf = _default_conf(n_atoms=n_atoms, n_tokens=n_tokens, with_pde=True)
        timing = {"inference": 45.2, "msa": 12.3}
        _make_seed_dir(tmp_path, "prot", seed=1, sample=1, agg=agg, conf=conf, timing=timing)

        metrics = compute_openfold_metrics(tmp_path, "prot", seed=1, sample=1)

        assert metrics["query_name"] == "prot"
        assert metrics["seed"] == 1
        assert metrics["sample"] == 1
        assert metrics["structure_path"] is not None
        assert metrics["avg_plddt"] == pytest.approx(87.5)
        assert metrics["ptm"] == pytest.approx(0.88)
        assert metrics["iptm"] == pytest.approx(0.76)
        assert metrics["n_atoms"] == n_atoms
        assert metrics["plddt_per_atom"] is not None
        assert metrics["plddt_per_atom"].shape == (n_atoms,)
        assert not np.isnan(metrics["max_pde"])
        assert not np.isnan(metrics["max_pae"])
        assert metrics["timing"]["inference"] == pytest.approx(45.2)

    def test_matrices_excluded_by_default(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        conf = _default_conf(with_pde=True)
        _make_seed_dir(tmp_path, "q", conf=conf)
        metrics = compute_openfold_metrics(tmp_path, "q")

        assert metrics["pde"] is None  # full matrices withheld by default
        assert metrics["pae"] is None
        assert not np.isnan(metrics["max_pde"])  # scalars computed regardless
        assert not np.isnan(metrics["max_pae"])

    def test_matrices_included_when_requested(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        n = 6
        conf = _default_conf(n_tokens=n, with_pde=True)
        _make_seed_dir(tmp_path, "q", conf=conf)
        metrics = compute_openfold_metrics(tmp_path, "q", include_matrices=True)

        assert metrics["pde"] is not None
        assert metrics["pde"].shape == (n, n)
        assert metrics["pae"] is not None
        assert metrics["pae"].shape == (n, n)

    def test_no_error_head(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        # Monomer prediction: no ptm/iptm in the aggregated file and no PDE/PAE
        # matrices persisted → interface error metrics are NaN.
        agg = {"avg_plddt": 80.0, "gpde": 1.5}
        conf = _default_conf(with_pde=False)
        _make_seed_dir(tmp_path, "monomer", agg=agg, conf=conf)
        metrics = compute_openfold_metrics(tmp_path, "monomer")

        assert np.isnan(metrics["ptm"])
        assert np.isnan(metrics["iptm"])
        assert np.isnan(metrics["max_pde"])
        assert np.isnan(metrics["max_pae"])
        assert metrics["avg_plddt"] == pytest.approx(80.0)

    def test_missing_output_dir(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        metrics = compute_openfold_metrics(tmp_path / "doesnotexist", "q")

        assert metrics["structure_path"] is None
        assert np.isnan(metrics["avg_plddt"])
        assert metrics["n_atoms"] == 0

    def test_avg_plddt_fallback_from_per_atom(self, tmp_path):
        """avg_plddt should be computed from plddt_per_atom if not in aggregated."""
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        plddt_vals = np.array([80.0, 90.0, 70.0])
        conf = {"plddt": plddt_vals, "gpde": 1.0}
        # No agg file → avg_plddt comes from per-atom data
        _make_seed_dir(tmp_path, "fallback", agg=None, conf=conf)
        metrics = compute_openfold_metrics(tmp_path, "fallback")

        assert metrics["avg_plddt"] == pytest.approx(float(np.mean(plddt_vals)))

    def test_seed_and_sample_selection(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        agg1 = {"avg_plddt": 70.0}
        agg2 = {"avg_plddt": 90.0}
        _make_seed_dir(tmp_path, "q", seed=1, sample=1, agg=agg1)
        _make_seed_dir(tmp_path, "q", seed=1, sample=2, agg=agg2)

        m1 = compute_openfold_metrics(tmp_path, "q", seed=1, sample=1)
        m2 = compute_openfold_metrics(tmp_path, "q", seed=1, sample=2)

        assert m1["avg_plddt"] == pytest.approx(70.0)
        assert m2["avg_plddt"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Tests: interface PAE / PDE slicing
# ---------------------------------------------------------------------------


class TestInterfacePaeStats:
    def _two_chain_atoms(self, n_a=2, n_b=3):
        """Minimal biotite AtomArray: one atom per residue, chains A then B."""
        struc = pytest.importorskip("biotite.structure")
        n = n_a + n_b
        atoms = struc.AtomArray(n)
        atoms.chain_id = np.array(["A"] * n_a + ["B"] * n_b)
        atoms.res_id = np.array(list(range(1, n_a + 1)) + list(range(1, n_b + 1)))
        atoms.res_name = np.array(["ALA"] * n)
        atoms.atom_name = np.array(["CA"] * n)
        atoms.element = np.array(["C"] * n)
        atoms.coord = np.zeros((n, 3), dtype=float)
        return atoms

    def test_slices_binder_receptor_block(self):
        from binding_metrics.metrics.openfold import _interface_pae_stats

        n_a, n_b = 2, 3
        atoms = self._two_chain_atoms(n_a, n_b)
        # tokens ordered A(0,1) then B(2,3,4); make the B×A block deterministic
        pae = np.zeros((5, 5), dtype=float)
        pae[2:5, 0:2] = 4.0  # receptor→? here binder=B rows, receptor=A cols
        pae[0:2, 2:5] = 2.0
        stats = _interface_pae_stats(pae, atoms, binder_chain="B", receptor_chain="A")

        assert stats["pae_interface"].shape == (n_b, n_a)
        # mean averages both slice directions: (2.0 + 4.0) / 2
        assert stats["mean_interface_pae"] == pytest.approx(3.0)
        assert stats["max_interface_pae"] == pytest.approx(4.0)
        assert stats["n_binder_tokens"] == n_b
        assert stats["n_receptor_tokens"] == n_a

    def test_missing_chain_raises(self):
        from binding_metrics.metrics.openfold import _interface_pae_stats

        atoms = self._two_chain_atoms()
        with pytest.raises(ValueError, match="Chains not found"):
            _interface_pae_stats(np.zeros((5, 5)), atoms, "B", "Z")


# ---------------------------------------------------------------------------
# Tests: token offsets against the PDE/PAE matrix size
# ---------------------------------------------------------------------------


def _protein_ligand_atoms():
    """Chains A (4 residues), B (3 residues) and a 5-atom ligand chain L (one residue)."""
    struc = pytest.importorskip("biotite.structure")
    atoms = []

    def add(chain, res_id, res_name, atom_name, xyz, hetero=False):
        atoms.append(
            struc.Atom(
                xyz,
                chain_id=chain,
                res_id=res_id,
                res_name=res_name,
                atom_name=atom_name,
                element="C",
                hetero=hetero,
            )
        )

    for i in range(4):
        add("A", i + 1, "ALA", "CA", [3.8 * i, 0.0, 0.0])
    for i in range(3):
        add("B", i + 1, "ALA", "CA", [3.8 * i, 5.0, 0.0])
    for k in range(5):
        add("L", 1, "LIG", f"C{k + 1}", [3.8 * k, 10.0, 0.0], hetero=True)
    return struc.array(atoms)


class TestTokenOffsetCheck:
    """AF3-style models tokenise a ligand per atom: 4 + 3 residue tokens + 5 ligand tokens."""

    @pytest.mark.parametrize(
        "name, func", [("PAE", "_interface_pae_stats"), ("PDE", "_interface_pde_stats")]
    )
    def test_ligand_chain_shifts_the_token_count_and_is_rejected(self, name, func):
        from binding_metrics.metrics import openfold

        atoms = _protein_ligand_atoms()
        matrix = np.zeros((12, 12))  # 4 + 3 + 5 tokens
        with pytest.raises(ValueError, match=rf"{name} matrix has 12 tokens .* 8 residues"):
            getattr(openfold, func)(matrix, atoms, binder_chain="B", receptor_chain="A")

    def test_message_names_the_residue_count_of_every_chain(self):
        from binding_metrics.metrics.openfold import _interface_pae_stats

        with pytest.raises(ValueError, match=r"A: 4, B: 3, L: 1"):
            _interface_pae_stats(np.zeros((12, 12)), _protein_ligand_atoms(), "B", "A")

    def test_matrix_matching_the_residue_count_is_sliced_as_before(self):
        from binding_metrics.metrics.openfold import _interface_pae_stats, _interface_pde_stats

        atoms = _protein_ligand_atoms()
        matrix = np.zeros((8, 8))  # one token per residue, ligand as one residue
        matrix[4:7, 0:4] = 6.0
        pae = _interface_pae_stats(matrix, atoms, "B", "A")
        pde = _interface_pde_stats(matrix, atoms, "B", "A")
        assert pae["pae_interface"].shape == (3, 4)
        assert pde["mean_interface_pde"] == pytest.approx(6.0)

    def test_matrix_smaller_than_the_residue_count_is_rejected(self):
        from binding_metrics.metrics.openfold import _interface_pde_stats

        with pytest.raises(ValueError, match="PDE matrix has 6 tokens"):
            _interface_pde_stats(np.zeros((6, 6)), _protein_ligand_atoms(), "B", "A")

    def test_non_square_matrix_is_rejected(self):
        from binding_metrics.metrics.openfold import _interface_pae_stats

        with pytest.raises(ValueError, match="must be square"):
            _interface_pae_stats(np.zeros((8, 5)), _protein_ligand_atoms(), "B", "A")

    def _write_run(self, tmp_path, n_tokens):
        pdbx = pytest.importorskip("biotite.structure.io.pdbx")
        atoms = _protein_ligand_atoms()
        conf = {
            "plddt": np.full(atoms.array_length(), 90.0),
            "gpde": 1.0,
            "pde": np.full((n_tokens, n_tokens), 2.0),
            "pae": np.full((n_tokens, n_tokens), 3.0),
        }
        root = _make_seed_dir(tmp_path, "lig", agg=_default_agg(n_chains=2), conf=conf)
        cif = pdbx.CIFFile()
        pdbx.set_structure(cif, atoms)
        cif.write(str(root / "lig" / "seed_1" / "lig_seed_1_sample_1_model.cif"))
        return root

    def test_compute_openfold_metrics_leaves_interface_values_nan_with_a_reason(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = self._write_run(tmp_path, n_tokens=12)
        with pytest.warns(UserWarning, match="interface PDE skipped"):
            metrics = compute_openfold_metrics(root, "lig", binder_chain="B", receptor_chain="A")
        assert np.isnan(metrics["mean_interface_pde"])
        assert np.isnan(metrics["mean_interface_pae"])
        assert "PDE matrix has 12 tokens" in metrics["reason"]
        assert "PAE matrix has 12 tokens" in metrics["reason"]
        # values that do not depend on the token layout are still reported
        assert metrics["binder_avg_plddt"] == pytest.approx(90.0)

    def test_no_reason_key_when_offsets_fit(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = self._write_run(tmp_path, n_tokens=8)
        metrics = compute_openfold_metrics(root, "lig", binder_chain="B", receptor_chain="A")
        assert metrics["mean_interface_pde"] == pytest.approx(2.0)
        assert metrics["mean_interface_pae"] == pytest.approx(3.0)
        assert "reason" not in metrics

    def test_compute_interface_pae_raises_for_a_ligand_complex(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_interface_pae

        root = self._write_run(tmp_path, n_tokens=12)
        seed_dir = root / "lig" / "seed_1"
        with pytest.raises(ValueError, match="PAE matrix has 12 tokens"):
            compute_interface_pae(
                seed_dir / "lig_seed_1_sample_1_confidences.json",
                seed_dir / "lig_seed_1_sample_1_model.cif",
                binder_chain="B",
                receptor_chain="A",
            )


# ---------------------------------------------------------------------------
# Tests: reason strings for values that could not be computed
# ---------------------------------------------------------------------------


def _write_dimer_run(tmp_path, n_plddt=7, pde_tokens=7, pae_tokens=7, structure=True):
    """Run directory for chains A (4 residues) and B (3 residues), one atom per residue.

    ``pde_tokens`` / ``pae_tokens`` of None leave the matrix out of the confidences file.
    """
    pdbx = pytest.importorskip("biotite.structure.io.pdbx")
    atoms = _protein_ligand_atoms()
    atoms = atoms[atoms.chain_id != "L"]
    conf = {"plddt": np.full(n_plddt, 90.0), "gpde": 1.0}
    if pde_tokens is not None:
        conf["pde"] = np.full((pde_tokens, pde_tokens), 2.0)
    if pae_tokens is not None:
        conf["pae"] = np.full((pae_tokens, pae_tokens), 3.0)
    root = _make_seed_dir(tmp_path, "dim", agg=_default_agg(n_chains=2), conf=conf)
    model = root / "dim" / "seed_1" / "dim_seed_1_sample_1_model.cif"
    if structure:
        cif = pdbx.CIFFile()
        pdbx.set_structure(cif, atoms)
        cif.write(str(model))
    else:
        model.unlink()
    return root


class TestFailureReasons:
    def test_complete_run_has_no_reason(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_dimer_run(tmp_path)
        metrics = compute_openfold_metrics(root, "dim", binder_chain="B", receptor_chain="A")
        assert "reason" not in metrics
        assert metrics["mean_interface_pde"] == pytest.approx(2.0)

    def test_missing_output_directory(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        metrics = compute_openfold_metrics(tmp_path / "nowhere", "q")
        assert "no confidence files found for query 'q'" in metrics["reason"]

    def test_missing_aggregated_file(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        _make_seed_dir(tmp_path, "q", agg=None, conf=_default_conf())
        assert compute_openfold_metrics(tmp_path, "q")["reason"] == (
            "aggregated confidences file not found"
        )

    def test_missing_per_atom_file(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        _make_seed_dir(tmp_path, "q", agg=_default_agg(), conf=None)
        assert compute_openfold_metrics(tmp_path, "q")["reason"] == (
            "per-atom confidences file not found"
        )

    def test_missing_structure_when_a_binder_chain_is_requested(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_dimer_run(tmp_path, structure=False)
        metrics = compute_openfold_metrics(root, "dim", binder_chain="B", receptor_chain="A")
        assert "structure file not found" in metrics["reason"]
        assert np.isnan(metrics["binder_avg_plddt"])

    def test_missing_matrices(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_dimer_run(tmp_path, pde_tokens=None, pae_tokens=None)
        metrics = compute_openfold_metrics(root, "dim", binder_chain="B", receptor_chain="A")
        assert "interface PDE: no PDE matrix" in metrics["reason"]
        assert "interface PAE: no PAE matrix" in metrics["reason"]
        assert metrics["binder_avg_plddt"] == pytest.approx(90.0)

    def test_plddt_that_does_not_match_the_atom_count(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_dimer_run(tmp_path, n_plddt=10)
        with pytest.warns(UserWarning, match="binder pLDDT skipped"):
            metrics = compute_openfold_metrics(root, "dim", binder_chain="B", receptor_chain="A")
        assert "binder pLDDT: plddt_per_atom length (10)" in metrics["reason"]
        assert np.isnan(metrics["binder_avg_plddt"])
        # the interface blocks do not depend on the pLDDT array
        assert metrics["mean_interface_pde"] == pytest.approx(2.0)

    def test_reference_with_a_different_binder_length(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        pdbx = pytest.importorskip("biotite.structure.io.pdbx")
        atoms = _protein_ligand_atoms()
        reference = atoms[(atoms.chain_id == "A") | ((atoms.chain_id == "B") & (atoms.res_id < 3))]
        ref_cif = pdbx.CIFFile()
        pdbx.set_structure(ref_cif, reference)
        ref_path = tmp_path / "ref.cif"
        ref_cif.write(str(ref_path))

        root = _write_dimer_run(tmp_path / "run")
        with pytest.warns(UserWarning, match="binder RMSD skipped"):
            metrics = compute_openfold_metrics(
                root,
                "dim",
                binder_chain="B",
                receptor_chain="A",
                reference_structure_path=ref_path,
            )
        assert "binder RMSD: Binder Cα count mismatch" in metrics["reason"]
        assert np.isnan(metrics["binder_ca_rmsd"])

    def test_missing_reference_file(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _write_dimer_run(tmp_path)
        with pytest.warns(UserWarning, match="binder RMSD skipped"):
            metrics = compute_openfold_metrics(
                root,
                "dim",
                binder_chain="B",
                reference_structure_path=tmp_path / "absent.cif",
            )
        assert metrics["reason"].startswith("binder RMSD:")

    def test_unparseable_structure_is_recorded(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        root = _make_seed_dir(tmp_path, "bad", agg=_default_agg(), conf=_default_conf())
        # the stub written by _make_seed_dir is not a CIF a parser can read
        with pytest.warns(UserWarning, match="structural analysis failed"):
            metrics = compute_openfold_metrics(root, "bad", binder_chain="B", receptor_chain="A")
        assert metrics["reason"].startswith("structural analysis failed:")
        assert metrics["avg_plddt"] == pytest.approx(87.5)  # scalar metrics survive

    def test_unexpected_errors_are_reported_by_the_outer_guard_not_swallowed_silently(
        self, tmp_path, monkeypatch
    ):
        from binding_metrics.metrics import openfold

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(openfold, "_binder_plddt_per_residue", _boom)
        root = _write_dimer_run(tmp_path)
        with pytest.warns(UserWarning, match="structural analysis failed"):
            metrics = openfold.compute_openfold_metrics(
                root, "dim", binder_chain="B", receptor_chain="A"
            )
        assert "RuntimeError: boom" in metrics["reason"]


# ---------------------------------------------------------------------------
# Tests: seeds in the query JSON and the seed index
# ---------------------------------------------------------------------------

_P53_MDM2 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


@pytest.fixture
def batch_sample():
    from binding_metrics.metrics.openfold import _BatchSample

    return _BatchSample(
        query_name="p53", complex_structure_path=_P53_MDM2, receptor_chain="A", binder_chain="B"
    )


class TestQuerySeeds:
    """The query JSON pins the seeds OpenFold3 samples with; 42 is only the default."""

    @pytest.fixture(autouse=True)
    def _require_gemmi(self):
        pytest.importorskip("gemmi")

    def _seeds(self, query_json: Path):
        return json.loads(query_json.read_text())["seeds"]

    def test_scoring_query_defaults_to_42(self, tmp_path):
        from binding_metrics.metrics.openfold import prepare_scoring_query

        path = prepare_scoring_query(_P53_MDM2, "A", "B", "q", tmp_path)
        assert self._seeds(path) == [42]

    def test_refolding_query_defaults_to_42(self, tmp_path):
        from binding_metrics.metrics.openfold import prepare_refolding_query

        path = prepare_refolding_query(_P53_MDM2, "A", "B", "q", tmp_path)
        assert self._seeds(path) == [42]

    def test_seeds_argument_reaches_the_json(self, tmp_path):
        from binding_metrics.metrics.openfold import (
            prepare_refolding_query,
            prepare_scoring_query,
        )

        scoring = prepare_scoring_query(_P53_MDM2, "A", "B", "q", tmp_path / "s", seeds=(7, 8, 9))
        refolding = prepare_refolding_query(_P53_MDM2, "A", "B", "q", tmp_path / "r", seeds=[3])
        assert self._seeds(scoring) == [7, 8, 9]
        assert self._seeds(refolding) == [3]

    def test_batched_queries_take_seeds(self, tmp_path, batch_sample):
        from binding_metrics.metrics.openfold import (
            prepare_batched_refolding_queries,
            prepare_batched_scoring_queries,
        )

        default = prepare_batched_scoring_queries([batch_sample], tmp_path / "d")
        scoring = prepare_batched_scoring_queries([batch_sample], tmp_path / "s", seeds=(1, 2))
        refolding = prepare_batched_refolding_queries([batch_sample], tmp_path / "r", seeds=(5,))
        assert self._seeds(default) == [42]
        assert self._seeds(scoring) == [1, 2]
        assert self._seeds(refolding) == [5]

    @pytest.mark.parametrize("bad", [(), []])
    def test_empty_seeds_are_rejected_before_anything_is_written(self, tmp_path, bad):
        from binding_metrics.metrics.openfold import prepare_scoring_query

        with pytest.raises(ValueError, match="at least one"):
            prepare_scoring_query(_P53_MDM2, "A", "B", "q", tmp_path / "out", seeds=bad)
        assert not (tmp_path / "out").exists()

    def test_a_string_is_not_a_seed_list(self, tmp_path):
        from binding_metrics.metrics.openfold import prepare_scoring_query

        with pytest.raises(TypeError, match="sequence of integers"):
            prepare_scoring_query(_P53_MDM2, "A", "B", "q", tmp_path, seeds="42")

    @pytest.mark.parametrize("runner", ["run_openfold_scoring", "run_openfold_refolding"])
    def test_run_wrappers_forward_seeds(self, tmp_path, monkeypatch, runner):
        from binding_metrics.metrics import openfold

        captured = {}

        def _fake_run(query_json, **kwargs):
            captured["seeds"] = self._seeds(Path(query_json))
            captured["num_model_seeds"] = kwargs["num_model_seeds"]
            return Path(kwargs["output_dir"])

        monkeypatch.setattr(openfold, "run_openfold", _fake_run)
        getattr(openfold, runner)(
            _P53_MDM2, "A", "B", "q", tmp_path, seeds=(11, 12), num_model_seeds=2
        )
        assert captured == {"seeds": [11, 12], "num_model_seeds": 2}

        getattr(openfold, runner)(_P53_MDM2, "A", "B", "q", tmp_path / "again")
        assert captured["seeds"] == [42]

    def test_batched_wrapper_forwards_seeds(self, tmp_path, monkeypatch, batch_sample):
        from binding_metrics.metrics import openfold

        captured = {}

        def _fake_run(query_json, **kwargs):
            captured["seeds"] = self._seeds(Path(query_json))
            return Path(kwargs["output_dir"])

        monkeypatch.setattr(openfold, "run_openfold", _fake_run)
        openfold.run_openfold_batched([batch_sample], tmp_path, mode="refold", seeds=(4, 5))
        assert captured["seeds"] == [4, 5]

    @pytest.mark.parametrize(
        "argv, expected",
        [([], [42]), (["--seeds", "5", "6"], [5, 6])],
    )
    def test_cli_seeds_flag(self, tmp_path, monkeypatch, argv, expected):
        from binding_metrics.metrics import openfold

        out = tmp_path / "out"
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "prepare-scoring-query", "--complex", str(_P53_MDM2), "--receptor-chain",
             "A", "--binder-chain", "B", "--query-name", "q", "--output-dir", str(out), *argv],
        )  # fmt: skip
        openfold.main()
        assert self._seeds(out / "q_query.json") == expected


class TestSeedIndex:
    def test_seed_index_selects_the_seed_directory_by_position(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        _make_seed_dir(tmp_path, "q", seed=1, agg={"avg_plddt": 70.0})
        _make_seed_dir(tmp_path, "q", seed=2, agg={"avg_plddt": 90.0})

        by_position = compute_openfold_metrics(tmp_path, "q", seed_index=2)
        assert by_position["avg_plddt"] == pytest.approx(90.0)
        assert by_position["seed"] == 2
        assert compute_openfold_metrics(tmp_path, "q", seed=1)["avg_plddt"] == pytest.approx(70.0)

    def test_seed_directory_names_are_not_seed_values(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        # OpenFold3 names directories after its own transformed seeds, e.g. seed_1234567
        _make_seed_dir(tmp_path, "q", seed=1234567, agg={"avg_plddt": 66.0})
        assert compute_openfold_metrics(tmp_path, "q", seed=1)["avg_plddt"] == pytest.approx(66.0)

    def test_seed_index_takes_precedence_over_seed(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        _make_seed_dir(tmp_path, "q", seed=1, agg={"avg_plddt": 70.0})
        _make_seed_dir(tmp_path, "q", seed=2, agg={"avg_plddt": 90.0})
        res = compute_openfold_metrics(tmp_path, "q", seed=1, seed_index=2)
        assert res["avg_plddt"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Tests: _write_runner_yaml
# ---------------------------------------------------------------------------


class TestWriteRunnerYaml:
    def test_default_presets(self, tmp_path):
        from binding_metrics.metrics.openfold import _write_runner_yaml

        yaml_path = _write_runner_yaml(tmp_path, ["predict", "pae_enabled", "low_mem"])

        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "predict" in content
        assert "pae_enabled" in content
        assert "low_mem" in content
        assert "model_update" in content

    def test_custom_presets(self, tmp_path):
        from binding_metrics.metrics.openfold import _write_runner_yaml

        yaml_path = _write_runner_yaml(tmp_path, ["predict", "pae_enabled"])
        content = yaml_path.read_text()

        assert "pae_enabled" in content
        assert "low_mem" not in content

    def test_predict_prepended_by_run_openfold(self, tmp_path):
        """run_openfold() must prepend 'predict' if absent from presets."""
        from binding_metrics.metrics.openfold import _write_runner_yaml

        yaml_path = _write_runner_yaml(tmp_path, ["predict", "pae_enabled", "low_mem"])
        content = yaml_path.read_text()
        for p in ("predict", "pae_enabled", "low_mem"):
            assert p in content

    def test_chain_metrics(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        agg = _default_agg(n_chains=2)
        _make_seed_dir(tmp_path, "complex", agg=agg)
        metrics = compute_openfold_metrics(tmp_path, "complex")

        assert "1" in metrics["chain_ptm"]
        assert "2" in metrics["chain_ptm"]
        assert "(1, 2)" in metrics["chain_pair_iptm"]


# ---------------------------------------------------------------------------
# Real-world integration tests using data/example_linear_p53_1YCR.pdb
# ---------------------------------------------------------------------------

EXAMPLE_PDB = Path("data/example_linear_p53_1YCR.pdb")
requires_example_pdb = pytest.mark.skipif(
    not EXAMPLE_PDB.exists(), reason="data/example_linear_p53_1YCR.pdb not found"
)


def _count_pdb_atoms(pdb_path: Path) -> int:
    return sum(
        1 for line in pdb_path.read_text().splitlines() if line.startswith(("ATOM  ", "HETATM"))
    )


def _count_pdb_residues(pdb_path: Path) -> int:
    residues = set()
    for line in pdb_path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                residues.add((line[21], int(line[22:26])))
            except ValueError:
                pass
    return len(residues)


def _pdb_chains(pdb_path: Path) -> set:
    return {
        line[21]
        for line in pdb_path.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    }


def _make_openfold3_dir_from_pdb(
    tmp_path: Path,
    query_name: str,
    pdb_src: Path,
    seed: int = 1,
    sample: int = 1,
    agg: dict | None = None,
    with_conf_json: bool = True,
) -> Path:
    """Create a realistic OpenFold3 output dir using a real PDB as structure."""
    import shutil

    seed_dir = tmp_path / query_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{query_name}_seed_{seed}_sample_{sample}"

    shutil.copy(pdb_src, seed_dir / f"{prefix}_model.pdb")

    n_atoms = _count_pdb_atoms(pdb_src)
    n_residues = _count_pdb_residues(pdb_src)

    if agg is None:
        chains = sorted(_pdb_chains(pdb_src))
        agg = {
            "avg_plddt": 82.5,
            "gpde": 1.45,
            "ptm": 0.86,
            "iptm": 0.73,
            "disorder": 0.08,
            "has_clash": 0.0,
            "sample_ranking_score": 0.79,
            "chain_ptm": {c: 0.85 for c in chains},
            "chain_pair_iptm": {f"({chains[0]}, {chains[1]})": 0.73} if len(chains) >= 2 else {},
            "bespoke_iptm": {},
        }
    (seed_dir / f"{prefix}_confidences_aggregated.json").write_text(json.dumps(agg))

    if with_conf_json:
        rng = np.random.default_rng(42)
        conf = {
            "plddt": rng.uniform(50, 100, n_atoms).tolist(),
            "pde": rng.uniform(0, 8, (n_residues, n_residues)).tolist(),
            "gpde": 1.45,
        }
        (seed_dir / f"{prefix}_confidences.json").write_text(json.dumps(conf))

    (seed_dir / "timing.json").write_text(json.dumps({"inference": 38.4}))
    return tmp_path


class TestRealWorldIntegration:
    """Integration tests that parse outputs built around data/example_linear_p53_1YCR.pdb."""

    @requires_example_pdb
    def test_parse_confidences_with_real_atom_count(self, tmp_path):
        """_parse_confidences handles a per-atom array sized to the real PDB."""
        from binding_metrics.metrics.openfold import _parse_confidences

        n_atoms = _count_pdb_atoms(EXAMPLE_PDB)
        n_residues = _count_pdb_residues(EXAMPLE_PDB)
        rng = np.random.default_rng(0)
        conf = {
            "plddt": rng.uniform(50, 100, n_atoms).tolist(),
            "pde": rng.uniform(0, 8, (n_residues, n_residues)).tolist(),
            "gpde": 1.2,
        }
        path = tmp_path / "conf.json"
        path.write_text(json.dumps(conf))

        result = _parse_confidences(path)

        assert result["plddt_per_atom"].shape == (n_atoms,)
        assert result["pde"].shape == (n_residues, n_residues)
        assert result["gpde"] == pytest.approx(1.2)

    @requires_example_pdb
    def test_find_prediction_files_locates_real_pdb(self, tmp_path):
        """_find_prediction_files finds the PDB copied into the output structure."""
        import shutil

        from binding_metrics.metrics.openfold import _find_prediction_files

        seed_dir = tmp_path / "example" / "seed_1"
        seed_dir.mkdir(parents=True)
        shutil.copy(EXAMPLE_PDB, seed_dir / "example_seed_1_sample_1_model.pdb")

        files = _find_prediction_files(tmp_path, "example", seed=1, sample=1)
        assert files["structure"] is not None
        assert files["structure"].suffix == ".pdb"

    @requires_example_pdb
    def test_compute_openfold_metrics_full_pipeline(self, tmp_path):
        """compute_openfold_metrics works end-to-end with a real PDB structure."""
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        n_atoms = _count_pdb_atoms(EXAMPLE_PDB)
        out_dir = _make_openfold3_dir_from_pdb(tmp_path, "example", EXAMPLE_PDB)
        metrics = compute_openfold_metrics(out_dir, "example")

        assert metrics["structure_path"] is not None
        assert metrics["structure_path"].endswith(".pdb")
        assert metrics["n_atoms"] == n_atoms
        assert metrics["avg_plddt"] == pytest.approx(82.5)
        assert metrics["ptm"] == pytest.approx(0.86)
        assert metrics["iptm"] == pytest.approx(0.73)
        assert not np.isnan(metrics["max_pde"])
        assert metrics["timing"]["inference"] == pytest.approx(38.4)

    @requires_example_pdb
    def test_per_atom_plddt_shape_matches_pdb(self, tmp_path):
        """plddt_per_atom length equals the atom count in the real PDB."""
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        n_atoms = _count_pdb_atoms(EXAMPLE_PDB)
        out_dir = _make_openfold3_dir_from_pdb(tmp_path, "example", EXAMPLE_PDB)
        metrics = compute_openfold_metrics(out_dir, "example")

        assert metrics["plddt_per_atom"] is not None
        assert len(metrics["plddt_per_atom"]) == n_atoms

    @requires_example_pdb
    def test_no_conf_json_still_returns_agg_metrics(self, tmp_path):
        """When no confidences.json, scalar metrics still come from aggregated JSON."""
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        out_dir = _make_openfold3_dir_from_pdb(
            tmp_path, "example", EXAMPLE_PDB, with_conf_json=False
        )
        metrics = compute_openfold_metrics(out_dir, "example")

        assert metrics["avg_plddt"] == pytest.approx(82.5)
        assert metrics["ptm"] == pytest.approx(0.86)
        assert metrics["plddt_per_atom"] is None
        assert metrics["n_atoms"] == 0

    @requires_example_pdb
    def test_chain_scores_match_pdb_chains(self, tmp_path):
        """chain_ptm keys match the actual chains present in the PDB."""
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        chains = sorted(_pdb_chains(EXAMPLE_PDB))
        out_dir = _make_openfold3_dir_from_pdb(tmp_path, "example", EXAMPLE_PDB)
        metrics = compute_openfold_metrics(out_dir, "example")

        for c in chains:
            assert c in metrics["chain_ptm"]
        pair_key = f"({chains[0]}, {chains[1]})"
        assert pair_key in metrics["chain_pair_iptm"]


# ---------------------------------------------------------------------------
# Tests: names that moved out of openfold.py stay importable from it
# ---------------------------------------------------------------------------


class TestModuleLayout:
    def test_console_script_target_is_the_cli_entry_point(self):
        from binding_metrics.metrics import _openfold_cli, openfold

        assert openfold.main is _openfold_cli.main

    @pytest.mark.parametrize("name", ["_add_parse_args", "_add_query_seeds_arg", "_print_metrics"])
    def test_cli_helpers_remain_importable_from_openfold(self, name):
        from binding_metrics.metrics import _openfold_cli, openfold

        assert getattr(openfold, name) is getattr(_openfold_cli, name)

    def test_cli_resolves_functions_on_the_openfold_module(self, tmp_path, monkeypatch):
        """Patching ``openfold.<name>`` redirects the command line, as before the move."""
        from binding_metrics.metrics import openfold

        called = {}

        def _fake_prepare(**kwargs):
            called.update(kwargs)
            return tmp_path / "q.json"

        monkeypatch.setattr(openfold, "prepare_scoring_query", _fake_prepare)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "prepare-scoring-query", "--complex", "c.cif", "--receptor-chain", "A"]
            + ["--binder-chain", "B", "--query-name", "q", "--output-dir", str(tmp_path)]
            + ["--seeds", "3"],
        )
        openfold.main()
        assert called["seeds"] == [3]
        assert called["query_name"] == "q"
