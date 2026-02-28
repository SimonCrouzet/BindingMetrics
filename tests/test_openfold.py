"""Tests for OpenFold3 metrics parsing."""

import json
import tempfile
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
        serialisable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in conf.items()
        }
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


def _default_conf(n_atoms=10, n_tokens=5, with_pae=True) -> dict:
    conf = {
        "plddt": np.random.uniform(70, 100, n_atoms),
        "pde": np.random.uniform(0, 3, (n_tokens, n_tokens)),
        "gpde": 1.23,
    }
    if with_pae:
        conf["pae"] = np.random.uniform(0, 5, (n_tokens, n_tokens))
    return conf


# ---------------------------------------------------------------------------
# Tests: _find_prediction_files
# ---------------------------------------------------------------------------


class TestFindPredictionFiles:
    def test_finds_all_files(self, tmp_path):
        from binding_metrics.metrics.openfold import _find_prediction_files

        out = _make_seed_dir(
            tmp_path, "myq", seed=1, sample=1,
            agg=_default_agg(), conf=_default_conf(), timing={"inference": 10.0},
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
    def test_json_with_pae(self, tmp_path):
        from binding_metrics.metrics.openfold import _parse_confidences

        conf = _default_conf(n_atoms=15, n_tokens=5, with_pae=True)
        path = tmp_path / "conf.json"
        path.write_text(json.dumps({
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in conf.items()
        }))
        result = _parse_confidences(path)

        assert result["plddt_per_atom"] is not None
        assert result["plddt_per_atom"].shape == (15,)
        assert result["pde"] is not None
        assert result["pde"].shape == (5, 5)
        assert result["pae"] is not None
        assert result["pae"].shape == (5, 5)

    def test_json_without_pae(self, tmp_path):
        from binding_metrics.metrics.openfold import _parse_confidences

        conf = _default_conf(with_pae=False)
        path = tmp_path / "conf_nopae.json"
        path.write_text(json.dumps({
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in conf.items()
        }))
        result = _parse_confidences(path)

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
        conf = _default_conf(n_atoms=n_atoms, n_tokens=n_tokens, with_pae=True)
        timing = {"inference": 45.2, "msa": 12.3}
        _make_seed_dir(tmp_path, "prot", seed=1, sample=1,
                       agg=agg, conf=conf, timing=timing)

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
        assert not np.isnan(metrics["max_pae"])
        assert metrics["timing"]["inference"] == pytest.approx(45.2)

    def test_matrices_excluded_by_default(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        conf = _default_conf(with_pae=True)
        _make_seed_dir(tmp_path, "q", conf=conf)
        metrics = compute_openfold_metrics(tmp_path, "q")

        assert metrics["pde"] is None
        assert metrics["pae"] is None
        assert not np.isnan(metrics["max_pae"])  # max_pae computed regardless

    def test_matrices_included_when_requested(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        n = 6
        conf = _default_conf(n_tokens=n, with_pae=True)
        _make_seed_dir(tmp_path, "q", conf=conf)
        metrics = compute_openfold_metrics(tmp_path, "q", include_matrices=True)

        assert metrics["pae"] is not None
        assert metrics["pae"].shape == (n, n)
        assert metrics["pde"] is not None

    def test_no_pae_head(self, tmp_path):
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        agg = {"avg_plddt": 80.0, "gpde": 1.5}
        conf = _default_conf(with_pae=False)
        _make_seed_dir(tmp_path, "monomer", agg=agg, conf=conf)
        metrics = compute_openfold_metrics(tmp_path, "monomer")

        assert np.isnan(metrics["ptm"])
        assert np.isnan(metrics["iptm"])
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
        # All three presets present
        for p in ("predict", "pae_enabled", "low_mem"):
            assert p in content
        from binding_metrics.metrics.openfold import compute_openfold_metrics

        agg = _default_agg(n_chains=2)
        _make_seed_dir(tmp_path, "complex", agg=agg)
        metrics = compute_openfold_metrics(tmp_path, "complex")

        assert "1" in metrics["chain_ptm"]
        assert "2" in metrics["chain_ptm"]
        assert "(1, 2)" in metrics["chain_pair_iptm"]
