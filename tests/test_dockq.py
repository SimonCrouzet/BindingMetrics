"""Tests for reference-based DockQ / CAPRI metrics.

The pure helpers (capri_class, _parse_dockq_json) are tested without DockQ
installed. The end-to-end run is exercised only when the DockQ CLI is on PATH.
"""

import shutil

import pytest

from binding_metrics.metrics.dockq import (
    _parse_dockq_json,
    capri_class,
    compute_dockq_metrics,
)

HAS_DOCKQ = shutil.which("DockQ") is not None
requires_dockq = pytest.mark.skipif(not HAS_DOCKQ, reason="DockQ CLI not installed")


class TestCapriClass:
    """CAPRI quality bins from a DockQ score (Basu & Wallner 2016)."""

    def test_incorrect(self):
        assert capri_class(0.0) == "Incorrect"
        assert capri_class(0.22) == "Incorrect"

    def test_acceptable_lower_boundary(self):
        # 0.23 is the Acceptable threshold (inclusive)
        assert capri_class(0.23) == "Acceptable"
        assert capri_class(0.48) == "Acceptable"

    def test_medium_lower_boundary(self):
        assert capri_class(0.49) == "Medium"
        assert capri_class(0.79) == "Medium"

    def test_high_lower_boundary(self):
        assert capri_class(0.80) == "High"
        assert capri_class(1.0) == "High"


class TestParseDockqJson:
    """Normalisation of DockQ's --json output into our flat dict."""

    def _sample(self):
        return {
            "model": "model.pdb",
            "native": "native.pdb",
            "best_dockq": 0.812,
            "GlobalDockQ": 0.812,
            "best_mapping": {"A": "B"},
            "best_mapping_str": "A:B",
            "best_result": {
                ("A", "B"): {
                    "DockQ": 0.812,
                    "fnat": 0.90,
                    "fnonnat": 0.10,
                    "iRMSD": 0.95,
                    "LRMSD": 1.80,
                    "clashes": 0,
                    "chain1": "A",
                    "chain2": "B",
                },
            },
        }

    def test_global_score_and_class(self):
        out = _parse_dockq_json(self._sample())
        assert out["dockq"] == pytest.approx(0.812)
        assert out["capri_class"] == "High"

    def test_interface_count_and_mapping(self):
        out = _parse_dockq_json(self._sample())
        assert out["n_interfaces"] == 1
        assert out["best_mapping"] == "A:B"

    def test_interface_fields_flattened(self):
        out = _parse_dockq_json(self._sample())
        iface = out["interfaces"][0]
        assert iface["chains"] == "AB"
        assert iface["fnat"] == pytest.approx(0.90)
        assert iface["fnonnat"] == pytest.approx(0.10)
        assert iface["iRMSD"] == pytest.approx(0.95)
        assert iface["LRMSD"] == pytest.approx(1.80)
        assert iface["capri_class"] == "High"

    def test_falls_back_to_best_dockq_when_no_global(self):
        data = self._sample()
        del data["GlobalDockQ"]
        out = _parse_dockq_json(data)
        assert out["dockq"] == pytest.approx(0.812)

    def test_empty_result_is_handled(self):
        data = {"best_result": {}, "GlobalDockQ": None}
        out = _parse_dockq_json(data)
        assert out["n_interfaces"] == 0
        assert out["dockq"] is None
        assert out["capri_class"] is None
        assert out["interfaces"] == []

    def test_string_keyed_interface(self):
        """Some DockQ versions may serialise the chain pair as a string key."""
        data = self._sample()
        data["best_result"] = {"AB": data["best_result"][("A", "B")]}
        out = _parse_dockq_json(data)
        assert out["interfaces"][0]["chains"] == "AB"


class TestComputeDockqInputValidation:
    """Argument validation happens before DockQ is invoked."""

    def test_missing_model_raises(self, tmp_path):
        ref = tmp_path / "ref.pdb"
        ref.write_text("REMARK\n")
        with pytest.raises(FileNotFoundError, match="Model structure not found"):
            compute_dockq_metrics(tmp_path / "nope.pdb", ref)

    def test_missing_reference_raises(self, tmp_path):
        model = tmp_path / "model.pdb"
        model.write_text("REMARK\n")
        with pytest.raises(FileNotFoundError, match="Reference structure not found"):
            compute_dockq_metrics(model, tmp_path / "nope.pdb")


@requires_dockq
class TestComputeDockqEndToEnd:
    """Full DockQ run — a perfect self-comparison must score 1.0."""

    def test_identical_structure_scores_one(self):
        from pathlib import Path

        example = Path("data/example.pdb")
        if not example.exists():
            pytest.skip("no example complex available")
        out = compute_dockq_metrics(example, example)
        if out["dockq"] is None:
            pytest.skip("DockQ found no interface in example structure")
        assert out["dockq"] == pytest.approx(1.0, abs=1e-3)
        assert out["capri_class"] == "High"
