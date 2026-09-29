"""--random-seed on binding-metrics-prep and binding-metrics-solvate."""

import json
import sys
from pathlib import Path

import pytest

from binding_metrics.core.system import DEFAULT_RANDOM_SEED, HAS_PDBFIXER
from binding_metrics.protocols import prep as prep_cli

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

requires_pdbfixer = pytest.mark.skipif(not HAS_PDBFIXER, reason="pdbfixer not installed")


def _run_prep(monkeypatch, capsys, out_path, *extra):
    monkeypatch.setattr(
        sys, "argv", ["binding-metrics-prep", "-i", str(EXAMPLE_1YCR), "-o", str(out_path), *extra]
    )
    prep_cli.main()
    return json.loads(capsys.readouterr().out)


class TestPrepSeed:
    @pytest.mark.parametrize(
        "extra,expected",
        [([], DEFAULT_RANDOM_SEED), (["--random-seed", "9"], 9), (["--random-seed", "none"], None)],
    )
    def test_seed_reaches_prep_structure_and_is_echoed(
        self, monkeypatch, capsys, tmp_path, extra, expected
    ):
        import binding_metrics.core.system as system

        seen = {}

        def fake_prep(topology, positions, **kwargs):
            seen.update(kwargs)
            return topology, positions

        monkeypatch.setattr(system, "prep_structure", fake_prep)
        monkeypatch.setattr(system, "HAS_PDBFIXER", True)
        summary = _run_prep(monkeypatch, capsys, tmp_path / "out.pdb", *extra)
        assert seen["random_seed"] == expected
        assert summary["random_seed"] == expected

    @requires_pdbfixer
    def test_same_seed_gives_the_same_file_and_another_seed_a_different_one(
        self, monkeypatch, capsys, tmp_path
    ):
        """Hydrogen placement is stochastic; the seed pins it (1YCR, about 2 s per run)."""
        outputs = {}
        for name, seed in (("a", "3"), ("b", "3"), ("c", "4")):
            out = tmp_path / f"{name}.pdb"
            _run_prep(monkeypatch, capsys, out, "--random-seed", seed)
            outputs[name] = out.read_text()
        assert outputs["a"] == outputs["b"]
        assert outputs["a"] != outputs["c"]
