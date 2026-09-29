"""--random-seed on binding-metrics-prep and binding-metrics-solvate."""

import json
import sys
from pathlib import Path

import pytest

from binding_metrics.core.system import DEFAULT_RANDOM_SEED, HAS_PDBFIXER
from binding_metrics.protocols import prep as prep_cli
from binding_metrics.protocols import solvate as solvate_cli

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
            outputs[name] = out.read_text(encoding="utf-8")
        assert outputs["a"] == outputs["b"]
        assert outputs["a"] != outputs["c"]


@pytest.fixture
def prepped_peptide(tmp_path) -> Path:
    """The 13-residue p53 peptide of 1YCR (chain B), protonated: a cheap solvation input."""
    atoms = [
        line
        for line in EXAMPLE_1YCR.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM") and line[21] == "B"
    ]
    raw = tmp_path / "peptide_raw.pdb"
    raw.write_text("\n".join(atoms) + "\nEND\n", encoding="utf-8")
    out = tmp_path / "peptide.pdb"
    from binding_metrics.core.system import prep_structure
    from binding_metrics.io.structures import load_structure, save_structure

    topology, positions = load_structure(raw)
    topology, positions = prep_structure(topology, positions, random_seed=1)
    save_structure(topology, positions, out, source_path=raw)
    return out


def _run_solvate(monkeypatch, capsys, src, out_path, *extra):
    argv = ["binding-metrics-solvate", "-i", str(src), "-o", str(out_path), "--padding", "0.6"]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    solvate_cli.main()
    return json.loads(capsys.readouterr().out)


@requires_pdbfixer
class TestSolvateSeed:
    def test_seed_is_echoed_and_defaults_to_the_library_seed(
        self, monkeypatch, capsys, tmp_path, prepped_peptide
    ):
        default = _run_solvate(monkeypatch, capsys, prepped_peptide, tmp_path / "d.pdb")
        assert default["random_seed"] == DEFAULT_RANDOM_SEED
        fresh = _run_solvate(
            monkeypatch, capsys, prepped_peptide, tmp_path / "f.pdb", "--random-seed", "none"
        )
        assert fresh["random_seed"] is None

    def test_same_seed_places_the_same_ions_and_another_seed_does_not(
        self, monkeypatch, capsys, tmp_path, prepped_peptide
    ):
        ions = {}
        for name, seed in (("a", "3"), ("b", "3"), ("c", "4")):
            out = tmp_path / f"{name}.pdb"
            summary = _run_solvate(monkeypatch, capsys, prepped_peptide, out, "--random-seed", seed)
            assert summary["n_ions"] > 0
            ions[name] = [
                line[30:54]
                for line in out.read_text(encoding="utf-8").splitlines()
                if line[12:16].strip() == "Na"
            ]
        assert ions["a"] and ions["a"] == ions["b"]
        assert ions["a"] != ions["c"]
