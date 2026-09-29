"""Argument parsing of ``binding-metrics-energy`` and how it reaches the API."""

import inspect
import sys
from pathlib import Path

import pytest

from binding_metrics.metrics import energy

# Read once at import: some tests below replace ``compute_interaction_energy``.
_API_PARAMETERS = inspect.signature(energy.compute_interaction_energy).parameters


def _api_default(name: str):
    return _API_PARAMETERS[name].default


class TestEnergyCliParser:
    def test_defaults_equal_api_defaults(self):
        """Without the new flags the CLI behaves exactly like the API defaults."""
        args = energy._build_parser().parse_args(["--input", "x.cif"])
        assert args.ph == _api_default("ph")
        assert args.random_seed == _api_default("random_seed")

    def test_random_seed_integer(self):
        args = energy._build_parser().parse_args(["-i", "x.cif", "--random-seed", "7"])
        assert args.random_seed == 7

    @pytest.mark.parametrize("token", ["none", "None", "random", "off"])
    def test_random_seed_none_disables_seeding(self, token):
        args = energy._build_parser().parse_args(["-i", "x.cif", "--random-seed", token])
        assert args.random_seed is None

    def test_random_seed_rejects_garbage(self, capsys):
        with pytest.raises(SystemExit):
            energy._build_parser().parse_args(["-i", "x.cif", "--random-seed", "abc"])
        assert "random-seed" in capsys.readouterr().err

    def test_ph_float(self):
        args = energy._build_parser().parse_args(["-i", "x.cif", "--ph", "6.5"])
        assert args.ph == 6.5


class TestEnergyMainForwardsSeedAndPh:
    @staticmethod
    def _run_main(monkeypatch, tmp_path: Path, extra: list[str]) -> dict:
        captured: dict = {}

        def fake_compute(path, **kwargs):
            captured["path"] = path
            captured.update(kwargs)
            return {"sample_id": Path(path).stem, "success": True, "error_message": None}

        monkeypatch.setattr(energy, "compute_interaction_energy", fake_compute)
        out_csv = tmp_path / "out.csv"
        monkeypatch.setattr(
            sys, "argv", ["binding-metrics-energy", "-i", "x.cif", "-o", str(out_csv), *extra]
        )
        energy.main()
        assert out_csv.exists()
        return captured

    def test_defaults_are_forwarded_unchanged(self, monkeypatch, tmp_path):
        captured = self._run_main(monkeypatch, tmp_path, [])
        assert captured["ph"] == _api_default("ph")
        assert captured["random_seed"] == _api_default("random_seed")

    def test_explicit_seed_and_ph_reach_the_call(self, monkeypatch, tmp_path):
        captured = self._run_main(monkeypatch, tmp_path, ["--random-seed", "123", "--ph", "5.0"])
        assert captured["random_seed"] == 123
        assert captured["ph"] == 5.0

    def test_none_seed_reaches_the_call_as_none(self, monkeypatch, tmp_path):
        captured = self._run_main(monkeypatch, tmp_path, ["--random-seed", "none"])
        assert captured["random_seed"] is None
