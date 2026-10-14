"""``binding-metrics-geometry --interface-cutoff``: one flag, two metrics with their own default.

The shape complementarity pre-selects the atoms it dots within 6.0 A of the other chain and
the buried void volume sizes its box from the atoms within 5.0 A. The flag used to default to
6.0 and fed both, so the command line reported a void volume that the function does not return
for the same structure. The flag now defaults to None, which leaves each metric on its own
default; a value given on the command line applies to both.
"""

import inspect
import re
import sys

import pytest
from conftest import DATA_DIR

from binding_metrics.metrics import geometry

P53 = DATA_DIR / "example_linear_p53_1YCR.pdb"
SFTI = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"
CYCLOSPORIN = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"

# void volume in A^3 at the default cutoff of the function (5.0) and at 6.0 A
VOID_BY_CUTOFF = {
    "1YCR": (P53, 55.875, 56.0),
    "3P8F": (SFTI, 31.875, 15.625),
    "1CWA": (CYCLOSPORIN, 16.25, 10.75),
}


def _run(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["binding-metrics-geometry", *map(str, args)])
    geometry.main()
    return capsys.readouterr().out


def _void_printed(out: str) -> float:
    return float(re.search(r"void_volume_A3: (\S+)", out).group(1))


@pytest.fixture
def seen(monkeypatch):
    """Keyword arguments that ``main`` passes to each metric function."""
    calls = {}

    def spy(name, result):
        def fake(*args, **kwargs):
            calls[name] = kwargs
            return result

        return fake

    void = {"void_volume_A3": 0.0}
    sc = {"sc": 0.0}
    monkeypatch.setattr(geometry, "compute_buried_void_volume", spy("void", void))
    monkeypatch.setattr(geometry, "compute_shape_complementarity", spy("sc", sc))
    return calls


class TestDefaultKeepsEachMetricsOwnCutoff:
    def test_the_functions_keep_their_own_defaults(self):
        default = {
            name: inspect.signature(getattr(geometry, name)).parameters["interface_cutoff"].default
            for name in ("compute_shape_complementarity", "compute_buried_void_volume")
        }
        assert default == {
            "compute_shape_complementarity": 6.0,
            "compute_buried_void_volume": 5.0,
        }

    @pytest.mark.parametrize("metric", ["sc", "void"])
    def test_no_flag_passes_no_cutoff(self, monkeypatch, capsys, seen, metric):
        _run(monkeypatch, capsys, "--input", P53, "--metric", metric)
        assert "interface_cutoff" not in seen[metric]

    @pytest.mark.parametrize("metric", ["sc", "void"])
    def test_a_value_on_the_command_line_reaches_the_metric(
        self, monkeypatch, capsys, seen, metric
    ):
        _run(monkeypatch, capsys, "--input", P53, "--metric", metric, "--interface-cutoff", "4.5")
        assert seen[metric]["interface_cutoff"] == 4.5

    def test_the_help_names_both_defaults(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["binding-metrics-geometry", "--help"])
        with pytest.raises(SystemExit):
            geometry.main()
        help_text = " ".join(capsys.readouterr().out.split())
        # the first mention of the flag is in the usage line
        flag = help_text[help_text.rindex("--interface-cutoff") :]
        flag = flag[: flag.index("--buried-cutoff")]
        assert "6.0" in flag
        assert "5.0" in flag


class TestVoidVolumeOnTheCommandLine:
    """The void volume printed by the command line, on the bundled examples."""

    @pytest.mark.parametrize("name", list(VOID_BY_CUTOFF))
    def test_the_default_prints_the_void_at_the_function_default(self, monkeypatch, capsys, name):
        path, at_5, _ = VOID_BY_CUTOFF[name]
        out = _run(monkeypatch, capsys, "--input", path, "--metric", "void")
        assert _void_printed(out) == pytest.approx(at_5)

    def test_the_default_matches_the_function(self, monkeypatch, capsys):
        out = _run(monkeypatch, capsys, "--input", P53, "--metric", "void")
        assert _void_printed(out) == pytest.approx(
            geometry.compute_buried_void_volume(P53)["void_volume_A3"]
        )

    @pytest.mark.parametrize("name", list(VOID_BY_CUTOFF))
    def test_an_explicit_cutoff_is_used(self, monkeypatch, capsys, name):
        path, _, at_6 = VOID_BY_CUTOFF[name]
        out = _run(
            monkeypatch, capsys, "--input", path, "--metric", "void", "--interface-cutoff", "6.0"
        )
        assert _void_printed(out) == pytest.approx(at_6)
