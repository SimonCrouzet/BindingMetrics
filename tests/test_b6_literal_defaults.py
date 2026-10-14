"""The pH and device defaults come from ``_constants``; no module writes them again.

``metrics/energy.py`` and ``metrics/receptor_quality.py`` wrote ``7.4`` and ``"cuda"`` as the
defaults of their signatures, of their command-line flags and of one hydrogen placement. They
now use ``DEFAULT_PH`` and ``DEFAULT_DEVICE``, and no default seen by a user changes.
"""

import ast
import inspect
from pathlib import Path

import pytest

from binding_metrics import _constants
from binding_metrics.metrics import energy, receptor_quality

SRC = Path(_constants.__file__).resolve().parent

SIGNATURES = [
    (energy._create_implicit_system, "ph", 7.4),
    (energy._get_platform, "device", "cuda"),
    (energy._evaluate_potential_energy, "device", "cuda"),
    (energy.compute_interaction_energy, "device", "cuda"),
    (energy.compute_interaction_energy, "ph", 7.4),
    (receptor_quality._receptor_energy, "device", "cuda"),
    (receptor_quality.compute_receptor_quality, "device", "cuda"),
]


SIGNATURE_IDS = [
    f"{function.__module__.rsplit('.', 1)[-1]}.{function.__name__}({parameter})"
    for function, parameter, _ in SIGNATURES
]


def test_the_constants_hold_the_values_users_see():
    assert _constants.DEFAULT_PH == 7.4
    assert _constants.DEFAULT_DEVICE == "cuda"


@pytest.mark.parametrize("function, parameter, expected", SIGNATURES, ids=SIGNATURE_IDS)
def test_signature_default_is_unchanged(function, parameter, expected):
    default = inspect.signature(function).parameters[parameter].default
    assert default == expected
    assert type(default) is type(expected)


def test_energy_flags_default_to_the_same_values():
    args = energy._build_parser().parse_args(["--input", "complex.cif"])
    assert args.ph == 7.4
    assert args.device == "cuda"


def test_energy_ph_help_still_names_the_default():
    parser = energy._build_parser()
    help_text = next(a.help for a in parser._actions if "--ph" in a.option_strings)
    assert help_text.endswith("(default: 7.4)")


def test_receptor_quality_flag_defaults_to_the_same_device(monkeypatch):
    seen = {}

    def record(*args, **kwargs):
        seen.update(kwargs)
        return {"error": "stopped by the test"}

    monkeypatch.setattr(receptor_quality, "compute_receptor_quality", record)
    monkeypatch.setattr("sys.argv", ["binding-metrics-receptor-quality", "--input", "x.pdb"])
    receptor_quality.main()
    assert seen["device"] == "cuda"


class TestNoModuleRepeatsTheLiteral:
    """No default of a signature, an argument parser or a hydrogen placement is written out."""

    VALUES = (7.4, "cuda")

    @classmethod
    def _is_value(cls, node):
        return (
            isinstance(node, ast.Constant)
            and not isinstance(node.value, bool)
            and node.value in cls.VALUES
            and type(node.value) in (float, str)
        )

    @classmethod
    def _repeats(cls, tree):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                defaults = [*node.args.defaults, *node.args.kw_defaults]
                for default in defaults:
                    if default is not None and cls._is_value(default):
                        yield default.lineno
            elif isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg in ("default", "pH", "ph", "device") and cls._is_value(
                        keyword.value
                    ):
                        yield keyword.value.lineno

    def test_the_scan_sees_a_literal_default(self):
        tree = ast.parse(
            "def f(ph: float = 7.4, device='cuda'): ...\nparser.add_argument(default=7.4)"
        )
        assert len(list(self._repeats(tree))) == 3

    def test_no_literal_default_in_src(self):
        found = []
        for path in sorted(SRC.rglob("*.py")):
            if path.name == "_constants.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "7.4" not in text and "cuda" not in text:
                continue
            found += [f"{path.relative_to(SRC)}:{line}" for line in self._repeats(ast.parse(text))]
        assert found == []
