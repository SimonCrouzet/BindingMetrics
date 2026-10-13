"""The text about custom weights names things that exist (``--prediction-weights``)."""

import inspect
from pathlib import Path

from binding_metrics.capabilities import preflight
from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.weights import CACHE_FILE
from tests.test_feat_c_docs import _capture_parser

ROOT = Path(__file__).parent.parent
METRICS = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
PREFLIGHT = (ROOT / "docs" / "preflight.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_documented_signature_of_preflight_is_the_real_one():
    line = next(
        text for text in PREFLIGHT.splitlines() if text.startswith("`preflight(profile, metrics")
    )
    documented = line[line.index("(") + 1 : line.index(")`")]
    parts = []
    for name, parameter in inspect.signature(preflight).parameters.items():
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and "*" not in parts:
            parts.append("*")
        parts.append(
            name
            if parameter.default is inspect.Parameter.empty
            else f"{name}={parameter.default!r}"
        )
    # the positional part is written without defaults in the docs; the keywords must match
    keywords = lambda text: text.replace("'", '"').split("*, ", 1)[1]  # noqa: E731
    assert keywords(documented) == keywords(", ".join(parts))


def test_the_runner_attributes_the_docs_name_exist_with_the_documented_defaults():
    assert PredictionRunner.supports_custom_weights is False
    assert PredictionRunner.weights_kind == "file"
    assert callable(PredictionRunner.check_weights)
    for text in ("`supports_custom_weights`", "`weights_kind`", "check_weights(request)"):
        assert text in METRICS


def test_the_documented_cache_file_is_the_real_one():
    assert f"`<store root>/{CACHE_FILE}`" in METRICS


def test_the_option_of_both_commands_is_documented_in_the_three_places(monkeypatch):
    for module in ("binding_metrics.cli.run", "binding_metrics.cli.batch"):
        parser = _capture_parser(module, monkeypatch)
        action = parser._option_string_actions["--prediction-weights"]
        usage = f"`--prediction-weights {action.metavar}`"
        for name, text in (
            ("docs/metrics.md", METRICS),
            ("docs/preflight.md", PREFLIGHT),
            ("README.md", README),
        ):
            assert usage in text, f"{usage} is not in {name} ({module})"
