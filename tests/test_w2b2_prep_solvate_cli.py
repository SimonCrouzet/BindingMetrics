"""``binding-metrics-prep`` and ``binding-metrics-solvate`` keep stdout a JSON document.

Scripts parse that JSON, so log records below WARNING must not reach stdout even
though the library now logs its progress instead of printing it.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

from binding_metrics.core.system import HAS_PDBFIXER
from binding_metrics.protocols import prep as prep_cli
from binding_metrics.protocols import solvate as solvate_cli
from binding_metrics.utils import _CurrentStreamHandler

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

requires_pdbfixer = pytest.mark.skipif(not HAS_PDBFIXER, reason="pdbfixer not installed")

PROGRESS_LINE = "  Repaired 1 wrong-side Cα hydrogen(s): ALA1/A"


@pytest.fixture(autouse=True)
def package_logging_restored():
    """Remove the console handlers ``main()`` installs, and restore the level."""
    package_logger = logging.getLogger("binding_metrics")
    saved_level = package_logger.level
    yield package_logger
    for name in ("binding_metrics", "__main__"):
        target = logging.getLogger(name)
        for handler in [h for h in target.handlers if isinstance(h, _CurrentStreamHandler)]:
            target.removeHandler(handler)
    package_logger.setLevel(saved_level)


def _log_progress():
    logging.getLogger("binding_metrics.core.system").info(PROGRESS_LINE)


@requires_pdbfixer
class TestPrepStdout:
    def test_info_records_do_not_reach_the_json_on_stdout(self, monkeypatch, capsys, tmp_path):
        import binding_metrics.core.system as system

        real_prep = system.prep_structure

        def prep_that_logs(topology, positions, **kwargs):
            _log_progress()
            return real_prep(topology, positions, **kwargs)

        monkeypatch.setattr(system, "prep_structure", prep_that_logs)
        monkeypatch.setattr(
            sys,
            "argv",
            ["binding-metrics-prep", "-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "out.pdb")],
        )
        prep_cli.main()
        out = capsys.readouterr().out
        summary = json.loads(out)
        assert summary["n_chains"] == 2
        assert PROGRESS_LINE not in out

    def test_main_configures_the_package_logger_at_warning(self, monkeypatch, tmp_path):
        import binding_metrics.core.system as system

        monkeypatch.setattr(system, "prep_structure", lambda top, pos, **kwargs: (top, pos))
        monkeypatch.setattr(
            sys,
            "argv",
            ["binding-metrics-prep", "-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "out.pdb")],
        )
        prep_cli.main()
        assert logging.getLogger("binding_metrics").level == logging.WARNING


@requires_pdbfixer
class TestSolvateStdout:
    def test_info_records_do_not_reach_the_json_on_stdout(
        self, monkeypatch, capsys, tmp_path, prepped_example_pdb
    ):
        import binding_metrics.core.system as system

        real_solvate = system.solvate

        def solvate_that_logs(*args, **kwargs):
            _log_progress()
            return real_solvate(*args, **kwargs)

        monkeypatch.setattr(system, "solvate", solvate_that_logs)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "binding-metrics-solvate",
                "-i",
                str(prepped_example_pdb),
                "-o",
                str(tmp_path / "solvated.pdb"),
                "--padding",
                "0.4",
            ],
        )
        solvate_cli.main()
        out = capsys.readouterr().out
        summary = json.loads(out)
        assert summary["n_waters"] > 0
        assert PROGRESS_LINE not in out
