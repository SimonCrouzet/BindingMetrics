"""Strict JSON export of the pipeline report: ``write_report``, ``-report`` and ``-run`` (#220).

The default ``nonfinite="keep"`` writes the report as it always was; ``"null"`` and
``--nonfinite null`` write ``null`` for every NaN and infinity, list the paths in
``nonfinite_fields`` and give text that a strict parser accepts.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.cli import run
from binding_metrics.protocols import report as pipeline_report
from binding_metrics.protocols.report import _json_default, write_report
from tests.test_feat_c_support import EXAMPLE_1YCR
from tests.test_t3_support import (  # noqa: F401  (the fixture is used by name)
    blank,
    read,
    restore_package_logger,
    strict_loads,
    value_at,
)

pytestmark = pytest.mark.usefixtures("restore_package_logger")

#: The paths ``pipeline_results`` lists, in the order of the traversal.
LISTED = [
    "relax.rmsd_md_final",
    "geometry.per_residue[1].phi",
    "geometry.series[1]",
    "geometry.series[2]",
    "geometry.pair[0]",
    "electrostatics.coulomb_energy_kJ",
]


def pipeline_results(sample_id="sample"):
    return {
        "sample_id": sample_id,
        "relax": {
            "rmsd_md_final": float("nan"),
            "energy": np.float32(-5.5),
            "count": np.int64(3),
            "flag": np.bool_(True),
            "path": Path("out/relaxed.cif"),
        },
        "geometry": {
            "per_residue": [{"phi": 1.0}, {"phi": np.float32("inf")}],
            "series": np.array([1.0, np.nan, -np.inf]),
            "pair": (float("nan"), 2.0),
        },
        "electrostatics": {"coulomb_energy_kJ": -np.inf, "ok": 1.25},
        "note": "NaN",
    }


class TestWriteReport:
    def test_the_default_json_is_the_text_json_dumps_gives(self, tmp_path):
        results = pipeline_results()
        text = read(write_report(results, tmp_path / "a", "sample"))
        assert text == json.dumps(results, indent=2, default=_json_default)
        assert results["nonfinite_fields"] == LISTED
        assert '"rmsd_md_final": NaN' in text and '"coulomb_energy_kJ": -Infinity' in text
        explicit = write_report(pipeline_results(), tmp_path / "b", "sample", nonfinite="keep")
        assert read(explicit) == text

    def test_null_writes_strict_json_and_lists_the_paths(self, tmp_path):
        results = pipeline_results()
        text = read(write_report(results, tmp_path, "sample", nonfinite="null"))
        data = strict_loads(text)
        assert data["nonfinite_fields"] == LISTED
        for path in LISTED:
            assert value_at(data, path) is None
        assert data["relax"]["energy"] == -5.5 and isinstance(data["relax"]["energy"], float)
        assert data["relax"]["count"] == 3 and data["relax"]["flag"] is True
        assert data["relax"]["path"] == str(Path("out/relaxed.cif"))
        assert data["geometry"]["series"] == [1.0, None, None]
        assert data["geometry"]["pair"] == [None, 2.0]
        assert data["electrostatics"]["ok"] == 1.25 and data["note"] == "NaN"

    def test_null_changes_no_other_value(self, tmp_path):
        kept = json.loads(read(write_report(pipeline_results(), tmp_path / "a", "sample")))
        nulled = strict_loads(
            read(write_report(pipeline_results(), tmp_path / "b", "sample", nonfinite="null"))
        )
        assert nulled == blank(kept)

    def test_the_results_of_the_caller_keep_their_values(self, tmp_path):
        results = pipeline_results()
        write_report(results, tmp_path, "sample", nonfinite="null")
        assert math.isnan(results["relax"]["rmsd_md_final"])
        assert results["nonfinite_fields"] == LISTED

    def test_writing_a_null_report_again_gives_the_same_text(self, tmp_path):
        first = read(write_report(pipeline_results(), tmp_path / "a", "sample", nonfinite="null"))
        again = read(write_report(strict_loads(first), tmp_path / "b", "sample", nonfinite="null"))
        assert again == first

    def test_a_report_without_a_nonfinite_value_lists_none(self, tmp_path):
        results = {"sample_id": "s", "relax": {"energy": -5.0}}
        data = strict_loads(read(write_report(results, tmp_path, "s", nonfinite="null")))
        assert data["nonfinite_fields"] == [] and data["relax"] == {"energy": -5.0}

    @pytest.mark.parametrize("fmt", ["json", "csv"])
    def test_the_csv_and_the_summary_do_not_depend_on_the_option(self, tmp_path, fmt):
        for mode in ("keep", "null"):
            write_report(
                pipeline_results(), tmp_path / mode, "sample", fmt=fmt, summary=True, nonfinite=mode
            )
        names = ["sample_report.md"] + (["sample_results.csv"] if fmt == "csv" else [])
        for name in names:
            assert read(tmp_path / "keep" / name) == read(tmp_path / "null" / name)

    def test_another_value_writes_nothing(self, tmp_path):
        with pytest.raises(ValueError, match=r"keep.*null"):
            write_report(pipeline_results(), tmp_path / "new", "sample", nonfinite="drop")
        assert not (tmp_path / "new").exists()


class TestReportCommand:
    @staticmethod
    def _run(monkeypatch, tmp_path, *extra):
        source = tmp_path / "in"
        write_report(pipeline_results(), source, "sample")
        argv = ["binding-metrics-report", "--results", str(source / "sample_results.json")]
        monkeypatch.setattr(sys, "argv", [*argv, "-o", str(tmp_path / "out"), *extra])
        pipeline_report.main()
        return read(tmp_path / "out" / "sample_results.json")

    def test_the_default_keeps_the_tokens(self, monkeypatch, tmp_path):
        text = self._run(monkeypatch, tmp_path)
        assert '"rmsd_md_final": NaN' in text

    def test_the_flag_writes_null(self, monkeypatch, tmp_path):
        data = strict_loads(self._run(monkeypatch, tmp_path, "--nonfinite", "null"))
        assert data["nonfinite_fields"] == LISTED
        assert data["relax"]["rmsd_md_final"] is None

    def test_a_null_report_is_read_and_written_again(self, monkeypatch, tmp_path):
        first = self._run(monkeypatch, tmp_path / "one", "--nonfinite", "null")
        source = tmp_path / "two" / "in"
        source.mkdir(parents=True)
        (source / "sample_results.json").write_text(first, encoding="utf-8")
        argv = ["binding-metrics-report", "--results", str(source / "sample_results.json")]
        monkeypatch.setattr(
            sys, "argv", [*argv, "-o", str(tmp_path / "two" / "out"), "--nonfinite", "null"]
        )
        pipeline_report.main()
        assert read(tmp_path / "two" / "out" / "sample_results.json") == first

    def test_another_choice_is_a_usage_error(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as stop:
            self._run(monkeypatch, tmp_path, "--nonfinite", "drop")
        assert stop.value.code == 2
        assert "invalid choice: 'drop'" in capsys.readouterr().err

    def test_the_help_names_the_flag_and_its_default(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["binding-metrics-report", "--help"])
        with pytest.raises(SystemExit):
            pipeline_report.main()
        text = " ".join(capsys.readouterr().out.split())
        assert "--nonfinite {keep,null}" in text and "(default: keep)" in text


class TestRunCommand:
    @staticmethod
    def _main(monkeypatch, tmp_path, *extra):
        monkeypatch.setattr(run, "run_pipeline", lambda **kwargs: pipeline_results())
        argv = ["binding-metrics-run", "-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "o")]
        monkeypatch.setattr(sys, "argv", [*argv, *extra])
        run.main()
        return read(tmp_path / "o" / f"{EXAMPLE_1YCR.stem}_results.json")

    def test_the_default_keeps_the_tokens(self, monkeypatch, tmp_path):
        assert '"rmsd_md_final": NaN' in self._main(monkeypatch, tmp_path)

    def test_the_flag_writes_null(self, monkeypatch, tmp_path):
        data = strict_loads(self._main(monkeypatch, tmp_path, "--nonfinite", "null"))
        assert data["nonfinite_fields"] == LISTED
        assert data["electrostatics"]["coulomb_energy_kJ"] is None

    def test_the_flag_is_read_from_a_config_file(self, monkeypatch, tmp_path):
        config = tmp_path / "run.toml"
        config.write_text('nonfinite = "null"\n', encoding="utf-8")
        strict_loads(self._main(monkeypatch, tmp_path, "--config", str(config)))

    def test_the_help_names_the_flag_and_its_default(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["binding-metrics-run", "--help"])
        with pytest.raises(SystemExit):
            run.main()
        text = " ".join(capsys.readouterr().out.split())
        assert "--nonfinite {keep,null}" in text and "(default: keep)" in text
