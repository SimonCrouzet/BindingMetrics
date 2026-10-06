"""write_pose_report, the Markdown summary and the flat CSV of a pose-set result."""

import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from binding_metrics.metrics.registry import METRICS_BY_NAME, MetricSpec
from binding_metrics.poses import write_pose_report as exported_writer
from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.report import FORMATS, summary_html, summary_markdown, write_pose_report
from binding_metrics.poses.score import PoseSetResult, score_pose_set
from binding_metrics.testing import make_synthetic_poses

pytest.importorskip("biotite.structure")

METRICS = ("delta_sasa_static", "hbonds")


def _stub_nan(cif_path, binder_chain=None, receptor_chain=None):
    return {"value": float("nan"), "other": 1.0}


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_poses()


@pytest.fixture(scope="module")
def corrupt(tmp_path_factory):
    path = tmp_path_factory.mktemp("corrupt") / "corrupt.pdb"
    path.write_text("this is not a structure\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def result(synthetic):
    return score_pose_set(synthetic.poses, "B", "A", metrics=METRICS)


@pytest.fixture(scope="module")
def result_with_a_failure(synthetic, corrupt):
    """The synthetic set plus a pose whose id has markup in it, and one that cannot be read."""
    markup = DockedPose.from_atoms(
        synthetic.poses[0].atoms(), id='<b>pose & "quote"</b>', backend="backend_a"
    )
    poses = list(synthetic.poses) + [markup, DockedPose.from_file(corrupt, id="zz_corrupt")]
    return score_pose_set(poses, "B", "A", metrics=METRICS)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestWriter:
    @pytest.mark.parametrize("fmt", FORMATS)
    def test_it_writes_the_file_and_returns_its_path(self, result, tmp_path, fmt):
        path = write_pose_report(result, fmt, tmp_path / f"report.{fmt}")
        assert isinstance(path, Path) and path == tmp_path / f"report.{fmt}"
        assert path.stat().st_size > 0

    def test_the_formats_are_the_four_of_the_design(self):
        assert FORMATS == ("json", "csv", "md", "html")

    def test_json_is_the_default_and_the_path_may_be_a_string(self, result, tmp_path):
        path = write_pose_report(result, path=str(tmp_path / "r.json"))
        assert json.loads(read(path))["schema"] == "binding_metrics.pose_set_result"

    def test_the_name_is_exported_from_the_package(self):
        assert exported_writer is write_pose_report

    def test_missing_parent_directories_are_made_and_nothing_else_is_written(
        self, result, tmp_path, monkeypatch
    ):
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        target = tmp_path / "out" / "deeper" / "r.md"
        write_pose_report(result, "md", target)
        assert target.is_file() and list(work.iterdir()) == []

    def test_an_existing_file_is_replaced(self, result, tmp_path):
        target = tmp_path / "r.csv"
        target.write_text("old", encoding="utf-8")
        write_pose_report(result, "csv", target)
        assert read(target).startswith("id,kind,status")

    @pytest.mark.parametrize(
        "kwargs, error, message",
        [
            ({"fmt": "pdf"}, ValueError, "fmt must be one of"),
            ({"fmt": "json", "path": None}, ValueError, "path is required"),
        ],
    )
    def test_a_bad_call_raises(self, result, tmp_path, kwargs, error, message):
        kwargs = {"path": tmp_path / "x", **kwargs}
        with pytest.raises(error, match=message):
            write_pose_report(result, **kwargs)

    def test_the_result_must_be_a_result(self, tmp_path):
        with pytest.raises(TypeError, match="PoseSetResult"):
            write_pose_report([1, 2], "json", tmp_path / "x")
        with pytest.raises(ValueError, match="not a pose-set result"):
            write_pose_report({"schema": "other"}, "json", tmp_path / "x")

    def test_the_dictionary_read_back_from_json_is_accepted(self, result, tmp_path):
        data = json.loads(result.to_json())
        a = write_pose_report(data, "md", tmp_path / "a.md")
        b = write_pose_report(result, "md", tmp_path / "b.md")
        assert read(a) == read(b)

    def test_a_writer_does_not_change_the_result_or_what_the_call_returns(
        self, synthetic, tmp_path
    ):
        result = score_pose_set(synthetic.poses[:3], "B", "A", metrics=["hbonds"])
        before = result.to_json()
        for fmt in FORMATS:
            write_pose_report(result, fmt, tmp_path / f"r.{fmt}")
        result.summary()
        result.to_flat_rows()
        assert type(result) is PoseSetResult and result.to_json() == before

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_the_same_result_gives_the_same_bytes(self, result, tmp_path, fmt):
        a = write_pose_report(result, fmt, tmp_path / f"a.{fmt}").read_bytes()
        b = write_pose_report(result, fmt, tmp_path / f"b.{fmt}").read_bytes()
        assert a == b

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_the_files_are_utf8_and_use_unix_newlines(self, result, tmp_path, fmt):
        raw = write_pose_report(result, fmt, tmp_path / f"r.{fmt}").read_bytes()
        raw.decode("utf-8")
        assert b"\r" not in raw and raw.endswith(b"\n")


class TestJson:
    def test_it_is_the_canonical_dictionary(self, result, tmp_path):
        path = write_pose_report(result, "json", tmp_path / "r.json")
        assert json.loads(read(path)) == json.loads(result.to_json())
        assert PoseSetResult.from_dict(json.loads(read(path))).to_json() == result.to_json()

    def test_the_timing_can_be_left_out(self, result, tmp_path):
        path = write_pose_report(result, "json", tmp_path / "r.json", include_timing=False)
        assert "timing" not in json.loads(read(path))
        assert "timing" in json.loads(read(write_pose_report(result, "json", tmp_path / "t.json")))

    def test_a_nan_of_a_metric_is_kept_as_a_token_and_listed(
        self, synthetic, tmp_path, monkeypatch
    ):
        monkeypatch.setitem(
            METRICS_BY_NAME,
            "stub_nan",
            MetricSpec(
                name="stub_nan",
                import_path=f"{__name__}:_stub_nan",
                description="stub",
                input_type="static_structure",
                chain_mode="interface",
                peptide_chain_arg="binder_chain",
                receptor_chain_arg="receptor_chain",
                cost_class="static",
            ),
        )
        result = score_pose_set(synthetic.poses[:2], "B", "A", metrics=["stub_nan"])
        text = read(write_pose_report(result, "json", tmp_path / "r.json"))
        assert re.search(r'"value": NaN', text)
        assert json.loads(text)["nonfinite_fields"] == [
            "poses[0].stub_nan.value",
            "poses[1].stub_nan.value",
        ]


class TestCsv:
    def rows(self, path):
        with open(path, encoding="utf-8", newline="") as handle:
            return list(csv.reader(handle))

    def test_the_columns_are_the_flat_columns_and_every_row_has_them(self, result, tmp_path):
        table = self.rows(write_pose_report(result, "csv", tmp_path / "r.csv"))
        flat = result.to_flat_rows()
        assert table[0] == list(flat[0])
        assert len(table) == 1 + len(flat)
        assert {len(line) for line in table} == {len(table[0])}
        assert [line[0] for line in table[1:]] == [row["id"] for row in flat]

    def test_the_values_read_back_as_they_were(self, result, tmp_path):
        table = self.rows(write_pose_report(result, "csv", tmp_path / "r.csv"))
        header = table[0]
        flat = result.to_flat_rows()[0]
        line = dict(zip(header, table[1]))
        assert float(line["delta_sasa_static.delta_sasa"]) == flat["delta_sasa_static.delta_sasa"]
        assert int(line["hbonds.hbonds"]) == flat["hbonds.hbonds"]
        assert line["status"] == "ok" and line["reason"] == ""
        assert line["profile.closures_as_expected"] == "True"
        assert line["independent_of_design"] == ""

    def test_no_cell_holds_a_list_or_a_dictionary(self, result, tmp_path):
        table = self.rows(write_pose_report(result, "csv", tmp_path / "r.csv"))
        for line in table[1:]:
            for cell in line:
                assert not cell.startswith(("[", "{")), cell
        assert "frame.transform.translation[0]" in table[0]
        assert not any(name.startswith("residue_table") for name in table[0])

    def test_a_failed_pose_has_empty_cells_and_its_reason(self, result_with_a_failure, tmp_path):
        table = self.rows(write_pose_report(result_with_a_failure, "csv", tmp_path / "r.csv"))
        line = dict(zip(table[0], next(row for row in table if row[0] == "zz_corrupt")))
        assert line["status"] == "failed" and line["reason"].startswith("ValueError")
        assert line["hbonds.hbonds"] == "" and line["frame.pairing"] == ""

    def test_non_ascii_ids_and_commas_and_quotes_survive(self, synthetic, tmp_path):
        pose = DockedPose.from_atoms(
            synthetic.poses[0].atoms(), id='pose,é "x"\nline', backend="backend_a"
        )
        result = score_pose_set([pose, synthetic.poses[1]], "B", "A")
        table = self.rows(write_pose_report(result, "csv", tmp_path / "r.csv"))
        assert 'pose,é "x"\nline' in [line[0] for line in table[1:]]

    def test_the_list_limit_is_a_parameter(self, result, tmp_path):
        short = self.rows(write_pose_report(result, "csv", tmp_path / "a.csv", max_list_items=0))
        long = self.rows(write_pose_report(result, "csv", tmp_path / "b.csv"))
        assert "frame.transform.translation[0]" not in short[0]
        assert "frame.transform.translation[0]" in long[0]

    def test_nan_is_written_as_nan_and_none_as_empty(self):
        from binding_metrics.poses.report import _csv_cell

        assert [_csv_cell(v) for v in (float("nan"), float("inf"), -float("inf"), None)] == [
            "nan",
            "inf",
            "-inf",
            "",
        ]
        assert [_csv_cell(v) for v in (True, False, 3, 0.1, "x")] == [
            "True",
            "False",
            "3",
            "0.1",
            "x",
        ]


class TestMarkdown:
    def test_summary_is_a_string_and_the_writer_writes_it(self, result, tmp_path):
        text = result.summary()
        assert isinstance(text, str) and text == summary_markdown(result)
        assert read(write_pose_report(result, "md", tmp_path / "r.md")) == text

    def test_it_has_a_table_per_back_end_with_the_headline_values(self, result):
        text = result.summary()
        assert "### backend_a (3 poses)" in text and "### backend_b (3 poses)" in text
        header = "| id | seed | status | receptor RMSD (A) | delta_sasa_static.delta_sasa |"
        assert header in text and "hbonds.hbond_energy" in text
        rows_a = re.findall(r"^\| backend_a-s\d \|", text, flags=re.M)
        assert len(rows_a) >= 3
        line = next(
            line for line in text.splitlines() if line.startswith("| backend_a-s1 | 1 | ok")
        )
        value = result["poses"][0]["delta_sasa_static"]["delta_sasa"]
        assert f"{value:.4g}" in line

    def test_it_names_the_reference_the_counts_and_the_version(self, result):
        text = result.summary()
        assert "6 poses in 2 groups: 6 ok, 0 partial, 0 failed." in text
        assert "the first readable pose (backend_a-s1)" in text
        assert f"input key {result.input_key[:12]}" in text
        assert "Binder chain B, target chain A" in text

    def test_the_failed_poses_are_listed_with_their_reasons(self, result_with_a_failure):
        text = result_with_a_failure.summary()
        assert "## Failed poses" in text
        assert "| zz_corrupt | load | ValueError" in text
        assert "1 failed" in text

    def test_a_clean_set_says_that_no_pose_failed(self, result):
        assert "No pose failed." in result.summary()

    def test_the_agreement_the_aggregates_and_the_representatives_are_there(self, result):
        text = result.summary()
        for expected in (
            "### Peptide RMSD, no refit",
            "| ca | within backend_a | 3 |",
            "| ca | between backend_a\\|backend_b | 9 |",
            "### Clusters",
            "| 1.0 | 2 | 3, 3 |",
            "### Medoids",
            "- backend_a: backend_a-s2",
            "### Interface overlap (Jaccard index)",
            "## Aggregates",
            "| delta_sasa_static.delta_sasa | 3 / 3 |",
            "## Representatives",
            "Rule: medoid.",
            "- all: backend_a-s3",
        ):
            assert expected in text, expected

    def test_a_pipe_in_an_id_is_escaped_in_a_table(self, synthetic):
        pose = DockedPose.from_atoms(synthetic.poses[0].atoms(), id="a|b", backend="backend_a")
        text = score_pose_set([pose, synthetic.poses[1]], "B", "A").summary()
        assert "| a\\|b |" in text

    @pytest.mark.parametrize(
        "poses",
        ["one", "single_backend", "failed", "all_failed", "no_metrics"],
    )
    def test_summary_is_a_string_for_every_shape_of_result(self, synthetic, corrupt, poses):
        cases = {
            "one": (synthetic.poses[:1], ("hbonds",)),
            "single_backend": (synthetic.poses[:3], ("hbonds",)),
            "failed": (
                list(synthetic.poses[:2]) + [DockedPose.from_file(corrupt, id="bad")],
                ("hbonds",),
            ),
            "all_failed": ([DockedPose.from_file(corrupt, id="bad")], ("hbonds",)),
            "no_metrics": (synthetic.poses[:2], ()),
        }
        chosen, metrics = cases[poses]
        result = score_pose_set(chosen, "B", "A", metrics=metrics)
        for text in (result.summary(), summary_html(result)):
            assert isinstance(text, str) and text.strip()

    def test_the_summary_says_that_the_spread_is_of_the_alpha_carbons_whatever_the_kind(
        self, synthetic
    ):
        result = score_pose_set(synthetic.poses, "B", "A", options={"agreement": {"kind": "heavy"}})
        text = result.summary()
        assert (
            "the clusters and the medoids use the heavy atoms (complete linkage), "
            "the spread is of the alpha carbons." in text
        )
        assert "use the heavy atoms (complete linkage)." not in text
        assert "the medoids and the spread" not in text
        assert "### Mean spread of the alpha carbons per position" in text

    def test_the_html_says_the_same(self, synthetic):
        result = score_pose_set(synthetic.poses, "B", "A", options={"agreement": {"kind": "heavy"}})
        page = summary_html(result)
        assert (
            "the clusters and the medoids use the heavy atoms (complete linkage), "
            "the spread is of the alpha carbons." in page
        )

    def test_one_pose_and_a_single_back_end_say_why_there_is_no_between_block(self, synthetic):
        result = score_pose_set(synthetic.poses[:3], "B", "A")
        assert "Between groups: only one back-end." in result.summary()

    def test_a_result_without_agreement_says_so(self, corrupt):
        result = score_pose_set([DockedPose.from_file(corrupt, id="bad")], "B", "A")
        text = result.summary()
        assert "Agreement was not computed" in text and "no pose could be read" in text
        assert "| bad | load |" in text

    def test_a_metric_that_reported_a_reason_is_listed_as_a_missing_value(
        self, synthetic, monkeypatch
    ):
        monkeypatch.setitem(
            METRICS_BY_NAME,
            "stub_reason",
            MetricSpec(
                name="stub_reason",
                import_path=f"{__name__}:_stub_reason",
                description="stub",
                input_type="static_structure",
                chain_mode="interface",
                peptide_chain_arg="binder_chain",
                receptor_chain_arg="receptor_chain",
                cost_class="static",
            ),
        )
        result = score_pose_set(synthetic.poses[:2], "B", "A", metrics=["stub_reason"])
        text = result.summary()
        assert "## Poses with a missing value" in text
        assert "| backend_a-s1 | stub_reason | half of it |" in text
        assert "stub_reason.value" in text  # the headline of a stub is its first number

    def test_the_warnings_and_the_nonfinite_fields_are_reported(self, synthetic, monkeypatch):
        monkeypatch.setitem(
            METRICS_BY_NAME,
            "stub_nan",
            MetricSpec(
                name="stub_nan",
                import_path=f"{__name__}:_stub_nan",
                description="stub",
                input_type="static_structure",
                chain_mode="interface",
                peptide_chain_arg="binder_chain",
                receptor_chain_arg="receptor_chain",
                cost_class="static",
            ),
        )
        result = score_pose_set(synthetic.poses[:2], "B", "A", metrics=["stub_nan"])
        text = result.summary()
        assert "## Warnings" in text and "2 value(s) are NaN or infinite" in text
        assert "| backend_a-s1 | 1 | ok | " in text and "NaN" in text


def _stub_reason(cif_path, binder_chain=None, receptor_chain=None):
    return {"value": 1.0, "reason": "half of it"}


class Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[str] = []
        self.attributes: list[tuple[str, str, str | None]] = []
        self.text: list[str] = []
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        for name, value in attrs:
            self.attributes.append((tag, name, value))
        if tag not in ("meta", "br", "link", "img", "input"):
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with {self.stack[-3:]}")
        else:
            self.stack.pop()

    def handle_data(self, data):
        self.text.append(data)


@pytest.fixture(scope="module")
def page(result):
    return summary_html(result)


@pytest.fixture(scope="module")
def parsed(page):
    collector = Collector()
    collector.feed(page)
    collector.close()
    return collector


class TestHtml:
    def test_it_is_a_complete_utf8_document(self, page):
        assert page.startswith("<!doctype html>\n")
        assert '<meta charset="utf-8">' in page
        assert "<title>Pose set</title>" in page and page.rstrip().endswith("</html>")

    def test_the_tags_are_balanced(self, parsed):
        assert parsed.errors == [] and parsed.stack == []

    def test_nothing_is_fetched_from_a_network(self, page, parsed):
        assert not re.search(r"https?://|//cdn|@import|url\(", page)
        forbidden = {"script", "link", "img", "iframe", "object", "embed", "video", "audio"}
        assert forbidden.isdisjoint(parsed.tags)
        assert not [a for a in parsed.attributes if a[1] in ("src", "href", "srcset", "action")]
        assert not [a for a in parsed.attributes if a[1].startswith("on")]

    def test_the_content_is_the_markdown_content(self, page, parsed, result):
        text = " ".join(parsed.text)
        for expected in ("backend_a (3 poses)", "No pose failed.", "Medoids", "backend_a-s2"):
            assert expected in text
        # one table in the page for each table of the Markdown
        assert page.count("<table>") == len(re.findall(r"^\|---", result.summary(), flags=re.M))

    def test_every_text_is_escaped(self, result_with_a_failure):
        page = summary_html(result_with_a_failure)
        assert "<b>pose" not in page
        assert "&lt;b&gt;pose &amp; &quot;quote&quot;&lt;/b&gt;" in page
        parsed = Collector()
        parsed.feed(page)
        assert "b" not in parsed.tags
        assert parsed.errors == []

    def test_the_failed_pose_and_its_reason_are_in_the_page(self, result_with_a_failure):
        page = summary_html(result_with_a_failure)
        assert "zz_corrupt" in page and "ValueError" in page

    def test_the_writer_writes_the_same_page(self, result, tmp_path, page):
        assert read(write_pose_report(result, "html", tmp_path / "r.html")) == page


class TestFlatRowsOfAFailure:
    def test_the_failed_row_is_in_the_flat_rows(self, result_with_a_failure):
        flat = result_with_a_failure.to_flat_rows()
        ids = [row["id"] for row in flat]
        assert ids == sorted(ids) and "zz_corrupt" in ids
        assert len({tuple(row) for row in flat}) == 1
