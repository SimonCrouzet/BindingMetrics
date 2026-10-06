"""What docs/pose_sets.md says about the pose-set layer is true.

The page is read from the file: the signature, the option tables with their defaults, the keys
of the result and of a row, the columns of the descriptors, the writers and the output of the
worked examples are compared with the code, so a rename in the code or a stale line in the page
fails here.
"""

import ast
import contextlib
import importlib
import inspect
import io
import re
from pathlib import Path

import pytest

from binding_metrics.poses import (
    DockedPose,
    FrameOptions,
    PoseSetResult,
    Reference,
    score_pose_set,
    write_pose_report,
)
from binding_metrics.poses.aggregate import (
    DEFAULT_EXCLUDED_BLOCKS,
    DEFAULT_STATISTICS,
    ENTRY_KEYS,
    RESERVED_GROUP_KEYS,
)
from binding_metrics.poses.descriptors import PARTNER_COLUMNS, SET_COLUMNS, TABLE_COLUMNS
from binding_metrics.poses.report import FORMATS
from binding_metrics.poses.score import (
    RESERVED_ROW_NAMES,
    RESULT_KEYS,
    SCHEMA,
    SCHEMA_VERSION,
    AggregationOptions,
    AgreementOptions,
    DescriptorOptions,
    RepresentativeOptions,
)

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "pose_sets.md"
TEXT = DOC.read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
METRICS = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
POSES = (ROOT / "docs" / "poses.md").read_text(encoding="utf-8")
README_HEADING = "### Pose sets: several seeds and models of one construct"


def _section(heading: str) -> str:
    """The text from ``heading`` (a line of the page) to the next heading of the same level."""
    start = TEXT.index(f"\n{heading}\n") + 1
    level = heading.split(" ", 1)[0]
    following = re.compile(rf"^{re.escape(level)} ", re.M).search(TEXT, start + len(heading))
    return TEXT[start : following.start() if following else len(TEXT)]


def _table_after(text: str, marker: str) -> dict[str, list[str]]:
    """The rows of the table that follows ``marker``, by the backticked name of the first cell."""
    lines = text[text.index(marker) :].splitlines()
    rows: dict[str, list[str]] = {}
    started = False
    for line in lines[1:]:
        if line.startswith("|"):
            started = True
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            names = re.findall(r"`([^`]+)`", cells[0])
            for name in names:
                rows[name] = cells[1:]
        elif started:
            break
    return rows


def _code_blocks(text: str) -> list[tuple[str, str]]:
    blocks, lines, language, buffer = [], text.splitlines(), None, []
    for line in lines:
        if line.startswith("```"):
            if language is None:
                language, buffer = line[3:].strip(), []
            else:
                blocks.append((language, "\n".join(buffer)))
                language = None
        elif language is not None:
            buffer.append(line)
    return blocks


def _examples(text: str) -> list[tuple[str, str]]:
    """``(code, printed output)`` for every python block that is followed by a text block."""
    blocks = _code_blocks(text)
    return [
        (code, blocks[i + 1][1])
        for i, (language, code) in enumerate(blocks[:-1])
        if language == "python" and blocks[i + 1][0] == "text"
    ]


def _run(code: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "<docs/pose_sets.md>", "exec"), {"__name__": "__docs_example__"})  # noqa: S102
    return out.getvalue().strip()


def _slug(heading: str) -> str:
    text = heading.lstrip("#").strip().replace("`", "").lower()
    return re.sub(r"\s", "-", re.sub(r"[^a-z0-9 _-]", "", text))


# ---------------------------------------------------------------------------- the call


def _default_text(value) -> str:
    if value is DEFAULT_STATISTICS:
        return "DEFAULT_STATISTICS"
    return repr(value)


def test_the_signature_in_the_docs_is_the_real_one():
    block = next(
        code for language, code in _code_blocks(_section("## The call")) if language == "python"
    )
    tree = ast.parse("def " + block + ": pass").body[0].args
    documented = [
        (arg.arg, ast.unparse(default) if default is not None else None)
        for arg, default in zip(
            tree.args + tree.kwonlyargs,
            [None] * (len(tree.args) - len(tree.defaults))
            + tree.defaults
            + [d for d in tree.kw_defaults],
        )
    ]
    real = [
        (name, None if p.default is inspect.Parameter.empty else _default_text(p.default))
        for name, p in inspect.signature(score_pose_set).parameters.items()
    ]
    assert documented == real


def test_every_argument_has_a_row_in_the_table():
    rows = _table_after(_section("## The call"), "| Argument | Meaning |")
    assert set(inspect.signature(score_pose_set).parameters) == set(rows)


@pytest.mark.parametrize(
    "marker, cls",
    [
        ('(`options["frame"]`, class `FrameOptions`)', FrameOptions),
        ('(`options["agreement"]`, class `AgreementOptions`)', AgreementOptions),
        ('(`options["aggregation"]`, class `AggregationOptions`)', AggregationOptions),
        ('(`options["descriptors"]`, class `DescriptorOptions`)', DescriptorOptions),
    ],
)
def test_the_option_tables_hold_every_option_and_its_default(marker, cls):
    rows = _table_after(TEXT, marker)
    expected = {
        name: tuple(value) if isinstance(value, list) else value
        for name, value in cls().to_dict().items()
    }
    assert set(rows) == set(expected), cls.__name__
    for name, value in expected.items():
        assert ast.literal_eval(rows[name][0].strip("`")) == value, (cls.__name__, name)


def test_the_representative_options_are_documented_with_their_defaults():
    defaults = RepresentativeOptions().to_dict()
    assert defaults == {"confidence_key": "ranking_score", "confidence_direction": None}
    text = _section("## The representative pose")
    assert '`confidence_key` (`"ranking_score"`)' in text
    assert "`confidence_direction` (`None`)" in text


def test_the_statistics_and_the_entry_keys_are_the_ones_of_the_code():
    text = _section("## Aggregation")
    for name in DEFAULT_STATISTICS:
        assert f"`{name}`" in text
    for name in ENTRY_KEYS:
        assert f"`{name}`" in text
    for name in DEFAULT_EXCLUDED_BLOCKS:
        assert f"`{name}`" in text
    for name in RESERVED_GROUP_KEYS:
        assert f"`{name}`" in _section("## Poses, groups and the pose kinds")


# ---------------------------------------------------------------------------- the result


def test_the_top_level_keys_are_documented_in_order():
    rows = _table_after(_section("## The result"), "| Key | Content |")
    documented = [name for name in rows if name in RESULT_KEYS]
    assert documented == list(RESULT_KEYS)
    assert f'`"{SCHEMA}"`' in TEXT
    assert rows["schema_version"][0] == f"`{SCHEMA_VERSION}`"


def test_the_row_keys_and_the_reserved_names_are_documented():
    section = _section("## The result")
    reserved_sentence = section[section.index("**A metric block next to the reserved names.**") :]
    for name in RESERVED_ROW_NAMES:
        assert f"`{name}`" in reserved_sentence, name


def test_the_descriptor_columns_are_documented():
    section = _section("## Reference-site descriptors")
    for column in (*SET_COLUMNS, *PARTNER_COLUMNS, *TABLE_COLUMNS):
        assert f"`{column}`" in section, column


def test_the_frame_block_keys_are_documented(synthetic_result):
    paragraph = _section("## The common frame")
    block = synthetic_result["poses"][0]["frame"]
    for key in block:
        assert f"`{key}`" in paragraph, key


def test_the_writers_are_the_ones_of_the_code():
    section = _section("## Writers")
    for fmt in FORMATS:
        assert f"| `{fmt}` |" in section
    assert "max_list_items=16" in section


# ---------------------------------------------------------------------------- the examples


@pytest.fixture(scope="module")
def synthetic_result():
    from binding_metrics.testing import make_synthetic_poses

    synthetic = make_synthetic_poses()
    return score_pose_set(synthetic.poses, metrics=["interface"])


def test_the_worked_examples_print_what_the_page_shows():
    examples = _examples(_section("## A worked example"))
    assert len(examples) == 2
    for code, shown in examples:
        assert _run(code) == shown.strip()


def test_the_markdown_summary_opens_as_the_page_shows(synthetic_result):
    shown = next(
        code for language, code in _code_blocks(_section("## Writers")) if language == "text"
    )
    assert "\n".join(synthetic_result.summary().splitlines()[: len(shown.splitlines())]) == shown


def test_the_engine_side_example_runs_and_prints_what_the_page_shows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ((code, shown),) = _examples(_section("## Using it from the engine side"))
    assert _run(code) == shown.strip()
    assert (tmp_path / "results" / "design_0042.json").is_file()


def test_the_pose_that_lost_its_peptide_is_a_failed_row_in_the_count():
    from binding_metrics.testing import make_synthetic_poses

    synthetic = make_synthetic_poses()
    receptor = synthetic.reference.atoms()
    lost = DockedPose.from_atoms(receptor[receptor.chain_id == "A"], id="x", program="backend_b")
    result = score_pose_set([*synthetic.poses, lost], metrics=["interface"])
    assert isinstance(result, PoseSetResult)
    assert result.row("x")["status"] == "failed"
    entry = result["aggregates"]["backend_b"]["interface.delta_sasa"]
    assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (3, 4, False)


# ---------------------------------------------------------------------------- links


def test_the_links_and_anchors_of_the_page_resolve():
    anchors = {_slug(line) for line in TEXT.splitlines() if re.match(r"#{1,6} ", line)}
    for target in re.findall(r"\]\(([^)]+)\)", TEXT):
        if target.startswith("#"):
            assert target[1:] in anchors, target
        elif not target.startswith("http"):
            path, _, fragment = target.partition("#")
            assert (DOC.parent / path).is_file(), target
            if fragment and path.endswith(".md"):
                other = (DOC.parent / path).read_text(encoding="utf-8")
                slugs = {_slug(line) for line in other.splitlines() if re.match(r"#{1,6} ", line)}
                assert fragment in slugs, target


# ---------------------------------------------------------------------------- the other pages


def _headings_of(text: str) -> set[str]:
    return {_slug(line) for line in text.splitlines() if re.match(r"#{1,6} ", line)}


def test_the_other_pages_point_at_the_pose_set_page():
    assert "[`docs/pose_sets.md`](docs/pose_sets.md)" in README
    assert "(pose_sets.md)" in METRICS
    assert "(pose_sets.md)" in POSES
    assert "pose_sets.md" in README.split("## Documentation", 1)[1]


def test_every_link_to_the_pose_set_page_resolves():
    slugs = _headings_of(TEXT)
    for label, text in (("README.md", README), ("metrics.md", METRICS), ("poses.md", POSES)):
        for target in re.findall(r"\]\(([^)]*pose_sets\.md[^)]*)\)", text):
            _, _, fragment = target.partition("#")
            assert not fragment or fragment in slugs, (label, target)


def test_the_readme_section_sits_between_the_evobind_and_the_batch_sections():
    headings = [line for line in README.splitlines() if line.startswith("### ")]
    position = headings.index(README_HEADING)
    assert headings[position - 1] == "### EvoBind scoring"
    assert headings[position + 1] == "### Batch scoring"


def test_the_readme_example_uses_names_and_keywords_that_exist():
    section = README.split(README_HEADING, 1)[1].split("\n### ", 1)[0]
    code = next(code for language, code in _code_blocks(section) if language == "python")
    calls = {"score_pose_set": score_pose_set, "Reference": Reference}
    calls["write_pose_report"] = write_pose_report
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom):
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), (node.module, alias.name)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) in calls:
            parameters = inspect.signature(calls[node.func.id]).parameters
            assert len(node.args) <= len(parameters), node.func.id
            for keyword in node.keywords:
                assert keyword.arg in parameters, (node.func.id, keyword.arg)
    from binding_metrics.predictors.record import SampleRef

    assert {"seed_index", "sample"} <= set(inspect.signature(SampleRef).parameters)
    assert "list_samples" in dir(
        importlib.import_module("binding_metrics.predictors.base").PredictionParser
    )


def test_the_page_names_the_cost_classes_the_registry_declares():
    from binding_metrics.metrics.registry import METRICS_BY_NAME

    classes = {spec.cost_class for spec in METRICS_BY_NAME.values()}
    section = _section("## The call")
    for name in ("static", "model", "structural", "md"):
        assert name in classes
        assert f"`{name}`" in section
