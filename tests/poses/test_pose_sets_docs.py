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
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
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


# ---------------------------------------------------------------------------- pickling


def test_the_objects_the_page_says_pickle_do_pickle():
    import copy
    import pickle

    from binding_metrics.capabilities import Capabilities
    from binding_metrics.poses import Partner, ResidueKey, superpose_receptor
    from binding_metrics.testing import bundled_complex_path, make_synthetic_poses

    path = bundled_complex_path()
    reference = Reference(
        path,
        "A",
        "B",
        residue_sets={"core": [("A", 51), ("A", 54)]},
        partners={"crystal_peptide": Partner(path, chains=["B"])},
    )
    pose = make_synthetic_poses().poses[1]
    frame = superpose_receptor(pose, reference, receptor_chain="A")
    capabilities = Capabilities(closures={"none"}, reasons={"closures": "a linear binder only"})
    for original in (reference, reference.partners["crystal_peptide"], frame, capabilities):
        for clone in (pickle.loads(pickle.dumps(original)), copy.deepcopy(original)):
            assert type(clone) is type(original)
    assert pickle.loads(pickle.dumps(reference)).content_hash() == reference.content_hash()
    assert pickle.loads(pickle.dumps(frame)).transform.to_dict() == frame.transform.to_dict()
    assert pickle.loads(pickle.dumps(FrameOptions())) == FrameOptions()
    assert pickle.loads(pickle.dumps(ResidueKey("A", 52, "A"))) == ResidueKey("A", 52, "A")
    for name in ("Reference", "Partner", "ResidueKey", "FrameOptions", "FrameResult", "Transform"):
        assert f"`{name}`" in _section("## Reference")
    assert "`Capabilities`" in _section("## Parallelism")


# ---------------------------------------------------------------------------- the changelog


def _changelog_section(heading: str) -> str:
    unreleased = CHANGELOG.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
    return unreleased.split(f"\n{heading}\n", 1)[1].split("\n### ", 1)[0]


#: module -> the names that the Added section of the changelog says it holds.
POSE_LAYER_NAMES = {
    "binding_metrics.poses.reference": (
        "Reference",
        "Partner",
        "ResidueKey",
        "normalize_references",
    ),
    "binding_metrics.poses.frame": (
        "FrameOptions",
        "FrameResult",
        "superpose_receptor",
        "place_in_reference_frame",
        "renumber_to_reference",
    ),
    "binding_metrics.poses.agreement": (
        "pairwise_peptide_rmsd",
        "pair_statistics",
        "complete_linkage_clusters",
        "medoid",
        "medoids",
        "jaccard_matrix",
        "interface_jaccard",
        "contact_map_jaccard",
        "heavy_atom_contacts",
        "interface_residues",
        "positional_spread",
    ),
    "binding_metrics.poses.aggregate": ("select_representatives", "rank_by"),
    "binding_metrics.poses.descriptors": (
        "residue_set_descriptors",
        "partner_descriptors",
        "interface_residue_table",
        "interface_areas",
        "sphere_overlap_volume",
    ),
    "binding_metrics.poses.reconcile": (
        "gap_between_backends",
        "consensus_interface",
        "compare_consensus_interfaces",
        "rank_agreement",
        "hotspot_consistency",
    ),
    "binding_metrics.poses.score": (
        "score_pose_set",
        "PoseSetResult",
        "PoseSetOptions",
        "DescriptorOptions",
        "PoseScoringError",
    ),
    "binding_metrics.poses.report": ("write_pose_report",),
    "binding_metrics.testing.synthetic": ("make_synthetic_poses",),
    "binding_metrics.core.residues": ("residue_number", "residue_label"),
}


def test_the_unreleased_section_comes_first_and_has_no_new_release_heading():
    headings = re.findall(r"^## \[.*$", CHANGELOG, re.M)
    assert headings[0] == "## [Unreleased]"
    assert "[Unreleased]" not in headings[1:]


@pytest.mark.parametrize("module", sorted(POSE_LAYER_NAMES))
def test_the_changelog_names_what_the_pose_layer_holds(module):
    added = _changelog_section("### Added")
    namespace = importlib.import_module(module)
    for name in POSE_LAYER_NAMES[module]:
        assert hasattr(namespace, name), (module, name)
        assert re.search(rf"`[\w.]*\b{name}\b", added), (module, name)


def test_the_calls_in_the_changelog_use_the_defaults_of_the_code():
    added = _changelog_section("### Added")
    functions = {
        name: getattr(importlib.import_module(module), name)
        for module, names in POSE_LAYER_NAMES.items()
        for name in names
        if callable(getattr(importlib.import_module(module), name))
    }
    checked = 0
    for name, call in re.findall(r"`(?:binding_metrics\.[a-z_.]+\.)?(\w+)(\([^`]*\))`", added):
        if name not in functions:
            continue
        try:
            node = ast.parse(name + call, mode="eval").body
        except SyntaxError:
            continue  # a call written with empty keywords (binder_chain=, ...) is not Python
        parameters = inspect.signature(functions[name]).parameters
        for argument in node.args:
            if isinstance(argument, ast.Name):
                assert argument.id in parameters, (name, argument.id)
        for keyword in node.keywords:
            assert keyword.arg in parameters, (name, keyword.arg)
            try:
                documented = ast.literal_eval(keyword.value)
            except ValueError:
                continue
            if parameters[keyword.arg].default is not inspect.Parameter.empty:
                assert documented == parameters[keyword.arg].default, (name, keyword.arg)
        checked += 1
    assert checked >= 8


def test_the_changelog_records_the_issues_of_the_residue_number_and_pose_fixes():
    unreleased = CHANGELOG.split("## [Unreleased]", 1)[1]
    for number in (116, 117, 118, 119, 120, 121, 122, 123, 124, 128, 133, 134, 135, 136):
        assert f"#{number}" in unreleased, number
    results = _changelog_section("### Results that change")
    assert "`save_cif` keeps the residue numbers and insertion codes" in results
    assert "Residue labels in logs and results carry the insertion code" in results
    assert "`receptor_interface_residues` accepts residue labels" in _changelog_section("### Added")
    assert "(#123)" in _changelog_section("### Fixed")
    assert "[`docs/pose_sets.md`](docs/pose_sets.md)" in _changelog_section("### Added")
    assert (ROOT / "docs" / "pose_sets.md").is_file()
