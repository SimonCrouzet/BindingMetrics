"""Result text that reaches the pipeline report is data: it never becomes markup (#175).

The HTML report is the Markdown summary run through a renderer that keeps raw HTML, and its
title holds the sample ID. A model error, a pre-flight reason or a name that carries tags or
Markdown delimiters must come out as text, and a table must keep its rows and cells. The
Markdown file keeps the original text.
"""

import re
from collections import Counter
from html.parser import HTMLParser

import pytest

pytest.importorskip("markdown")

from binding_metrics.protocols import report  # noqa: E402
from binding_metrics.protocols.report import (  # noqa: E402
    _build_html,
    _build_summary,
    write_report,
)

SCRIPT = "<script>alert(1)</script>"
PAYLOADS = {
    "script": SCRIPT,
    "tags_and_ampersand": "<b>x</b> & <img src=x onerror=alert(1)>",
    "links": "[click](javascript:alert(1)) ![i](http://e.example/x.png) <http://e.example/>",
    "inline_delimiters": "*em* _it_ **strong** `code` ``d`` ~~s~~ a|b | c",
    "line_start_syntax": "# head\n- item\n1. one\n> quote\n---\n\nnext paragraph",
    "escapes_and_entities": "\\ back\\slash \\| &amp; &lt;b&gt; &#60; <!-- c -->",
}
BENIGN = "ok"


class _Page(HTMLParser):
    """Elements, attributes, visible text and table shapes of an HTML page."""

    def __init__(self, page: str):
        super().__init__(convert_charrefs=True)
        self.tags: Counter = Counter()
        self.attributes: list[tuple[str, str]] = []
        self.chunks: list[str] = []
        self.tables: list[list[int]] = []
        self.feed(page)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.tags[tag] += 1
        self.attributes += [(tag, name) for name, _ in attrs]
        if tag == "table":
            self.tables.append([])
        elif tag == "tr":
            self.tables[-1].append(0)
        elif tag in ("td", "th"):
            self.tables[-1][-1] += 1

    def handle_data(self, data):
        self.chunks.append(data)

    @property
    def text(self) -> str:
        return " ".join("".join(self.chunks).split())


def _failures(p: str) -> dict:
    """Every section in its failure form, which prints the error text on a line."""
    return {
        "sample_id": p,
        "input": p,
        "total_elapsed_s": 12.3,
        "chains": {
            "peptide_chain": p,
            "receptor_chain": p,
            "peptide_n_residues": p,
            "receptor_n_residues": p,
        },
        "preflight": {
            "status": p,
            "policy": p,
            "reason": p,
            "skipped_steps": {p: p},
            "skipped_geometry": {p: p},
            "report": {"warnings": [p], "notes": [p]},
        },
        "relax": {
            "success": False,
            "error_message": p,
            "peptide_cyclic_bonds": [{"type": p, "atom1": p, "atom2": p}],
        },
        "energy": {"success": False, "error_message": p},
        "prediction": {"model": p, "error": p},
        "dockq": {"error": p},
    }


def _successes(p: str) -> dict:
    """Every section in its success form, with the text in lines, headings and table cells."""
    return {
        "sample_id": p,
        "relax": {"success": True, "rmsd_md_final": p},
        "energy": {"success": True, "relaxed_interaction_energy": p, "num_contacts": p},
        "interface": {
            "peptide_chain": "B",
            "delta_sasa": p,
            "hbonds": p,
            "per_residue": [
                {
                    "chain": "B",
                    "res_name": p,
                    "res_id": 3,
                    "ins_code": p,
                    "buried_sasa": 1.0,
                    "polar_area": 1.0,
                    "apolar_area": 1.0,
                    "delta_g_res": 0.1,
                }
            ],
        },
        "geometry": {
            "ramachandran": {
                "ramachandran_favoured_pct": p,
                "ramachandran_outlier_count": p,
                "per_residue": [
                    {
                        "region": "outlier",
                        "res_name": p,
                        "res_id": 1,
                        "ins_code": p,
                        "phi": 1.0,
                        "psi": 2.0,
                    }
                ],
            },
            "omega": {
                "per_residue": [
                    {
                        "is_outlier": True,
                        "res_name": p,
                        "res_id": 2,
                        "ins_code": "",
                        "omega": 1.0,
                        "deviation": 2.0,
                    }
                ],
            },
            "shape_complementarity": {"sc": p},
        },
        "electrostatics": {"coulomb_energy_kJ": p},
        "openfold": {
            "avg_plddt": p,
            "binder_cyclic": True,
            "templates": {p: {"requested": True, "used": False, "cause": p}},
            "weights": {"name": p, "custom": True, "sha256": p},
            "binder_plddt_per_residue": [50.0, 90.0],
        },
        "prediction": {
            "model": p,
            "avg_plddt": 80.0,
            "mean_interface_pae": p,
            "reason": p,
            "cache": {"runs": p, "hits": p, "adopted": p},
            "templates": {p: {"requested": True, "used": True}},
            "weights": {"name": p, "custom": False},
        },
        "dockq": {
            "dockq": 0.5,
            "capri_class": p,
            "n_interfaces": p,
            "best_mapping": p,
            "interfaces": [{"chains": p, "DockQ": p, "capri_class": p}],
        },
    }


BUILDERS = {"failures": _failures, "successes": _successes}


def _tables_of_markdown(summary: str) -> list[list[int]]:
    """Table shapes of a Markdown summary rendered with the same extensions as the report."""
    import markdown

    return _Page(markdown.markdown(summary, extensions=["tables", "nl2br"])).tables


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=PAYLOADS)
class TestHtmlReport:
    def test_no_element_comes_from_the_data(self, builder, payload):
        page = _Page(_build_html(BUILDERS[builder](payload)))
        reference = _Page(_build_html(BUILDERS[builder](BENIGN)))
        assert page.tags == reference.tags
        assert page.attributes == reference.attributes

    def test_the_tables_keep_their_rows_and_cells(self, builder, payload):
        page = _Page(_build_html(BUILDERS[builder](payload)))
        reference = _Page(_build_html(BUILDERS[builder](BENIGN)))
        assert page.tables == reference.tables

    def test_the_text_shows_as_written(self, builder, payload):
        page = _Page(_build_html(BUILDERS[builder](payload)))
        assert " ".join(payload.split()) in page.text


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=PAYLOADS)
def test_the_markdown_tables_keep_their_rows_and_cells(builder, payload):
    summary = _build_summary(BUILDERS[builder](payload))
    reference = _build_summary(BUILDERS[builder](BENIGN))
    assert _tables_of_markdown(summary) == _tables_of_markdown(reference)


class TestNamedCases:
    def test_a_model_error_with_a_script_is_text(self):
        results = {"sample_id": "s", "prediction": {"model": "boltz2", "error": SCRIPT}}
        page = _build_html(results)
        assert "<script" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page

    def test_the_title_and_the_sample_row_escape_the_sample_id(self):
        page = _build_html({"sample_id": "<b>x</b> & co"})
        assert "<title>Binding Metrics Report — &lt;b&gt;x&lt;/b&gt; &amp; co</title>" in page
        assert "<b>" not in page
        assert "<code>&lt;b&gt;x&lt;/b&gt; &amp; co</code>" in page

    def test_a_quote_cannot_leave_the_title(self):
        page = _build_html({"sample_id": '"></title><script>alert(1)</script>'})
        assert "<script" not in page
        assert page.count("</title>") == 1

    def test_a_backtick_in_a_name_does_not_end_its_code_span(self):
        page = _build_html({"sample_id": "a` <script>alert(1)</script> `b"})
        assert "<script" not in page

    def test_a_pipe_and_a_newline_in_a_cell_keep_one_row(self):
        results = {
            "sample_id": "s",
            "relax": {"peptide_cyclic_bonds": [{"type": "a|b\nc", "atom1": "B:1:N", "atom2": "x"}]},
        }
        summary = _build_summary(results)
        row = next(line for line in summary.splitlines() if "B:1:N" in line)
        assert row.startswith("| a\\|b c ")
        assert len(re.findall(r"(?<!\\)\|", row)) == 4
        assert _tables_of_markdown(summary)[1] == [3, 3]

    def test_the_renderer_does_not_pass_raw_html(self):
        body = report._markdown_to_html("<script>alert(1)</script>\n\n<div onclick=x>t</div>\n")
        assert "<script" not in body
        assert "<div" not in body


class TestMarkdownFile:
    def test_the_summary_keeps_the_text_as_written(self):
        results = {
            "sample_id": "<b>x</b> & co",
            "prediction": {"model": "boltz2", "error": f"{SCRIPT} *a* _b_"},
        }
        summary = _build_summary(results)
        assert f"_Failed: {SCRIPT} *a* _b__" in summary
        assert "| Sample ID | `<b>x</b> & co` |" in summary
        assert "&lt;" not in summary
        assert "&amp;" not in summary

    def test_the_file_written_for_a_summary_keeps_the_text(self, tmp_path):
        results = {"sample_id": "s", "prediction": {"model": "boltz2", "error": SCRIPT}}
        write_report(results, tmp_path, "s", summary=True, summary_format="md")
        written = (tmp_path / "s_report.md").read_text(encoding="utf-8")
        assert f"_Failed: {SCRIPT}_" in written

    def test_the_html_file_written_for_a_summary_has_no_script(self, tmp_path):
        results = {"sample_id": "<b>x</b>", "prediction": {"model": "boltz2", "error": SCRIPT}}
        write_report(results, tmp_path, "s", summary=True, summary_format="html")
        written = (tmp_path / "s_report.html").read_text(encoding="utf-8")
        assert _Page(written).tags["script"] == 0
        assert _Page(written).tags["b"] == 0


class TestOrdinaryText:
    def results(self) -> dict:
        return {
            "sample_id": "1YCR_B-sample.v2",
            "input": "/data/run 1/1YCR.pdb",
            "total_elapsed_s": 12.34,
            "chains": {
                "peptide_chain": "B",
                "receptor_chain": "A",
                "peptide_n_residues": 13,
                "receptor_n_residues": 85,
            },
            "preflight": {
                "status": "warn",
                "policy": "skip",
                "reason": "no receptor chain was given; pass --receptor-chain",
                "skipped_steps": {"interface": "needs two protein chains (found 1)"},
                "report": {"warnings": ["Boltz-2 (seed 42): 1.5 A"], "notes": []},
            },
            "relax": {
                "success": True,
                "rmsd_md_final": 1.234,
                "peptide_rmsf_per_residue": [0.5, 2.0],
                "peptide_cyclic_bonds": [
                    {"type": "disulfide", "atom1": "B:3:SG", "atom2": "B:11:SG"}
                ],
            },
            "energy": {"success": False, "error_message": "OpenMM failed: bad system (code 3)."},
            "interface": {"peptide_chain": "B", "delta_sasa": 1266.8, "hbonds": 3},
            "geometry": {
                "ramachandran": {
                    "ramachandran_favoured_pct": 91.7,
                    "per_residue": [
                        {
                            "region": "outlier",
                            "res_name": "ALA",
                            "res_id": 12,
                            "ins_code": "A",
                            "phi": 60.0,
                            "psi": -120.0,
                        }
                    ],
                },
                "omega": {
                    "per_residue": [
                        {
                            "is_outlier": True,
                            "res_name": "PRO",
                            "res_id": 4,
                            "ins_code": "",
                            "omega": 150.0,
                            "deviation": 30.0,
                        }
                    ]
                },
            },
            "prediction": {
                "model": "boltz2",
                "avg_plddt": 88.5,
                "iptm": 0.85,
                "reason": "no PAE file for sample 0.",
                "binder_plddt_per_residue": [60.0, 90.0],
                "cache": {"runs": 1, "hits": 0, "adopted": 0},
            },
        }

    @staticmethod
    def without_time(text: str) -> str:
        text = re.sub(r"\| Generated \|[^\n]*\n", "", text)
        return re.sub(r"<tr>\s*<td>Generated</td>\s*<td>[^<]*</td>\s*</tr>", "", text)

    def test_the_html_body_is_the_rendering_of_the_markdown_text(self):
        import markdown

        results = self.results()
        body = markdown.markdown(_build_summary(results), extensions=["tables", "nl2br"])
        page = _build_html(results)
        assert self.without_time(body) in self.without_time(page)

    def test_the_summary_has_no_added_backslash_or_entity(self):
        summary = _build_summary(self.results())
        assert "\\" not in summary
        assert not re.search(r"&(amp|lt|gt|#\d+);", summary)
