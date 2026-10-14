"""The interaction-energy section names a mode that failed while another succeeded (#179).

``compute_interaction_energy`` returns ``success`` true with an ``error_message`` when one of
its modes failed. The table shows the mode that has a value; the line under it carries the
message, escaped in the HTML report like the other result text.
"""

from html.parser import HTMLParser

import pytest

pytest.importorskip("markdown")

from binding_metrics.protocols.report import _build_html, _build_summary, _md_energy  # noqa: E402

MODE_FAILED = "relaxed: minimisation did not converge"

ENERGY = {
    "success": True,
    "error_message": MODE_FAILED,
    "raw_interaction_energy": -310.5,
    "relaxed_interaction_energy": None,
    "raw_e_complex": -1.0,
    "raw_e_peptide": -2.0,
    "raw_e_receptor": -3.0,
    "num_contacts": 5,
    "num_close_contacts": 1,
}


class _Text(HTMLParser):
    def __init__(self, page: str):
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.chunks: list[str] = []
        self.feed(page)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)

    def handle_data(self, data):
        self.chunks.append(data)

    @property
    def text(self) -> str:
        return " ".join("".join(self.chunks).split())


class TestMarkdown:
    def test_a_failed_mode_is_named_under_the_table(self):
        section = _md_energy(ENERGY)
        lines = section.rstrip("\n").splitlines()
        assert "| E_int (raw)" in section
        assert lines[-1] == f"⚠️ **A mode failed:** {MODE_FAILED}"
        assert not lines[-2].startswith("|") and lines[-2] == ""

    @pytest.mark.parametrize("message", [None, ""])
    def test_a_section_without_a_message_has_no_such_line(self, message):
        section = _md_energy({**ENERGY, "error_message": message})
        assert "mode failed" not in section
        assert section.rstrip("\n").splitlines()[-1].startswith("| Close contacts")

    def test_the_failure_form_is_unchanged(self):
        section = _md_energy({"success": False, "error_message": "no template"})
        assert section == "## Interaction Energy\n_Failed: no template_\n"

    def test_the_markdown_file_keeps_the_text_as_written(self):
        message = "relaxed: <b>x</b> & *y*"
        section = _md_energy({**ENERGY, "error_message": message})
        assert section.rstrip("\n").endswith(f"**A mode failed:** {message}")

    def test_the_summary_of_a_run_carries_the_line_once(self):
        summary = _build_summary({"sample_id": "s", "energy": ENERGY})
        assert summary.count(MODE_FAILED) == 1


class TestHtml:
    def test_the_line_is_in_the_page(self):
        page = _Text(_build_html({"sample_id": "s", "energy": ENERGY}))
        assert f"A mode failed: {MODE_FAILED}" in page.text

    def test_the_message_is_text(self):
        message = "relaxed: <script>alert(1)</script> [x](javascript:alert(1)) a|b"
        page = _build_html({"sample_id": "s", "energy": {**ENERGY, "error_message": message}})
        assert "<script" not in page
        shown = _Text(page)
        assert message in shown.text
        reference = _Text(_build_html({"sample_id": "s", "energy": ENERGY}))
        assert shown.tags == reference.tags
