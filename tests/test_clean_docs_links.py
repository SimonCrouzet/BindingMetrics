"""Every relative link and anchor of the Markdown documents points at something that exists.

A heading that is renamed or a file that moves leaves a dead link behind. The check reads the
links of README.md, CHANGELOG.md, METRICS.md and docs/*.md (not those inside code fences), looks
for the target file next to the document and, for a ``#fragment``, for a heading of the target
with the anchor that GitHub gives it (lower case, punctuation dropped, spaces to hyphens, a
numeric suffix for a repeated heading). Links to other sites are not followed (no network).
An image link has to point at a file that exists, too. Every page of ``docs/`` has to be linked
from another document, so that no page is left out of the navigation.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"#{1,6}\s+(.*?)\s*#*\s*$")
EXPLICIT_ANCHOR = re.compile(r"""<a\s+(?:name|id)=["']([^"']+)["']""")


def _documents() -> list[Path]:
    names = ["README.md", "CHANGELOG.md", "METRICS.md"]
    found = [ROOT / name for name in names if (ROOT / name).is_file()]
    return found + sorted((ROOT / "docs").glob("*.md"))


def _lines_outside_fences(path: Path):
    in_fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            yield number, line


def _slug(heading: str) -> str:
    text = heading.replace("`", "").strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    anchors, seen = set(), {}
    for _, line in _lines_outside_fences(path):
        match = HEADING.match(line)
        if match:
            slug = _slug(match.group(1))
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            anchors.add(slug if count == 0 else f"{slug}-{count}")
        anchors.update(EXPLICIT_ANCHOR.findall(line))
    return anchors


def broken_links(documents: list[Path]) -> list[str]:
    """``document:line: target (reason)`` for each relative link that does not resolve."""
    broken = []
    for document in documents:
        for number, line in _lines_outside_fences(document):
            for target in LINK.findall(line):
                if re.match(r"[a-z][a-z0-9+.-]*:", target):  # https:, mailto:, ...
                    continue
                relative, _, fragment = target.partition("#")
                destination = (document.parent / relative).resolve() if relative else document
                where = f"{document.relative_to(ROOT)}:{number}: {target}"
                if not destination.exists():
                    broken.append(f"{where} (no such file)")
                elif fragment and destination.suffix == ".md":
                    if fragment not in _anchors(destination):
                        broken.append(f"{where} (no such heading)")
    return broken


def test_every_relative_link_and_anchor_of_the_documents_resolves():
    documents = _documents()
    assert documents, "no Markdown document found"
    assert not broken_links(documents)


def test_every_page_of_the_docs_is_linked_from_another_document():
    documents = _documents()
    linked = set()
    for document in documents:
        for _, line in _lines_outside_fences(document):
            for target in LINK.findall(line):
                if not re.match(r"[a-z][a-z0-9+.-]*:", target):
                    destination = (document.parent / target.partition("#")[0]).resolve()
                    if destination != document:
                        linked.add(destination)
    pages = sorted((ROOT / "docs").glob("*.md"))
    assert pages, "no page in docs/"
    assert not [page.name for page in pages if page.resolve() not in linked]
