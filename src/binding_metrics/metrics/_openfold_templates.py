"""What became of the templates of an OpenFold3 run, and what OpenFold3 says on its streams.

OpenFold3 0.5.0 takes a template as an alignment file (or a list of CIF files) per chain and
preprocesses it before the model runs. Three things can leave a chain without its template while
the run goes on, exits with status 0 and reports "Successful Queries":

* with the ColabFold MSA server on (the default), OpenFold3 overwrites the template alignment of
  every chain with the alignment the server returns (a ``UserWarning`` from
  ``colabfold_msa_server.py``), and since the toolkit runs with ``fetch_missing_structures:
  false`` none of the structures it lists is found: the chain ends with no template at all;
* a template file that OpenFold3 cannot read makes the preprocessing of the chain raise, which
  OpenFold3 catches and prints (``Failed to preprocess template alignment ...``, on stdout);
* the hits of an alignment can all be filtered out, which is not reported anywhere.

A ``score`` result then equals a template-free prediction with the same keys. The evidence is in
``<predictions>/inference_query_set.json``, which OpenFold3 rewrites after the preprocessing: a
chain that declared a template source has ``template_entry_chain_ids`` set to the entries it kept
(an empty list when none), a chain that declared none keeps ``null``.

This module reads that file, adds why a chain that asked for a template did not get one (from the
messages above and from whether the server was on), and keeps the result in
``<predictions>/template_accounting.json`` so that it can be read again from a stored run. It also
reads the warning that OpenFold3 prints when a query fails while its features are built (before
the model runs), for which it writes no log file. It imports the standard library only.

Record of one chain (``chain ID -> record``, below the query name)::

    {"requested": True | False | None,   # the query asked for a template (None: not known)
     "source": "alignment" | "structure" | None,
     "used": True | False,               # OpenFold3 kept at least one template entry
     "cause": None | "replaced_by_msa_server" | "preprocessing_failed" | "no_template_kept"
              | "not_requested" | "not_recorded",
     "detail": None | "<what OpenFold3 printed>",
     "entry_ids": ["receptor_A"]}
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: File that the toolkit writes next to OpenFold3's output: the accounting of every query.
TEMPLATE_ACCOUNTING_FILE = "template_accounting.json"

#: File that OpenFold3 rewrites after the template preprocessing (v0.5.0).
_QUERY_SET_FILE = "inference_query_set.json"

#: Causes of a chain without a template (``record["cause"]``).
REPLACED_BY_MSA_SERVER = "replaced_by_msa_server"
PREPROCESSING_FAILED = "preprocessing_failed"
NO_TEMPLATE_KEPT = "no_template_kept"
NOT_REQUESTED = "not_requested"
NOT_RECORDED = "not_recorded"

#: One sentence per cause for a log line or a ``reason``: what happened, and what to do.
CAUSE_TEXT = {
    REPLACED_BY_MSA_SERVER: (
        "the ColabFold MSA server replaced the template alignment of the chain, and OpenFold3 "
        "found none of the structures it lists in the template directory. Run without the "
        "server to keep the template (use_msa_server=False, --openfold-no-msa-server)"
    ),
    PREPROCESSING_FAILED: (
        "OpenFold3 could not preprocess the template of the chain and went on without it"
    ),
    NO_TEMPLATE_KEPT: (
        "OpenFold3 kept no template for the chain (every hit failed its filters, or the "
        "template structure could not be read)"
    ),
    NOT_RECORDED: "the run's messages were not kept, so the cause is not known",
}

#: Longest text of ``detail`` (characters).
_DETAIL_CHARS = 300

#: Longest unfinished line that is carried to the next chunk (characters). The warning of the
#: MSA server is one line of about four thousand characters; a progress bar has no newline.
_PARTIAL_LINE_CHARS = 100_000

_OVERWRITE_WARNING = re.compile(
    r"UserWarning: Query (?P<query>\S+) chain .*?chain_ids=\[(?P<chains>[^\]]*)\]"
    r".*?overwritten with a path to the template alignment"
)
_PREPROCESS_FAILED = re.compile(r"Failed to preprocess template alignment (?P<path>.*?):?\s*$")
_PREPROCESS_TIMED_OUT = re.compile(
    r"Template preprocessing TIMED OUT after (?P<seconds>\S+?)s for (?P<path>.*?)\.?\s*$"
)
_QUERY_FAILED = re.compile(
    r"Failed to process (?P<query>\S+) with preferredException type: (?P<kind>\S+)"
)
_DASHES = re.compile(r"^-{20,}\s*$")

#: Lines after a header in which the rest of a block is looked for.
_BLOCK_LINES = 40

#: Lines of a traceback after which a failed-query block is closed although no closing line came.
_TRACEBACK_LINES = 400


class StreamNotes:
    """The lines of OpenFold3's output that explain a lost template or a failed query.

    Fed with the text of one stream as it arrives (``feed``), because the tail of stderr that
    the error message keeps is cut to a few kilobytes and these lines come before the progress
    bars. Three kinds are noted:

    * ``overwritten``: the ``(query, chain ID)`` pairs whose template alignment the MSA server
      replaced;
    * ``failed_templates``: ``alignment path -> error`` for a template preprocessing that
      raised or timed out;
    * ``failed_queries``: ``query -> "ExceptionType: message"`` for a query whose features
      could not be built. OpenFold3 writes ``logs/predict_err_rank<N>.log`` only for a failure in
      the model's forward pass; a failure while the features are built (``single_datasets/
      inference.py``) is a logger warning and nothing else.

    One object per stream; ``merge`` combines them.
    """

    def __init__(self) -> None:
        self.overwritten: set[tuple[str, str]] = set()
        self.failed_templates: dict[str, str] = {}
        self.failed_queries: dict[str, str] = {}
        self._partial = ""
        self._template: Optional[dict[str, Any]] = None
        self._query: Optional[dict[str, Any]] = None

    def feed(self, text: str) -> None:
        """Take the next chunk of the stream; a line is read once it is complete."""
        pieces = (self._partial + text).split("\n")
        self._partial = pieces.pop()[-_PARTIAL_LINE_CHARS:]
        for piece in pieces:
            self._read_line(piece.split("\r")[-1].rstrip())

    def finish(self) -> None:
        """Read the unfinished last line and close an open block."""
        if self._partial:
            self._read_line(self._partial.split("\r")[-1].rstrip())
            self._partial = ""
        self._close_template()
        self._close_query()

    def merge(self, other: "StreamNotes") -> "StreamNotes":
        """Add what ``other`` noted to this object and return it."""
        self.overwritten |= other.overwritten
        self.failed_templates.update(other.failed_templates)
        self.failed_queries.update(other.failed_queries)
        return self

    # ---- one line

    def _read_line(self, line: str) -> None:
        if self._query is not None:
            self._continue_query(line)
        if self._template is not None:
            self._continue_template(line)
        if "overwritten with a path to the template alignment" in line:
            found = _OVERWRITE_WARNING.search(line)
            if found:
                for chain in re.findall(r"'([^']*)'", found.group("chains")):
                    self.overwritten.add((found.group("query"), chain))
            return
        failed = _PREPROCESS_FAILED.search(line) if "Failed to preprocess" in line else None
        if failed:
            self._close_template()
            self._template = {"path": failed.group("path").strip(), "lines": 0, "state": None}
            return
        timed_out = _PREPROCESS_TIMED_OUT.search(line) if "TIMED OUT" in line else None
        if timed_out:
            path = timed_out.group("path").strip()
            self.failed_templates[path] = f"timed out after {timed_out.group('seconds')} s"
            return
        query = _QUERY_FAILED.search(line) if "Failed to process" in line else None
        if query:
            self._close_query()
            self._query = {
                "query": query.group("query"),
                "kind": query.group("kind"),
                "last": "",
                "lines": 0,
            }

    # ---- "Failed to preprocess template alignment <path>:" (a print on stdout)
    # followed by the blank-separated blocks "Exception:", the message, "Type:", the type name.

    def _continue_template(self, line: str) -> None:
        block = self._template
        block["lines"] += 1
        text = line.strip()
        if text == "Exception:":
            block["state"] = "message"
        elif text == "Type:":
            block["state"] = "type"
        elif text == "Traceback:" or block["lines"] > _BLOCK_LINES:
            self._close_template()
        elif text and block["state"] == "message" and "message" not in block:
            block["message"] = text
        elif text and block["state"] == "type" and "type" not in block:
            block["type"] = text

    def _close_template(self) -> None:
        block, self._template = self._template, None
        if block is None:
            return
        kind, message = block.get("type"), block.get("message")
        error = ": ".join(part for part in (kind, message) if part) or "no message"
        self.failed_templates[block["path"]] = error[:_DETAIL_CHARS]

    # ---- "Failed to process <query> with preferredException type: <T>" (a logger warning)
    # followed by the traceback and a closing line of dashes; the last line of the traceback that
    # starts with the exception type is the exception.

    def _continue_query(self, line: str) -> None:
        block = self._query
        block["lines"] += 1
        if _DASHES.match(line) or block["lines"] > _TRACEBACK_LINES:
            self._close_query()
        elif line.startswith(block["kind"]):
            block["last"] = line.strip()

    def _close_query(self) -> None:
        block, self._query = self._query, None
        if block is None:
            return
        self.failed_queries[block["query"]] = (block["last"] or block["kind"])[:_DETAIL_CHARS]


# ---------------------------------------------------------------------------
# The accounting
# ---------------------------------------------------------------------------


def _read_json(path: Path, not_before: float = 0.0) -> Optional[dict]:
    """The JSON object in ``path``; None when it is absent, older than ``not_before``, or bad."""
    try:
        if path.stat().st_mtime < not_before:
            return None
        with open(path, encoding="utf-8") as handle:
            content = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.debug("%s could not be read: %s", path, exc)
        return None
    return content if isinstance(content, dict) else None


def requested_templates(
    query_json: str | Path,
) -> Optional[dict[str, dict[str, tuple[str, Optional[str]]]]]:
    """The chains of a query file that ask for a template: ``{query: {chain ID: (source, path)}}``.

    ``source`` is ``"alignment"`` for a ``template_alignment_file_path`` (``path`` is that
    path) and ``"structure"`` for ``template_cif_paths`` (``path`` is None). Chains that ask for
    none are not listed. None when the file cannot be read.
    """
    content = _read_json(Path(query_json))
    if content is None:
        return None
    asked: dict[str, dict[str, tuple[str, Optional[str]]]] = {}
    for query, body in (content.get("queries") or {}).items():
        for chain in (body or {}).get("chains") or []:
            alignment = chain.get("template_alignment_file_path")
            structures = chain.get("template_cif_paths")
            if alignment:
                source = ("alignment", str(alignment))
            elif structures:
                source = ("structure", None)
            else:
                continue
            for chain_id in chain.get("chain_ids") or []:
                asked.setdefault(query, {})[str(chain_id)] = source
    return asked


def account_for_templates(
    output_dir: str | Path,
    *,
    query_json: Optional[str | Path] = None,
    use_msa_server: Optional[bool] = None,
    notes: Optional[StreamNotes] = None,
    not_before: float = 0.0,
) -> dict[str, dict[str, dict]]:
    """Say for every chain of every query whether OpenFold3 used a template, and why not.

    Args:
        output_dir: The predictions folder of the run (it holds ``inference_query_set.json``).
        query_json: The query file the run was given. It says which chains asked for a template;
            without it (or when it cannot be read) ``requested`` is None and a chain without a
            template has the cause ``not_recorded``.
        use_msa_server: Whether the run used the MSA server; None when not known. With the
            server on, a chain that asked for an alignment and ended without a template is
            taken as replaced by the server's alignment even when its warning was not seen.
        notes: What the run printed (:class:`StreamNotes`, finished).
        not_before: A modification time; a ``inference_query_set.json`` older than this is from
            an earlier run and is ignored.

    Returns:
        ``{query name: {chain ID: record}}`` with the records of the module docstring; empty when
        OpenFold3 wrote no ``inference_query_set.json``.
    """
    processed = _read_json(Path(output_dir) / _QUERY_SET_FILE, not_before)
    if not processed:
        return {}
    asked = None if query_json is None else requested_templates(query_json)
    notes = notes or StreamNotes()
    accounting: dict[str, dict[str, dict]] = {}
    for query, body in (processed.get("queries") or {}).items():
        for chain in (body or {}).get("chains") or []:
            entries = chain.get("template_entry_chain_ids")
            used = bool(entries)
            for chain_id in chain.get("chain_ids") or []:
                chain_id = str(chain_id)
                request = None if asked is None else asked.get(query, {}).get(chain_id)
                record = {
                    "requested": None if asked is None else request is not None,
                    "source": None if request is None else request[0],
                    "used": used,
                    "cause": None,
                    "detail": None,
                    "entry_ids": list(entries or []),
                }
                if not used:
                    record["cause"], record["detail"] = _cause(
                        query, chain_id, request, asked is None, use_msa_server, notes
                    )
                accounting.setdefault(query, {})[chain_id] = record
    return accounting


def _cause(
    query: str,
    chain_id: str,
    request: Optional[tuple[str, Optional[str]]],
    request_unknown: bool,
    use_msa_server: Optional[bool],
    notes: StreamNotes,
) -> tuple[str, Optional[str]]:
    """``(cause, detail)`` of a chain that ended without a template."""
    if request_unknown:
        return NOT_RECORDED, None
    if request is None:
        return NOT_REQUESTED, None
    source, alignment = request
    if (query, chain_id) in notes.overwritten:
        return REPLACED_BY_MSA_SERVER, "OpenFold3 warned that it overwrote the alignment path"
    if alignment is not None:
        failed = {os.path.normpath(path): error for path, error in notes.failed_templates.items()}
        if os.path.normpath(alignment) in failed:
            return PREPROCESSING_FAILED, failed[os.path.normpath(alignment)]
    if use_msa_server and source == "alignment":
        return REPLACED_BY_MSA_SERVER, "the MSA server was on (its warning was not seen)"
    return NO_TEMPLATE_KEPT, None


def write_template_accounting(output_dir: str | Path, accounting: dict) -> Optional[Path]:
    """Keep ``accounting`` in ``<output_dir>/template_accounting.json``; None if that fails."""
    path = Path(output_dir) / TEMPLATE_ACCOUNTING_FILE
    try:
        path.write_text(json.dumps({"queries": accounting}, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("%s could not be written: %s", path, exc)
        return None
    return path


def read_template_accounting(
    output_dir: str | Path, query_name: Optional[str] = None
) -> dict[str, dict]:
    """What became of the templates of a stored or adopted OpenFold3 output.

    Reads ``template_accounting.json``, which the toolkit writes when it runs OpenFold3 (it has
    the cause of a lost template). An output that has none (one made elsewhere, or by an older
    version) is read from ``inference_query_set.json`` alone: ``used`` is known, ``requested``
    is None and a chain without a template has the cause ``not_recorded``.

    Args:
        output_dir: The predictions folder (the one that holds ``<query>/seed_*/``).
        query_name: The query to return; None returns every query.

    Returns:
        ``{chain ID: record}`` of the query, or ``{query: {chain ID: record}}`` without a name;
        empty when nothing is known.
    """
    kept = _read_json(Path(output_dir) / TEMPLATE_ACCOUNTING_FILE)
    accounting: dict[str, dict] = dict((kept or {}).get("queries") or {})
    if query_name is not None:
        if query_name in accounting:
            return accounting[query_name]
        return account_for_templates(output_dir).get(query_name, {})
    for query, chains in account_for_templates(output_dir).items():
        accounting.setdefault(query, chains)
    return accounting


def missing_templates(chains: dict[str, dict]) -> dict[str, dict]:
    """The chains of one query that asked for a template and did not get one."""
    return {
        chain_id: record
        for chain_id, record in chains.items()
        if record.get("requested") and not record.get("used")
    }


def describe_missing(chains: dict[str, dict]) -> Optional[str]:
    """One sentence for the chains of a query that asked for a template and got none, or None.

    Chains with the same cause and detail share a clause (``chains A, B: ...``).
    """
    groups: dict[tuple, list[str]] = {}
    for chain_id, record in missing_templates(chains).items():
        groups.setdefault((record.get("cause"), record.get("detail")), []).append(chain_id)
    if not groups:
        return None
    parts = []
    for (cause, detail), chain_ids in groups.items():
        text = CAUSE_TEXT.get(cause, "it is not known why")
        extra = f" ({detail})" if detail else ""
        label = "chain" if len(chain_ids) == 1 else "chains"
        parts.append(f"{label} {', '.join(chain_ids)}: {text}{extra}")
    return "OpenFold3 used no template where the query asked for one: " + "; ".join(parts)


def warn_about_missing_templates(accounting: dict[str, dict[str, dict]]) -> None:
    """Log a warning for every chain that asked for a template and did not get one."""
    for query, chains in accounting.items():
        for chain_id, record in missing_templates(chains).items():
            detail = f" ({record['detail']})" if record.get("detail") else ""
            logger.warning(
                "OpenFold3 used no template for chain %s of '%s' although the query asked for "
                "one: %s%s. The prediction does not use the structure that was given as a "
                "template.",
                chain_id,
                query,
                CAUSE_TEXT.get(record.get("cause"), "it is not known why"),
                detail,
            )
