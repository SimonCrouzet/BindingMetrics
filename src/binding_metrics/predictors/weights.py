"""Custom model weights as part of a prediction request: what they are, and a hash that is cheap.

A fine-tuned model is a different model. The prediction store keys a request by everything that
changes the output, so the weights a run used must be in the key, by content: a moved or renamed
identical file gives the same key, and a changed file another. This module turns a path into a
``WeightsRef`` (path, kind, SHA-256, size) and keeps the hashing affordable.

Public names and signatures (the module imports the standard library only)::

    WEIGHTS_KINDS = ("file", "directory")

    @dataclass(frozen=True)
    class WeightsRef:
        path: Path; kind: str; sha256: str; size: int; n_files: int = 1
        .key_fields() -> dict             # {"kind", "sha256", "size"}: what the request key hashes
        .to_dict() -> dict                # JSON-ready, with the path (request.json, results)

    weights_reference(path, *, cache_dir=None, expect=None) -> WeightsRef

What is hashed. A file is hashed by content. A directory is described by a manifest, the sorted
list of ``(relative path, size, SHA-256)`` of every file below it, and its digest is the SHA-256
of the canonical JSON of that list, so renaming, adding, removing or changing any file changes it
while the directory's own location does not. The size of a directory is the total of its files.

The hash cache. A checkpoint of 2 GB is not read again on every run. With ``cache_dir`` (the
prediction store root, see ``PredictionStore.weights_reference``) the SHA-256 of each file is kept
in ``<cache_dir>/.weights-sha256.json`` under ``(absolute path, size, mtime_ns)``: a file with
the same three values is not read again. The file is written atomically (a temporary file
renamed over it) while holding an advisory lock (``fcntl.flock``) on ``.weights-sha256.lock``,
and merged with what another process wrote meanwhile. A corrupt or unreadable cache is ignored
and rebuilt; a store on a read-only file system hashes every time and says nothing.

Limit of the cache. A file edited in place with an unchanged size AND an unchanged ``mtime_ns``
is not detected: its old hash is used. That needs the modification time to be reset on purpose
(``os.utime``, ``touch -r``, a copy that preserves it over a same-sized edit); an editor, a
download and a training run all change it. Delete ``.weights-sha256.json`` to force a re-read.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

logger = logging.getLogger(__name__)

#: How a model takes its weights: one checkpoint file, or a directory of files.
WEIGHTS_KINDS: tuple[str, ...] = ("file", "directory")

#: Name of the hash cache in the cache directory, and of its lock file.
CACHE_FILE = ".weights-sha256.json"
_LOCK_FILE = ".weights-sha256.lock"
_CACHE_FORMAT = 1

#: Files of at least this many bytes are announced in the log before they are hashed.
_ANNOUNCE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class WeightsRef:
    """A reference to custom weights, identified by content.

    Attributes:
        path: The absolute path the weights were found at. Not part of the key.
        kind: ``"file"`` or ``"directory"``.
        sha256: SHA-256 of the file, or of the manifest of a directory.
        size: Bytes (a directory: the total of its files).
        n_files: Number of files (1 for a file).
    """

    path: Path
    kind: str
    sha256: str
    size: int
    n_files: int = 1

    def __post_init__(self):
        if self.kind not in WEIGHTS_KINDS:
            raise ValueError(f"kind must be one of {WEIGHTS_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "path", Path(self.path))

    def key_fields(self) -> dict[str, Any]:
        """What the request key hashes: the content, never the path."""
        return {"kind": self.kind, "sha256": self.sha256, "size": self.size}

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready description, with the path."""
        return {
            "path": str(self.path),
            "kind": self.kind,
            "sha256": self.sha256,
            "size": self.size,
            "n_files": self.n_files,
        }


# ---------------------------------------------------------------------------- the hash cache


class _HashCache:
    """``(absolute path, size, mtime_ns)`` to SHA-256, in a JSON file under a lock."""

    def __init__(self, directory: Optional[Path]):
        self.directory = None if directory is None else Path(directory)
        self._known: dict[str, dict[str, Any]] = {}
        self._added: dict[str, dict[str, Any]] = {}
        if self.directory is not None:
            self._known = self._read()

    @property
    def _path(self) -> Path:
        return self.directory / CACHE_FILE

    def _read(self) -> dict[str, dict[str, Any]]:
        try:
            with open(self._path, encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            logger.warning("ignoring the unreadable weights hash cache %s: %s", self._path, exc)
            return {}
        files = data.get("files") if isinstance(data, dict) else None
        if not isinstance(files, dict) or data.get("format") != _CACHE_FORMAT:
            logger.warning("ignoring the malformed weights hash cache %s", self._path)
            return {}
        return {
            key: value
            for key, value in files.items()
            if isinstance(key, str)
            and isinstance(value, dict)
            and isinstance(value.get("size"), int)
            and isinstance(value.get("mtime_ns"), int)
            and isinstance(value.get("sha256"), str)
            and len(value["sha256"]) == 64
        }

    def lookup(self, path: Path, size: int, mtime_ns: int) -> Optional[str]:
        entry = self._known.get(str(path))
        if entry and entry["size"] == size and entry["mtime_ns"] == mtime_ns:
            return entry["sha256"]
        return None

    def remember(self, path: Path, size: int, mtime_ns: int, sha256: str) -> None:
        self._added[str(path)] = {"size": size, "mtime_ns": mtime_ns, "sha256": sha256}

    def save(self) -> None:
        """Merge what was hashed into the file; never raises (a cache is an optimisation)."""
        if self.directory is None or not self._added:
            return
        try:
            with _flocked(self.directory / _LOCK_FILE):
                merged = self._read()  # what another process wrote while this one hashed
                merged.update(self._added)
                payload = {"format": _CACHE_FORMAT, "files": merged}
                temporary = self._path.with_name(f"{CACHE_FILE}.tmp-{uuid.uuid4().hex[:8]}")
                try:
                    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                    os.replace(temporary, self._path)
                finally:
                    temporary.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("the weights hash cache in %s was not written: %s", self.directory, exc)
        else:
            self._known.update(self._added)
            self._added = {}


@contextlib.contextmanager
def _flocked(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``lock_path`` (created, with its directory)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)  # closing the descriptor releases the lock


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_digest(path: Path, cache: _HashCache) -> tuple[str, int]:
    """SHA-256 and size of ``path``, from the cache when its size and mtime_ns are unchanged."""
    stat = path.stat()
    cached = cache.lookup(path, stat.st_size, stat.st_mtime_ns)
    if cached is not None:
        return cached, stat.st_size
    if stat.st_size >= _ANNOUNCE_BYTES:
        logger.info(
            "hashing the weights file %s (%.1f GB); the hash is kept per path, size and "
            "modification time",
            path,
            stat.st_size / 1e9,
        )
    sha256 = _sha256_of_file(path)
    cache.remember(path, stat.st_size, stat.st_mtime_ns, sha256)
    return sha256, stat.st_size


# ---------------------------------------------------------------------------- the reference


def weights_reference(
    path: str | Path,
    *,
    cache_dir: Optional[str | Path] = None,
    expect: Optional[str] = None,
) -> WeightsRef:
    """The ``WeightsRef`` of a file or directory of weights.

    Args:
        path: The weights; ``~`` is expanded and the path made absolute.
        cache_dir: Where the hash cache lives (the store root); None hashes without a cache.
        expect: ``"file"`` or ``"directory"`` to require that kind; None accepts either.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: ``path`` is not the expected kind, or is a directory without files.
        OSError: A file cannot be read.
    """
    if expect is not None and expect not in WEIGHTS_KINDS:
        raise ValueError(f"expect must be one of {WEIGHTS_KINDS} or None, got {expect!r}")
    resolved = Path(path).expanduser().absolute()
    if not resolved.exists():
        raise FileNotFoundError(f"the weights {path} do not exist")
    kind = "directory" if resolved.is_dir() else "file"
    if expect is not None and kind != expect:
        raise ValueError(f"the weights {path} are a {kind}, and a {expect} is expected")
    cache = _HashCache(None if cache_dir is None else Path(cache_dir))
    if kind == "file":
        sha256, size = _file_digest(resolved, cache)
        cache.save()
        return WeightsRef(resolved, "file", sha256, size, 1)

    manifest = []
    total = 0
    for member in resolved.rglob("*"):
        if member.is_file():
            sha256, size = _file_digest(member, cache)
            manifest.append([member.relative_to(resolved).as_posix(), size, sha256])
            total += size
    manifest.sort(key=lambda row: row[0])  # one order on every machine
    cache.save()
    if not manifest:
        raise ValueError(f"the weights directory {path} holds no files")
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    ).hexdigest()
    return WeightsRef(resolved, "directory", digest, total, len(manifest))
