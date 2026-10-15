"""Helpers of the strict JSON tests (#220): a strict parser and the paths of a document."""

import json
import logging
import math
import re
from pathlib import Path

import pytest

from binding_metrics.utils import _CurrentStreamHandler


@pytest.fixture
def restore_package_logger():
    """The ``main`` functions install console handlers; do not leave them for later tests."""
    package_logger = logging.getLogger("binding_metrics")
    saved_level = package_logger.level
    yield
    for handler in [h for h in package_logger.handlers if isinstance(h, _CurrentStreamHandler)]:
        package_logger.removeHandler(handler)
    package_logger.setLevel(saved_level)


def _refuse(token):
    raise AssertionError(f"the text holds the non-standard JSON constant {token}")


def strict_loads(text):
    """``json.loads`` as a strict parser reads: ``NaN`` and ``Infinity`` are errors."""
    return json.loads(text, parse_constant=_refuse)


def blank(value):
    """A loaded document with every NaN and infinity replaced by None."""
    if isinstance(value, dict):
        return {key: blank(child) for key, child in value.items()}
    if isinstance(value, list):
        return [blank(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def value_at(document, path):
    """The value at a path such as ``geometry.series[1]``."""
    for name, index in re.findall(r"([^.\[\]]+)|\[(\d+)\]", path):
        document = document[int(index)] if index else document[name]
    return document


def read(path):
    return Path(path).read_text(encoding="utf-8")
