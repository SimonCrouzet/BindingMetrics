"""``ChainNotFoundError`` is defined in ``metrics._common`` and re-exported by ``cli.run`` (#172).

The metric modules cannot import ``cli.run`` at module level (the registry would form a cycle),
so the class lives where both can import it.
"""

import inspect
import pickle

import pytest
from conftest import DATA_DIR

pytest.importorskip("biotite.structure")

from binding_metrics.cli import run
from binding_metrics.metrics import _common, polar_contacts
from binding_metrics.metrics._common import load_structure

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"


def test_both_import_paths_give_one_class():
    assert run.ChainNotFoundError is _common.ChainNotFoundError
    assert issubclass(_common.ChainNotFoundError, ValueError)


def test_the_polar_contacts_module_binds_the_class_at_module_level():
    assert polar_contacts.ChainNotFoundError is _common.ChainNotFoundError
    assert "cli.run" not in inspect.getsource(polar_contacts.require_chain_pair)


def test_a_metric_raises_the_class_that_the_run_catches():
    with pytest.raises(run.ChainNotFoundError, match="peptide chain 'Z'"):
        polar_contacts.compute_hbonds(load_structure(YCR), "Z", "A")


def test_the_class_survives_a_round_trip_through_pickle():
    again = pickle.loads(pickle.dumps(_common.ChainNotFoundError("chain 'Z' not found")))
    assert type(again) is run.ChainNotFoundError and str(again) == "chain 'Z' not found"
