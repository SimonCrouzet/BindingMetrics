"""``PredictionRequest`` takes ``None`` for ``sequences``, ``extra_files`` and ``options``.

The signature in the module docstring of ``predictors/store.py`` gives ``None`` as the empty value
of the three mappings. A request built with ``None`` is the request built without the argument:
same attributes, same canonical fields, same key.
"""

from __future__ import annotations

import pickle

import pytest

from binding_metrics.predictors.store import PredictionRequest

#: Key of ``PredictionRequest("stub", "s1", sequences={"A": "GGG", "B": "AAA"}, options={})``.
KEY_OF_SEQUENCES = "44e4e705ee6ffe99daaa8ce7f1341f538ccba24bf3fce3ee8f7e88fef5a45b8a"


@pytest.fixture
def complex_file(tmp_path):
    path = tmp_path / "complex.cif"
    path.write_bytes(b"data_x\n#pinned complex\n")
    return path


class TestNoneIsTheEmptyMapping:
    @pytest.mark.parametrize("field", ["sequences", "extra_files", "options"])
    def test_none_gives_the_request_of_the_default(self, complex_file, field):
        default = PredictionRequest("stub", "s1", input_path=complex_file)
        with_none = PredictionRequest("stub", "s1", input_path=complex_file, **{field: None})
        assert getattr(with_none, field) == {}
        assert with_none.canonical() == default.canonical()
        assert with_none.key() == default.key()
        assert with_none.batch_signature() == default.batch_signature()
        assert with_none == default

    def test_all_three_none_on_a_request_of_sequences(self):
        default = PredictionRequest("stub", "s1", sequences={"A": "GGG", "B": "AAA"})
        with_none = PredictionRequest(
            "stub", "s1", sequences={"A": "GGG", "B": "AAA"}, extra_files=None, options=None
        )
        assert with_none.key() == default.key() == KEY_OF_SEQUENCES
        assert with_none.extra_files == {} and with_none.options == {}

    def test_sequences_none_with_an_input_file_has_the_key_of_the_default(self, complex_file):
        with_none = PredictionRequest("stub", "s1", input_path=complex_file, sequences=None)
        assert with_none.canonical()["sequences"] == {}
        assert with_none.key() == PredictionRequest("stub", "s1", input_path=complex_file).key()

    def test_the_normalised_mappings_are_plain_dicts_that_pickle(self, complex_file):
        request = PredictionRequest(
            "stub", "s1", input_path=complex_file, sequences=None, extra_files=None, options=None
        )
        assert type(request.sequences) is dict
        assert type(request.extra_files) is dict
        assert type(request.options) is dict
        assert pickle.loads(pickle.dumps(request)).key() == request.key()

    def test_none_does_not_lift_the_need_for_an_input(self):
        with pytest.raises(ValueError, match="needs an input file or sequences"):
            PredictionRequest("stub", "s1", sequences=None)

    def test_a_non_mapping_still_raises(self, complex_file):
        with pytest.raises((TypeError, ValueError)):
            PredictionRequest("stub", "s1", input_path=complex_file, options=3)
