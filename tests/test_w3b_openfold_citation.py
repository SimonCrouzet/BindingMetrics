"""Reference entries in the module docstring of ``metrics.openfold``."""

from binding_metrics.metrics import openfold


def test_module_docstring_cites_the_openfold3_0_5_0_record():
    doc = openfold.__doc__
    assert "OpenFold3, v0.5.0" in doc
    assert "10.5281/zenodo.22042719" in doc  # the v0.5.0 record, as in the README
    assert "10.5281/zenodo.19001000" not in doc  # the 0.4.0 record
    assert "OpenFold3-preview" not in doc
    assert "Abramson" in doc


def test_the_openfold_v1_paper_is_not_presented_as_the_openfold3_reference():
    assert "OpenFold3: An open-source, trainable implementation" not in openfold.__doc__
