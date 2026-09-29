"""The OpenFold3 module cites the OpenFold3 software, not the OpenFold (v1) paper."""

from binding_metrics.metrics import openfold


def test_module_docstring_cites_openfold3_preview():
    doc = openfold.__doc__
    assert "OpenFold3-preview" in doc
    assert "10.5281/zenodo.19001000" in doc
    assert "Abramson" in doc  # the repository asks that AlphaFold 3 is cited with it


def test_the_openfold_v1_paper_is_not_presented_as_the_openfold3_reference():
    assert "OpenFold3: An open-source, trainable implementation" not in openfold.__doc__
