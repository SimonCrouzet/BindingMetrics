"""The per-residue table of the report follows the peptide chain the result names (#115)."""

from binding_metrics.protocols.report import _md_interface


def _row(chain, res_id, name="GLY"):
    return {
        "chain": chain,
        "res_id": res_id,
        "res_name": name,
        "buried_sasa": 10.0,
        "polar_area": 4.0,
        "apolar_area": 6.0,
        "delta_g_res": -0.5,
    }


ROWS = [_row("A", 1, "SER"), _row("C", 1, "ALA"), _row("C", 2, "VAL"), _row("B", 7, "LEU")]


def test_the_table_lists_the_residues_of_the_named_chain():
    text = _md_interface({"peptide_chain": "C", "per_residue": ROWS})
    assert "Per-residue buried SASA" in text
    assert "ALA:1" in text and "VAL:2" in text
    assert "SER:1" not in text and "LEU:7" not in text


def test_without_the_chain_no_table_is_written():
    """No chain B is assumed: 1CWA has none, and another file may have a receptor B."""
    text = _md_interface({"per_residue": ROWS})
    assert "Per-residue buried SASA" not in text
    assert "LEU:7" not in text


def test_an_error_entry_has_no_table():
    text = _md_interface({"error": "chain 'Z' not found", "per_residue": []})
    assert "Per-residue buried SASA" not in text


def test_a_named_chain_without_rows_has_no_table():
    assert "Per-residue buried SASA" not in _md_interface(
        {"peptide_chain": "Z", "per_residue": ROWS}
    )
