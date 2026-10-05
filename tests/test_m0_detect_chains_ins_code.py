"""detect_chains_from_file counts residues by number and insertion code."""

from pathlib import Path

import pytest

pytest.importorskip("biotite")

from binding_metrics.io.structures import detect_chains_from_file  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"

_BACKBONE = [
    ("N", "N", (-1.2, 0.8, 0.0)),
    ("CA", "C", (0.0, 0.0, 0.0)),
    ("C", "C", (1.3, 0.6, 0.0)),
]


def _write_two_chains(path: Path, chains: dict) -> Path:
    """PDB file with the given chains; each chain is a list of (number, insertion code)."""
    lines = []
    serial = 1
    for chain_id, ids in chains.items():
        for position, (number, ins_code) in enumerate(ids):
            for name, element, (dx, dy, dz) in _BACKBONE:
                x = 3.8 * position + dx
                lines.append(
                    f"ATOM  {serial:5d} {name:<4s} ALA {chain_id}{number:4d}{ins_code:1s}   "
                    f"{x:8.3f}{dy:8.3f}{dz:8.3f}  1.00  0.00          {element:>2s}"
                )
                serial += 1
        lines.append("TER")
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return path


def test_the_chain_with_more_residues_is_the_receptor_whatever_its_numbering(tmp_path):
    """Chain A has six residues (3, 3A and 3B among them) and chain B has five.

    Regression: A counted four residue numbers, so it was taken as the peptide.
    """
    path = _write_two_chains(
        tmp_path / "ins.pdb",
        {
            "A": [(1, ""), (2, ""), (3, ""), (3, "A"), (3, "B"), (4, "")],
            "B": [(n, "") for n in range(1, 6)],
        },
    )
    info = detect_chains_from_file(path, verbose=False)
    assert (info["peptide_chain"], info["receptor_chain"]) == ("B", "A")
    assert (info["peptide_n_residues"], info["receptor_n_residues"]) == (5, 6)
    assert info["all_chains"] == [{"id": "B", "n_residues": 5}, {"id": "A", "n_residues": 6}]


def test_chains_without_insertion_codes_are_counted_as_before(tmp_path):
    path = _write_two_chains(
        tmp_path / "plain.pdb",
        {"A": [(n, "") for n in range(1, 7)], "B": [(n, "") for n in range(1, 6)]},
    )
    info = detect_chains_from_file(path, verbose=False)
    assert (info["peptide_chain"], info["receptor_chain"]) == ("B", "A")
    assert info["all_chains"] == [{"id": "B", "n_residues": 5}, {"id": "A", "n_residues": 6}]


def test_bundled_trypsin_with_insertion_codes_counts_its_residues():
    """3P8F chain A has 227 residue numbers and 241 residues (60A-60I, 184A, ...)."""
    path = DATA / "example_bicyclic_sfti1_3P8F.cif"
    if not path.exists():
        pytest.skip("3P8F is not bundled")
    info = detect_chains_from_file(path, verbose=False)
    sizes = {chain["id"]: chain["n_residues"] for chain in info["all_chains"]}
    assert sizes == {"I": 14, "A": 241}
    assert (info["peptide_chain"], info["receptor_chain"]) == ("I", "A")
