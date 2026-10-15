"""receptor_interface_residues takes residue numbers, labels such as "52A" and pairs."""

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402

from binding_metrics.metrics.evobind import compute_evobind_score  # noqa: E402

# receptor residues (number, insertion code, x): Cβ atoms 10 Å apart along x, 52A being
# the one nearest the binder atom at x = 22 and 53 the one farthest from it.
RECEPTOR = [(51, "", 0.0), (52, "", 10.0), (52, "A", 20.0), (53, "", -30.0)]


@pytest.fixture
def complex_path(tmp_path):
    atoms = []
    for number, ins_code, x in RECEPTOR:
        atoms.append(
            struc.Atom(
                [x, 0.0, 0.0],
                chain_id="A",
                res_id=number,
                ins_code=ins_code,
                res_name="ALA",
                atom_name="CB",
                element="C",
            )
        )
    atoms.append(
        struc.Atom(
            [22.0, 0.0, 0.0], chain_id="B", res_id=1, res_name="ALA", atom_name="CB", element="C"
        )
    )
    path = tmp_path / "complex.pdb"
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, struc.array(atoms))
    pdb_file.write(str(path))
    return path


def _select(path, residues):
    return compute_evobind_score(path, None, "B", "A", receptor_interface_residues=residues)


def test_an_int_selects_every_residue_with_that_number(complex_path):
    """Unchanged: 52 selects 52 and 52A."""
    result = _select(complex_path, [52])
    assert result["n_interface_receptor_residues"] == 2
    assert result["if_dist_rec_to_pep"] == pytest.approx(np.mean([12.0, 2.0]), abs=1e-2)


def test_a_label_selects_one_residue(complex_path):
    """52A alone: it could not be asked for before."""
    result = _select(complex_path, ["52A"])
    assert result["n_interface_receptor_residues"] == 1
    assert result["if_dist_rec_to_pep"] == pytest.approx(2.0, abs=1e-2)


def test_a_label_without_a_code_is_the_residue_without_one(complex_path):
    result = _select(complex_path, ["52"])
    assert result["n_interface_receptor_residues"] == 1
    assert result["if_dist_rec_to_pep"] == pytest.approx(12.0, abs=1e-2)


def test_a_pair_means_the_same_as_the_label(complex_path):
    by_pair = _select(complex_path, [(52, "A")])
    assert by_pair["if_dist_rec_to_pep"] == pytest.approx(2.0, abs=1e-2)
    without_code = _select(complex_path, [(52, "")])
    assert without_code["if_dist_rec_to_pep"] == pytest.approx(12.0, abs=1e-2)


def test_kinds_can_be_mixed(complex_path):
    result = _select(complex_path, [51, "52A", (53, "")])
    assert result["n_interface_receptor_residues"] == 3
    assert result["if_dist_rec_to_pep"] == pytest.approx(np.mean([22.0, 2.0, 52.0]), abs=1e-2)


def test_numpy_integers_are_numbers(complex_path):
    result = _select(complex_path, np.array([51, 53]))
    assert result["n_interface_receptor_residues"] == 2


def test_a_label_that_names_no_residue_raises(complex_path):
    with pytest.raises(ValueError, match="receptor_interface_residues"):
        _select(complex_path, ["53A"])


@pytest.mark.parametrize("entry", ["A52", "52AB", "", 1.5, (52,), None, True])
def test_a_malformed_entry_raises(complex_path, entry):
    with pytest.raises(ValueError, match="not a residue number"):
        _select(complex_path, [entry])
