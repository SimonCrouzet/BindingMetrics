"""Phosphopeptide complexes read from a raw wwPDB mmCIF.

The fixture is written here from the bundled ``data/example_phospho_1QJB.pdb`` (chain Q
of 1QJB, residues 3-10: ALA ARG SER HIS SEP TYR PRO ALA), with the numbering that makes a
wwPDB file awkward for OpenMM: ``label_seq_id`` 1-8 against ``auth_seq_id`` 3-10, and the
HIS6-SEP7 peptide bond only in ``_struct_conn``, keyed on the label numbers.
"""

from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("gemmi")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PHOSPHO_PDB = DATA_DIR / "example_phospho_1QJB.pdb"


def _pdb_atoms():
    """(name, resname, resseq, element, x, y, z) of every atom of the bundled peptide."""
    atoms = []
    for line in PHOSPHO_PDB.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atoms.append(
                (
                    line[12:16].strip(),
                    line[17:20].strip(),
                    int(line[22:26]),
                    line[76:78].strip(),
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
    return atoms


def _write_mmcif(path: Path, chains, links=(), water_of_auth=None):
    """Write a wwPDB-style mmCIF.

    ``chains`` is a list of ``(label_asym_id, auth_asym_id, n_residues, dx)``: the
    first ``n_residues`` residues of the peptide, shifted by ``dx`` angstrom along x.
    ``links`` is a list of ``(chain_index, (resseq1, atom1), (resseq2, atom2))`` in AUTH
    numbering; they are written as ``covale`` rows keyed on label and author IDs.
    ``water_of_auth``, when given, appends one water whose author chain is that ID and
    whose label chain is a new one, as in a wwPDB file, which gives the file more label
    IDs than author IDs.
    """
    atoms = _pdb_atoms()
    first = min(a[2] for a in atoms)
    lines = [
        "data_test",
        "loop_",
        *(
            f"_atom_site.{c}"
            for c in (
                "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id "
                "label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code "
                "Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv "
                "auth_seq_id auth_comp_id auth_asym_id auth_atom_id pdbx_PDB_model_num"
            ).split()
        ),
    ]
    serial = 0
    resname_of = {}
    for label, auth, n_res, dx in chains:
        for name, res, seq, elem, x, y, z in atoms:
            if seq - first >= n_res:
                continue
            serial += 1
            group = "ATOM" if res != "SEP" else "HETATM"
            resname_of[(label, seq)] = res
            lines.append(
                f"{group} {serial} {elem} {name} . {res} {label} 1 {seq - first + 1} ? "
                f"{x + dx:.3f} {y:.3f} {z:.3f} 1.00 20.00 {seq} {res} {auth} {name} 1"
            )
    if water_of_auth is not None:
        serial += 1
        lines.append(
            f"HETATM {serial} O O . HOH W 2 . ? 60.000 60.000 60.000 1.00 20.00 "
            f"900 HOH {water_of_auth} O 1"
        )
    if links:
        lines += ["#", "loop_"] + [
            f"_struct_conn.{c}"
            for c in (
                "id conn_type_id ptnr1_label_asym_id ptnr1_label_comp_id ptnr1_label_seq_id "
                "ptnr1_label_atom_id ptnr2_label_asym_id ptnr2_label_comp_id "
                "ptnr2_label_seq_id ptnr2_label_atom_id ptnr1_auth_asym_id "
                "ptnr1_auth_comp_id ptnr1_auth_seq_id ptnr2_auth_asym_id "
                "ptnr2_auth_comp_id ptnr2_auth_seq_id"
            ).split()
        ]
        for i, (ci, (s1, a1), (s2, a2)) in enumerate(links, 1):
            label, auth = chains[ci][0], chains[ci][1]
            lines.append(
                f"covale{i} covale {label} {resname_of[(label, s1)]} {s1 - first + 1} {a1} "
                f"{label} {resname_of[(label, s2)]} {s2 - first + 1} {a2} "
                f"{auth} {resname_of[(label, s1)]} {s1} {auth} {resname_of[(label, s2)]} {s2}"
            )
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def phospho_mmcif(tmp_path):
    if not PHOSPHO_PDB.exists():
        pytest.skip("bundled phospho example not found")
    return _write_mmcif(
        tmp_path / "phospho.cif",
        [("C", "Q", 8, 0.0)],
        links=[(0, (6, "C"), (7, "N")), (0, (7, "C"), (8, "N"))],
    )


def _bond_names(topology):
    """{frozenset({(resname, resid, atom), (resname, resid, atom)})} of every bond."""
    return {
        frozenset(
            {
                (b.atom1.residue.name, b.atom1.residue.id, b.atom1.name),
                (b.atom2.residue.name, b.atom2.residue.id, b.atom2.name),
            }
        )
        for b in topology.bonds()
    }


HIS_SEP = frozenset({("HIS", "6", "C"), ("SEP", "7", "N")})
HIS_SEP_BY_NAME = frozenset({("HIS", "C"), ("SEP", "N")})


def _n_internal_bonds(topology, residue) -> int:
    return sum(
        1 for b in topology.bonds() if b.atom1.residue is residue and b.atom2.residue is residue
    )


SEP_TYR = frozenset({("SEP", "7", "C"), ("TYR", "8", "N")})


class TestStructConnWithOffsetNumbering:
    def test_load_cif_bonds_the_link_into_a_nonstandard_residue(self, phospho_mmcif):
        """OpenMM keys atoms by auth_seq_id and _struct_conn rows by label_seq_id (4, 5 here)."""
        from binding_metrics.io.structures import load_structure

        topology, _ = load_structure(phospho_mmcif)
        bonds = _bond_names(topology)
        assert HIS_SEP in bonds
        assert SEP_TYR in bonds

    def test_link_is_bonded_once(self, phospho_mmcif):
        from binding_metrics.io.structures import load_structure

        topology, _ = load_structure(phospho_mmcif)
        pairs = [
            frozenset({(b.atom1.residue.id, b.atom1.name), (b.atom2.residue.id, b.atom2.name)})
            for b in topology.bonds()
        ]
        assert pairs.count(frozenset({("6", "C"), ("7", "N")})) == 1

    def test_prepped_cif_keeps_the_link_after_save_and_reload(self, phospho_mmcif, tmp_path):
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure, save_structure

        topology, positions = load_structure(phospho_mmcif)
        topology, positions = prep_structure(topology, positions)
        out = tmp_path / "prepped.cif"
        save_structure(topology, positions, out, source_path=phospho_mmcif)
        reloaded, _ = load_structure(out)
        # Compared by residue and atom name: prep renumbers the residues.
        assert HIS_SEP_BY_NAME in {
            frozenset((res, atom) for res, _, atom in bond) for bond in _bond_names(reloaded)
        }

    def test_save_cif_rewrites_struct_conn_residue_numbers_with_the_auth_numbers(
        self, phospho_mmcif, tmp_path
    ):
        """A bond present in the topology survives save_cif with the source numbering.

        OpenMM keys atoms by ``auth_seq_id`` and ``_struct_conn`` partners by
        ``label_seq_id``. ``save_cif`` restores the source ``auth_seq_id`` (HIS is 6,
        not 4), so the rows and ``_atom_site.label_seq_id`` must carry the same numbers.
        OpenMM's own reader is used on both sides so the loader is out of the picture.
        """
        import gemmi
        from openmm.app import PDBxFile

        from binding_metrics.io.structures import save_structure

        source = PDBxFile(str(phospho_mmcif))
        topology, positions = source.topology, source.positions
        atoms = {(a.residue.id, a.name): a for a in topology.atoms()}
        topology.addBond(atoms[("6", "C")], atoms[("7", "N")])  # what a correct loader gives
        out = tmp_path / "saved.cif"
        save_structure(topology, positions, out, source_path=phospho_mmcif)
        assert HIS_SEP in _bond_names(PDBxFile(str(out)).topology)

        block = gemmi.cif.read(str(out)).sole_block()
        site = block.find("_atom_site.", ["auth_seq_id", "label_seq_id"])
        assert all(row[0] == row[1] for row in site)
        conn = block.find("_struct_conn.", ["ptnr1_label_seq_id", "ptnr2_label_seq_id"])
        assert ("6", "7") in {(row[0], row[1]) for row in conn}


class TestStripHeterogensKeepsPhosphoResidues:
    def _two_chain_topology(self):
        """The peptide as chain Q plus a second copy, moved 3 nm, as chain S."""
        from openmm import unit
        from openmm.app import Modeller

        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(PHOSPHO_PDB)
        modeller = Modeller(topology, positions)
        shifted = [p + unit.Quantity((3.0, 0.0, 0.0), unit.nanometer) for p in positions]
        modeller.add(topology, shifted)
        for chain in modeller.topology.chains():
            if chain.index == 1:
                chain.id = "S"
        return modeller.topology, modeller.positions

    def test_sep_of_an_unselected_chain_stays(self):
        from binding_metrics.io.structures import strip_heterogens

        topology, positions = self._two_chain_topology()
        report: dict = {}
        top, _ = strip_heterogens(
            topology, positions, peptide_chain="Q", receptor_chain=None, report=report
        )
        assert sum(1 for r in top.residues() if r.name == "SEP") == 2
        assert "removed_heterogens" not in report or not report["removed_heterogens"]

    def test_a_ligand_of_an_unselected_chain_is_still_removed(self):
        """The phospho residues are exempt; other heterogens are not."""
        from openmm import Vec3, unit
        from openmm.app import Modeller, Topology, element

        from binding_metrics.io.structures import strip_heterogens

        topology, positions = self._two_chain_topology()
        ligand = Topology()
        chain = ligand.addChain("L")
        residue = ligand.addResidue("LIG", chain)
        ligand.addAtom("C1", element.carbon, residue)
        modeller = Modeller(topology, positions)
        modeller.add(ligand, unit.Quantity([Vec3(9.0, 9.0, 9.0)], unit.nanometer))
        top, _ = strip_heterogens(
            modeller.topology, modeller.positions, peptide_chain="Q", receptor_chain=None
        )
        names = [r.name for r in top.residues()]
        assert "LIG" not in names
        assert names.count("SEP") == 2


class TestBondsInsideNonstandardResidues:
    """A raw mmCIF lists no bond inside SEP; the PDB file has them as CONECT records."""

    def test_load_cif_gives_sep_the_bonds_the_pdb_conect_records_give(self, phospho_mmcif):
        from binding_metrics.io.structures import load_structure

        cif_top, _ = load_structure(phospho_mmcif)
        pdb_top, _ = load_structure(PHOSPHO_PDB)
        cif_bonds, pdb_bonds = _bond_names(cif_top), _bond_names(pdb_top)
        assert cif_bonds == pdb_bonds
        sep_internal = [b for b in cif_bonds if {r for r, _, _ in b} == {"SEP"}]
        assert len(sep_internal) == 9  # N-CA, CA-CB, CA-C, C-O, CB-OG, OG-P and P-O1P/O2P/O3P

    def test_every_chain_is_repaired_not_only_the_peptide(self, tmp_path):
        """Two chains, both with a SEP: the spectator chain gets its bonds too."""
        from binding_metrics.io.structures import load_structure

        cif = _write_mmcif(
            tmp_path / "two_chains.cif",
            [("C", "Q", 8, 0.0), ("D", "S", 8, 30.0)],
            links=[(0, (6, "C"), (7, "N")), (1, (6, "C"), (7, "N"))],
        )
        topology, _ = load_structure(cif)
        for chain in topology.chains():
            sep = next(r for r in chain.residues() if r.name == "SEP")
            assert _n_internal_bonds(topology, sep) == 9, chain.id

    def test_a_fully_bonded_structure_gains_no_bond(self, tmp_path):
        """Standard residues already carry their bonds; the repair leaves them alone."""
        from openmm.app import PDBFile, PDBxFile

        from binding_metrics.io.structures import load_structure

        pdb = PDBFile(str(DATA_DIR / "example_linear_p53_1YCR.pdb"))
        cif = tmp_path / "p53.cif"
        with open(cif, "w") as handle:
            PDBxFile.writeFile(pdb.topology, pdb.positions, handle)
        assert len(list(load_structure(cif)[0].bonds())) == len(
            list(PDBxFile(str(cif)).topology.bonds())
        )

    def test_repair_takes_an_explicit_residue_list(self):
        """``residues`` replaces the chain-ID lookup, which finds only the first chain of an ID."""
        from openmm.app import PDBFile, Topology

        from binding_metrics.core.cyclic import reconstruct_intraresidue_bonds

        pdb = PDBFile(str(PHOSPHO_PDB))
        topology, positions = pdb.topology, pdb.positions
        sep = next(r for r in topology.residues() if r.name == "SEP")
        assert _n_internal_bonds(topology, sep) == 9  # CONECT records
        # Rebuild the topology without the SEP internal bonds, as a raw mmCIF loads it.
        bare = Topology()
        atoms = {}
        for chain in topology.chains():
            new_chain = bare.addChain(chain.id)
            for res in chain.residues():
                new_res = bare.addResidue(res.name, new_chain, res.id)
                for atom in res.atoms():
                    atoms[atom.index] = bare.addAtom(atom.name, atom.element, new_res)
        for b in topology.bonds():
            if not (b.atom1.residue is sep and b.atom2.residue is sep):
                bare.addBond(atoms[b.atom1.index], atoms[b.atom2.index])
        residues = [r for r in bare.residues() if r.name == "SEP"]
        # The chain ID lookup would find the first chain of that name; the list is explicit.
        assert (
            reconstruct_intraresidue_bonds(bare, positions, "no-such-chain", residues=residues) == 9
        )
