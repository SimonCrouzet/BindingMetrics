"""Tests for the I/O module."""

from pathlib import Path

import pytest

from binding_metrics.io.structures import (
    detect_chains,
    get_chain_atom_indices,
    get_residue_info,
    load_complex,
    load_structure,
    merge_cif_models,
    save_cif,
    save_structure,
)

# ---------------------------------------------------------------------------
# Bundled example structures used by the round-trip / bond-preservation tests.
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
SFTI_CIF = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"        # head_to_tail + disulfide
CYCLOSPORIN_CIF = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"  # NCAA ring (BMT/ABA/...)
LINEAR_PDB = DATA_DIR / "example_linear_p53_1YCR.pdb"          # linear, all-standard control


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"bundled example not found: {path}")
    return path


class TestInternalResidueRenameBack:
    """save_cif must restore force-field-internal residue names on output."""

    def test_lactam_and_disulfide_names_restored(self, tmp_path: Path):
        """GLUL/LYSL/ASPL/CYX in a written CIF are renamed back to GLU/LYS/ASP/CYS.

        Regression: prep renames closing residues to the lactam templates
        (GLUL/LYSL/ASPL) and disulfide CYS to CYX. If those internal names leak
        into the saved file, detect_cyclization no longer recognises the closure
        on reload and relaxation raises a spurious CyclizationError. The disulfide
        rename was handled; the lactam ones were not, until this guard.
        """
        from binding_metrics.io.structures import _rename_internal_residues_to_standard

        cif = tmp_path / "internal.cif"
        # Minimal fragment carrying each internal residue code as a whole word,
        # plus a standard GLU/LYS/ASP that must NOT be touched.
        cif.write_text(
            "data_test\n"
            "_struct_conn.ptnr1_label_comp_id GLUL\n"
            "_struct_conn.ptnr2_label_comp_id LYSL\n"
            "ATOM 1 N N . CYX A 1 ?\n"
            "ATOM 2 N N . ASPL A 2 ?\n"
            "ATOM 3 N N . GLU A 3 ?\n"
        )
        _rename_internal_residues_to_standard(cif)
        out = cif.read_text()

        for internal in ("CYX", "GLUL", "LYSL", "ASPL"):
            assert internal not in out, f"{internal} leaked into output"
        for standard in ("CYS", "GLU", "LYS", "ASP"):
            assert standard in out, f"{standard} missing after rename-back"


def _coords_nm(positions):
    """Return atom positions as an (N, 3) numpy array in nanometers."""
    import numpy as np
    from openmm.unit import nanometer

    return np.array(positions.value_in_unit(nanometer))


def _nonseq_intra_bonds(topology):
    """Set of non-sequential intra-chain bonds, keyed chain-agnostically.

    Each bond becomes a frozenset of (residue_name, atom_name) endpoints, so the
    signature survives the chain-ID relabelling (label_asym → auth_asym) that a
    CIF round-trip performs. Only bonds whose residues are >1 apart within the
    same chain are kept — i.e. covalent closures, disulfides and NCAA links,
    never ordinary sequential peptide bonds.
    """
    out = []
    for bond in topology.bonds():
        r1, r2 = bond.atom1.residue, bond.atom2.residue
        if r1.chain.id == r2.chain.id and abs(r1.index - r2.index) > 1:
            out.append(
                frozenset((
                    (r1.name, bond.atom1.name),
                    (r2.name, bond.atom2.name),
                ))
            )
    return out


def _has_ring_closure(topology):
    """True if a backbone N–C amide closes a ring (non-sequential, same chain).

    This is the head-to-tail / macrocyclic closure signature (SFTI-1's
    GLY1.N–ASP14.C, cyclosporin's DAL1.N–ALA11.C). Chain-agnostic so it holds
    across the label→auth relabelling of a CIF round-trip.
    """
    for bond in topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        r1, r2 = a1.residue, a2.residue
        if (
            r1.chain.id == r2.chain.id
            and abs(r1.index - r2.index) > 1
            and {a1.name, a2.name} == {"N", "C"}
        ):
            return True
    return False


def _n_disulfides(topology):
    return sum(
        1 for b in topology.bonds() if b.atom1.name == "SG" and b.atom2.name == "SG"
    )


_WATER_NAMES = {"HOH", "WAT", "TIP", "TIP3", "SOL"}


def _residue_sequence(topology, exclude_water=False):
    """Flat list of residue names in topology order (structure fingerprint)."""
    return [
        r.name
        for r in topology.residues()
        if not (exclude_water and r.name in _WATER_NAMES)
    ]


def _n_water_residues(topology):
    return sum(1 for r in topology.residues() if r.name in _WATER_NAMES)


class TestLoadComplex:
    """Tests for load_complex function."""

    @pytest.mark.integration
    def test_load_valid_pdb(self, sample_pdb_path: Path):
        """Should successfully load a valid PDB file."""
        pdb = load_complex(sample_pdb_path)

        assert pdb is not None
        assert pdb.topology is not None
        assert pdb.positions is not None

    def test_load_nonexistent_file_raises(self, tmp_path: Path):
        """Should raise FileNotFoundError for missing file."""
        fake_path = tmp_path / "nonexistent.pdb"

        with pytest.raises(FileNotFoundError, match="not found"):
            load_complex(fake_path)

    @pytest.mark.integration
    def test_load_invalid_pdb_raises(self, tmp_path: Path):
        """Should raise ValueError for invalid PDB content."""
        bad_pdb = tmp_path / "bad.pdb"
        bad_pdb.write_text("This is not valid PDB content\nJust random text")

        with pytest.raises(ValueError, match="Failed to parse"):
            load_complex(bad_pdb)

    @pytest.mark.integration
    def test_load_accepts_path_object(self, sample_pdb_path: Path):
        """Should accept Path objects."""
        pdb = load_complex(sample_pdb_path)
        assert pdb is not None

    @pytest.mark.integration
    def test_load_accepts_string_path(self, sample_pdb_path: Path):
        """Should accept string paths."""
        pdb = load_complex(str(sample_pdb_path))
        assert pdb is not None


class TestGetChainAtomIndices:
    """Tests for get_chain_atom_indices function."""

    @pytest.mark.integration
    def test_get_single_chain_indices(self, sample_pdb_path: Path):
        """Should return indices for a single chain."""
        indices = get_chain_atom_indices(sample_pdb_path, ["A"])

        assert isinstance(indices, list)
        assert len(indices) > 0
        assert all(isinstance(i, int) for i in indices)

    @pytest.mark.integration
    def test_get_multiple_chain_indices(self, sample_pdb_path: Path):
        """Should return indices for multiple chains."""
        indices_a = get_chain_atom_indices(sample_pdb_path, ["A"])
        indices_b = get_chain_atom_indices(sample_pdb_path, ["B"])
        indices_ab = get_chain_atom_indices(sample_pdb_path, ["A", "B"])

        assert len(indices_ab) == len(indices_a) + len(indices_b)

    @pytest.mark.integration
    def test_nonexistent_chain_returns_empty(self, sample_pdb_path: Path):
        """Should return empty list for non-existent chain."""
        indices = get_chain_atom_indices(sample_pdb_path, ["Z"])

        assert indices == []

    @pytest.mark.integration
    def test_indices_are_zero_based(self, sample_pdb_path: Path):
        """Atom indices should be 0-based."""
        indices = get_chain_atom_indices(sample_pdb_path, ["A", "B"])

        if indices:
            assert min(indices) >= 0

    @pytest.mark.integration
    def test_chain_a_has_23_atoms(self, sample_pdb_path: Path):
        """Chain A (receptor) should have 23 atoms (with hydrogens)."""
        indices = get_chain_atom_indices(sample_pdb_path, ["A"])
        assert len(indices) == 23

    @pytest.mark.integration
    def test_chain_b_has_10_atoms(self, sample_pdb_path: Path):
        """Chain B (ligand) should have 10 atoms (with hydrogens)."""
        indices = get_chain_atom_indices(sample_pdb_path, ["B"])
        assert len(indices) == 10


class TestGetResidueInfo:
    """Tests for get_residue_info function."""

    @pytest.mark.integration
    def test_returns_list_of_dicts(self, sample_pdb_path: Path):
        """Should return a list of residue dictionaries."""
        residues = get_residue_info(sample_pdb_path)

        assert isinstance(residues, list)
        assert len(residues) > 0
        assert all(isinstance(r, dict) for r in residues)

    @pytest.mark.integration
    def test_residue_dict_has_required_keys(self, sample_pdb_path: Path):
        """Each residue dict should have name, index, chain, n_atoms."""
        residues = get_residue_info(sample_pdb_path)

        required_keys = {"name", "index", "chain", "n_atoms"}
        for residue in residues:
            assert required_keys.issubset(residue.keys())

    @pytest.mark.integration
    def test_detects_correct_residue_names(self, sample_pdb_path: Path):
        """Should detect ALA and GLY residues."""
        residues = get_residue_info(sample_pdb_path)
        names = {r["name"] for r in residues}

        assert "ALA" in names
        assert "GLY" in names

    @pytest.mark.integration
    def test_detects_correct_chains(self, sample_pdb_path: Path):
        """Should detect chains A and B."""
        residues = get_residue_info(sample_pdb_path)
        chains = {r["chain"] for r in residues}

        assert "A" in chains
        assert "B" in chains

    @pytest.mark.integration
    def test_residue_count(self, sample_pdb_path: Path):
        """Should find 3 residues total (2 ALA + 1 GLY)."""
        residues = get_residue_info(sample_pdb_path)
        assert len(residues) == 3


class TestLoadStructure:
    """Tests for the generalized load_structure function."""

    @pytest.mark.integration
    def test_load_pdb(self, sample_pdb_path: Path):
        """Should load a PDB file and return (topology, positions)."""
        topology, positions = load_structure(sample_pdb_path)
        assert topology is not None
        assert positions is not None

    def test_load_nonexistent_raises(self, tmp_path: Path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_structure(tmp_path / "missing.cif")

    def test_unsupported_format_raises(self, tmp_path: Path):
        """Should raise ValueError for unsupported format."""
        f = tmp_path / "structure.xyz"
        f.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_structure(f)

    @pytest.mark.integration
    def test_load_cif(self):
        """Should load a CIF file from test data."""
        cif_path = Path("data/example_ncaa_cyclosporin_1CWA.cif")
        if not cif_path.exists():
            pytest.skip("Test CIF not available")
        topology, positions = load_structure(cif_path)
        assert topology is not None
        assert len(list(topology.atoms())) > 0


class TestDetectChains:
    """Tests for detect_chains function."""

    @pytest.mark.integration
    def test_detect_two_chains(self, sample_pdb_path: Path):
        """Should detect ligand (smallest) and receptor (largest) chains."""
        topology, _ = load_structure(sample_pdb_path)
        ligand, receptor = detect_chains(topology)
        assert ligand is not None
        assert receptor is not None
        # Chain B (1 GLY) is smaller than chain A (2 ALA)
        assert ligand == "B"
        assert receptor == "A"

    @pytest.mark.integration
    def test_detect_chains_from_cif(self):
        """Should detect chains from a CIF file."""
        cif_path = Path("data/example_ncaa_cyclosporin_1CWA.cif")
        if not cif_path.exists():
            pytest.skip("Test CIF not available")
        topology, _ = load_structure(cif_path)
        ligand, receptor = detect_chains(topology)
        assert ligand is not None
        assert receptor is not None


class TestSaveCif:
    """Tests for save_cif function."""

    @pytest.mark.integration
    def test_save_cif_creates_file(self, sample_pdb_path: Path, tmp_path: Path):
        """Should create a CIF file at the specified path."""
        topology, positions = load_structure(sample_pdb_path)
        out_path = tmp_path / "output.cif"
        save_cif(topology, positions, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    @pytest.mark.integration
    def test_save_cif_roundtrips(self, sample_pdb_path: Path, tmp_path: Path):
        """The saved CIF must be a real, reloadable structure — not just non-empty.

        ``test_save_cif_creates_file`` only checks the file exists and has bytes, so
        a bug that wrote a malformed/truncated CIF would still pass. Reading it back
        with the same loader and comparing atom counts and chain IDs proves the
        downstream consumer actually accepts what save_cif produced.
        """
        topology, positions = load_structure(sample_pdb_path)
        out_path = tmp_path / "roundtrip.cif"
        save_cif(topology, positions, out_path)

        rt_topology, rt_positions = load_structure(out_path)

        orig_atoms = list(topology.atoms())
        rt_atoms = list(rt_topology.atoms())
        assert len(rt_atoms) == len(orig_atoms)
        assert len(rt_positions) == len(positions)

        orig_chains = {c.id for c in topology.chains()}
        rt_chains = {c.id for c in rt_topology.chains()}
        assert rt_chains == orig_chains


class TestRoundTripStructure:
    """load -> save_structure -> load must preserve the molecule.

    These exercise the real pipeline path (minimized structures are written via
    ``save_structure`` and later reloaded). Atoms, elements, residue structure
    and coordinates must survive both .cif and .pdb output unchanged.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "src",
        [SFTI_CIF, CYCLOSPORIN_CIF, LINEAR_PDB],
        ids=["sfti_cif", "cyclosporin_cif", "linear_pdb"],
    )
    def test_cif_output_preserves_atoms_elements_coords(self, src, tmp_path):
        """CIF round-trip keeps atom count, element identity, residue sequence, coords."""
        import numpy as np

        src = _require(src)
        topo, pos = load_structure(src)
        out = tmp_path / "rt.cif"
        save_structure(topo, pos, out, source_path=src)
        rt_topo, rt_pos = load_structure(out)

        assert rt_topo.getNumAtoms() == topo.getNumAtoms()
        assert len(rt_pos) == len(pos)
        # Element identity, in order.
        assert [a.element for a in rt_topo.atoms()] == [a.element for a in topo.atoms()]
        # Residue structure (names, in order), excluding water. Chain IDs are
        # legitimately relabelled by the CIF writer (label_asym → auth_asym) and
        # several label chains may collapse onto one auth chain, so the raw
        # chain-ID set and chain *count* are not comparable across the round-trip.
        # Water residue boundaries are separately fragile (see
        # test_cif_preserves_water_residue_count); the meaningful structural
        # fingerprint is the non-water residue sequence.
        assert _residue_sequence(rt_topo, exclude_water=True) == _residue_sequence(
            topo, exclude_water=True
        )
        # Coordinates, tight tolerance (CIF carries full precision here).
        dev = np.abs(_coords_nm(rt_pos) - _coords_nm(pos)).max()
        assert dev < 1e-4, f"coord deviation {dev:.2e} nm too large"

    @pytest.mark.integration
    @pytest.mark.xfail(
        strict=True,
        reason="save_cif merges two water residues on round-trip: its positional "
        "auth_seq_id restoration can assign the same residue number to two "
        "adjacent single-atom HOH residues, so OpenMM reads them back as one "
        "residue (SFTI-1: 101 waters -> 100). Atoms and coordinates are intact; "
        "only the water residue boundary is lost. Harmless in-pipeline (water is "
        "stripped) but still a silent structural mutation.",
    )
    def test_cif_preserves_water_residue_count(self, tmp_path):
        """CIF round-trip must not merge distinct water residues."""
        src = _require(SFTI_CIF)
        topo, pos = load_structure(src)
        assert _n_water_residues(topo) > 0  # sanity: source has waters
        out = tmp_path / "rt.cif"
        save_structure(topo, pos, out, source_path=src)
        rt_topo, _ = load_structure(out)
        assert _n_water_residues(rt_topo) == _n_water_residues(topo)

    @pytest.mark.integration
    def test_pdb_output_preserves_atoms_chains_coords(self, tmp_path):
        """.pdb round-trip (linear 1YCR control) keeps atoms, chains and coords."""
        import numpy as np

        src = _require(LINEAR_PDB)
        topo, pos = load_structure(src)
        out = tmp_path / "rt.pdb"
        save_structure(topo, pos, out, source_path=src)
        rt_topo, rt_pos = load_structure(out)

        assert rt_topo.getNumAtoms() == topo.getNumAtoms()
        assert {c.id for c in rt_topo.chains()} == {c.id for c in topo.chains()}
        assert [a.element for a in rt_topo.atoms()] == [a.element for a in topo.atoms()]
        # PDB stores 3 decimals in Angstrom → 1e-4 nm precision; allow a margin.
        dev = np.abs(_coords_nm(rt_pos) - _coords_nm(pos)).max()
        assert dev < 2e-4, f"coord deviation {dev:.2e} nm too large"


class TestRoundTripBonds:
    """The regression that matters: non-standard / covalent bonds must survive
    a save_structure round-trip. If they don't, a reloaded minimized cyclic /
    NCAA structure is silently missing its ring closure, disulfide or backbone
    links, and would be relaxed as an open chain.
    """

    # --- behaviours that already work today ---------------------------------

    @pytest.mark.integration
    def test_cif_preserves_disulfide(self, tmp_path):
        """SFTI-1's Cys3–Cys11 disulfide survives a CIF round-trip."""
        src = _require(SFTI_CIF)
        topo, pos = load_structure(src)
        assert _n_disulfides(topo) >= 1  # sanity: source has it
        out = tmp_path / "rt.cif"
        save_structure(topo, pos, out, source_path=src)
        rt_topo, _ = load_structure(out)
        assert _n_disulfides(rt_topo) >= 1

    @pytest.mark.integration
    def test_cif_preserves_ncaa_ring(self, tmp_path):
        """Cyclosporin's NCAA macrocycle closure (DAL1.N–ALA11.C) survives CIF."""
        src = _require(CYCLOSPORIN_CIF)
        topo, pos = load_structure(src)
        assert _has_ring_closure(topo)  # sanity: source has it
        out = tmp_path / "rt.cif"
        save_structure(topo, pos, out, source_path=src)
        rt_topo, _ = load_structure(out)
        assert _has_ring_closure(rt_topo), (
            "cyclosporin NCAA ring closure lost on CIF round-trip"
        )

    @pytest.mark.integration
    def test_pdb_preserves_ncaa_ring(self, tmp_path):
        """Cyclosporin's NCAA ring closure survives a .pdb round-trip."""
        src = _require(CYCLOSPORIN_CIF)
        topo, pos = load_structure(src)
        out = tmp_path / "rt.pdb"
        save_structure(topo, pos, out, source_path=src)
        rt_topo, _ = load_structure(out)
        assert _has_ring_closure(rt_topo), (
            "cyclosporin NCAA ring closure lost on PDB round-trip"
        )

    @pytest.mark.integration
    def test_pdb_preserves_disulfide(self, tmp_path):
        """SFTI-1's disulfide survives a .pdb round-trip (CONECT / distance)."""
        src = _require(SFTI_CIF)
        topo, pos = load_structure(src)
        out = tmp_path / "rt.pdb"
        save_structure(topo, pos, out, source_path=src)
        rt_topo, _ = load_structure(out)
        assert _n_disulfides(rt_topo) >= 1

    # --- the broken behaviour: head-to-tail between STANDARD residues -------
    #
    # SFTI-1's head-to-tail amide (GLY1.N–ASP14.C) joins two *standard* residues.
    # OpenMM's PDBxFile.writeFile only emits non-standard-touching or Cys–Cys
    # bonds to _struct_conn, so this bond is added by our own
    # _patch_nonstd_bonds_in_cif — but that patch writes the topology chain ID
    # and a chain-local residue index into the covale row, while save_cif has
    # already rewritten _atom_site to auth_asym_id / original auth_seq_id. The
    # covale row's keys therefore no longer match any atom on reload and OpenMM
    # silently drops the bond. (Cyclosporin escapes this because every ring bond
    # touches an NCAA and is emitted+patched correctly by the normal path.)

    @pytest.mark.integration
    def test_cif_preserves_head_to_tail(self, tmp_path):
        """SFTI-1 head-to-tail closure must survive a CIF round-trip.

        Regression guard for the fix to ``_patch_nonstd_bonds_in_cif``: the covale
        _struct_conn row must carry the auth_asym_id / auth_seq_id that actually
        appear in the written _atom_site (which is what OpenMM keys atoms on when
        reloading), not the topology chain id and a chain-local residue index.
        Two standard residues (GLY, ASP) form this bond, so OpenMM's own writer
        never emits it — only our patch does.
        """
        src = _require(SFTI_CIF)
        topo, pos = load_structure(src)
        assert _has_ring_closure(topo)  # sanity: source has it
        out = tmp_path / "rt.cif"
        save_structure(topo, pos, out, source_path=src)
        rt_topo, _ = load_structure(out)
        assert _has_ring_closure(rt_topo), (
            "SFTI-1 head-to-tail closure lost on CIF round-trip"
        )

    @pytest.mark.integration
    @pytest.mark.xfail(
        strict=True,
        reason="save_structure(.pdb) loses SFTI-1 head-to-tail closure: "
        "PDBFile.writeFile emits CONECT only for non-standard residues, and the "
        "GLY-ASP amide joins two standard residues, so no CONECT is written and "
        "the geometric distance is too long for OpenMM's disulfide/standard-bond "
        "reconstruction to recover it.",
    )
    def test_pdb_preserves_head_to_tail(self, tmp_path):
        """SFTI-1 head-to-tail closure must survive a .pdb round-trip."""
        src = _require(SFTI_CIF)
        topo, pos = load_structure(src)
        assert _has_ring_closure(topo)  # sanity: source has it
        out = tmp_path / "rt.pdb"
        save_structure(topo, pos, out, source_path=src)
        rt_topo, _ = load_structure(out)
        assert _has_ring_closure(rt_topo), (
            "SFTI-1 head-to-tail closure lost on PDB round-trip"
        )


class TestCrossFormat:
    """A structure loaded from CIF and the same structure written to PDB and
    reloaded must agree on the things PDB can represent (atoms, elements, coords).
    """

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "src",
        [CYCLOSPORIN_CIF, SFTI_CIF],
        ids=["cyclosporin", "sfti"],
    )
    def test_cif_vs_saved_pdb_agree_on_atoms_and_coords(self, src, tmp_path):
        import numpy as np

        src = _require(src)
        cif_topo, cif_pos = load_structure(src)
        out = tmp_path / "cross.pdb"
        save_structure(cif_topo, cif_pos, out, source_path=src)
        pdb_topo, pdb_pos = load_structure(out)

        assert pdb_topo.getNumAtoms() == cif_topo.getNumAtoms()
        assert [a.element for a in pdb_topo.atoms()] == [
            a.element for a in cif_topo.atoms()
        ]
        dev = np.abs(_coords_nm(pdb_pos) - _coords_nm(cif_pos)).max()
        assert dev < 2e-4, f"cross-format coord deviation {dev:.2e} nm too large"


class TestStructConnLandmine:
    """Guard the OpenMM PDBxFile struct_conn heuristic (openmm/app/pdbxfile.py
    ~L105-111): it keys atoms by auth_* but reads struct_conn via ptnr*_label_*,
    reconciling only when there are strictly more distinct label_asym_id values
    than auth_asym_id values. If that heuristic ever flips, the NCAA backbone
    bonds — which exist ONLY in struct_conn, since the residues are non-standard
    — vanish silently and the molecule would be relaxed with an open ring.
    """

    @pytest.mark.integration
    def test_load_cyclosporin_recovers_ncaa_bonds(self):
        """Loading cyclosporin resolves the NCAA backbone links from struct_conn."""
        src = _require(CYCLOSPORIN_CIF)
        topo, _ = load_structure(src)

        # Build a set of inter-residue bonds within the peptide as
        # ((resname, resid, atom), (resname, resid, atom)) unordered pairs.
        links = set()
        for bond in topo.bonds():
            r1, r2 = bond.atom1.residue, bond.atom2.residue
            if r1.chain.id == r2.chain.id and r1.index != r2.index:
                a = (r1.name, r1.id, bond.atom1.name)
                b = (r2.name, r2.id, bond.atom2.name)
                links.add(frozenset((a, b)))

        # NCAA residues must be covalently wired into the backbone. These bonds
        # exist ONLY via struct_conn (BMT/ABA/MVA/... are non-standard, so
        # createStandardBonds cannot build them).
        expected = [
            frozenset({("MVA", "4", "C"), ("BMT", "5", "N")}),
            frozenset({("BMT", "5", "C"), ("ABA", "6", "N")}),
            frozenset({("ABA", "6", "C"), ("SAR", "7", "N")}),
        ]
        for e in expected:
            assert e in links, (
                f"NCAA backbone bond {tuple(e)} missing after load — the OpenMM "
                "struct_conn label/auth heuristic may have flipped."
            )

        # And the macrocycle must be closed (DAL1.N–ALA11.C).
        assert _has_ring_closure(topo), (
            "cyclosporin macrocycle not closed after load — struct_conn heuristic "
            "may have flipped."
        )


class TestMergeCifModels:
    """merge_cif_models must key _atom_site columns by tag, not by position."""

    @staticmethod
    def _write_single_model_cif(path: Path, tags: list[str], rows: list[list[str]]) -> None:
        """Write a minimal CIF whose _atom_site loop uses the given column order."""
        lines = ["data_test", "loop_"]
        lines += [f"_atom_site.{t}" for t in tags]
        lines += [" ".join(r) for r in rows]
        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def _read_atom_site(path: Path) -> tuple[list[str], list[list[str]]]:
        gemmi = pytest.importorskip("gemmi")
        loop = gemmi.cif.read(str(path)).sole_block().find_loop("_atom_site.id").get_loop()
        tags = list(loop.tags)
        width = loop.width()
        vals = list(loop.values)
        rows = [vals[r * width : (r + 1) * width] for r in range(len(vals) // width)]
        return tags, rows

    @pytest.mark.integration
    def test_merges_models_with_divergent_column_order(self, tmp_path: Path):
        """A second model whose _atom_site columns are ordered differently must
        still land in the right columns.

        Nothing in the CIF spec fixes _atom_site column order, and different
        writers emit different orders. Merging by column index rather than by
        tag silently transposes fields — y coordinates into x, element into
        atom name — producing a file that parses cleanly and is geometrically
        wrong, which is the worst possible failure mode here.
        """
        tags_a = ["id", "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num"]
        # Same six columns, coordinates permuted relative to model 1.
        tags_b = ["id", "type_symbol", "Cartn_z", "Cartn_x", "Cartn_y", "pdbx_PDB_model_num"]

        model_a = tmp_path / "m1.cif"
        model_b = tmp_path / "m2.cif"
        self._write_single_model_cif(
            model_a, tags_a, [["1", "C", "1.000", "2.000", "3.000", "1"]]
        )
        # Written in tags_b order, this row still means x=4, y=5, z=6.
        self._write_single_model_cif(
            model_b, tags_b, [["1", "C", "6.000", "4.000", "5.000", "1"]]
        )

        out = tmp_path / "merged.cif"
        merge_cif_models([(1, model_a), (2, model_b)], out)

        tags, rows = self._read_atom_site(out)
        assert len(rows) == 2, "merged file should hold one row per input model"

        def field(row: list[str], tag: str) -> str:
            return row[tags.index(f"_atom_site.{tag}")]

        assert [field(rows[0], t) for t in ("Cartn_x", "Cartn_y", "Cartn_z")] == [
            "1.000", "2.000", "3.000",
        ]
        assert [field(rows[1], t) for t in ("Cartn_x", "Cartn_y", "Cartn_z")] == [
            "4.000", "5.000", "6.000",
        ], "model 2 coordinates were read positionally instead of by column tag"

        assert field(rows[0], "pdbx_PDB_model_num") == "1"
        assert field(rows[1], "pdbx_PDB_model_num") == "2"

    @pytest.mark.integration
    def test_rejects_model_missing_a_column(self, tmp_path: Path):
        """A model lacking a column the first model has must fail loudly."""
        tags_a = ["id", "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num"]
        tags_b = ["id", "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num"]

        model_a = tmp_path / "m1.cif"
        model_b = tmp_path / "m2.cif"
        self._write_single_model_cif(
            model_a, tags_a, [["1", "C", "1.000", "2.000", "3.000", "1"]]
        )
        self._write_single_model_cif(model_b, tags_b, [["1", "4.000", "5.000", "6.000", "1"]])

        with pytest.raises(ValueError, match="type_symbol"):
            merge_cif_models([(1, model_a), (2, model_b)], tmp_path / "merged.cif")
