"""Tests for cyclic peptide detection and topology patching.

All tests run without OpenMM (pure-Python or lightweight) except those that
build a real topology, which are marked @pytest.mark.integration.
"""

import numpy as np
import pytest

from binding_metrics.core.cyclic import (
    _XML_ASPL,
    _XML_GLUL,
    _XML_LYSL,
    CyclicBondInfo,
    CyclizationError,
)

# ---------------------------------------------------------------------------
# CyclicBondInfo
# ---------------------------------------------------------------------------


class TestCyclicBondInfo:
    def test_fields(self):
        info = CyclicBondInfo(
            cyclic_type="head_to_tail",
            atom1_id=("A", 4, "C"),
            atom2_id=("A", 0, "N"),
        )
        assert info.cyclic_type == "head_to_tail"
        assert info.extra_ff_xmls == []
        assert info.omega_ids is None

    def test_extra_ff_xmls_populated(self):
        info = CyclicBondInfo(
            cyclic_type="lactam_n_asp",
            atom1_id=("A", 2, "CG"),
            atom2_id=("A", 0, "N"),
            extra_ff_xmls=[_XML_ASPL],
        )
        assert len(info.extra_ff_xmls) == 1
        assert "ASPL" in info.extra_ff_xmls[0]


# ---------------------------------------------------------------------------
# XML template sanity checks (no OpenMM needed)
# ---------------------------------------------------------------------------


class TestLactamXMLTemplates:
    """Verify the XML strings are well-formed and contain expected tokens."""

    def _check_xml(self, xml_str, residue_name, amide_c_atom, amide_o_atom, closure_atom):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_str)
        residues = root.findall(".//Residue")
        names = [r.get("name") for r in residues]
        assert residue_name in names, f"{residue_name} missing from XML"

        res = next(r for r in residues if r.get("name") == residue_name)

        atoms = {a.get("name"): a for a in res.findall("Atom")}
        assert amide_c_atom in atoms, f"{amide_c_atom} missing"
        assert amide_o_atom in atoms, f"{amide_o_atom} missing"
        assert atoms[amide_c_atom].get("type") == "protein-C"
        assert atoms[amide_o_atom].get("type") == "protein-O"

        ext_bonds = [e.get("atomName") for e in res.findall("ExternalBond")]
        assert closure_atom in ext_bonds, f"ExternalBond for {closure_atom} missing"
        assert "N" in ext_bonds
        assert "C" in ext_bonds

        # Check charge neutrality
        total = sum(float(a.get("charge", 0)) for a in atoms.values())
        assert abs(total) < 1e-3, f"Net charge {total:.4f} ≠ 0 for {residue_name}"

    def test_aspl_xml(self):
        self._check_xml(_XML_ASPL, "ASPL", "CG", "OD1", "CG")

    def test_glul_xml(self):
        self._check_xml(_XML_GLUL, "GLUL", "CD", "OE1", "CD")

    def test_lysl_xml(self):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(_XML_LYSL)
        res = root.find(".//Residue[@name='LYSL']")
        assert res is not None
        atoms = {a.get("name"): a for a in res.findall("Atom")}
        assert atoms["NZ"].get("type") == "protein-N"
        ext_bonds = [e.get("atomName") for e in res.findall("ExternalBond")]
        assert "NZ" in ext_bonds
        total = sum(float(a.get("charge", 0)) for a in atoms.values())
        assert abs(total) < 1e-3


# ---------------------------------------------------------------------------
# Detection logic (requires OpenMM topology mocks)
# ---------------------------------------------------------------------------


def _make_minimal_topology(residue_names, atom_specs, bonds=None):
    """Build a minimal OpenMM Topology for testing.

    Args:
        residue_names: list of residue name strings.
        atom_specs: list of lists of (atom_name, element_symbol) per residue.
        bonds: list of (chain_id, res_idx, atom_name, chain_id, res_idx, atom_name)
               or None. Backbone N-C bonds between adjacent residues added auto.

    Returns:
        (topology, positions) where positions is an (N,3) numpy array in nm.
    """
    pytest.importorskip("openmm", reason="OpenMM required")
    from openmm import app
    from openmm.app import element as elem

    _elem = {
        "N": elem.nitrogen,
        "C": elem.carbon,
        "O": elem.oxygen,
        "S": elem.sulfur,
        "H": elem.hydrogen,
    }

    topology = app.Topology()
    chain = topology.addChain(id="A")
    atom_map = {}  # (res_idx, atom_name) → Atom
    pos_list = []
    atom_count = 0

    for ri, (res_name, specs) in enumerate(zip(residue_names, atom_specs)):
        res = topology.addResidue(res_name, chain)
        for aname, esym in specs:
            el = _elem.get(esym.upper(), elem.carbon)
            atom = topology.addAtom(aname, el, res)
            atom_map[(ri, aname)] = atom
            # Spread atoms 2 Å apart in x for no accidental close contacts
            pos_list.append([atom_count * 1.0, 0.0, 0.0])
            atom_count += 1

    # Add backbone N–C bonds between adjacent residues
    for ri in range(len(residue_names) - 1):
        c_atom = atom_map.get((ri, "C"))
        n_next = atom_map.get((ri + 1, "N"))
        if c_atom and n_next:
            topology.addBond(c_atom, n_next)

    # Add intra-residue bonds (simple: connect atoms in order within each residue)
    # (not needed for distance-based tests, skip)

    positions = np.array(pos_list, dtype=float)
    return topology, positions, atom_map


@pytest.mark.integration
class TestDetectCyclization:
    """Test detect_cyclization with synthetic minimal topologies."""

    def test_head_to_tail_detected(self):
        from binding_metrics.core.cyclic import detect_cyclization

        res_names = ["ALA", "GLY", "ALA"]
        specs = [
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
        ]
        topology, positions, atom_map = _make_minimal_topology(res_names, specs)
        positions[atom_map[(2, "C")].index] = positions[atom_map[(0, "N")].index] + np.array(
            [0.133, 0, 0]
        )
        result = detect_cyclization(topology, positions, "A")
        assert len(result) == 1
        assert result[0].cyclic_type == "head_to_tail"

    def test_no_cyclization_returns_empty_list(self):
        from binding_metrics.core.cyclic import detect_cyclization

        res_names = ["ALA", "GLY"]
        specs = [
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
        ]
        topology, positions, _ = _make_minimal_topology(res_names, specs)
        result = detect_cyclization(topology, positions, "A")
        assert result == []

    def test_disulfide_detected(self):
        from binding_metrics.core.cyclic import detect_cyclization

        res_names = ["CYS", "ALA", "CYS"]
        specs = [
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"), ("SG", "S")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"), ("SG", "S")],
        ]
        topology, positions, atom_map = _make_minimal_topology(res_names, specs)
        sg0_idx = atom_map[(0, "SG")].index
        sg2_idx = atom_map[(2, "SG")].index
        positions[sg2_idx] = positions[sg0_idx] + np.array([0.205, 0, 0])
        result = detect_cyclization(topology, positions, "A")
        assert len(result) == 1
        assert result[0].cyclic_type == "disulfide"

    def test_bicyclic_head_to_tail_plus_disulfide(self):
        """Bicyclic: both head-to-tail AND disulfide detected in one call."""
        from binding_metrics.core.cyclic import detect_cyclization

        res_names = ["CYS", "ALA", "CYS", "GLY"]
        specs = [
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"), ("SG", "S")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"), ("SG", "S")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
        ]
        topology, positions, atom_map = _make_minimal_topology(res_names, specs)
        # Head-to-tail: C(GLY3) close to N(CYS0)
        positions[atom_map[(3, "C")].index] = positions[atom_map[(0, "N")].index] + np.array(
            [0.133, 0, 0]
        )
        # Disulfide: SG(CYS0) close to SG(CYS2)
        positions[atom_map[(2, "SG")].index] = positions[atom_map[(0, "SG")].index] + np.array(
            [0.205, 0, 0]
        )
        result = detect_cyclization(topology, positions, "A")
        types = {r.cyclic_type for r in result}
        assert "head_to_tail" in types
        assert "disulfide" in types
        assert len(result) == 2

    def test_unsupported_cyclization_raises(self):
        # detect_cyclization raises only on a non-sequential intra-chain
        # topology bond that matches no supported pattern (a covalent record,
        # not a mere close contact — distance scanning was dropped to avoid
        # false positives from tight backbone geometry). An S–C thioether
        # bridge is a genuinely unsupported cross-link (an all-carbon C–C
        # side-chain bridge is now recognised as a hydrocarbon staple).
        from binding_metrics.core.cyclic import detect_cyclization

        res_names = ["CYS", "ALA", "ALA"]
        specs = [
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"), ("SG", "S")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")],
        ]
        topology, positions, atom_map = _make_minimal_topology(res_names, specs)
        # Encode an unsupported thioether (S–C) bond between non-adjacent
        # residues 0 and 2.
        topology.addBond(atom_map[(0, "SG")], atom_map[(2, "CB")])
        with pytest.raises(CyclizationError, match="Unsupported"):
            detect_cyclization(topology, positions, "A")

    def test_lactam_asp_detected(self):
        from binding_metrics.core.cyclic import detect_cyclization

        res_names = ["ALA", "ASP", "GLY"]
        specs = [
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
            [
                ("N", "N"),
                ("CA", "C"),
                ("C", "C"),
                ("O", "O"),
                ("CB", "C"),
                ("CG", "C"),
                ("OD1", "O"),
                ("OD2", "O"),
            ],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
        ]
        topology, positions, atom_map = _make_minimal_topology(res_names, specs)
        positions[atom_map[(1, "CG")].index] = positions[atom_map[(0, "N")].index] + np.array(
            [0.133, 0, 0]
        )
        result = detect_cyclization(topology, positions, "A")
        assert len(result) == 1
        assert result[0].cyclic_type == "lactam_n_asp"
        assert _XML_ASPL in result[0].extra_ff_xmls


# ---------------------------------------------------------------------------
# RelaxationConfig: cyclic_bond_hints field
#
# Cyclization is always auto-detected (there is no is_cyclic flag). The only
# cyclic-related config field is cyclic_bond_hints, an optional list of
# CyclicBondInfo used as a fallback when a prepped file has lost its
# STRUCT_CONN records and geometry is too strained for distance detection.
# ---------------------------------------------------------------------------


class TestRelaxationConfigCyclic:
    def test_default_cyclic_bond_hints_none(self):
        from binding_metrics.protocols.relaxation import RelaxationConfig

        config = RelaxationConfig()
        assert config.cyclic_bond_hints is None

    def test_cyclic_bond_hints_settable(self):
        from binding_metrics.protocols.relaxation import RelaxationConfig

        hint = CyclicBondInfo(
            cyclic_type="head_to_tail",
            atom1_id=("A", 4, "C"),
            atom2_id=("A", 0, "N"),
        )
        config = RelaxationConfig(cyclic_bond_hints=[hint])
        assert config.cyclic_bond_hints == [hint]


# ---------------------------------------------------------------------------
# _internal_h_list and get_addh_variants
# ---------------------------------------------------------------------------


class TestInternalHList:
    """Tests for _internal_h_list: no terminal H atoms should be returned."""

    def test_gly_no_terminal_h(self):
        pytest.importorskip("openmm", reason="OpenMM required")
        from binding_metrics.core.cyclic import _internal_h_list

        h_list = _internal_h_list("GLY")
        assert h_list is not None
        names = [h for h, _ in h_list]
        # Internal GLY H: H (amide NH), HA2, HA3 — no H2/H3 (N-terminal only)
        assert "H" in names
        assert "HA2" in names
        assert "HA3" in names
        assert "H2" not in names
        assert "H3" not in names

    def test_ala_no_terminal_h(self):
        pytest.importorskip("openmm", reason="OpenMM required")
        from binding_metrics.core.cyclic import _internal_h_list

        h_list = _internal_h_list("ALA")
        assert h_list is not None
        names = [h for h, _ in h_list]
        assert "H" in names
        assert "H2" not in names
        assert "H3" not in names
        # OXT / HXT are C-terminal only — must not appear
        assert "HXT" not in names

    def test_unknown_residue_returns_none(self):
        pytest.importorskip("openmm", reason="OpenMM required")
        from binding_metrics.core.cyclic import _internal_h_list

        assert _internal_h_list("XYZ") is None


@pytest.mark.integration
class TestGetAddhVariants:
    """Tests for get_addh_variants: GLY-first head-to-tail peptide (SFTI-1 case)."""

    def _make_sfti_like_topology(self):
        """Minimal bicyclic topology: GLY-ARG-CYS-... (head-to-tail + disulfide)."""
        pytest.importorskip("openmm", reason="OpenMM required")
        from openmm import app
        from openmm.app import element as elem

        _elem = {
            "N": elem.nitrogen,
            "C": elem.carbon,
            "O": elem.oxygen,
            "S": elem.sulfur,
            "H": elem.hydrogen,
        }

        topology = app.Topology()
        chain = topology.addChain(id="B")
        pos_list = []
        atom_count = [0]

        def add_res(name, specs):
            res = topology.addResidue(name, chain)
            atoms = {}
            for aname, esym in specs:
                a = topology.addAtom(aname, _elem.get(esym, elem.carbon), res)
                atoms[aname] = a
                pos_list.append([atom_count[0] * 0.5, 0.0, 0.0])
                atom_count[0] += 1
            return res, atoms

        # SFTI-1 sequence starts with GLY
        r0, m0 = add_res("GLY", [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")])
        r1, m1 = add_res("ARG", [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")])
        r2, m2 = add_res(
            "CYS", [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"), ("SG", "S")]
        )
        r3, m3 = add_res(
            "CYS", [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"), ("SG", "S")]
        )

        # Backbone bonds
        topology.addBond(m0["C"], m1["N"])
        topology.addBond(m1["C"], m2["N"])
        topology.addBond(m2["C"], m3["N"])

        positions = np.array(pos_list, dtype=float)

        # Head-to-tail: C(r3) close to N(r0)
        positions[m3["C"].index] = positions[m0["N"].index] + np.array([0.133, 0, 0])
        # Disulfide: SG(r2) close to SG(r3)
        positions[m2["SG"].index] = positions[m3["SG"].index] + np.array([0.205, 0, 0])

        return topology, positions

    def test_gly_first_residue_gets_list_variant(self):
        from binding_metrics.core.cyclic import (
            CyclicBondInfo,
            get_addh_variants,
        )

        topology, positions = self._make_sfti_like_topology()
        bond_info = [
            CyclicBondInfo(
                cyclic_type="head_to_tail",
                atom1_id=("B", 3, "C"),
                atom2_id=("B", 0, "N"),
            )
        ]
        variants = get_addh_variants(topology, bond_info, "B")

        # First residue (GLY at index 0) must get a list variant, not a string
        gly_res = next(r for r in topology.residues() if r.name == "GLY")
        v = variants[gly_res.index]
        assert v is not None, "GLY first residue must get a variant"
        assert isinstance(v, list), f"Expected list variant for GLY, got {type(v)}: {v}"
        names = [h for h, _ in v]
        assert "H" in names
        assert "H2" not in names, "H2 (N-terminal) must be excluded for cyclic GLY"
        assert "H3" not in names, "H3 (N-terminal) must be excluded for cyclic GLY"

    def test_addhydrogens_does_not_crash_on_gly_first(self):
        """Regression: addHydrogens must not raise 'Illegal variant for GLY'.

        Uses a minimal synthetic topology — the call may fail for other reasons
        (incomplete residue geometry), but must NOT fail with 'Illegal variant'.
        """
        from openmm import Vec3, app

        from binding_metrics.core.cyclic import (
            get_addh_variants,
            load_extra_xmls,
            patch_cyclic_topology,
        )

        topology, positions = self._make_sfti_like_topology()
        # Pass positions WITHOUT units so _CellList comparisons stay in plain floats
        pos_vec = [Vec3(p[0], p[1], p[2]) for p in positions]

        topology, pos_vec, bond_info = patch_cyclic_topology(topology, pos_vec, "B")

        ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
        load_extra_xmls(ff, bond_info)

        variants = get_addh_variants(topology, bond_info, "B")
        modeller = app.Modeller(topology, pos_vec)
        try:
            modeller.addHydrogens(pH=7.0, variants=variants)
        except Exception as exc:
            # Acceptable to fail for geometry/template reasons on a fake topology,
            # but must NOT be an "Illegal variant" error for GLY.
            assert "Illegal variant" not in str(exc), (
                f"addHydrogens raised 'Illegal variant' — GLY fix is broken: {exc}"
            )


# ---------------------------------------------------------------------------
# Side-chain-to-side-chain lactam staples  (#1: lactam_sc_lys_asp / _glu)
# ---------------------------------------------------------------------------

_BB = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
_LYS_SPEC = _BB + [("CB", "C"), ("CG", "C"), ("CD", "C"), ("CE", "C"), ("NZ", "N")]
_ASP_SPEC = _BB + [("CB", "C"), ("CG", "C"), ("OD1", "O"), ("OD2", "O")]
_GLU_SPEC = _BB + [("CB", "C"), ("CG", "C"), ("CD", "C"), ("OE1", "O"), ("OE2", "O")]


@pytest.mark.integration
class TestDetectSidechainLactam:
    """detect_cyclization for internal LYS–ASP / LYS–GLU side-chain lactams."""

    def test_lys_asp_sidechain_detected(self):
        from binding_metrics.core.cyclic import detect_cyclization

        res_names = ["ALA", "LYS", "ALA", "ALA", "ALA", "ASP", "ALA"]
        specs = [_BB, _LYS_SPEC, _BB, _BB, _BB, _ASP_SPEC, _BB]
        topology, positions, atom_map = _make_minimal_topology(res_names, specs)
        # Place LYS NZ within amide-bond distance of ASP CG (i,i+4 staple)
        positions[atom_map[(1, "NZ")].index] = positions[atom_map[(5, "CG")].index] + np.array(
            [0.133, 0, 0]
        )
        result = detect_cyclization(topology, positions, "A")
        assert len(result) == 1
        info = result[0]
        assert info.cyclic_type == "lactam_sc_lys_asp"
        assert info.atom1_id == ("A", 1, "NZ")
        assert info.atom2_id == ("A", 5, "CG")
        assert _XML_LYSL in info.extra_ff_xmls
        assert _XML_ASPL in info.extra_ff_xmls
        assert info.omega_ids is not None

    def test_lys_glu_sidechain_detected(self):
        from binding_metrics.core.cyclic import detect_cyclization

        res_names = ["ALA", "LYS", "ALA", "ALA", "ALA", "GLU", "ALA"]
        specs = [_BB, _LYS_SPEC, _BB, _BB, _BB, _GLU_SPEC, _BB]
        topology, positions, atom_map = _make_minimal_topology(res_names, specs)
        positions[atom_map[(1, "NZ")].index] = positions[atom_map[(5, "CD")].index] + np.array(
            [0.133, 0, 0]
        )
        result = detect_cyclization(topology, positions, "A")
        assert len(result) == 1
        info = result[0]
        assert info.cyclic_type == "lactam_sc_lys_glu"
        assert info.atom2_id == ("A", 5, "CD")
        assert _XML_GLUL in info.extra_ff_xmls

    def test_no_false_positive_when_far(self):
        """Far-apart LYS and ASP side chains must not be flagged as a staple."""
        from binding_metrics.core.cyclic import detect_cyclization

        res_names = ["ALA", "LYS", "ALA", "ALA", "ALA", "ASP", "ALA"]
        specs = [_BB, _LYS_SPEC, _BB, _BB, _BB, _ASP_SPEC, _BB]
        topology, positions, atom_map = _make_minimal_topology(res_names, specs)
        result = detect_cyclization(topology, positions, "A")
        assert result == []


class TestLactamInternalHList:
    """_internal_h_list must return the correct H set for the lactam residues."""

    def test_lysl(self):
        pytest.importorskip("openmm", reason="OpenMM required")
        from binding_metrics.core.cyclic import _internal_h_list

        names = {n for n, _ in _internal_h_list("LYSL")}
        assert "HZ1" in names  # single amide proton kept
        assert "HZ2" not in names and "HZ3" not in names
        assert "H" in names  # backbone amide NH

    def test_aspl(self):
        pytest.importorskip("openmm", reason="OpenMM required")
        from binding_metrics.core.cyclic import _internal_h_list

        names = {n for n, _ in _internal_h_list("ASPL")}
        assert names == {"H", "HA", "HB2", "HB3"}  # HD2 dropped (OD2 gone)

    def test_glul(self):
        pytest.importorskip("openmm", reason="OpenMM required")
        from binding_metrics.core.cyclic import _internal_h_list

        names = {n for n, _ in _internal_h_list("GLUL")}
        assert names == {"H", "HA", "HB2", "HB3", "HG2", "HG3"}


class TestLactamTemplateConnectivity:
    """Regression guard: every atom in a lactam template must be bonded.

    The H atoms were originally declared without bonds to their parent heavy
    atoms, which made createSystem reject the built residue ('too many H-C
    bonds').  This test fails if that omission recurs.
    """

    @pytest.mark.parametrize(
        "xml_str",
        [_XML_ASPL, _XML_GLUL, _XML_LYSL],
        ids=["ASPL", "GLUL", "LYSL"],
    )
    def test_no_unbonded_atoms(self, xml_str):
        import xml.etree.ElementTree as ET

        res = ET.fromstring(xml_str).find(".//Residue")
        atom_names = {a.get("name") for a in res.findall("Atom")}
        bonded = set()
        for b in res.findall("Bond"):
            bonded.add(b.get("atomName1"))
            bonded.add(b.get("atomName2"))
        external = {e.get("atomName") for e in res.findall("ExternalBond")}
        unbonded = atom_names - bonded - external
        assert not unbonded, f"{res.get('name')}: unbonded atoms {sorted(unbonded)}"


@pytest.mark.integration
class TestSidechainLactamEndToEnd:
    """Full patch -> addHydrogens -> createSystem for a real heavy-atom peptide."""

    def _heavy_atom_peptide(self, seq):
        import os
        import tempfile

        pytest.importorskip("openmm", reason="OpenMM required")
        chem = pytest.importorskip("rdkit.Chem", reason="RDKit required")
        from openmm.app import Modeller, PDBFile
        from rdkit.Chem import AllChem

        mol = chem.MolFromSequence(seq)
        mol = chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=1)
        mol = chem.RemoveHs(mol)  # heavy-atom-only input
        fd, path = tempfile.mkstemp(suffix=".pdb")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(chem.MolToPDBBlock(mol))
            pdb = PDBFile(path)
        finally:
            os.unlink(path)
        modeller = Modeller(pdb.topology, pdb.positions)
        chain_id = next(iter(modeller.topology.chains())).id
        return modeller.topology, modeller.positions, chain_id

    def _run(self, seq, lys_idx, acid_idx, acid_atom, expect_name):
        from openmm.app import ForceField, HBonds, Modeller, NoCutoff

        from binding_metrics.core.cyclic import (
            get_addh_variants,
            load_extra_xmls,
            patch_cyclic_topology,
            resolve_closure_atoms,
        )

        top, pos, chain_id = self._heavy_atom_peptide(seq)

        # Introduce the closure bond as a prepped staple structure would carry it.
        residues = list(next(iter(top.chains())).residues())
        a1 = next(a for a in residues[lys_idx].atoms() if a.name == "NZ")
        a2 = next(a for a in residues[acid_idx].atoms() if a.name == acid_atom)
        top.addBond(a1, a2)

        top, pos, bond_info = patch_cyclic_topology(top, pos, chain_id)
        names = [r.name for r in next(iter(top.chains())).residues()]
        assert "LYSL" in names and expect_name in names

        ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        load_extra_xmls(ff, bond_info)
        modeller = Modeller(top, pos)
        variants = get_addh_variants(modeller.topology, bond_info, chain_id)
        modeller.addHydrogens(ff, pH=7.4, variants=variants)

        system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
        assert system.getNumParticles() == modeller.topology.getNumAtoms()

        i1, i2 = resolve_closure_atoms(modeller.topology, bond_info[0], chain_id)
        assert any({b.atom1.index, b.atom2.index} == {i1, i2} for b in modeller.topology.bonds())

    def test_lys_asp_staple_builds(self):
        # G-K-A-A-A-D-G : LYS(1) NZ — ASP(5) CG  (i,i+4)
        self._run("GKAAADG", 1, 5, "CG", "ASPL")

    def test_lys_glu_staple_builds(self):
        # G-K-A-A-A-E-G : LYS(1) NZ — GLU(5) CD  (i,i+4)
        self._run("GKAAAEG", 1, 5, "CD", "GLUL")


@pytest.mark.integration
class TestDisulfideRenameApplies:
    """Regression: rename_disulfide_cys_to_cyx must rewrite res.name to CYX (not a
    dead ._name), so the CYX force-field template — which requires the HG-less,
    externally-SG-bonded residue — matches at createSystem. test_disulfide_detected
    only covers detection, not the applied rename."""

    _CYS = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"), ("SG", "S")]
    _ALA = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]

    def test_ss_bonded_cys_become_cyx(self):
        from binding_metrics.core.cyclic import rename_disulfide_cys_to_cyx

        topology, positions, atom_map = _make_minimal_topology(
            ["CYS", "ALA", "CYS"], [self._CYS, self._ALA, self._CYS]
        )
        # Put the two SG atoms at a disulfide distance (~2.05 Å).
        positions[atom_map[(2, "SG")].index] = positions[atom_map[(0, "SG")].index] + np.array(
            [0.205, 0, 0]
        )
        top2, _ = rename_disulfide_cys_to_cyx(topology, positions)
        assert [r.name for r in top2.residues()] == ["CYX", "ALA", "CYX"]

    def test_free_cys_not_renamed(self):
        # No close SG contact (atoms spread far apart) → CYS stays CYS, no spurious CYX.
        from binding_metrics.core.cyclic import rename_disulfide_cys_to_cyx

        topology, positions, atom_map = _make_minimal_topology(
            ["CYS", "ALA", "CYS"], [self._CYS, self._ALA, self._CYS]
        )
        top2, _ = rename_disulfide_cys_to_cyx(topology, positions)
        assert [r.name for r in top2.residues()] == ["CYS", "ALA", "CYS"]


# ---------------------------------------------------------------------------
# Regression: non-standard residues lose their intra-residue bonds at CIF load
# (createStandardBonds only bonds standard names).  This broke the head-to-tail
# cyclosporin junction (a D-Ala renamed to ALA whose bonds were never built) and
# the N-methyl / GAFF residues.  Covered by reconstruct_intraresidue_bonds and
# the NMe hydrogen-variant support.
# ---------------------------------------------------------------------------


def _make_bondless_ala_chain():
    """A one-residue ALA chain with realistic heavy-atom geometry but NO bonds.

    Mirrors how a residue that was non-standard at load time (e.g. DAL renamed to
    ALA) arrives: correct atoms/coordinates, zero intra-residue bonds.
    """
    pytest.importorskip("openmm", reason="OpenMM required")
    import openmm.unit as unit
    from openmm import Vec3, app
    from openmm.app import element as elem

    top = app.Topology()
    chain = top.addChain(id="B")
    res = top.addResidue("ALA", chain)
    coords_nm = {
        "N": (0.000, 0.000, 0.0),
        "CA": (0.147, 0.000, 0.0),
        "C": (0.147, 0.153, 0.0),
        "O": (0.270, 0.153, 0.0),
        "CB": (0.147, -0.153, 0.0),
    }
    els = {
        "N": elem.nitrogen,
        "CA": elem.carbon,
        "C": elem.carbon,
        "O": elem.oxygen,
        "CB": elem.carbon,
    }
    pos = []
    amap = {}
    for name, xyz in coords_nm.items():
        a = top.addAtom(name, els[name], res)
        amap[name] = a
        pos.append(Vec3(*xyz))
    positions = unit.Quantity(pos, unit.nanometer)
    return top, positions, res, amap


@pytest.mark.integration
class TestReconstructIntraresidueBonds:
    """reconstruct_intraresidue_bonds restores bonds only for bond-less residues."""

    def test_bondless_residue_gets_standard_connectivity(self):
        from binding_metrics.core.cyclic import reconstruct_intraresidue_bonds

        top, positions, res, amap = _make_bondless_ala_chain()
        assert top.getNumBonds() == 0

        added = reconstruct_intraresidue_bonds(top, positions, "B")

        bonded = {frozenset((b.atom1.name, b.atom2.name)) for b in top.bonds()}
        assert added == 4
        assert frozenset(("N", "CA")) in bonded
        assert frozenset(("CA", "C")) in bonded
        assert frozenset(("CA", "CB")) in bonded
        assert frozenset(("C", "O")) in bonded
        # No spurious 1,3 bonds (e.g. N–C, N–CB, CA–O).
        assert frozenset(("N", "C")) not in bonded
        assert frozenset(("N", "CB")) not in bonded
        assert frozenset(("CA", "O")) not in bonded

    def test_already_bonded_residue_untouched(self):
        from binding_metrics.core.cyclic import reconstruct_intraresidue_bonds

        top, positions, res, amap = _make_bondless_ala_chain()
        # Pre-bond the residue as a standard residue would be.
        top.addBond(amap["N"], amap["CA"])
        top.addBond(amap["CA"], amap["C"])
        top.addBond(amap["CA"], amap["CB"])
        top.addBond(amap["C"], amap["O"])
        before = top.getNumBonds()

        added = reconstruct_intraresidue_bonds(top, positions, "B")

        assert added == 0
        assert top.getNumBonds() == before  # no duplicates


@pytest.mark.integration
class TestDResidueBondRestore:
    """patch_nonstandard restores the intra-residue bonds of a renamed D-amino acid.

    Regression for the cyclosporin head-to-tail junction: DAL is loaded under its
    D name (so createStandardBonds skips it, leaving 0 intra bonds); after the
    DAL->ALA rename the internal ALA template must still match.
    """

    def test_dal_rename_restores_bonds(self):
        import openmm.unit as unit
        from openmm import Vec3, app
        from openmm.app import element as elem

        from binding_metrics.core.nonstandard import detect_nonstandard, patch_nonstandard

        top = app.Topology()
        chain = top.addChain(id="B")
        res = top.addResidue("DAL", chain)  # D-alanine, no bonds (as loaded)
        coords_nm = {
            "N": (0.0, 0.0, 0.0),
            "CA": (0.147, 0.0, 0.0),
            "C": (0.147, 0.153, 0.0),
            "O": (0.270, 0.153, 0.0),
            "CB": (0.147, -0.153, 0.0),
        }
        els = {
            "N": elem.nitrogen,
            "CA": elem.carbon,
            "C": elem.carbon,
            "O": elem.oxygen,
            "CB": elem.carbon,
        }
        pos = []
        for name, xyz in coords_nm.items():
            top.addAtom(name, els[name], res)
            pos.append(Vec3(*xyz))
        positions = unit.Quantity(pos, unit.nanometer)

        info = detect_nonstandard(top, "B")
        assert info.has_d_residues
        top2, _ = patch_nonstandard(top, positions, "B", info)

        r = next(top2.residues())
        assert r.name == "ALA"  # renamed
        bonded = {frozenset((b.atom1.name, b.atom2.name)) for b in top2.bonds()}
        assert frozenset(("N", "CA")) in bonded  # bonds restored
        assert frozenset(("CA", "C")) in bonded
        assert frozenset(("CA", "CB")) in bonded
        assert frozenset(("C", "O")) in bonded


class TestNMeInternalHList:
    """_internal_h_list must supply H specs for N-methylated NCAA residues so
    addHydrogens builds their heavy+H form (matching the NMG/NMA/MVA/MLE FF
    templates)."""

    @pytest.mark.parametrize(
        "name,parent_backbone",
        [
            ("NMG", "GLY"),
            ("NMA", "ALA"),
            ("MVA", "VAL"),
            ("MLE", "LEU"),
        ],
    )
    def test_nme_h_list(self, name, parent_backbone):
        pytest.importorskip("openmm", reason="OpenMM required")
        from binding_metrics.core.cyclic import _internal_h_list

        h = _internal_h_list(name)
        assert h is not None
        names = [n for n, _ in h]
        parents = {n: p for n, p in h}
        # The three N-methyl protons on CN are present.
        for hn in ("HN1", "HN2", "HN3"):
            assert hn in names
            assert parents[hn] == "CN"
        # The backbone amide H is dropped (tertiary N in N-methyl residues).
        assert not any(n == "H" and p == "N" for n, p in h)
