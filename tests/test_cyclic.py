"""Tests for cyclic peptide detection and topology patching.

All tests run without OpenMM (pure-Python or lightweight) except those that
build a real topology, which are marked @pytest.mark.integration.
"""

import math
import pytest
import numpy as np

from binding_metrics.core.cyclic import (
    CyclicBondInfo,
    CyclizationError,
    _XML_ASPL,
    _XML_GLUL,
    _XML_LYSL,
    _AMIDE_BOND_THRESH,
    _DISULFIDE_THRESH,
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
        "N": elem.nitrogen, "C": elem.carbon, "O": elem.oxygen,
        "S": elem.sulfur,   "H": elem.hydrogen,
    }

    topology = app.Topology()
    chain = topology.addChain(id="A")
    atom_map = {}   # (res_idx, atom_name) → Atom
    pos_list = []
    atom_count = 0

    for ri, (res_name, specs) in enumerate(zip(residue_names, atom_specs)):
        res = topology.addResidue(res_name, chain)
        for (aname, esym) in specs:
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
        positions[atom_map[(2, "C")].index] = positions[atom_map[(0, "N")].index] + np.array([0.133, 0, 0])
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
        positions[atom_map[(3, "C")].index] = (
            positions[atom_map[(0, "N")].index] + np.array([0.133, 0, 0])
        )
        # Disulfide: SG(CYS0) close to SG(CYS2)
        positions[atom_map[(2, "SG")].index] = (
            positions[atom_map[(0, "SG")].index] + np.array([0.205, 0, 0])
        )
        result = detect_cyclization(topology, positions, "A")
        types = {r.cyclic_type for r in result}
        assert "head_to_tail" in types
        assert "disulfide" in types
        assert len(result) == 2

    def test_unsupported_cyclization_raises(self):
        from binding_metrics.core.cyclic import detect_cyclization
        res_names = ["ALA", "ALA"]
        specs = [
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")],
        ]
        topology, positions, atom_map = _make_minimal_topology(res_names, specs)
        positions[atom_map[(1, "CB")].index] = (
            positions[atom_map[(0, "CB")].index] + np.array([0.15, 0, 0])
        )
        with pytest.raises(CyclizationError, match="Unsupported"):
            detect_cyclization(topology, positions, "A")

    def test_lactam_asp_detected(self):
        from binding_metrics.core.cyclic import detect_cyclization
        res_names = ["ALA", "ASP", "GLY"]
        specs = [
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"),
             ("CB", "C"), ("CG", "C"), ("OD1", "O"), ("OD2", "O")],
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
        ]
        topology, positions, atom_map = _make_minimal_topology(res_names, specs)
        positions[atom_map[(1, "CG")].index] = (
            positions[atom_map[(0, "N")].index] + np.array([0.133, 0, 0])
        )
        result = detect_cyclization(topology, positions, "A")
        assert len(result) == 1
        assert result[0].cyclic_type == "lactam_n_asp"
        assert _XML_ASPL in result[0].extra_ff_xmls


# ---------------------------------------------------------------------------
# RelaxationConfig: is_cyclic field
# ---------------------------------------------------------------------------

class TestRelaxationConfigCyclic:
    def test_default_is_cyclic_false(self):
        from binding_metrics.protocols.relaxation import RelaxationConfig
        config = RelaxationConfig()
        assert config.is_cyclic is False

    def test_is_cyclic_true(self):
        from binding_metrics.protocols.relaxation import RelaxationConfig
        config = RelaxationConfig(is_cyclic=True)
        assert config.is_cyclic is True


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

        _elem = {"N": elem.nitrogen, "C": elem.carbon, "O": elem.oxygen,
                 "S": elem.sulfur, "H": elem.hydrogen}

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
        r0, m0 = add_res("GLY", [("N","N"),("CA","C"),("C","C"),("O","O")])
        r1, m1 = add_res("ARG", [("N","N"),("CA","C"),("C","C"),("O","O")])
        r2, m2 = add_res("CYS", [("N","N"),("CA","C"),("C","C"),("O","O"),("CB","C"),("SG","S")])
        r3, m3 = add_res("CYS", [("N","N"),("CA","C"),("C","C"),("O","O"),("CB","C"),("SG","S")])

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
            _internal_h_list,
        )
        topology, positions = self._make_sfti_like_topology()
        bond_info = [CyclicBondInfo(
            cyclic_type="head_to_tail",
            atom1_id=("B", 3, "C"),
            atom2_id=("B", 0, "N"),
        )]
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
        from openmm import app, Vec3
        from openmm.unit import nanometers
        from binding_metrics.core.cyclic import (
            get_addh_variants,
            patch_cyclic_topology,
            load_extra_xmls,
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
