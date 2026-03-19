"""Tests for D-amino acid and N-methylated residue handling.

Pure-Python tests (no OpenMM) run always.
Integration tests (require OpenMM topology) are marked @pytest.mark.integration.
"""

import xml.etree.ElementTree as ET

import pytest

from binding_metrics.core.nonstandard import (
    D_AA_MAP,
    NME_AA_MAP,
    NonstandardInfo,
    _XML_NMG,
    _XML_NMA,
    _XML_MVA,
    _XML_MLE,
    detect_nonstandard,
    is_d_residue,
    patch_nonstandard,
)


# ---------------------------------------------------------------------------
# D-AA registry
# ---------------------------------------------------------------------------


class TestDAminoAcidRegistry:
    def test_known_codes_present(self):
        expected = ["DAL", "DLE", "DVA", "DGL", "DAR", "DPR", "DCY", "DPN", "DTR"]
        for code in expected:
            assert code in D_AA_MAP, f"{code} missing from D_AA_MAP"

    def test_l_names_are_standard(self):
        standard = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
            "SER", "THR", "TRP", "TYR", "VAL",
        }
        for d_code, l_name in D_AA_MAP.items():
            assert l_name in standard, f"{d_code} → {l_name} is not a standard L-AA"

    def test_no_glycine(self):
        # Glycine is achiral; no D-form in the registry
        assert "DGY" not in D_AA_MAP
        assert "GLY" not in D_AA_MAP.values()

    def test_is_d_residue_helper(self):
        assert is_d_residue("DAL") is True
        assert is_d_residue("ALA") is False
        assert is_d_residue("SAR") is False
        assert is_d_residue("XXX") is False


# ---------------------------------------------------------------------------
# NMe-AA registry
# ---------------------------------------------------------------------------


class TestNMeAminoAcidRegistry:
    def test_sarcosine_codes(self):
        assert NME_AA_MAP["SAR"] == "NMG"
        assert NME_AA_MAP["NMG"] == "NMG"

    def test_standard_codes_present(self):
        for code in ("NMA", "MAA", "MVA", "MLE"):
            assert code in NME_AA_MAP, f"{code} missing from NME_AA_MAP"

    def test_canonical_names(self):
        assert set(NME_AA_MAP.values()) <= {"NMG", "NMA", "MVA", "MLE"}


# ---------------------------------------------------------------------------
# XML template sanity checks
# ---------------------------------------------------------------------------


class TestNMeXMLTemplates:
    """Verify charge neutrality, atom types, and bond topology."""

    def _parse(self, xml_str, name):
        root = ET.fromstring(xml_str)
        res = root.find(f".//Residue[@name='{name}']")
        assert res is not None, f"Residue {name} not found in XML"
        return res

    def _check_template(self, xml_str, res_name, n_methyl_c="CN"):
        res = self._parse(xml_str, res_name)

        atoms = {a.get("name"): a for a in res.findall("Atom")}
        bonds = [(b.get("atomName1"), b.get("atomName2")) for b in res.findall("Bond")]
        ext = [e.get("atomName") for e in res.findall("ExternalBond")]

        # Charge neutrality
        total = sum(float(a.get("charge", 0)) for a in atoms.values())
        assert abs(total) < 1e-3, f"Net charge {total:.4f} ≠ 0 for {res_name}"

        # N-methyl group present and correctly typed
        assert "N" in atoms
        assert n_methyl_c in atoms
        assert "HN1" in atoms and "HN2" in atoms and "HN3" in atoms
        assert atoms["N"].get("type") == "N"
        assert atoms[n_methyl_c].get("type") == "CT"
        assert atoms["HN1"].get("type") == "H1"  # H adjacent to N (one EWG)

        # No backbone H on N (it's N-methylated)
        assert "H" not in atoms, f"Unexpected backbone H in {res_name} template"

        # N-methyl bond present
        assert (n_methyl_c, "HN1") in bonds or ("HN1", n_methyl_c) in bonds
        assert ("N", n_methyl_c) in bonds or (n_methyl_c, "N") in bonds

        # Standard backbone externals
        assert "N" in ext and "C" in ext

        # Standard backbone carbonyl
        assert atoms["C"].get("type") == "C"
        assert atoms["O"].get("type") == "O"

    def test_nmg(self):
        self._check_template(_XML_NMG, "NMG")
        # GLY-specific: two alpha Hs
        res = self._parse(_XML_NMG, "NMG")
        atom_names = {a.get("name") for a in res.findall("Atom")}
        assert "HA2" in atom_names and "HA3" in atom_names
        assert "CB" not in atom_names  # no sidechain

    def test_nma(self):
        self._check_template(_XML_NMA, "NMA")
        res = self._parse(_XML_NMA, "NMA")
        atom_names = {a.get("name") for a in res.findall("Atom")}
        assert "CB" in atom_names  # ALA has CB

    def test_mva(self):
        self._check_template(_XML_MVA, "MVA")
        res = self._parse(_XML_MVA, "MVA")
        atom_names = {a.get("name") for a in res.findall("Atom")}
        assert "CG1" in atom_names and "CG2" in atom_names  # VAL isopropyl

    def test_mle(self):
        self._check_template(_XML_MLE, "MLE")
        res = self._parse(_XML_MLE, "MLE")
        atom_names = {a.get("name") for a in res.findall("Atom")}
        assert "CD1" in atom_names and "CD2" in atom_names  # LEU isobutyl
        assert "CG" in atom_names

    def test_charges_from_forcefield_ncaa(self):
        """Spot-check a few ForceField_NCAA charge values and verify neutrality."""
        # NMG backbone N charge from ForceField_NCAA
        root = ET.fromstring(_XML_NMG)
        atoms = {a.get("name"): float(a.get("charge", 0))
                 for a in root.find(".//Residue[@name='NMG']").findall("Atom")}
        assert abs(atoms["N"] - (-0.058784)) < 1e-5
        assert abs(atoms["CN"] - (-0.311738)) < 1e-5
        assert abs(sum(atoms.values())) < 1e-3

        # MVA: verify CG1 and CG2 are symmetric (as expected by symmetry)
        root = ET.fromstring(_XML_MVA)
        atoms = {a.get("name"): float(a.get("charge", 0))
                 for a in root.find(".//Residue[@name='MVA']").findall("Atom")}
        assert abs(atoms["CG1"] - atoms["CG2"]) < 1e-6
        assert abs(sum(atoms.values())) < 1e-3


# ---------------------------------------------------------------------------
# NonstandardInfo dataclass
# ---------------------------------------------------------------------------


class TestNonstandardInfo:
    def test_empty(self):
        info = NonstandardInfo()
        assert info.is_empty
        assert not info.has_d_residues
        assert not info.has_nmethyl

    def test_with_d_residue(self):
        info = NonstandardInfo(d_residues=[{"res_idx": 0, "original_name": "DAL", "l_name": "ALA"}])
        assert info.has_d_residues
        assert not info.has_nmethyl
        assert not info.is_empty

    def test_with_nmethyl(self):
        info = NonstandardInfo(
            nmethyl_residues=[{"res_idx": 2, "original_name": "SAR", "template_name": "NMG"}],
            extra_ff_xmls=[_XML_NMG],
        )
        assert info.has_nmethyl
        assert not info.has_d_residues
        assert not info.is_empty


# ---------------------------------------------------------------------------
# Detection (requires OpenMM topology)
# ---------------------------------------------------------------------------


def _make_minimal_topology(residue_names):
    """Build a minimal one-chain topology with CA-only residues."""
    pytest.importorskip("openmm", reason="OpenMM required")
    from openmm import app
    from openmm.app import element as elem

    topology = app.Topology()
    chain = topology.addChain(id="A")
    for name in residue_names:
        res = topology.addResidue(name, chain)
        topology.addAtom("CA", elem.carbon, res)
    return topology


@pytest.mark.integration
class TestDetectNonstandard:
    def test_detects_d_ala(self):
        topology = _make_minimal_topology(["GLY", "DAL", "GLY"])
        info = detect_nonstandard(topology, "A")
        assert info.has_d_residues
        assert len(info.d_residues) == 1
        assert info.d_residues[0]["original_name"] == "DAL"
        assert info.d_residues[0]["l_name"] == "ALA"
        assert info.d_residues[0]["res_idx"] == 1

    def test_detects_sarcosine(self):
        topology = _make_minimal_topology(["ALA", "SAR", "VAL"])
        info = detect_nonstandard(topology, "A")
        assert info.has_nmethyl
        assert len(info.nmethyl_residues) == 1
        assert info.nmethyl_residues[0]["original_name"] == "SAR"
        assert info.nmethyl_residues[0]["template_name"] == "NMG"
        assert len(info.extra_ff_xmls) == 1
        assert "NMG" in info.extra_ff_xmls[0]

    def test_mixed_d_and_nme(self):
        topology = _make_minimal_topology(["DAL", "MLE", "DVA"])
        info = detect_nonstandard(topology, "A")
        assert len(info.d_residues) == 2
        assert len(info.nmethyl_residues) == 1
        assert info.nmethyl_residues[0]["template_name"] == "MLE"

    def test_standard_residues_ignored(self):
        topology = _make_minimal_topology(["ALA", "GLY", "LEU", "PRO"])
        info = detect_nonstandard(topology, "A")
        assert info.is_empty

    def test_duplicate_nme_template_loaded_once(self):
        # Two SAR residues → only one NMG XML should be added
        topology = _make_minimal_topology(["SAR", "ALA", "SAR"])
        info = detect_nonstandard(topology, "A")
        assert len(info.nmethyl_residues) == 2
        assert len(info.extra_ff_xmls) == 1  # NMG XML loaded once


# ---------------------------------------------------------------------------
# Ramachandran D-AA awareness
# ---------------------------------------------------------------------------


class TestRamachandranDAA:
    def test_l_alpha_helix_favoured(self):
        from binding_metrics.metrics.geometry import _classify_ramachandran
        # L-α-helix
        assert _classify_ramachandran(-60, -45) == "favoured"

    def test_d_alpha_helix_favoured_with_flag(self):
        from binding_metrics.metrics.geometry import _classify_ramachandran
        # D-α-helix: φ≈+60°, ψ≈+45° → negated → −60°, −45° → L-α-helix
        assert _classify_ramachandran(60, 45, is_d=True) == "favoured"

    def test_d_beta_sheet_outlier_without_flag(self):
        from binding_metrics.metrics.geometry import _classify_ramachandran
        # D-β-sheet (φ≈+120°, ψ≈−120°) is an outlier without the D-AA flag
        # because positive phi is outside all L-AA allowed regions
        assert _classify_ramachandran(120, -120, is_d=False) == "outlier"

    def test_d_beta_sheet_favoured(self):
        from binding_metrics.metrics.geometry import _classify_ramachandran
        # D-β-sheet: φ≈+120°, ψ≈−120° → negated → −120°, +120° → L-β-sheet
        assert _classify_ramachandran(120, -120, is_d=True) == "favoured"

    def test_nan_returns_none(self):
        from binding_metrics.metrics.geometry import _classify_ramachandran
        import math
        assert _classify_ramachandran(float("nan"), 0.0) is None
        assert _classify_ramachandran(0.0, float("nan"), is_d=True) is None
