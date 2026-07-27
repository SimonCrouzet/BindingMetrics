"""Tests for hydrocarbon-staple detection and end-to-end parameterization.

A hydrocarbon staple (Aileron-type) is an all-carbon side-chain cross-link
between two non-canonical olefin residues (e.g. S5/R8 = MK8/0EH joined by
ring-closing metathesis). Detection must *recognise* it (not reject it as an
unsupported cyclization); the staple residues are then parameterised by the
GAFF hybrid path (core.gaff_ncaa), whose GAFF↔GAFF cross-link terms already
cover the staple bond — so no dedicated force-field template is needed.
"""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent.parent / "data"


def _find_example(*tokens):
    """Return the first bundled example file whose name contains a token."""
    if not DATA_DIR.is_dir():
        return None
    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() in (".pdb", ".cif", ".mmcif") and any(
            t.lower() in path.name.lower() for t in tokens
        ):
            return path
    return None


def _minimal_topology(residue_names, atom_specs, extra_bonds=()):
    """Build a minimal OpenMM Topology (heavy atoms spread far apart in x).

    ``extra_bonds`` is a list of ((res_idx, atom_name), (res_idx, atom_name))
    added as explicit topology bonds (e.g. the cross-link), on top of the
    automatic sequential backbone C–N bonds.
    """
    pytest.importorskip("openmm", reason="OpenMM required")
    from openmm import app
    from openmm.app import element as elem

    _elem = {"N": elem.nitrogen, "C": elem.carbon, "O": elem.oxygen, "S": elem.sulfur}
    top = app.Topology()
    chain = top.addChain(id="A")
    amap = {}
    pos = []
    n = 0
    for ri, (rname, specs) in enumerate(zip(residue_names, atom_specs)):
        res = top.addResidue(rname, chain)
        for aname, esym in specs:
            a = top.addAtom(aname, _elem.get(esym, elem.carbon), res)
            amap[(ri, aname)] = a
            pos.append([n * 1.0, 0.0, 0.0])
            n += 1
    for ri in range(len(residue_names) - 1):
        c, nx = amap.get((ri, "C")), amap.get((ri + 1, "N"))
        if c and nx:
            top.addBond(c, nx)
    for (r1, a1), (r2, a2) in extra_bonds:
        top.addBond(amap[(r1, a1)], amap[(r2, a2)])
    return top, np.array(pos, dtype=float), amap


_BB = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]


# ---------------------------------------------------------------------------
# Detection (fast, synthetic topologies)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDetectHydrocarbonStaple:
    def test_all_carbon_sidechain_crosslink_detected(self):
        from binding_metrics.core.cyclic import detect_cyclization

        # Two NCAA olefin residues joined by an all-carbon side-chain cross-link.
        res = ["ALA", "0EH", "ALA", "ALA", "MK8", "ALA"]
        oeh = _BB + [("CB", "C"), ("CG", "C"), ("CAT", "C")]
        mk8 = _BB + [("CB", "C"), ("CE", "C")]
        specs = [_BB, oeh, _BB, _BB, mk8, _BB]
        top, pos, _ = _minimal_topology(res, specs, extra_bonds=[((1, "CAT"), (4, "CE"))])
        result = detect_cyclization(top, pos, "A")
        assert len(result) == 1
        info = result[0]
        assert info.cyclic_type == "hydrocarbon_staple"
        assert info.atom1_id == ("A", 1, "CAT")
        assert info.atom2_id == ("A", 4, "CE")
        assert info.omega_ids is None
        assert info.extra_ff_xmls == []

    def test_thioether_crosslink_still_rejected(self):
        """An S–C cross-link is NOT a hydrocarbon staple and must still raise."""
        from binding_metrics.core.cyclic import CyclizationError, detect_cyclization

        res = ["ALA", "CYS", "ALA", "ALA", "SED", "ALA"]
        cys = _BB + [("CB", "C"), ("SG", "S")]
        sed = _BB + [("CB", "C"), ("CX", "C")]
        specs = [_BB, cys, _BB, _BB, sed, _BB]
        top, pos, _ = _minimal_topology(res, specs, extra_bonds=[((1, "SG"), (4, "CX"))])
        with pytest.raises(CyclizationError):
            detect_cyclization(top, pos, "A")

    def test_backbone_carbon_crosslink_not_a_staple(self):
        """A cross-link touching a backbone carbonyl C is not a side-chain staple."""
        from binding_metrics.core.cyclic import CyclizationError, detect_cyclization

        res = ["ALA", "XAA", "ALA", "ALA", "YAA", "ALA"]
        xaa = _BB + [("CB", "C"), ("CG", "C")]
        yaa = _BB + [("CB", "C")]
        specs = [_BB, xaa, _BB, _BB, yaa, _BB]
        # CG(sidechain) -- C(backbone carbonyl): not all-side-chain → unsupported
        top, pos, _ = _minimal_topology(res, specs, extra_bonds=[((1, "CG"), (4, "C"))])
        with pytest.raises(CyclizationError):
            detect_cyclization(top, pos, "A")


class TestIsHydrocarbonStapleBond:
    def test_carbon_carbon_sidechain(self):
        pytest.importorskip("openmm", reason="OpenMM required")
        from openmm import app
        from openmm.app import element as elem

        from binding_metrics.core.cyclic import _is_hydrocarbon_staple_bond

        top = app.Topology()
        c = top.addChain()
        r = top.addResidue("MK8", c)
        ca = top.addAtom("CAT", elem.carbon, r)
        cb = top.addAtom("CE", elem.carbon, r)
        bb_c = top.addAtom("C", elem.carbon, r)  # backbone carbonyl
        sg = top.addAtom("SG", elem.sulfur, r)
        assert _is_hydrocarbon_staple_bond(ca, cb) is True
        assert _is_hydrocarbon_staple_bond(ca, bb_c) is False  # backbone name
        assert _is_hydrocarbon_staple_bond(ca, sg) is False  # sulfur


# ---------------------------------------------------------------------------
# End-to-end: real 3V3B stapled peptide → detect → patch → GAFF → createSystem
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestStapleEndToEnd:
    def test_3v3b_staple_builds_system(self):
        pytest.importorskip("openmm", reason="OpenMM required")
        pytest.importorskip("openmmforcefields", reason="openmmforcefields required")
        pytest.importorskip("openff.toolkit", reason="openff-toolkit required")
        pytest.importorskip("rdkit", reason="RDKit required")
        from openmm import HarmonicBondForce, app
        from openmm.app import ForceField, Modeller, PDBFile

        from binding_metrics.core.cyclic import (
            detect_cyclization,
            get_addh_variants,
            load_extra_xmls,
            patch_cyclic_topology,
        )
        from binding_metrics.core.gaff_ncaa import parameterize_ncaa_residues

        fixture = _find_example("staple_3V3B", "staple")
        if fixture is None:
            pytest.skip("bundled staple example not found")

        pdb = PDBFile(str(fixture))
        mod = Modeller(pdb.topology, pdb.positions)
        mod.delete([r for r in mod.topology.residues() if r.name in ("HOH", "WAT")])
        top, pos = mod.topology, mod.positions
        chain_id = next(iter(top.chains())).id

        # 1. Detection recognises the staple (does not raise).
        infos = detect_cyclization(top, np.array([[v.x, v.y, v.z] for v in pos]), chain_id)
        assert any(i.cyclic_type == "hydrocarbon_staple" for i in infos)

        # 2. Patch keeps the staple residues (no rename), keeps the cross-link.
        top, pos, bond_info = patch_cyclic_topology(top, pos, chain_id)
        names = [r.name for r in next(iter(top.chains())).residues()]
        assert "0EH" in names and "MK8" in names

        # 3. GAFF-parameterize the NCAAs, add H, build the system.
        ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml", "implicit/obc2.xml")
        load_extra_xmls(ff, bond_info)
        top, pos, _ = parameterize_ncaa_residues(top, pos, ff, verbose=False)
        mod = Modeller(top, pos)
        variants = get_addh_variants(mod.topology, bond_info, chain_id)
        mod.addHydrogens(ff, pH=7.4, variants=variants)
        top = mod.topology
        system = ff.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None)
        assert system.getNumParticles() == top.getNumAtoms()

        # 4. The staple cross-link carries a real bond spring (not silently dropped).
        def _find(rn, an):
            for r in top.residues():
                if r.name == rn:
                    for a in r.atoms():
                        if a.name == an:
                            return a
            return None

        cat, ce = _find("0EH", "CAT"), _find("MK8", "CE")
        assert cat is not None and ce is not None
        hbf = next(f for f in system.getForces() if isinstance(f, HarmonicBondForce))
        has_term = any(
            {hbf.getBondParameters(i)[0], hbf.getBondParameters(i)[1]} == {cat.index, ce.index}
            for i in range(hbf.getNumBonds())
        )
        assert has_term, "staple cross-link bond has no force term"
