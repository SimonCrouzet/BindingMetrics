"""Tests for AMBER phosaa parameterization of phosphorylated residues.

The generic GAFF NCAA path perceives phospho residues at neutral charge and
protonates the phosphate to net 0 (verified in the module docstring); phosaa is
the correct, RESP-fitted source. These tests lock in that the adapter:
  * preserves phosaa's charges exactly (net −2 for SEP/TPO/PTR),
  * remaps every base type onto amber14-all's prefixed scheme + injects P,
  * builds a real phosphopeptide with no spurious phosphate H and no dropped
    bonded terms, and
  * reproduces AMBER's phosaa energy to < 0.01 kJ/mol (tleap reference).
"""

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from binding_metrics.core import phosaa

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Unit: the adapter XML (no OpenMM needed)
# ---------------------------------------------------------------------------


class TestAdapterXml:
    def test_phospho_residue_names(self):
        names = phosaa.phospho_residue_names()
        for r in ("SEP", "TPO", "PTR", "S1P", "T1P", "Y1P"):
            assert r in names

    def test_all_base_types_remapped(self):
        """No bare base type may remain in a phospho template: every type is
        either a phosaa phosphate type, phosphorus P, or a protein-* type."""
        root = ET.fromstring(phosaa.build_adapter_ffxml())
        phosphate_types = {t.get("name") for t in root.find("AtomTypes").findall("Type")}
        for res in root.find("Residues").findall("Residue"):
            for atom in res.findall("Atom"):
                t = atom.get("type")
                assert t in phosphate_types or t.startswith("protein-"), (
                    f"{res.get('name')}.{atom.get('name')} has unremapped type {t!r}"
                )

    def test_phosphorus_type_and_lj_injected(self):
        root = ET.fromstring(phosaa.build_adapter_ffxml())
        assert root.find('.//AtomTypes/Type[@name="P"]') is not None
        lj = [a for a in root.find("NonbondedForce").findall("Atom") if a.get("class") == "P"]
        assert lj and float(lj[0].get("sigma")) > 0 and float(lj[0].get("epsilon")) > 0

    def test_charges_identical_to_source_and_net_minus_two(self):
        """Adapter must not alter any charge, and SEP/TPO/PTR net to −2."""
        import os

        import openmmforcefields

        src = ET.parse(
            os.path.join(
                os.path.dirname(openmmforcefields.__file__), "ffxml", "amber", "phosaa14SB.xml"
            )
        ).getroot()
        src_q = {
            res.get("name"): {a.get("name"): float(a.get("charge")) for a in res.findall("Atom")}
            for res in src.find("Residues").findall("Residue")
        }
        adapted = ET.fromstring(phosaa.build_adapter_ffxml())
        for res in adapted.find("Residues").findall("Residue"):
            name = res.get("name")
            for a in res.findall("Atom"):
                assert float(a.get("charge")) == pytest.approx(src_q[name][a.get("name")])
        for res_name in ("SEP", "TPO", "PTR"):
            net = sum(src_q[res_name].values())
            assert net == pytest.approx(-2.0, abs=1e-4), f"{res_name} net {net}"

    def test_gaff_skips_phospho(self):
        from binding_metrics.core.gaff_ncaa import GAFF_SKIP_RESIDUES

        assert phosaa.phospho_residue_names() <= GAFF_SKIP_RESIDUES


# ---------------------------------------------------------------------------
# Integration: build a real phosphopeptide
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPhosphoPeptideBuild:
    def _fixture(self):
        f = DATA_DIR / "example_phospho_1QJB.pdb"
        if not f.exists():
            pytest.skip("bundled phospho example not found")
        return f

    def test_sep_peptide_net_minus_two_no_spurious_h(self):
        pytest.importorskip("openmm", reason="OpenMM required")
        pytest.importorskip("pdbfixer", reason="pdbfixer required")
        from openmm import HarmonicBondForce, NonbondedForce, unit
        from openmm.app import ForceField, Modeller, NoCutoff
        from pdbfixer import PDBFixer

        fixer = PDBFixer(filename=str(self._fixture()))
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        ff = ForceField("amber14-all.xml", "implicit/obc2.xml")
        phosaa.register(ff)
        phosaa.ensure_hydrogen_definitions()
        mod = Modeller(fixer.topology, fixer.positions)
        mod.addHydrogens(ff, pH=7.4)
        top = mod.topology
        system = ff.createSystem(top, nonbondedMethod=NoCutoff)

        sep = next(r for r in top.residues() if r.name == "SEP")
        # No proton on any phosphate oxygen / phosphorus.
        for a in sep.atoms():
            if a.name in ("O1P", "O2P", "O3P", "P"):
                bonded_h = [
                    b
                    for b in top.bonds()
                    if a in (b.atom1, b.atom2)
                    and (b.atom1.element.atomic_number == 1 or b.atom2.element.atomic_number == 1)
                ]
                assert not bonded_h, f"spurious H on {a.name}"

        nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
        q = sum(
            nb.getParticleParameters(a.index)[0].value_in_unit(unit.elementary_charge)
            for a in sep.atoms()
        )
        assert q == pytest.approx(-2.0, abs=1e-3)

        # No heavy–heavy bond silently dropped.
        adj = {}
        for b in top.bonds():
            adj.setdefault(b.atom1.index, set()).add(b.atom2.index)
            adj.setdefault(b.atom2.index, set()).add(b.atom1.index)
        heavy = {a.index for a in top.atoms() if a.element.atomic_number > 1}
        hbf = next(f for f in system.getForces() if isinstance(f, HarmonicBondForce))
        sysb = {
            frozenset((hbf.getBondParameters(i)[0], hbf.getBondParameters(i)[1]))
            for i in range(hbf.getNumBonds())
        }
        missing = [
            (x, y)
            for x in heavy
            for y in adj[x]
            if y > x and y in heavy and frozenset((x, y)) not in sysb
        ]
        assert not missing, f"{len(missing)} heavy-heavy bonds missing a term"


# ---------------------------------------------------------------------------
# Reference: adapter energy must match AMBER tleap/phosaa exactly
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTleapReference:
    @pytest.mark.parametrize("res_name", ["SEP", "TPO", "PTR"])
    def test_energy_matches_tleap(self, res_name, tmp_path):
        pytest.importorskip("openmm", reason="OpenMM required")
        if shutil.which("tleap") is None:
            pytest.skip("AmberTools tleap not available")
        import openmm
        from openmm import unit
        from openmm.app import AmberInpcrdFile, AmberPrmtopFile, ForceField, NoCutoff

        # Build a capped ACE–X–NME tripeptide with ff14SB + phosaa.
        leap = tmp_path / "leap.in"
        prm = tmp_path / "cap.prmtop"
        rst = tmp_path / "cap.rst7"
        leap.write_text(
            "source leaprc.protein.ff14SB\n"
            "source leaprc.phosaa14SB\n"
            f"pep = sequence {{ ACE {res_name} NME }}\n"
            f"saveamberparm pep {prm} {rst}\nquit\n"
        )
        subprocess.run(["tleap", "-f", str(leap)], cwd=tmp_path, capture_output=True, check=True)
        if not prm.exists():
            pytest.skip("tleap did not produce a prmtop")

        def energy(system, positions):
            ctx = openmm.Context(
                system,
                openmm.VerletIntegrator(1 * unit.femtosecond),
                openmm.Platform.getPlatformByName("Reference"),
            )
            ctx.setPositions(positions)
            e = (
                ctx.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )
            del ctx
            return e

        prmtop = AmberPrmtopFile(str(prm))
        coords = AmberInpcrdFile(str(rst)).positions
        e_ref = energy(prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=None), coords)

        ff = ForceField("amber14-all.xml")
        phosaa.register(ff)
        e_adapter = energy(
            ff.createSystem(prmtop.topology, nonbondedMethod=NoCutoff, constraints=None),
            coords,
        )
        assert e_adapter == pytest.approx(e_ref, abs=0.01), (
            f"{res_name}: adapter {e_adapter:.4f} vs tleap {e_ref:.4f} kJ/mol"
        )
