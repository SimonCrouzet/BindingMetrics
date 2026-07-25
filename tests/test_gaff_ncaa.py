"""Tests for automatic GAFF2 parameterisation of non-canonical amino acids.

Fast unit tests (no antechamber / OpenMM force field) run always.
Integration tests generate real GAFF templates (antechamber / AM1-BCC) from the
cyclosporin A structure (1CWA: backbone-embedded BMT and ABA) and are marked
``@pytest.mark.integration`` — they are slower and need openmmforcefields +
AmberTools.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_cuda

from binding_metrics.core.gaff_ncaa import (
    GAFF_SKIP_RESIDUES,
    _hydrogen_names,
    _template_net_charge,
    parameterize_ncaa_residues,
)

# Cyclosporin A — the canonical backbone-NCAA test case (BMT, ABA + curated NMe).
# Must come from data/, which is tracked: manuscripts/ is gitignored, so pointing
# here at a copy under it silently skipped this whole module everywhere but the
# author's machine.
CYCLOSPORIN_CIF = Path(__file__).parent.parent / "data" / "example_ncaa_cyclosporin_1CWA.cif"

try:
    import openmmforcefields  # noqa: F401

    HAS_OMMFF = True
except ImportError:
    HAS_OMMFF = False

requires_ommff = pytest.mark.skipif(not HAS_OMMFF, reason="openmmforcefields not installed")
requires_cyclosporin = pytest.mark.skipif(
    not CYCLOSPORIN_CIF.exists(), reason="1CWA.cif example not available"
)


# ---------------------------------------------------------------------------
# Fast unit tests
# ---------------------------------------------------------------------------


class TestSkipSet:
    def test_standard_amino_acids_skipped(self):
        for name in ("ALA", "GLY", "VAL", "LEU", "PRO", "CYX", "HIE"):
            assert name in GAFF_SKIP_RESIDUES

    def test_curated_nonstandard_skipped(self):
        # These have hand-curated XML templates elsewhere and must NOT be
        # double-handled by the GAFF generator.
        for name in ("NMG", "NMA", "MVA", "MLE", "ASPL", "GLUL", "LYSL"):
            assert name in GAFF_SKIP_RESIDUES

    def test_water_and_ions_skipped(self):
        for name in ("HOH", "WAT", "NA", "CL", "ZN"):
            assert name in GAFF_SKIP_RESIDUES

    def test_exotic_ncaas_not_skipped(self):
        # BMT / ABA are the exotic cyclosporin building blocks that DO need GAFF.
        assert "BMT" not in GAFF_SKIP_RESIDUES
        assert "ABA" not in GAFF_SKIP_RESIDUES


class TestTemplateNetCharge:
    def test_sums_residue_atom_charges(self):
        xml = (
            "<ForceField><Residues><Residue name='X'>"
            "<Atom name='A' type='t' charge='0.5'/>"
            "<Atom name='B' type='t' charge='-0.5'/>"
            "</Residue></Residues></ForceField>"
        )
        assert abs(_template_net_charge(xml)) < 1e-9

    def test_nonzero_sum(self):
        xml = (
            "<ForceField><Residues><Residue name='X'>"
            "<Atom name='A' type='t' charge='0.3'/>"
            "<Atom name='B' type='t' charge='0.4'/>"
            "</Residue></Residues></ForceField>"
        )
        assert _template_net_charge(xml) == pytest.approx(0.7)


class TestHydrogenNames:
    def test_names_are_unique(self):
        # Two H on CB, one on CA → HB, HB2, HA — all unique.
        keep_h = [(10, 0), (11, 0), (12, 1)]
        rd_res_names = {0: "CB", 1: "CA"}
        names = _hydrogen_names(keep_h, rd_res_names)
        assert len(set(names.values())) == len(names)
        assert all(n.startswith("H") for n in names.values())

    def test_all_hydrogens_named(self):
        keep_h = [(5, 0), (6, 0), (7, 0)]
        rd_res_names = {0: "N"}
        names = _hydrogen_names(keep_h, rd_res_names)
        assert set(names.keys()) == {5, 6, 7}
        assert len(set(names.values())) == 3


# ---------------------------------------------------------------------------
# Integration tests — real GAFF templates from cyclosporin A
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cyclosporin_ncaa_result():
    """Run the pre-GAFF prep + parameterize_ncaa_residues on cyclosporin once.

    Returns ``(topology, positions, ff, ncaa_xmls, peptide_chain)``.
    """
    import openmm.app as app
    from pdbfixer import PDBFixer

    from binding_metrics.core.cyclic import (
        load_extra_xmls,
        patch_cyclic_topology,
        rename_disulfide_cys_to_cyx,
    )
    from binding_metrics.core.nonstandard import (
        detect_nonstandard,
        load_nonstandard_xmls,
        patch_nonstandard,
    )

    fixer = PDBFixer(filename=str(CYCLOSPORIN_CIF))
    topology, positions = fixer.topology, fixer.positions

    # Peptide chain = smallest non-water chain.
    water = {"HOH", "WAT", "H2O"}
    sizes = sorted(
        ((c.id, sum(1 for r in c.residues() if r.name not in water)) for c in topology.chains()),
        key=lambda t: t[1],
    )
    peptide_chain = next(cid for cid, n in sizes if n > 0)

    ns = detect_nonstandard(topology, peptide_chain)
    topology, positions = patch_nonstandard(topology, positions, peptide_chain, ns)
    topology, positions, bond_info = patch_cyclic_topology(topology, positions, peptide_chain)
    topology, positions = rename_disulfide_cys_to_cyx(topology, positions)

    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml", "implicit/obc2.xml")
    load_nonstandard_xmls(ff, ns)
    load_extra_xmls(ff, bond_info)

    topology, positions, ncaa_xmls = parameterize_ncaa_residues(topology, positions, ff)
    return topology, positions, ff, ncaa_xmls, peptide_chain, bond_info


@requires_ommff
@requires_cyclosporin
@pytest.mark.integration
class TestGaffTemplateGeneration:
    def test_templates_generated_for_exotic_ncaas(self, cyclosporin_ncaa_result):
        _, _, _, ncaa_xmls, _, _ = cyclosporin_ncaa_result
        names = {ET.fromstring(x).find(".//Residue").get("name") for x in ncaa_xmls}
        assert "BMT" in names
        assert "ABA" in names

    def test_templates_declare_external_bonds(self, cyclosporin_ncaa_result):
        _, _, _, ncaa_xmls, _, _ = cyclosporin_ncaa_result
        for xml in ncaa_xmls:
            resel = ET.fromstring(xml).find(".//Residue")
            ext = resel.findall("ExternalBond")
            assert ext, f"{resel.get('name')} template has no <ExternalBond>"
            ext_atoms = {e.get("atomName") for e in ext}
            # Backbone N and C carry the peptide (external) bonds.
            assert "N" in ext_atoms and "C" in ext_atoms

    def test_templates_are_net_neutral(self, cyclosporin_ncaa_result):
        _, _, _, ncaa_xmls, _, _ = cyclosporin_ncaa_result
        for xml in ncaa_xmls:
            net = _template_net_charge(xml)
            name = ET.fromstring(xml).find(".//Residue").get("name")
            assert abs(net) < 1e-3, f"{name} template net charge {net} is not integer-neutral"

    def test_template_atom_names_match_topology_residue(self, cyclosporin_ncaa_result):
        topology, _, _, ncaa_xmls, _, _ = cyclosporin_ncaa_result
        # Every heavy-atom name in the topology NCAA residue must appear in its
        # generated template (H are injected, so the template is a superset).
        tmpl_atoms = {}
        for xml in ncaa_xmls:
            resel = ET.fromstring(xml).find(".//Residue")
            tmpl_atoms[resel.get("name")] = {a.get("name") for a in resel.findall("Atom")}
        for res in topology.residues():
            if res.name in tmpl_atoms:
                heavy = {
                    a.name
                    for a in res.atoms()
                    if a.element is not None and a.element.atomic_number > 1
                }
                missing = heavy - tmpl_atoms[res.name]
                assert not missing, f"{res.name} heavy atoms {missing} absent from template"

    def test_templates_load_and_build_system(self, cyclosporin_ncaa_result):
        """The generated templates must let ff14SB createSystem succeed."""
        import openmm.app as app

        from binding_metrics.core.cyclic import get_addh_variants

        topology, positions, ff, _, peptide_chain, bond_info = cyclosporin_ncaa_result
        modeller = app.Modeller(topology, positions)
        variants = (
            get_addh_variants(modeller.topology, bond_info, peptide_chain) if bond_info else None
        )
        modeller.addHydrogens(ff, pH=7.4, variants=variants)
        system = ff.createSystem(
            modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )
        assert system.getNumParticles() > 0
        # No NCAA residue should have been left unparameterised (particle count
        # must cover every atom, including the exotic residues).
        assert system.getNumParticles() == modeller.topology.getNumAtoms()


# ---------------------------------------------------------------------------
# Structural sanity — cyclosporin minimises to a sane structure (not exploded)
# ---------------------------------------------------------------------------


# Every test here goes through the `relaxed` fixture, which runs a real
# minimisation — so the requirement belongs on the class, not inside the fixture
# where no marker-based selection can see it.
@requires_ommff
@requires_cyclosporin
@requires_cuda
@pytest.mark.integration
class TestCyclosporinRelaxSanity:
    @pytest.fixture(scope="class")
    def relaxed(self, tmp_path_factory):
        """Prep + minimise-only relaxation of cyclosporin; returns (result, in, out)."""
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure, save_structure
        from binding_metrics.protocols.relaxation import (
            ImplicitRelaxation,
            RelaxationConfig,
        )

        out = tmp_path_factory.mktemp("cyclo")
        top, pos = load_structure(str(CYCLOSPORIN_CIF))
        top, pos = prep_structure(top, pos, ph=7.4)
        prepped = out / "prepped.cif"
        save_structure(top, pos, prepped)

        config = RelaxationConfig(
            md_duration_ps=0.0,
            min_steps_initial=200,
            min_steps_restrained=100,
            min_steps_final=200,
            device="cuda",
            small_molecules="auto",
        )
        result = ImplicitRelaxation(config).run(prepped, out)
        return result, prepped, out

    def test_relaxation_succeeds_with_finite_energy(self, relaxed):
        result, _, _ = relaxed
        assert result.success, result.error_message
        assert result.potential_energy_minimized is not None
        assert np.isfinite(result.potential_energy_minimized)

    def test_minimised_structure_is_sane(self, relaxed):
        """Coordinates finite, no exploded geometry, no egregious clashes."""
        import openmm.app as app

        result, prepped, _ = relaxed
        assert result.minimized_structure_path is not None
        min_path = Path(result.minimized_structure_path)
        assert min_path.exists()

        pre = app.PDBxFile(str(prepped))
        post = app.PDBxFile(str(min_path))
        post_xyz = np.array([[v.x, v.y, v.z] for v in post.positions]) * 10.0  # Å

        assert np.all(np.isfinite(post_xyz)), "minimised coordinates contain NaN/inf"

        # Heavy-atom RMSD (no alignment) vs the prepped pose: minimise-only must
        # not translate/explode the structure.
        pre_heavy = (
            np.array(
                [
                    [v.x, v.y, v.z]
                    for a, v in zip(pre.topology.atoms(), pre.positions)
                    if a.element is not None and a.element.symbol != "H"
                ]
            )
            * 10.0
        )
        post_heavy = (
            np.array(
                [
                    [v.x, v.y, v.z]
                    for a, v in zip(post.topology.atoms(), post.positions)
                    if a.element is not None and a.element.symbol != "H"
                ]
            )
            * 10.0
        )
        if pre_heavy.shape == post_heavy.shape:
            rmsd = float(np.sqrt(np.mean(np.sum((pre_heavy - post_heavy) ** 2, axis=1))))
            assert rmsd < 5.0, f"minimised heavy-atom RMSD {rmsd:.2f} Å too large (exploded)"

        # No unphysically short non-bonded heavy-atom contact within the peptide.
        pep_chain = min(
            post.topology.chains(),
            key=lambda c: sum(1 for r in c.residues() if r.name not in ("HOH", "WAT")),
        )
        pep_heavy_idx = [
            a.index for a in pep_chain.atoms() if a.element is not None and a.element.symbol != "H"
        ]
        bonded = {frozenset((b.atom1.index, b.atom2.index)) for b in post.topology.bonds()}
        p = post_xyz[pep_heavy_idx]
        n = len(pep_heavy_idx)
        min_d = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                if frozenset((pep_heavy_idx[i], pep_heavy_idx[j])) in bonded:
                    continue
                d = float(np.linalg.norm(p[i] - p[j]))
                min_d = min(min_d, d)
        assert min_d > 0.8, f"egregious clash: closest non-bonded heavy pair {min_d:.2f} Å"
