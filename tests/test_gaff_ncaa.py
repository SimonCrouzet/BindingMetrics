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

    def test_net_charge_is_recorded_per_residue(self, cyclosporin_ncaa_result):
        """BMT and ABA carry no acid or base, so they are neutral and raise no warning."""
        _, _, _, ncaa_xmls, _, _ = cyclosporin_ncaa_result
        assert set(ncaa_xmls.net_charge_by_residue) == {"BMT", "ABA"}
        assert all(abs(q) < 1e-3 for q in ncaa_xmls.net_charge_by_residue.values())
        assert ncaa_xmls.neutral_ionizable_groups == {}

    def test_bond_orders_come_from_the_component_dictionary(self, cyclosporin_ncaa_result):
        _, _, _, ncaa_xmls, _, _ = cyclosporin_ncaa_result
        assert ncaa_xmls.bond_order_source_by_residue == {"BMT": "ccd", "ABA": "ccd"}

    @staticmethod
    def _build_again(cyclosporin_ncaa_result, residue_name):
        """Template XML of ``residue_name`` built a second time from the fixture's topology."""
        from binding_metrics.core.gaff_ncaa import (
            _amber_backbone_types,
            _generate_residue_template,
            _pos_to_angstrom,
        )

        topology, positions, ff, _, _, _ = cyclosporin_ncaa_result
        residue = next(r for r in topology.residues() if r.name == residue_name)
        return _generate_residue_template(
            residue,
            topology,
            _pos_to_angstrom(positions),
            "gaff-2.2.20",
            _amber_backbone_types(ff),
        )[0]

    @classmethod
    def _rebuilt_xml(cls, cyclosporin_ncaa_result, residue_name):
        """``(first, rebuilt)``: the template the fixture made and a second build of it."""
        ncaa_xmls = cyclosporin_ncaa_result[3]
        first = next(
            x for x in ncaa_xmls if ET.fromstring(x).find(".//Residue").get("name") == residue_name
        )
        return first, cls._build_again(cyclosporin_ncaa_result, residue_name)

    def test_a_second_build_of_abu_is_identical(self, cyclosporin_ncaa_result):
        """Same residue, same seed: same charges and atom types, byte for byte."""
        first, rebuilt = self._rebuilt_xml(cyclosporin_ncaa_result, "ABA")
        assert rebuilt == first

    @pytest.mark.slow
    def test_a_second_build_of_mebmt_is_identical(self, cyclosporin_ncaa_result):
        """MeBmt is the residue whose charges used to change between builds (sqm timing)."""
        first, rebuilt = self._rebuilt_xml(cyclosporin_ncaa_result, "BMT")
        assert rebuilt == first

    def test_mebmt_stereochemistry_reaches_the_charge_calculation(
        self, cyclosporin_ncaa_result, monkeypatch
    ):
        """MeBmt is (2S,3R,4R) with an E double bond in the structure; AM1-BCC must see that.

        Without it the conformer that sqm minimises is a random stereoisomer: seed 1 gave a
        diastereomer with a Z double bond.
        """
        from binding_metrics.core import gaff_ncaa

        seen: dict = {}

        def charges(molecule, random_seed=None):
            seen["molecule"] = molecule
            return np.linspace(-0.01, 0.01, molecule.n_atoms)

        monkeypatch.setattr(gaff_ncaa, "_am1bcc_charges", charges)
        self._build_again(cyclosporin_ncaa_result, "BMT")
        molecule = seen["molecule"]
        assert [a.stereochemistry for a in molecule.atoms if a.stereochemistry] == ["S", "R", "R"]
        assert [b.stereochemistry for b in molecule.bonds if b.stereochemistry] == ["E"]

    @staticmethod
    def _template(ncaa_xmls, residue_name):
        for xml in ncaa_xmls:
            root = ET.fromstring(xml)
            resel = root.find(".//Residue")
            if resel.get("name") == residue_name:
                return root, resel
        raise AssertionError(f"no template for {residue_name}")

    def test_bmt_template_carries_the_alkene(self, cyclosporin_ncaa_result):
        """MeBmt (C10H19NO3 as a free acid) has 17 H in the chain and a CE=CZ double bond.

        With every bond perceived single the template had 19 H and typed the alkene
        carbons as sp3 (c3).
        """
        _, _, _, ncaa_xmls, _, _ = cyclosporin_ncaa_result
        _, resel = self._template(ncaa_xmls, "BMT")
        types = {a.get("name"): a.get("type") for a in resel.findall("Atom")}
        hydrogens = [name for name in types if name.startswith("H")]

        assert len(hydrogens) == 17
        sp2_carbon_types = {"c2", "ce", "cf"}
        assert types["CE"] in sp2_carbon_types and types["CZ"] in sp2_carbon_types
        for name in ("CB", "CG2", "CD1", "CD2", "CH", "CN"):
            assert types[name] == "c3", f"{name} is sp3 in MeBmt"

    def test_aba_template_is_saturated(self, cyclosporin_ncaa_result):
        _, _, _, ncaa_xmls, _, _ = cyclosporin_ncaa_result
        _, resel = self._template(ncaa_xmls, "ABA")
        types = {a.get("name"): a.get("type") for a in resel.findall("Atom")}

        assert sum(name.startswith("H") for name in types) == 7
        assert types["CB"] == "c3" and types["CG"] == "c3"

    def test_backbone_carbonyl_is_a_carbonyl_without_extra_hydrogens(self, cyclosporin_ncaa_result):
        """No hydrogen on the backbone C or O, and a carbonyl-sized charge on C and O."""
        _, _, _, ncaa_xmls, _, _ = cyclosporin_ncaa_result
        for name in ("BMT", "ABA"):
            _, resel = self._template(ncaa_xmls, name)
            bonded_to_backbone_co = {
                (b.get("atomName1"), b.get("atomName2")) for b in resel.findall("Bond")
            }
            for atom1, atom2 in bonded_to_backbone_co:
                for backbone, other in ((atom1, atom2), (atom2, atom1)):
                    if backbone in ("C", "O"):
                        assert not other.startswith("H"), f"{name}: H on backbone {backbone}"
            charges = {a.get("name"): float(a.get("charge")) for a in resel.findall("Atom")}
            assert charges["C"] > 0.4, f"{name} carbonyl carbon charge {charges['C']:.2f}"
            assert charges["O"] < -0.4, f"{name} carbonyl oxygen charge {charges['O']:.2f}"

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


# ---------------------------------------------------------------------------
# Log output of the template step (no antechamber: the template builders are stubbed)
# ---------------------------------------------------------------------------

_STUB_TEMPLATE = (
    '<ForceField><Residues><Residue name="X">'
    '<Atom name="C1" type="c3" charge="0.06"/><Atom name="C2" type="c3" charge="0.04"/>'
    "</Residue></Residues></ForceField>"
)

# What the step used to print, one line per event, in order.
_EXPECTED_LINES = [
    "  [warning] could not read ff14SB backbone types; "
    "NCAA backbones stay on GAFF (junctions may be under-parameterised).",
    "  Auto-GAFF2: 'BMT' template generated (2 H, net charge +0.1000)",
    "  [warning] 'BMT' instance differs from first template; reusing first (H 1 vs 2).",
    "  [warning] GAFF NCAA template failed for 'ABA': antechamber not found",
]


class _FakeResidue:
    def __init__(self, name, index):
        self.name = name
        self.index = index


class _FakeTopology:
    def __init__(self, residues):
        self._residues = residues

    def residues(self):
        return iter(self._residues)


@pytest.fixture
def stubbed_template_step(monkeypatch):
    """Run ``parameterize_ncaa_residues`` on three fake residues: BMT twice and ABA."""
    from binding_metrics.core import gaff_ncaa

    seeds: list = []

    def generate(res, topology, pos_A, gaff_version, backbone_amber, random_seed=None):
        seeds.append(random_seed)
        if res.name == "ABA":
            raise RuntimeError("antechamber not found")
        hydrogens = [("H1", "C1", (0, 0, 0)), ("H2", "C2", (0, 0, 0))]
        if res.index == 1:  # the second BMT is perceived with one hydrogen fewer
            hydrogens = hydrogens[:1]
        return _STUB_TEMPLATE, hydrogens, [], None

    monkeypatch.setattr(gaff_ncaa, "_is_ncaa", lambda res: True)
    monkeypatch.setattr(gaff_ncaa, "_pos_to_angstrom", lambda positions: np.zeros((3, 3)))
    monkeypatch.setattr(gaff_ncaa, "_amber_backbone_types", lambda ff: None)
    monkeypatch.setattr(gaff_ncaa, "_generate_residue_template", generate)
    monkeypatch.setattr(gaff_ncaa, "_load_ffxml", lambda ff, ffxml: None)
    monkeypatch.setattr(
        gaff_ncaa, "_rebuild_topology_with_injected_h", lambda top, pos, h: (top, pos)
    )
    topology = _FakeTopology(
        [_FakeResidue("BMT", 0), _FakeResidue("BMT", 1), _FakeResidue("ABA", 2)]
    )

    def run(**kwargs):
        return parameterize_ncaa_residues(topology, [None] * 3, ff=None, **kwargs)

    run.seeds = seeds
    return run


@requires_ommff
class TestTemplateStepLogging:
    def test_events_reach_the_module_logger_at_their_level(self, stubbed_template_step, caplog):
        with caplog.at_level("INFO", logger="binding_metrics"):
            stubbed_template_step()
        records = [r for r in caplog.records if r.name == "binding_metrics.core.gaff_ncaa"]
        assert [r.getMessage() for r in records] == _EXPECTED_LINES
        assert [r.levelname for r in records] == ["WARNING", "INFO", "WARNING", "WARNING"]

    def test_random_seed_reaches_every_template_build(self, stubbed_template_step):
        from binding_metrics._constants import DEFAULT_RANDOM_SEED

        stubbed_template_step()
        assert stubbed_template_step.seeds == [DEFAULT_RANDOM_SEED] * 3
        stubbed_template_step.seeds.clear()
        stubbed_template_step(random_seed=7)
        assert stubbed_template_step.seeds == [7] * 3
        stubbed_template_step.seeds.clear()
        stubbed_template_step(random_seed=None)
        assert stubbed_template_step.seeds == [None] * 3

    def test_verbose_false_stays_silent(self, stubbed_template_step, caplog):
        with caplog.at_level("INFO", logger="binding_metrics"):
            stubbed_template_step(verbose=False)
        assert not [r for r in caplog.records if r.name == "binding_metrics.core.gaff_ncaa"]

    def test_console_text_matches_the_former_prints(self, stubbed_template_step, capsys):
        from binding_metrics.utils import configure_logging

        configure_logging()
        stubbed_template_step()
        captured = capsys.readouterr()
        assert captured.out == "\n".join(_EXPECTED_LINES) + "\n"
        assert captured.err == ""
