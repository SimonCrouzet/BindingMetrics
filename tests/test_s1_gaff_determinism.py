"""The GAFF2 route gives the same charges in every run.

``antechamber -c bcc`` starts sqm, and by default sqm times seven diagonalisation routines and
keeps the fastest. One of them (dsyev) ends the AM1 minimisation of MeBmt in another minimum than
the others, so the charges of the template changed by up to 0.013 e between builds of the same
residue, depending on the load of the machine. ``_am1bcc_charges`` fixes the routine and seeds the
conformer.

Tests without sqm check the command that is run, the seed and the normalisation. The tests that
run sqm use propanol, which takes a few seconds. The last test runs the 1CWA pipeline twice on
the GPU.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_cuda

from binding_metrics import _constants
from binding_metrics.core import gaff_ncaa

pytest.importorskip("rdkit")
openff_molecule = pytest.importorskip("openff.toolkit").Molecule

DATA = Path(__file__).parent.parent / "data"
CYCLOSPORIN_CIF = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI1_CIF = DATA / "example_bicyclic_sfti1_3P8F.cif"
P53_PDB = DATA / "example_linear_p53_1YCR.pdb"

requires_antechamber = pytest.mark.skipif(
    shutil.which("antechamber") is None, reason="AmberTools (antechamber, sqm) not installed"
)


@pytest.fixture
def fake_antechamber(monkeypatch):
    """Replace ``_run_antechamber``: record each call, write 0.01 e per atom as the charges."""
    calls: list = []

    def fake(args, workdir):
        sdf = Path(workdir, "molecule.sdf")
        calls.append({"args": list(args), "sdf": sdf.read_text() if sdf.exists() else None})
        if "charges.txt" in args:
            n_atoms = _atom_count(calls[0]["sdf"])
            Path(workdir, "charges.txt").write_text(" ".join(["0.01"] * n_atoms))

    monkeypatch.setattr(gaff_ncaa, "_run_antechamber", fake)
    return calls


def _atom_count(sdf_text: str) -> int:
    """Atom count from the counts line of a V2000 mol block."""
    counts_line = sdf_text.splitlines()[3]
    return int(counts_line[:3])


def _sdf_coordinates(sdf_text: str) -> np.ndarray:
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(sdf_text, removeHs=False)
    return mol.GetConformer().GetPositions()


class TestAm1bccCommand:
    def test_sqm_diagonaliser_is_fixed(self, fake_antechamber):
        gaff_ncaa._am1bcc_charges(openff_molecule.from_smiles("CCCO"))
        args = fake_antechamber[0]["args"]
        assert args[args.index("-c") + 1] == "bcc"
        keywords = args[args.index("-ek") + 1]
        assert f"diag_routine={gaff_ncaa._SQM_DIAG_ROUTINE_INTERNAL}" in keywords
        # -ek replaces antechamber's own keywords, so the AM1-BCC defaults must be repeated.
        assert "qm_theory='AM1'" in keywords and "scfconv=1.d-10" in keywords

    @pytest.mark.parametrize(
        "smiles, total_charge", [("CCCO", 0), ("CC(=O)[O-]", -1), ("CC[NH3+]", 1)]
    )
    def test_charges_sum_to_the_formal_charge(self, fake_antechamber, smiles, total_charge):
        molecule = openff_molecule.from_smiles(smiles)
        charges = gaff_ncaa._am1bcc_charges(molecule)
        args = fake_antechamber[0]["args"]
        assert args[args.index("-nc") + 1] == str(total_charge)
        assert charges.shape == (molecule.n_atoms,)
        assert charges.sum() == pytest.approx(total_charge, abs=1e-9)

    def test_wrong_number_of_charges_is_an_error(self, monkeypatch):
        def fake(args, workdir):
            if "charges.txt" in args:
                Path(workdir, "charges.txt").write_text("0.1 0.2")

        monkeypatch.setattr(gaff_ncaa, "_run_antechamber", fake)
        with pytest.raises(RuntimeError, match="2 AM1-BCC charges for 12 atoms"):
            gaff_ncaa._am1bcc_charges(openff_molecule.from_smiles("CCCO"))


class TestConformerSeed:
    """The conformer that sqm minimises comes from the seed, as the docstring says."""

    def _build(self, monkeypatch, **kwargs):
        """Coordinates of the conformer that reaches antechamber for propanol."""
        sdf_texts: list = []

        def fake(args, workdir):
            if "molecule.sdf" in args:
                sdf_texts.append(Path(workdir, "molecule.sdf").read_text())
            if "charges.txt" in args:
                Path(workdir, "charges.txt").write_text(" ".join(["0.0"] * 12))

        monkeypatch.setattr(gaff_ncaa, "_run_antechamber", fake)
        gaff_ncaa._am1bcc_charges(openff_molecule.from_smiles("CCCO"), **kwargs)
        return _sdf_coordinates(sdf_texts[0])

    def test_same_seed_same_conformer(self, monkeypatch):
        first = self._build(monkeypatch, random_seed=3)
        second = self._build(monkeypatch, random_seed=3)
        np.testing.assert_array_equal(first, second)

    def test_another_seed_another_conformer(self, monkeypatch):
        first = self._build(monkeypatch, random_seed=3)
        second = self._build(monkeypatch, random_seed=4)
        assert not np.allclose(first, second, atol=1e-2)

    def test_none_draws_a_new_conformer_each_time(self, monkeypatch):
        first = self._build(monkeypatch, random_seed=None)
        second = self._build(monkeypatch, random_seed=None)
        assert not np.allclose(first, second, atol=1e-2)

    def test_default_seed_is_the_toolkits_conformer(self, monkeypatch):
        """Seed 1 is what the OpenFF toolkit hard-codes, so the default changes no geometry."""
        from openff.units import unit

        assert _constants.DEFAULT_RANDOM_SEED == 1
        ours = self._build(monkeypatch)
        molecule = openff_molecule.from_smiles("CCCO")
        molecule.generate_conformers(n_conformers=1, rms_cutoff=0.25 * unit.angstrom)
        theirs = molecule.conformers[0].m_as(unit.angstrom)
        # The mol block keeps four decimals.
        np.testing.assert_allclose(ours, theirs, atol=1e-3)


class TestRunAntechamber:
    def test_runs_on_one_thread_in_the_work_directory(self, monkeypatch, tmp_path):
        seen: dict = {}

        def fake_run(cmd, **kwargs):
            seen.update(cmd=cmd, **kwargs)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/antechamber")
        monkeypatch.setattr(subprocess, "run", fake_run)
        gaff_ncaa._run_antechamber(["-h"], str(tmp_path))
        assert seen["cmd"] == ["antechamber", "-h"]
        assert seen["cwd"] == str(tmp_path)
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert seen["env"][variable] == "1"

    def test_failure_names_the_output(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/antechamber")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "Error: cannot read mol2"),
        )
        with pytest.raises(RuntimeError, match="cannot read mol2"):
            gaff_ncaa._run_antechamber(["-i", "x"], str(tmp_path))

    def test_missing_antechamber_says_how_to_install_it(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="ambertools"):
            gaff_ncaa._run_antechamber(["-h"], str(tmp_path))


@pytest.fixture(scope="module")
def two_propanol_builds():
    """Two builds of propanol with sqm, and the sqm.out text of each."""
    outputs: list = []
    original = gaff_ncaa._run_antechamber

    def spy(args, workdir):
        original(args, workdir)
        if "bcc" in args:
            outputs.append(Path(workdir, "sqm.out").read_text())

    molecule = openff_molecule.from_smiles("CCCO")
    gaff_ncaa._run_antechamber = spy
    try:
        charges = [gaff_ncaa._am1bcc_charges(molecule) for _ in range(2)]
    finally:
        gaff_ncaa._run_antechamber = original
    return charges, outputs


@requires_antechamber
@pytest.mark.integration
class TestWithSqm:
    def test_sqm_does_not_pick_its_diagonaliser_by_timing(self, two_propanol_builds):
        _, outputs = two_propanol_builds
        for sqm_out in outputs:
            assert "Auto diagonalization routine selection is disabled" in sqm_out
            assert "Using internal diagonalization routine (diag_routine=1)" in sqm_out
            assert "Timing diagonalization routines" not in sqm_out

    def test_two_builds_give_identical_charges(self, two_propanol_builds):
        (first, second), _ = two_propanol_builds
        np.testing.assert_array_equal(first, second)

    def test_charges_are_am1_bcc_like(self, two_propanol_builds):
        """Propanol: hydroxyl O about -0.7 e, hydroxyl H about +0.4 e, neutral overall."""
        (charges, _), _ = two_propanol_builds
        molecule = openff_molecule.from_smiles("CCCO")
        oxygen = next(a.molecule_atom_index for a in molecule.atoms if a.atomic_number == 8)
        hydroxyl_h = next(
            n.molecule_atom_index
            for n in molecule.atoms[oxygen].bonded_atoms
            if n.atomic_number == 1
        )
        assert -0.85 < charges[oxygen] < -0.55
        assert 0.3 < charges[hydroxyl_h] < 0.5
        assert charges.sum() == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("smiles, total_charge", [("CC(=O)[O-]", -1), ("CC[NH3+]", 1)])
    def test_charged_molecules_match_the_toolkit(self, smiles, total_charge):
        """``-nc`` reaches sqm as ``qmcharge``; an ion gets the charges the toolkit gives it."""
        molecule = openff_molecule.from_smiles(smiles)
        charges = gaff_ncaa._am1bcc_charges(molecule)
        molecule.assign_partial_charges("am1bcc")
        assert charges.sum() == pytest.approx(total_charge, abs=1e-9)
        assert np.abs(charges - molecule.partial_charges.magnitude).max() < 5e-3

    def test_agrees_with_the_toolkit_protocol(self, two_propanol_builds):
        """Same protocol as ``assign_partial_charges("am1bcc")``; only rounding may differ."""
        (charges, _), _ = two_propanol_builds
        molecule = openff_molecule.from_smiles("CCCO")
        molecule.assign_partial_charges("am1bcc")
        reference = molecule.partial_charges.magnitude
        assert np.abs(charges - reference).max() < 5e-3


def _aba_between_two_neighbours(smiles: str):
    """Topology and coordinates (Å) of ``ACE-ABA-NME`` heavy atoms, embedded from ``smiles``.

    ``smiles`` must have the layout ``CC(=O)N[C@@H](CC)C(=O)NC``. Only the outer atom of each
    neighbour (the carbonyl C before, the N after) is kept, which is what the caps use.
    """
    from openmm.app import Topology, element
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=11)
    xyz = mol.GetConformer().GetPositions()
    # (residue, atom name, element, SMILES index)
    layout = [
        ("XXA", "C", element.carbon, 1),
        ("ABA", "N", element.nitrogen, 3),
        ("ABA", "CA", element.carbon, 4),
        ("ABA", "C", element.carbon, 7),
        ("ABA", "O", element.oxygen, 8),
        ("ABA", "CB", element.carbon, 5),
        ("ABA", "CG", element.carbon, 6),
        ("XXB", "N", element.nitrogen, 9),
    ]
    topology = Topology()
    chain = topology.addChain("A")
    residues: dict = {}
    atoms: dict = {}
    for res_name, name, elem, _ in layout:
        residue = residues.setdefault(res_name, topology.addResidue(res_name, chain))
        atoms[(res_name, name)] = topology.addAtom(name, elem, residue)
    for first, second in [
        (("XXA", "C"), ("ABA", "N")),
        (("ABA", "N"), ("ABA", "CA")),
        (("ABA", "CA"), ("ABA", "C")),
        (("ABA", "C"), ("ABA", "O")),
        (("ABA", "CA"), ("ABA", "CB")),
        (("ABA", "CB"), ("ABA", "CG")),
        (("ABA", "C"), ("XXB", "N")),
    ]:
        topology.addBond(atoms[first], atoms[second])
    positions = np.array([xyz[index] for *_, index in layout])
    return topology, positions, residues["ABA"]


@requires_antechamber
@pytest.mark.integration
class TestTheChargeConformerHasTheInputStereochemistry:
    """The AM1-BCC conformer must be the stereoisomer of the residue, not a random one."""

    @staticmethod
    def _sdf_of_the_charge_calculation(monkeypatch, smiles, seed):
        sdf_texts: list = []

        def fake(args, workdir):
            if "molecule.sdf" in args:
                sdf_texts.append(Path(workdir, "molecule.sdf").read_text())
            if "charges.txt" in args:
                # Non-zero charges: all-zero ones make the template generator run its own
                # AM1-BCC calculation.
                n_atoms = _atom_count(sdf_texts[0])
                charges = np.linspace(-0.01, 0.01, n_atoms)
                Path(workdir, "charges.txt").write_text(" ".join(str(q) for q in charges))

        monkeypatch.setattr(gaff_ncaa, "_run_antechamber", fake)
        topology, positions, residue = _aba_between_two_neighbours(smiles)
        gaff_ncaa._generate_residue_template(
            residue, topology, positions, "gaff-2.2.20", random_seed=seed
        )
        return sdf_texts[0]

    @pytest.mark.parametrize(
        "smiles, cip_label",
        [
            ("CC(=O)N[C@@H](CC)C(=O)NC", "S"),
            ("CC(=O)N[C@H](CC)C(=O)NC", "R"),
        ],
    )
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_every_seed_embeds_the_input_enantiomer(self, monkeypatch, smiles, cip_label, seed):
        from rdkit import Chem

        sdf = self._sdf_of_the_charge_calculation(monkeypatch, smiles, seed)
        mol = Chem.MolFromMolBlock(sdf, removeHs=False)
        Chem.AssignStereochemistryFrom3D(mol)
        labels = [a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")]
        assert labels == [cip_label]


class TemplateStepReachedError(Exception):
    """Raised by a stub to stop a step once it calls the template generator."""


@pytest.fixture
def recorded_seed(monkeypatch):
    """Stop the pipeline step at ``parameterize_ncaa_residues`` and record its ``random_seed``."""
    seen: dict = {}

    def stop(topology, positions, ff, **kwargs):
        seen.update(kwargs)
        raise TemplateStepReachedError

    monkeypatch.setattr(gaff_ncaa, "parameterize_ncaa_residues", stop)
    return seen


@pytest.mark.integration
class TestTheSeedReachesTheTemplateBuild:
    def test_prep(self, recorded_seed):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(SFTI1_CIF)
        with pytest.raises(TemplateStepReachedError):
            prep_structure(topology, positions, random_seed=5)
        assert recorded_seed["random_seed"] == 5

    def test_relaxation(self, recorded_seed):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        config = RelaxationConfig(
            small_molecules="auto", random_seed=6, peptide_chain_id="B", receptor_chain_id="A"
        )
        with pytest.raises(TemplateStepReachedError):
            ImplicitRelaxation(config)._setup_system(P53_PDB)
        assert recorded_seed["random_seed"] == 6

    def test_energy(self, recorded_seed):
        from binding_metrics.io.structures import load_structure, strip_heterogens
        from binding_metrics.metrics.energy import _create_implicit_system

        topology, positions = load_structure(P53_PDB)
        topology, positions = strip_heterogens(topology, positions, "B", "A")
        with pytest.raises(TemplateStepReachedError):
            _create_implicit_system(topology, positions, peptide_chain="B", random_seed=7)
        assert recorded_seed["random_seed"] == 7


@requires_cuda
@requires_antechamber
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not CYCLOSPORIN_CIF.exists(), reason="1CWA example not available")
def test_1cwa_pipeline_run_twice_gives_the_same_numbers(tmp_path):
    """Prep, relaxation, energy and interface of cyclosporin A do not depend on the run.

    Before the fix, two runs of the same command gave relaxed E_int of -298.6 and -297.1
    kJ/mol and buried areas of 1010.3 and 1006.4 A^2, because MeBmt was parameterised
    with different charges in prep.
    """
    from binding_metrics.cli.run import run_pipeline

    runs = []
    for name in ("first", "second"):
        runs.append(
            run_pipeline(
                CYCLOSPORIN_CIF,
                tmp_path / name,
                peptide_chain="C",
                receptor_chain="A",
                md_duration_ps=0.0,
                metrics=frozenset({"energy", "interface", "geometry", "electrostatics"}),
                energy_modes=("raw", "relaxed"),
                random_seed=1,
            )
        )
    first, second = runs

    # Prep runs on the Reference platform: the prepped structure is identical byte for byte.
    assert Path(first["prep"]["output"]).read_bytes() == Path(second["prep"]["output"]).read_bytes()

    def numbers(results):
        return {
            "minimised complex energy": results["relax"]["potential_energy_minimized"],
            "raw E_int": results["energy"]["raw_interaction_energy"],
            "relaxed E_int": results["energy"]["relaxed_interaction_energy"],
            "buried area": results["interface"]["delta_sasa"],
            "Sc": results["geometry"]["shape_complementarity"]["sc"],
        }

    # CUDA mixed precision may differ in the last digits; the template change was 1e-3 in kJ/mol.
    assert numbers(second) == pytest.approx(numbers(first), rel=1e-6)
