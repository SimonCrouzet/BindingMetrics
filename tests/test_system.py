"""Tests for core.system: hydrogen placement determinism and Cα H repair.

``core.system`` holds structure prep, and prep is where a wrong-side Cα
hydrogen silently inverted stereocenters downstream (see
``repair_ca_hydrogen_chirality``). Both the repair and the determinism guard are
correctness-critical and cheap to test directly, so they are unit-tested here on
synthetic geometry rather than only exercised incidentally through fixtures.

The synthetic tests build an ideal tetrahedral Cα by hand. For a regular
tetrahedron the four unit vectors sum to zero, so placing N/C/CB along three of
them puts the correct H along the fourth — which is exactly the vertex the
repair should compute.
"""

import numpy as np
import pytest
from openmm import Vec3, unit
from openmm.app import Topology, element

from binding_metrics.core.system import (
    deterministic_hydrogen_placement,
    repair_ca_hydrogen_chirality,
)

# Four unit vectors of a regular tetrahedron centred on the origin (they sum to 0).
TETRA = [
    np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
    np.array([1.0, -1.0, -1.0]) / np.sqrt(3),
    np.array([-1.0, 1.0, -1.0]) / np.sqrt(3),
    np.array([-1.0, -1.0, 1.0]) / np.sqrt(3),
]
# Representative bond lengths (nm).
D_N, D_C, D_CB, D_HA = 0.145, 0.152, 0.153, 0.109


def _signed_volume(n, ca, c, cb):
    return float(np.dot(n - ca, np.cross(c - ca, cb - ca)))


def _build_ca_residue(ha_dir, resname="ALA", mirror=False, with_cb=True):
    """One residue with an ideal tetrahedral Cα; H placed along ``ha_dir``.

    ``mirror=True`` reflects every coordinate through x, turning the L centre
    into its D enantiomer without disturbing the geometry.
    """
    ca = np.zeros(3)
    coords = {
        "N": ca + TETRA[0] * D_N,
        "CA": ca,
        "C": ca + TETRA[1] * D_C,
        "HA": ca + ha_dir * D_HA,
    }
    if with_cb:
        coords["CB"] = ca + TETRA[2] * D_CB
    if mirror:
        coords = {k: v * np.array([-1.0, 1.0, 1.0]) for k, v in coords.items()}

    top = Topology()
    chain = top.addChain("A")
    res = top.addResidue(resname, chain)
    order = ["N", "CA", "C", "CB", "HA"] if with_cb else ["N", "CA", "C", "HA"]
    els = {"N": element.nitrogen, "CA": element.carbon, "C": element.carbon,
           "CB": element.carbon, "HA": element.hydrogen}
    for name in order:
        top.addAtom(name, els[name], res)
    positions = unit.Quantity([Vec3(*coords[n]) for n in order], unit.nanometer)
    return top, positions, order


def _faces(topology, positions):
    """Return (v_cb, v_ha): signed volumes of CB and HA about the N/CA/C plane."""
    pos = np.array(positions.value_in_unit(unit.nanometer))
    idx = {a.name: a.index for a in topology.atoms()}
    ca = pos[idx["CA"]]
    return (
        _signed_volume(pos[idx["N"]], ca, pos[idx["C"]], pos[idx["CB"]]),
        _signed_volume(pos[idx["N"]], ca, pos[idx["C"]], pos[idx["HA"]]),
    )


class TestRepairCaHydrogenChirality:
    """Unit tests for the wrong-side Cα hydrogen repair."""

    @pytest.mark.parametrize("mirror,label", [(False, "L"), (True, "D")])
    def test_noop_on_correct_centre(self, mirror, label):
        """A correct centre must be left exactly alone — for L and D alike."""
        top, pos, _ = _build_ca_residue(TETRA[3], mirror=mirror)
        v_cb, v_ha = _faces(top, pos)
        assert (v_cb > 0) != (v_ha > 0), f"{label}: fixture is not a correct centre"

        out = repair_ca_hydrogen_chirality(top, pos, verbose=False)

        before = np.array(pos.value_in_unit(unit.nanometer))
        after = np.array(out.value_in_unit(unit.nanometer))
        assert np.allclose(before, after, atol=0.0), (
            f"{label}: repair moved atoms on an already-correct structure"
        )

    @pytest.mark.parametrize("mirror,label", [(False, "L"), (True, "D")])
    def test_repairs_wrong_side_hydrogen(self, mirror, label):
        """HA on CB's face is moved back to the opposite face."""
        # Put HA along CB's direction => same face as CB (chemically impossible).
        top, pos, _ = _build_ca_residue(TETRA[2], mirror=mirror)
        v_cb, v_ha = _faces(top, pos)
        assert (v_cb > 0) == (v_ha > 0), f"{label}: fixture should start broken"

        out = repair_ca_hydrogen_chirality(top, pos, verbose=False)

        v_cb2, v_ha2 = _faces(top, out)
        assert (v_cb2 > 0) != (v_ha2 > 0), f"{label}: HA still on the wrong face"

    @pytest.mark.parametrize("mirror,label", [(False, "L"), (True, "D")])
    def test_repair_moves_only_the_hydrogen(self, mirror, label):
        """The heavy atoms define the handedness and must never be touched.

        This is the property that makes the repair safe for D-amino acids: it
        reads the correct side off the actual CB rather than assuming L, so it
        can never convert a D residue into an L one.
        """
        top, pos, order = _build_ca_residue(TETRA[2], mirror=mirror)
        before = np.array(pos.value_in_unit(unit.nanometer))
        v_cb_before, _ = _faces(top, pos)

        out = repair_ca_hydrogen_chirality(top, pos, verbose=False)
        after = np.array(out.value_in_unit(unit.nanometer))

        for i, name in enumerate(order):
            if name != "HA":
                assert np.allclose(before[i], after[i], atol=0.0), (
                    f"{label}: repair moved heavy atom {name}"
                )
        v_cb_after, _ = _faces(top, out)
        assert np.sign(v_cb_before) == np.sign(v_cb_after), (
            f"{label}: CB handedness changed — repair must not re-assign chirality"
        )

    def test_repaired_hydrogen_sits_at_the_ideal_vertex(self):
        """The moved H lands on the 4th tetrahedral vertex at a real C-H length."""
        top, pos, order = _build_ca_residue(TETRA[2])
        out = repair_ca_hydrogen_chirality(top, pos, verbose=False)
        after = np.array(out.value_in_unit(unit.nanometer))
        idx = {name: i for i, name in enumerate(order)}

        ha, ca = after[idx["HA"]], after[idx["CA"]]
        assert np.allclose(ha, TETRA[3] * D_HA, atol=1e-6), "H not at the ideal vertex"
        # CA-HA must satisfy the length that constraints=HBonds will impose.
        assert np.isclose(np.linalg.norm(ha - ca), D_HA, atol=1e-6)

    def test_skips_residue_without_cb(self):
        """Glycine has no Cα stereocenter, so there is nothing to repair."""
        top, pos, _ = _build_ca_residue(TETRA[2], resname="GLY", with_cb=False)
        before = np.array(pos.value_in_unit(unit.nanometer))
        out = repair_ca_hydrogen_chirality(top, pos, verbose=False)
        after = np.array(out.value_in_unit(unit.nanometer))
        assert np.allclose(before, after, atol=0.0), "repair touched a CB-less residue"

    def test_returns_vec3_not_tuples(self):
        """Downstream consumers index positions as p.x/p.y/p.z."""
        top, pos, _ = _build_ca_residue(TETRA[2])
        out = repair_ca_hydrogen_chirality(top, pos, verbose=False)
        first = out[0]
        assert hasattr(first, "x") and hasattr(first, "y") and hasattr(first, "z")


class TestDeterministicHydrogenPlacement:
    """The seeding guard that makes prep reproducible."""

    def test_restores_caller_rng_state(self):
        """Seeding inside the block must not perturb randomness outside it."""
        import random

        random.seed(1234)
        expected = [random.random() for _ in range(3)]

        random.seed(1234)
        with deterministic_hydrogen_placement():
            random.random()  # burn draws inside the block
            random.random()
        actual = [random.random() for _ in range(3)]

        assert actual == expected, "prep leaked RNG state to the caller"

    def test_block_is_reproducible(self):
        """The same block yields the same draws on every entry."""
        import random

        def draws():
            with deterministic_hydrogen_placement():
                return [random.random() for _ in range(5)]

        assert draws() == draws()

    def test_seed_is_configurable(self):
        """A different seed gives a different (still reproducible) sequence."""
        import random

        def draws(seed):
            with deterministic_hydrogen_placement(seed=seed):
                return [random.random() for _ in range(5)]

        assert draws(0) == draws(0)
        assert draws(0) != draws(1)


@pytest.mark.integration
class TestPrepDeterminism:
    """End-to-end: identical input must give a bit-identical prepped structure."""

    def test_prep_is_reproducible(self, example_pdb_path):
        """Regression: prep must give the same structure for the same input.

        ``addHydrogens`` used to jitter every new hydrogen by up to 0.05 nm
        (0.5 Å) from an *unseeded* global RNG, and ``addMissingAtoms`` minimized
        rebuilt atoms with an unseeded stochastic integrator. Identical input
        therefore produced a different structure — and a different minimized
        energy — on every run.

        Two different bars, because two different things are going on:

        * **Heavy atoms must be exactly reproducible.** Measured at 0.000000 Å
          across repeats for both the PDBFixer path (1YCR) and the cyclic path
          (3P8F). This is the assertion that carries the reproducibility claim.
        * **Hydrogens get a loose bound.** On the PDBFixer path they still shift
          by up to ~0.25 Å between runs, exclusively on soft/rotatable groups
          (lysine HG/HE, a threonine hydroxyl HG1, leucine HB...). That is the
          50-step hydrogen minimization landing in a marginally different spot
          because GPU reduction order is not deterministic — a hydroxyl rotamer,
          not randomness, and physically irrelevant. Unseeded, the spread was
          2.28 Å, so 0.5 Å still catches a regression by a wide margin.
        """
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        raw_t, raw_p = load_structure(example_pdb_path)

        def prepped():
            t, p = prep_structure(
                raw_t, raw_p, ph=7.4, keep_water=False, canonicalize=False
            )
            return t, np.array(p.value_in_unit(unit.nanometer))

        topo, first = prepped()
        _topo2, second = prepped()
        assert first.shape == second.shape, "prep returned different atom counts"

        delta_ang = np.linalg.norm(first - second, axis=1) * 10.0
        is_heavy = np.array([a.element.symbol != "H" for a in topo.atoms()])

        heavy_max = float(delta_ang[is_heavy].max())
        assert heavy_max < 1e-3, (
            f"heavy-atom geometry is not reproducible: moved up to {heavy_max:.6f} Å "
            f"between identical runs"
        )
        all_max = float(delta_ang.max())
        assert all_max < 0.5, (
            f"prep is not reproducible: atoms moved up to {all_max:.4f} Å between "
            f"identical runs (unseeded hydrogen placement jitters ~2.3 Å)"
        )

    def test_prep_leaves_no_wrong_side_ca_hydrogens(self, example_pdb_path):
        """Prep output must never contain a Cα whose HA and CB share a face."""
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        raw_t, raw_p = load_structure(example_pdb_path)
        top, pos = prep_structure(
            raw_t, raw_p, ph=7.4, keep_water=False, canonicalize=False
        )
        coords = np.array(pos.value_in_unit(unit.nanometer))

        bad = []
        for res in top.residues():
            idx = {a.name: a.index for a in res.atoms()}
            if not {"N", "CA", "C", "CB", "HA"} <= set(idx):
                continue
            ca = coords[idx["CA"]]
            v_cb = _signed_volume(coords[idx["N"]], ca, coords[idx["C"]], coords[idx["CB"]])
            v_ha = _signed_volume(coords[idx["N"]], ca, coords[idx["C"]], coords[idx["HA"]])
            if (v_cb > 0) == (v_ha > 0):
                bad.append(f"{res.name}{res.id}/{res.chain.id}")
        assert not bad, f"prep emitted wrong-side Cα hydrogens: {bad}"
