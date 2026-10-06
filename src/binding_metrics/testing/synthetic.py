"""Synthetic pose sets whose pairwise RMSD is known in closed form.

``make_synthetic_poses`` builds poses of one complex from several back-ends and seeds, the
way a model run would, without a model. The receptor of the bundled 1YCR complex (MDM2,
chain A) is left as it is and the peptide (p53, chain B) is moved as a rigid body, so what a
pose-set analysis should report (pairwise peptide RMSD after superposition on the receptor,
clusters, medoids) is known in advance and no code under test is involved in computing it.

Construction
------------
Let ``c`` be the centroid of the peptide's C-alpha atoms in the file. Pose ``i`` moves every
peptide atom ``x`` to ``R_i (x - c) + c + t_i``: a rotation ``R_i`` about an axis through
``c`` (``rotation_degrees``, none by default) and a shift ``t_i`` of ``offset_i`` angstrom along
a fixed unit vector (``offsets_angstrom``). With rotations off, the RMSD of poses ``i`` and
``j`` is ``|offset_i - offset_j|`` for every atom set, exactly. With rotations, for any set of
``n`` atoms with offsets ``d_k = x_k - c`` from ``c``, mean offset ``m`` and second moment
``M = (1/n) sum_k d_k d_k^T``::

    RMSD_ij^2 = tr(dR M dR^T) + 2 dt . (dR m) + |dt|^2,    dR = R_i - R_j,  dt = t_i - t_j

This is evaluated from the file's coordinates and the motions alone, in float64, and held in
``SyntheticTruth.rmsd`` for the atom sets ``"ca"``, ``"backbone"`` (N, CA, C, O) and
``"heavy"`` (every non-hydrogen peptide atom).

The pairwise RMSD "after receptor superposition" equals this value because every pose has
the same receptor up to a rigid motion: a Kabsch fit on any set of receptor atoms undoes the
motion exactly, and the peptide then sits where the construction put it.

With ``frame_noise=True`` (the default) each pose is also given its own random rigid
transform of the whole complex, so that raw coordinates are in arbitrary frames and the
superposition has work to do. The transforms are seeded (``rng_seed``, one stream per pose
that does not depend on the other poses) and kept in ``SyntheticTruth.frame_transforms``.

Tolerance
---------
The closed form is exact in float64. biotite stores coordinates as float32, so the poses
hold the construction rounded to float32 (about ``6e-8 * |x|``, a few ``1e-6`` angstrom at the
size of this complex after a frame transform): an RMSD computed from the poses agrees with
the closed form to ``1e-4`` angstrom (the observed difference is about ``1e-6``). The mmCIF
files of ``SyntheticPoses.write`` keep the float32 values exactly; a PDB file would round to
0.001 angstrom, so a test on PDB files should use ``atol=2e-3``.

Where the complex comes from
----------------------------
By default the bundled copy of ``example_linear_p53_1YCR.pdb`` (PDB entry 1YCR, MDM2 bound to
the p53 transactivation peptide). The package ships it as data in
``binding_metrics/testing/data/``, declared in ``pyproject.toml``, and it is found through
``importlib.resources``, so it is there in an installed wheel, an sdist install and an editable
install. A source tree that lacks that copy falls back to the repository's ``data/`` directory;
if neither exists ``FileNotFoundError`` names ``complex_path``. Pass ``complex_path`` to use any
other two-chain structure. Only the atoms of the two chains are kept.

Poses are ``DockedPose`` objects built from in-memory atoms, with ``backend`` and ``seed`` in
their provenance (the group key of the pose-set aggregates) and a round-number
``ranking_score`` as native score. Real model output never enters this module.

Importing this module imports numpy and the standard library only; biotite is imported when
the structure is read or written.
"""

from __future__ import annotations

import atexit
import contextlib
import io
import math
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.experimental import ExperimentalPose

#: Name of the bundled complex file.
BUNDLED_COMPLEX_NAME = "example_linear_p53_1YCR.pdb"

#: Files that ``importlib.resources`` had to extract from an archive (a zipped install); they
#: must outlive the call that returns their path, so they are removed when the process exits.
_EXTRACTED_RESOURCES = contextlib.ExitStack()
atexit.register(_EXTRACTED_RESOURCES.close)

DEFAULT_BACKENDS = ("backend_a", "backend_b")
DEFAULT_SEEDS = (1, 2, 3)

#: Offsets in angstrom along ``direction`` of the default set (one per seed). Within
#: ``backend_a`` the pairs are 0.4, 0.8 and 0.4 angstrom apart, within ``backend_b`` 0.3, 0.9
#: and 0.6, and the nearest pair across the two is 1.2 (a3 and b1).
DEFAULT_OFFSETS_ANGSTROM = MappingProxyType(
    {"backend_a": (0.0, 0.4, 0.8), "backend_b": (2.0, 2.3, 2.9)}
)

#: Spacing of the offsets for other backends and seeds:
#: ``BACKEND_GAP_ANGSTROM * backend_index + SEED_STEP_ANGSTROM * seed_position``.
BACKEND_GAP_ANGSTROM = 2.0
SEED_STEP_ANGSTROM = 0.4

#: A unit vector that is not a coordinate axis, so no component hides an axis mix-up.
DEFAULT_DIRECTION = (2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0)
DEFAULT_ROTATION_AXIS = (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0)

#: Atom sets of the peptide with a closed-form RMSD.
ATOM_SETS = ("ca", "backbone", "heavy")
_BACKBONE_NAMES = ("N", "CA", "C", "O")

_FRAME_TRANSLATION_ANGSTROM = 25.0
_BACKEND_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


# ---------------------------------------------------------------------- result types


@dataclass(frozen=True, eq=False)
class SyntheticTruth:
    """What the construction guarantees about a synthetic pose set.

    Attributes:
        ids: Pose ids, in the order of ``SyntheticPoses.poses`` and of the matrices.
        backends: The back-end of each pose.
        seeds: The seed of each pose.
        offsets_angstrom: The shift of the peptide along ``direction`` for each pose.
        rotation_degrees: The rotation of the peptide about the axis through its C-alpha
            centroid for each pose (zeros when there is none).
        direction, rotation_axis: The unit vectors of the shift and the rotation axis.
        rmsd: ``{"ca" | "backbone" | "heavy": (n, n) array}`` of the pairwise peptide RMSD
            in angstrom after receptor superposition, no refit of the peptide; closed form.
        frame_transforms: ``None`` without ``frame_noise``; otherwise, for each pose, the
            rotation (3 x 3) and translation (3) that map the pose's own coordinates from
            the frame of the file to the frame the pose is stored in (``x @ R.T + t``).
        scores: The ``ranking_score`` of each pose.
    """

    ids: tuple[str, ...]
    backends: tuple[str, ...]
    seeds: tuple[int, ...]
    offsets_angstrom: tuple[float, ...]
    rotation_degrees: tuple[float, ...]
    direction: tuple[float, float, float]
    rotation_axis: tuple[float, float, float]
    rmsd: Mapping[str, np.ndarray]
    frame_transforms: Optional[tuple[tuple[np.ndarray, np.ndarray], ...]]
    scores: tuple[float, ...]

    def pair(self, id_a: str, id_b: str, kind: str = "ca") -> float:
        """The RMSD of two poses in angstrom for the atom set ``kind``."""
        index = {pose_id: i for i, pose_id in enumerate(self.ids)}
        return float(self.rmsd[kind][index[id_a], index[id_b]])

    def ids_of(self, backend: str) -> list[str]:
        """The ids of the poses of one back-end, in matrix order."""
        return [i for i, b in zip(self.ids, self.backends) if b == backend]


@dataclass(frozen=True, eq=False)
class SyntheticPoses:
    """A synthetic pose set: the poses, the unmoved reference complex and the truth.

    Attributes:
        poses: One ``DockedPose`` per (back-end, seed), back-end by back-end in the order
            given, seeds in the order given. ``provenance`` has ``backend``, ``seed``,
            ``offset_angstrom``, ``rotation_degrees`` and ``synthetic=True``.
        reference: The complex as it is in the file (an ``ExperimentalPose``; for the
            bundled file its PDB ID, method and resolution are set). With ``frame_noise``
            the poses are in other frames than the reference.
        truth: See ``SyntheticTruth``.
        binder_chain, target_chain: The chain IDs of the peptide and the receptor.
    """

    poses: list[DockedPose]
    reference: ExperimentalPose
    truth: SyntheticTruth
    binder_chain: str
    target_chain: str

    def write(self, directory: Union[str, Path]) -> list[Path]:
        """Write each pose as ``<id>.cif`` in ``directory`` (created if needed).

        Returns the paths in the order of ``poses``. The coordinates keep their float32
        values exactly, so reading a file back gives the pose's atoms.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for pose in self.poses:
            path = directory / f"{pose.id}.cif"
            write_cif(pose.atoms(), path)
            paths.append(path)
        return paths


# ---------------------------------------------------------------------- the complex


def _packaged_complex_path() -> Optional[Path]:
    """Path of the copy shipped as package data, found through ``importlib.resources``.

    None when this package has no such file. For an install from an archive the file is
    extracted once to a temporary location that lives until the process exits.
    """
    try:
        resource = resources.files(__package__).joinpath("data", BUNDLED_COMPLEX_NAME)
        if not resource.is_file():
            return None
    except (ModuleNotFoundError, FileNotFoundError, NotADirectoryError):
        return None
    if isinstance(resource, Path):
        return resource
    return Path(_EXTRACTED_RESOURCES.enter_context(resources.as_file(resource)))


def _candidate_paths() -> list[Path]:
    """Where the bundled complex may be, best first: package data, then the source tree."""
    candidates = []
    packaged = _packaged_complex_path()
    if packaged is not None:
        candidates.append(packaged)
    candidates.append(Path(__file__).resolve().parents[3] / "data" / BUNDLED_COMPLEX_NAME)
    return candidates


def bundled_complex_path() -> Path:
    """Path of the bundled 1YCR file (MDM2 chain A, p53 peptide chain B).

    The copy in the package data (``binding_metrics/testing/data/``) is used when it exists,
    else the repository's ``data/`` directory of a source tree.

    Raises:
        FileNotFoundError: If neither has it; pass ``complex_path`` to ``make_synthetic_poses``
            then.
    """
    candidates = _candidate_paths()
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"the bundled complex {BUNDLED_COMPLEX_NAME} was not found (looked in "
        f"{[str(path) for path in candidates]}); it ships as package data of "
        "binding_metrics.testing and, in a source tree, is read from the repository's data/ "
        "directory. Pass complex_path=<a two-chain structure file>."
    )


def write_cif(atoms: Any, path: Union[str, Path]) -> None:
    """Write an ``AtomArray`` as mmCIF with biotite, in UTF-8; coordinates keep float32 values."""
    from binding_metrics.metrics._common import import_biotite

    _, pdbx, _ = import_biotite("writing synthetic poses")
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atoms)
    buffer = io.StringIO()
    cif.write(buffer)
    Path(path).write_text(buffer.getvalue(), encoding="utf-8")


def _load_complex(path: Path, binder_chain: str, target_chain: str):
    from binding_metrics.metrics._common import load_structure

    atoms = load_structure(path, purpose="synthetic poses")
    chains = {str(chain) for chain in np.unique(atoms.chain_id)}
    for role, chain in (("binder_chain", binder_chain), ("target_chain", target_chain)):
        if chain not in chains:
            raise ValueError(f"{role} {chain!r} is not in {path} (chains: {sorted(chains)})")
    kept = atoms[np.isin(atoms.chain_id, [binder_chain, target_chain]) & (atoms.element != "H")]
    binder = kept.chain_id == binder_chain
    if not np.any(binder & (kept.atom_name == "CA")):
        raise ValueError(f"chain {binder_chain!r} of {path} has no C-alpha atom")
    if np.count_nonzero(~binder & (kept.atom_name == "CA")) < 3:
        raise ValueError(f"chain {target_chain!r} of {path} has fewer than 3 C-alpha atoms")
    return kept


# ---------------------------------------------------------------------- arguments


def _check_names(
    backends: Sequence[str], seeds: Sequence[int]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    backends, seeds = tuple(backends), tuple(seeds)
    if not backends or not seeds:
        raise ValueError("backends and seeds must not be empty")
    for backend in backends:
        if not isinstance(backend, str) or not _BACKEND_NAME.match(backend):
            raise ValueError(f"backend names must match {_BACKEND_NAME.pattern}, got {backend!r}")
    if len(set(backends)) != len(backends) or len(set(seeds)) != len(seeds):
        raise ValueError("backends and seeds must each be unique")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seeds must be integers, got {seed!r}")
    return backends, tuple(int(seed) for seed in seeds)


def _per_pose_values(
    given: Optional[Mapping[str, Sequence[float]]],
    default: Mapping[str, Sequence[float]],
    backends: tuple[str, ...],
    n_seeds: int,
    label: str,
) -> list[list[float]]:
    table = default if given is None else given
    rows = []
    for backend in backends:
        if backend not in table:
            raise ValueError(f"{label} has no entry for backend {backend!r}")
        values = [float(value) for value in table[backend]]
        if len(values) != n_seeds:
            raise ValueError(f"{label}[{backend!r}] has {len(values)} values for {n_seeds} seeds")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{label}[{backend!r}] must be finite, got {values}")
        rows.append(values)
    return rows


def _default_offsets(backends: tuple[str, ...], n_seeds: int) -> dict[str, tuple[float, ...]]:
    return {
        backend: tuple(BACKEND_GAP_ANGSTROM * i + SEED_STEP_ANGSTROM * j for j in range(n_seeds))
        for i, backend in enumerate(backends)
    }


def _unit_vector(vector: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array)) if array.shape == (3,) else 0.0
    if array.shape != (3,) or not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{label} must be a non-zero vector of 3 numbers, got {vector!r}")
    return array / norm


# ---------------------------------------------------------------------- geometry


def _rotation_matrix(axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues' formula: the proper rotation by ``degrees`` about the unit vector ``axis``."""
    theta = math.radians(degrees)
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(theta) * cross + (1.0 - math.cos(theta)) * (cross @ cross)


def _random_frame(seed: int, backend_index: int, seed_position: int):
    """A seeded random rotation and translation; one stream per pose, independent of the others."""
    stream = np.random.SeedSequence(entropy=seed, spawn_key=(backend_index, seed_position))
    rng = np.random.default_rng(stream)
    quaternion = rng.normal(size=4)
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    translation = rng.uniform(-_FRAME_TRANSLATION_ANGSTROM, _FRAME_TRANSLATION_ANGSTROM, size=3)
    return rotation, translation


def _atom_set_masks(atoms, binder: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ca": binder & (atoms.atom_name == "CA"),
        "backbone": binder & np.isin(atoms.atom_name, _BACKBONE_NAMES),
        "heavy": binder.copy(),
    }


def _closed_form_rmsd(
    points: np.ndarray,
    centre: np.ndarray,
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
) -> np.ndarray:
    """Pairwise RMSD of rigid motions ``x -> R (x - centre) + centre + t`` of ``points``.

    ``RMSD^2 = tr(dR M dR^T) + 2 dt . (dR m) + |dt|^2`` with ``m`` the mean and ``M`` the
    second moment of the offsets of the points from ``centre``.
    """
    offsets = points - centre
    mean = offsets.mean(axis=0)
    second_moment = offsets.T @ offsets / len(offsets)
    n_poses = len(rotations)
    matrix = np.zeros((n_poses, n_poses))
    for i in range(n_poses):
        for j in range(i + 1, n_poses):
            delta_rotation = rotations[i] - rotations[j]
            delta_shift = translations[i] - translations[j]
            squared = (
                np.trace(delta_rotation @ second_moment @ delta_rotation.T)
                + 2.0 * delta_shift @ (delta_rotation @ mean)
                + delta_shift @ delta_shift
            )
            matrix[i, j] = matrix[j, i] = math.sqrt(max(squared, 0.0))
    return matrix


# ---------------------------------------------------------------------- the generator


def make_synthetic_poses(
    complex_path: Union[str, Path, None] = None,
    *,
    binder_chain: str = "B",
    target_chain: str = "A",
    backends: Sequence[str] = DEFAULT_BACKENDS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    offsets_angstrom: Optional[Mapping[str, Sequence[float]]] = None,
    rotation_degrees: Optional[Mapping[str, Sequence[float]]] = None,
    direction: Sequence[float] = DEFAULT_DIRECTION,
    rotation_axis: Sequence[float] = DEFAULT_ROTATION_AXIS,
    frame_noise: bool = True,
    rng_seed: int = 0,
) -> SyntheticPoses:
    """Build a deterministic pose set with a known pairwise RMSD (see the module docstring).

    The same arguments give the same coordinates and the same truth on every call: the only
    randomness is the frame noise, seeded by ``rng_seed``.

    Args:
        complex_path: A structure file with the two chains; None for the bundled 1YCR (see
            ``bundled_complex_path`` for where it is read from).
        binder_chain, target_chain: The peptide (moved) and the receptor (left as it is).
        backends: Names of the back-ends, letters, digits, ``_``, ``.`` and ``-``; default
            two. One pose is made for every (back-end, seed).
        seeds: Seed values, integers; default three. They only label the poses: the
            perturbation of a pose is set by its offset, not by the seed value.
        offsets_angstrom: ``{backend: one offset per seed}`` in angstrom along ``direction``.
            None gives ``DEFAULT_OFFSETS_ANGSTROM`` for the default back-ends and seeds, and
            ``2.0 * backend_index + 0.4 * seed_position`` for any other choice.
        rotation_degrees: ``{backend: one angle per seed}``, a rotation of the peptide about
            ``rotation_axis`` through its C-alpha centroid; None means no rotation.
        direction, rotation_axis: Vectors of 3 numbers; normalised to unit length.
        frame_noise: Give every pose its own random rigid transform of the whole complex.
        rng_seed: Seed of the frame noise, a non-negative integer.

    Returns:
        ``SyntheticPoses``; ``truth.rmsd`` holds the closed-form pairwise RMSD matrices.

    Raises:
        FileNotFoundError: If ``complex_path`` is None and the bundled file is not available.
        ValueError: For a bad argument, or a chain the structure does not have.
        ImportError: If biotite is not installed.
    """
    backends, seeds = _check_names(backends, seeds)
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, (int, np.integer)) or rng_seed < 0:
        raise ValueError(f"rng_seed must be a non-negative integer, got {rng_seed!r}")
    if binder_chain == target_chain:
        raise ValueError("binder_chain and target_chain must differ")
    n_seeds = len(seeds)
    use_default_offsets = offsets_angstrom is None and (backends, seeds) == (
        DEFAULT_BACKENDS,
        DEFAULT_SEEDS,
    )
    offsets = _per_pose_values(
        offsets_angstrom,
        DEFAULT_OFFSETS_ANGSTROM if use_default_offsets else _default_offsets(backends, n_seeds),
        backends,
        n_seeds,
        "offsets_angstrom",
    )
    angles = _per_pose_values(
        rotation_degrees,
        {backend: (0.0,) * n_seeds for backend in backends},
        backends,
        n_seeds,
        "rotation_degrees",
    )
    shift_direction = _unit_vector(direction, "direction")
    axis = _unit_vector(rotation_axis, "rotation_axis")

    bundled = complex_path is None
    path = bundled_complex_path() if bundled else Path(complex_path)
    complex_atoms = _load_complex(path, binder_chain, target_chain)
    binder = complex_atoms.chain_id == binder_chain
    reference = ExperimentalPose.from_atoms(
        complex_atoms.copy(),
        id="reference",
        **(
            {"method": "X-RAY DIFFRACTION", "pdb_id": "1YCR", "resolution_angstrom": 2.6}
            if bundled
            else {}
        ),
    )

    coordinates = complex_atoms.coord.astype(np.float64)
    masks = _atom_set_masks(complex_atoms, binder)
    centre = coordinates[masks["ca"]].mean(axis=0)

    poses, ids, rotations, shifts, frames = [], [], [], [], []
    pose_backends, pose_seeds, pose_offsets, pose_angles, scores = [], [], [], [], []
    for backend_index, backend in enumerate(backends):
        for seed_position, seed in enumerate(seeds):
            offset = offsets[backend_index][seed_position]
            angle = angles[backend_index][seed_position]
            rotation = _rotation_matrix(axis, angle)
            shift = offset * shift_direction
            moved = coordinates.copy()
            moved[binder] = (coordinates[binder] - centre) @ rotation.T + centre + shift
            if frame_noise:
                frame_rotation, frame_shift = _random_frame(rng_seed, backend_index, seed_position)
                moved = moved @ frame_rotation.T + frame_shift
                frames.append((frame_rotation, frame_shift))
            atoms = complex_atoms.copy()
            atoms.coord = moved.astype(np.float32)
            score = round(0.55 + 0.05 * (backend_index * n_seeds + seed_position), 2)
            pose_id = f"{backend}-s{seed}"
            poses.append(
                DockedPose.from_atoms(
                    atoms,
                    program=backend,
                    scores={"ranking_score": score},
                    id=pose_id,
                    backend=backend,
                    seed=seed,
                    offset_angstrom=offset,
                    rotation_degrees=angle,
                    synthetic=True,
                )
            )
            ids.append(pose_id)
            rotations.append(rotation)
            shifts.append(shift)
            pose_backends.append(backend)
            pose_seeds.append(seed)
            pose_offsets.append(offset)
            pose_angles.append(angle)
            scores.append(score)

    truth = SyntheticTruth(
        ids=tuple(ids),
        backends=tuple(pose_backends),
        seeds=tuple(pose_seeds),
        offsets_angstrom=tuple(pose_offsets),
        rotation_degrees=tuple(pose_angles),
        direction=tuple(float(v) for v in shift_direction),
        rotation_axis=tuple(float(v) for v in axis),
        rmsd=MappingProxyType(
            {
                kind: _closed_form_rmsd(coordinates[masks[kind]], centre, rotations, shifts)
                for kind in ATOM_SETS
            }
        ),
        frame_transforms=tuple(frames) if frame_noise else None,
        scores=tuple(scores),
    )
    return SyntheticPoses(poses, reference, truth, binder_chain, target_chain)
