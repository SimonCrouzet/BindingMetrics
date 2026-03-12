"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow `from conftest import ...` in test files
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# GPU / CUDA availability
# ---------------------------------------------------------------------------

try:
    import openmm as _openmm
    _openmm.Platform.getPlatformByName("CUDA")
    HAS_CUDA = True
except Exception:
    HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")

# Best available OpenMM platform: CUDA if available, otherwise CPU
BEST_PLATFORM = "CUDA" if HAS_CUDA else "CPU"


# Path to real example PDB for integration tests
EXAMPLE_PDB_PATH = Path(__file__).parent.parent / "data" / "example.pdb"


@pytest.fixture
def sample_pdb_content() -> str:
    """Minimal valid PDB content with two chains for testing.

    Chain A: Alanine dipeptide (receptor) - with hydrogens
    Chain B: Glycine (ligand) - with hydrogens

    Note: Coordinates are approximate but chemically reasonable.
    """
    return """\
HEADER    TEST STRUCTURE
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  ALA A   1      -0.350   0.840   0.420  1.00  0.00           H
ATOM      3  H2  ALA A   1      -0.350  -0.840   0.420  1.00  0.00           H
ATOM      4  H3  ALA A   1      -0.350   0.000  -0.950  1.00  0.00           H
ATOM      5  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      6  HA  ALA A   1       1.780   0.000   1.040  1.00  0.00           H
ATOM      7  C   ALA A   1       2.009   1.240  -0.700  1.00  0.00           C
ATOM      8  O   ALA A   1       1.400   1.700  -1.670  1.00  0.00           O
ATOM      9  CB  ALA A   1       1.986  -1.280  -0.680  1.00  0.00           C
ATOM     10  HB1 ALA A   1       1.660  -1.280  -1.720  1.00  0.00           H
ATOM     11  HB2 ALA A   1       3.070  -1.280  -0.640  1.00  0.00           H
ATOM     12  HB3 ALA A   1       1.610  -2.150  -0.150  1.00  0.00           H
ATOM     13  N   ALA A   2       3.150   1.770  -0.240  1.00  0.00           N
ATOM     14  H   ALA A   2       3.620   1.350   0.550  1.00  0.00           H
ATOM     15  CA  ALA A   2       3.800   2.950  -0.820  1.00  0.00           C
ATOM     16  HA  ALA A   2       3.080   3.450  -1.470  1.00  0.00           H
ATOM     17  C   ALA A   2       4.300   3.900   0.270  1.00  0.00           C
ATOM     18  O   ALA A   2       4.100   3.620   1.450  1.00  0.00           O
ATOM     19  OXT ALA A   2       4.850   4.940  -0.080  1.00  0.00           O
ATOM     20  CB  ALA A   2       4.980   2.540  -1.700  1.00  0.00           C
ATOM     21  HB1 ALA A   2       5.700   3.350  -1.780  1.00  0.00           H
ATOM     22  HB2 ALA A   2       5.470   1.680  -1.240  1.00  0.00           H
ATOM     23  HB3 ALA A   2       4.640   2.280  -2.700  1.00  0.00           H
TER
ATOM     24  N   GLY B   1      10.000  10.000  10.000  1.00  0.00           N
ATOM     25  H1  GLY B   1       9.650  10.840  10.420  1.00  0.00           H
ATOM     26  H2  GLY B   1       9.650   9.160  10.420  1.00  0.00           H
ATOM     27  H3  GLY B   1       9.650  10.000   9.050  1.00  0.00           H
ATOM     28  CA  GLY B   1      11.458  10.000  10.000  1.00  0.00           C
ATOM     29  HA2 GLY B   1      11.780  10.000  11.040  1.00  0.00           H
ATOM     30  HA3 GLY B   1      11.780  10.900   9.480  1.00  0.00           H
ATOM     31  C   GLY B   1      12.009  11.240   9.300  1.00  0.00           C
ATOM     32  O   GLY B   1      11.400  11.700   8.330  1.00  0.00           O
ATOM     33  OXT GLY B   1      13.050  11.800   9.700  1.00  0.00           O
TER
END
"""


@pytest.fixture
def sample_pdb_path(sample_pdb_content: str, tmp_path: Path) -> Path:
    """Create a temporary PDB file for testing."""
    pdb_path = tmp_path / "test_complex.pdb"
    pdb_path.write_text(sample_pdb_content)
    return pdb_path


@pytest.fixture
def minimal_trajectory_data() -> dict:
    """Minimal mock trajectory data for metric testing."""
    n_frames = 10
    n_atoms = 33  # Updated for PDB with hydrogens

    # Random positions with small fluctuations
    np.random.seed(42)
    base_positions = np.random.rand(n_atoms, 3) * 5  # 5 nm box

    positions = np.zeros((n_frames, n_atoms, 3))
    for i in range(n_frames):
        positions[i] = base_positions + np.random.randn(n_atoms, 3) * 0.01

    return {
        "positions": positions,  # nm
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "ligand_indices": list(range(23, 33)),  # Chain B (atoms 24-33, 0-indexed)
        "receptor_indices": list(range(23)),  # Chain A (atoms 1-23, 0-indexed)
    }


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def example_pdb_path() -> Path:
    """Path to real example PDB for integration tests.

    Skips test if the example PDB is not available.
    """
    if not EXAMPLE_PDB_PATH.exists():
        pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")
    return EXAMPLE_PDB_PATH


@pytest.fixture
def example_pdb_chains() -> dict:
    """Chain IDs for the example PDB structure.

    Returns dict with 'ligand' and 'receptor' chain IDs.
    Tests should use these rather than hard-coding chain IDs.
    """
    return {
        "ligand": "M",
        "receptor": ["R"],
    }


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring OpenMM/MDTraj installation"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU acceleration"
    )
