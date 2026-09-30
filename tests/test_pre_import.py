"""Importing ``binding_metrics.capabilities`` is cheap: no OpenMM, no torch, no biotite."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "example_bicyclic_sfti1_3P8F.cif"
HEAVY = ("openmm", "simtk", "torch", "biotite", "numpy", "scipy", "gemmi", "mdtraj", "pdbfixer")


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_import_loads_none_of_the_heavy_packages():
    out = _run(
        f"""
        import sys
        import binding_metrics.capabilities as capabilities
        capabilities.Capabilities()
        heavy = {HEAVY!r}
        print(sorted(m for m in sys.modules if m.split(".")[0] in heavy))
        """
    )
    assert out.strip() == "[]"


def test_building_and_checking_a_profile_by_hand_stays_light():
    out = _run(
        f"""
        import sys
        from binding_metrics.capabilities import Capabilities, InputProfile
        caps = Capabilities(closures={{"none"}}, reasons={{"closures": "No ring supported."}})
        caps.check(InputProfile("B", closures={{"disulfide"}}))
        heavy = {HEAVY!r}
        print(sorted(m for m in sys.modules if m.split(".")[0] in heavy))
        """
    )
    assert out.strip() == "[]"


def test_profiling_a_structure_needs_biotite_but_never_openmm():
    out = _run(
        f"""
        import sys
        from pathlib import Path
        from binding_metrics.capabilities import profile_input
        profile_input(Path({str(DATA)!r}), "I")
        print(sorted(m for m in sys.modules if m.split(".")[0] in ("openmm", "simtk", "torch")))
        """
    )
    assert out.strip() == "[]"
