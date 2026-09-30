"""Writers of synthetic AlphaFold2 and ColabFold output (layouts in ``predictors/af2.py``).

``write_bare`` writes a structure alone, as AlphaFold2 and ColabFold write one: the pLDDT of a
residue in the B-factor column of each of its atoms. The models give one pLDDT per residue, so the
writer takes the mean over the atoms of the residue in the ``SyntheticComplex``; ``PLDDT_ATOL`` is
the largest difference between an atom and its residue mean in the default complex.
"""

from pathlib import Path

import numpy as np

from tests.predictors import synth

#: The default complex has residues whose two atoms differ by up to 4, so a residue mean is
#: within 2 of each atom.
PLDDT_ATOL = 2.01

#: The pLDDT of a residue is repeated over its atoms by reading the structure file, so a test
#: cannot replace that file by a stub (``contract.check_scalars_parse_without_biotite`` does).
PLDDT_FROM_STRUCTURE = True


# ---------------------------------------------------------------------- what the model writes


def residue_starts(complex_: synth.SyntheticComplex) -> np.ndarray:
    return synth.struc.get_residue_starts(complex_.atoms)


def plddt_per_residue(complex_: synth.SyntheticComplex) -> np.ndarray:
    """The mean pLDDT of the atoms of each residue, in the order of the structure."""
    starts = residue_starts(complex_)
    counts = np.diff(np.append(starts, complex_.n_atoms))
    return np.add.reduceat(complex_.plddt_per_atom, starts) / counts


def af2_atoms(complex_: synth.SyntheticComplex):
    """The structure with the pLDDT of its residue in the B-factor of each atom."""
    starts = residue_starts(complex_)
    counts = np.diff(np.append(starts, complex_.n_atoms))
    atoms = complex_.atoms.copy()
    atoms.set_annotation("b_factor", np.repeat(plddt_per_residue(complex_), counts))
    return atoms


# ---------------------------------------------------------------------- bare structure


def write_bare(
    directory: Path,
    name: str,
    complex_: synth.SyntheticComplex,
    *,
    suffix: str = ".pdb",
    bfactor_scale: float = 1.0,
) -> Path:
    """A structure alone, with the pLDDT of each residue in the B-factor column."""
    return synth.write_structure(
        af2_atoms(complex_), Path(directory) / f"{name}{suffix}", bfactor_scale=bfactor_scale
    )
