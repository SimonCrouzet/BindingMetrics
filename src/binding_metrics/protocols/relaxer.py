"""The contract a structure relaxer fulfils, so the pipeline can take any of them.

``ImplicitRelaxation`` (OpenMM, implicit solvent) is the relaxer the package ships.
``run_pipeline(..., relaxer=...)`` accepts any other implementation of ``Relaxer``:
a different force field, an explicit-solvent protocol, or a stub in a test.

The module imports nothing heavy, so a class can subclass ``Relaxer`` on an install
without OpenMM.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from binding_metrics.protocols.relaxation import RelaxationResult


class Relaxer(ABC):
    """Relaxes one structure file and reports what happened.

    A relaxer carries its own configuration (force field, MD length, chain IDs,
    device, seed) from construction; ``run`` takes only the input and where to
    write. ``run_pipeline`` does not configure an injected relaxer.
    """

    @abstractmethod
    def run(
        self,
        input_path: Path,
        output_dir: Path,
        sample_id: Optional[str] = None,
    ) -> "RelaxationResult":
        """Relax the structure in ``input_path`` and write the outputs to ``output_dir``.

        Args:
            input_path: Complex structure (CIF or PDB).
            output_dir: Directory for the relaxed structures; created when missing.
            sample_id: Name for this run (default: the input file stem).

        Returns:
            A ``RelaxationResult``. The pipeline reads ``success`` and
            ``error_message``, takes the structure to analyse from
            ``md_final_structure_path`` (else ``minimized_structure_path``), and
            records ``to_dict()`` under ``results["relax"]``. A failure is reported
            with ``success=False`` and an ``error_message``, not raised: the
            pipeline then continues on the unrelaxed input.
        """
