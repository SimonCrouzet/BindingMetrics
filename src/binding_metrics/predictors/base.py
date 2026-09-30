"""The contract of a predictor adapter: from the files of one model to a ``PredictionRecord``.

An adapter reads the output that one model wrote and never runs the model (runners are
separate and optional). It has two abstract steps: ``find_files`` locates the files of one
sample, ``parse`` reads them. ``load`` chains the two and applies a chain map; a model whose
files have nothing in common with the others' needs no more than these two methods.

Contract, checked for every registered adapter by ``tests/predictors/contract.py``:

* a missing file leaves the fields it feeds at NaN (None for an array) and adds a sentence to
  ``record.reasons``; a directory with no output at all gives such a record too, it does not
  raise;
* a corrupt file raises;
* parsing the scalars imports no biotite and does not open the structure file (a per-atom
  array that needs the atoms, such as the per-atom expansion of a per-token pLDDT, may be
  left None with a reason when the structure cannot be read);
* pLDDT is per atom on 0-100 in the atom order of the structure file, PAE and PDE are in
  angstrom with ``pae[i, j]`` the error of token ``j`` aligned on token ``i``, and scalars the
  model lacks are NaN (see ``binding_metrics.predictors.record``);
* ``seed_index`` and ``sample`` are 1-based positions in the model's natural order of
  outputs; ``sample=1`` is the first one, not the best-ranked one.

Adding a model: write ``predictors/<name>.py`` with a subclass of ``PredictionParser``, add
one ``ParserSpec`` line to ``predictors/registry.py``, and add
``tests/predictors/synth_<name>.py`` with a ``write_prediction`` function, after which the
contract tests cover the new adapter without further edits.

The module imports numpy only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional

from binding_metrics.predictors.record import (
    PredictionFiles,
    PredictionRecord,
    SampleRef,
    _not_provided_problems,
    check_chain_map,
)

#: The two families of confidence outputs. ``af2``: one token per residue (AlphaFold2,
#: ColabFold, AlphaFold-Multimer). ``af3``: one token per standard residue and one per heavy
#: atom of a ligand or modified residue (AlphaFold3, OpenFold3, Boltz, Protenix, Chai-1).
FAMILIES: tuple[str, ...] = ("af2", "af3")

#: Upper bound of the samples ``list_samples`` returns, so an adapter whose ``find_files``
#: never comes back empty cannot loop for long.
_PROBE_LIMIT = 1000


class PredictionParser(ABC):
    """Reads the output of one structure-prediction model.

    Subclasses set the class attributes ``name``, ``display_name`` and ``family`` and
    implement ``find_files`` and ``parse``. A parser holds no state: one instance can load
    any number of predictions.
    """

    #: Registry key and ``PredictionRecord.model`` value, lower case (``"of3"``, ``"boltz2"``).
    name: ClassVar[str]
    #: Name for reports (``"OpenFold3"``).
    display_name: ClassVar[str]
    #: ``"af2"`` or ``"af3"`` (see ``FAMILIES``).
    family: ClassVar[str]
    #: What inputs the model can be given, as a ``binding_metrics.capabilities.Capabilities``
    #: (peptide closures, residue classes, binder size, ...). None declares no constraint.
    #: The class does not exist yet; a later change defines it, and the contract tests check
    #: that this attribute is None or an instance of it. A pre-flight check reads the value
    #: to refuse an input the model cannot handle before anything runs.
    capabilities: ClassVar[Optional[Any]] = None
    #: Names of the ``PredictionRecord`` fields the model never provides (AlphaFold2 and
    #: ColabFold write no ``pde``): ``load`` copies them to ``record.not_provided``, and
    #: ``summarize_prediction`` gives no reason for them. Anything a model writes and a file
    #: lacks is still explained. Only fields listed in ``record._PROVIDABLE`` are accepted.
    not_provided: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        family = cls.__dict__.get("family")
        if family is not None and family not in FAMILIES:
            raise TypeError(f"{cls.__name__}.family must be one of {FAMILIES}, got {family!r}")
        problems = _not_provided_problems(cls.__dict__.get("not_provided", ()))
        if problems:
            raise TypeError(f"{cls.__name__}: {problems[0]}")

    @abstractmethod
    def find_files(
        self,
        prediction_dir: Path,
        name: str,
        *,
        seed_index: int = 1,
        sample: int = 1,
    ) -> PredictionFiles:
        """Locate the files of one sample; a file that does not exist is None.

        Never raises for a missing file or directory. Only looks at file names, never at
        their content.
        """

    @abstractmethod
    def parse(
        self,
        files: PredictionFiles,
        *,
        name: str,
        seed_index: int = 1,
        sample: int = 1,
    ) -> PredictionRecord:
        """Read the located files into a record (see the module docstring for the rules)."""

    def load(
        self,
        prediction_dir: str | Path,
        name: str,
        *,
        seed_index: int = 1,
        sample: int = 1,
        chain_map: Optional[Mapping[str, str]] = None,
    ) -> PredictionRecord:
        """Find and parse one sample.

        Args:
            prediction_dir: The directory the model wrote (the layout inside it is the
                adapter's business).
            name: The prediction name.
            seed_index, sample: 1-based positions in the model's natural order.
            chain_map: Model chain ID to user chain ID, applied to ``record.atoms()``. A
                chain the map does not mention keeps its ID.

        A missing file raises nothing; a corrupt file raises what its reader raises, usually
        a ``ValueError``.

        Raises:
            ValueError: If ``chain_map`` is not a valid map (see ``check_chain_map``).
        """
        checked_map = check_chain_map(chain_map) if chain_map else {}
        files = self.find_files(Path(prediction_dir), name, seed_index=seed_index, sample=sample)
        record = self.parse(files, name=name, seed_index=seed_index, sample=sample)
        if record.files is None:
            record.files = files
        if self.not_provided:
            record.not_provided = record.not_provided | self.not_provided
        if checked_map:
            record.chain_map = checked_map
        return record

    def list_samples(self, prediction_dir: str | Path, name: str) -> list[SampleRef]:
        """The samples present, in the model's natural order, with their ranking scores.

        The default probes seed positions 1, 2, ... and, inside each, sample positions 1, 2,
        ... until ``find_files`` finds no file of the sample (``PredictionFiles.has_output``),
        and parses each sample for its ranking score, which reads its arrays; an adapter that
        can read the score more cheaply overrides it. At most ``_PROBE_LIMIT`` samples are
        returned.
        """
        directory = Path(prediction_dir)
        refs: list[SampleRef] = []
        seed_index = 0
        while len(refs) < _PROBE_LIMIT:
            seed_index += 1
            sample = 0
            found_in_seed = False
            while len(refs) < _PROBE_LIMIT:
                sample += 1
                files = self.find_files(directory, name, seed_index=seed_index, sample=sample)
                if not files.has_output():
                    break
                found_in_seed = True
                record = self.parse(files, name=name, seed_index=seed_index, sample=sample)
                refs.append(SampleRef(seed_index, sample, record.ranking_score))
            if not found_in_seed:
                break
        return refs
