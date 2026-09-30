"""Adapters of the stub model that each break one rule of the parser contract.

They exist to show that the contract checks catch what they claim to: each class names the
check that must fail on it (``FAILS_CHECK``). The module imports no biotite, so the class can
be loaded where biotite is blocked.
"""

import numpy as np

from tests.predictors.synth_stub import StubParser


class PlddtLeftOnZeroToOne(StubParser):
    """Forgets to rescale the per-atom pLDDT from 0-1 to 0-100."""

    FAILS_CHECK = "check_load_valid_record"

    def parse(self, files, *, name, seed_index=1, sample=1):
        record = super().parse(files, name=name, seed_index=seed_index, sample=sample)
        if record.plddt_per_atom is not None:
            record.plddt_per_atom = record.plddt_per_atom / 100.0
        return record


class TransposedPae(StubParser):
    """Reads PAE with rows and columns exchanged."""

    FAILS_CHECK = "check_load_valid_record"

    def parse(self, files, *, name, seed_index=1, sample=1):
        record = super().parse(files, name=name, seed_index=seed_index, sample=sample)
        if record.pae is not None:
            record.pae = record.pae.T.copy()
        return record


class DropsOneAtom(StubParser):
    """Returns one pLDDT value too few for the structure."""

    FAILS_CHECK = "check_load_valid_record"

    def parse(self, files, *, name, seed_index=1, sample=1):
        record = super().parse(files, name=name, seed_index=seed_index, sample=sample)
        record.plddt_per_atom = record.plddt_per_atom[:-1]
        return record


class ForgetsTheReason(StubParser):
    """Leaves a missing file without a reason."""

    FAILS_CHECK = "check_missing_files"

    def parse(self, files, *, name, seed_index=1, sample=1):
        record = super().parse(files, name=name, seed_index=seed_index, sample=sample)
        record.reasons.clear()
        return record


class RaisesOnAnEmptyDirectory(StubParser):
    """Raises for a directory without output instead of returning a record with a reason."""

    FAILS_CHECK = "check_missing_files"

    def parse(self, files, *, name, seed_index=1, sample=1):
        if not files.any_found():
            raise FileNotFoundError(files.directory)
        return super().parse(files, name=name, seed_index=seed_index, sample=sample)


class SwallowsACorruptFile(StubParser):
    """Turns any read error into an empty record."""

    FAILS_CHECK = "check_corrupt_files"

    def parse(self, files, *, name, seed_index=1, sample=1):
        try:
            return super().parse(files, name=name, seed_index=seed_index, sample=sample)
        except (ValueError, OSError, EOFError):
            return super().parse(
                type(files)(directory=files.directory, structure=files.structure),
                name=name,
                seed_index=seed_index,
                sample=sample,
            )


class CrashesOnACorruptFile(StubParser):
    """Fails with a programming error, not a data error, on a corrupt file."""

    FAILS_CHECK = "check_corrupt_files"

    def parse(self, files, *, name, seed_index=1, sample=1):
        try:
            return super().parse(files, name=name, seed_index=seed_index, sample=sample)
        except (ValueError, OSError, EOFError) as exc:
            raise AttributeError("unexpected structure of the file") from exc


class IgnoresTheSample(StubParser):
    """Always finds the first sample."""

    FAILS_CHECK = "check_sample_and_seed_selection"

    def find_files(self, prediction_dir, name, *, seed_index=1, sample=1):
        return super().find_files(prediction_dir, name, seed_index=1, sample=1)


class NeedsBiotiteToParse(StubParser):
    """Imports biotite while parsing the scalars."""

    FAILS_CHECK = "check_scalars_parse_without_biotite"

    def parse(self, files, *, name, seed_index=1, sample=1):
        import biotite.structure  # noqa: F401 - the defect under test

        return super().parse(files, name=name, seed_index=seed_index, sample=sample)


class OpensTheStructureToParse(StubParser):
    """Reads the structure file while parsing."""

    FAILS_CHECK = "check_scalars_parse_without_biotite"

    def parse(self, files, *, name, seed_index=1, sample=1):
        if files.structure is not None:
            text = files.structure.read_text(encoding="utf-8")
            if not text.startswith("data_"):
                raise ValueError("structure file is not an mmCIF")
        return super().parse(files, name=name, seed_index=seed_index, sample=sample)


class IgnoresTheChainMap(StubParser):
    """Drops the chain map that ``load`` was given."""

    FAILS_CHECK = "check_chain_map"

    def load(self, prediction_dir, name, *, seed_index=1, sample=1, chain_map=None):
        return super().load(prediction_dir, name, seed_index=seed_index, sample=sample)


class WrongName(StubParser):
    """Its ``name`` attribute is not the registry key."""

    FAILS_CHECK = "check_class_attributes"
    name = "not_the_key"


class ConstantPlddt(StubParser):
    """Returns the right shape with every value equal (a scale that hides an order bug)."""

    FAILS_CHECK = "check_load_valid_record"

    def parse(self, files, *, name, seed_index=1, sample=1):
        record = super().parse(files, name=name, seed_index=seed_index, sample=sample)
        if record.plddt_per_atom is not None:
            record.plddt_per_atom = np.full_like(record.plddt_per_atom, record.avg_plddt)
        return record


class CompleteRaises(StubParser):
    """Raises from ``complete`` instead of recording a reason."""

    FAILS_CHECK = "check_completion"

    def complete(self, record):
        raise ValueError("the structure does not fit")


class CompleteReturnsACopy(StubParser):
    """Returns another record from ``complete``."""

    FAILS_CHECK = "check_completion"

    def complete(self, record):
        import copy

        return copy.copy(record)


class CompleteGrowsTheReasons(StubParser):
    """Adds the same sentence to the reasons on every call."""

    FAILS_CHECK = "check_completion"

    def complete(self, record):
        record.reasons.append("completed")
        return record


DEFECTS = [
    PlddtLeftOnZeroToOne,
    TransposedPae,
    DropsOneAtom,
    ForgetsTheReason,
    RaisesOnAnEmptyDirectory,
    SwallowsACorruptFile,
    CrashesOnACorruptFile,
    IgnoresTheSample,
    NeedsBiotiteToParse,
    OpensTheStructureToParse,
    IgnoresTheChainMap,
    WrongName,
    ConstantPlddt,
    CompleteRaises,
    CompleteReturnsACopy,
    CompleteGrowsTheReasons,
]
