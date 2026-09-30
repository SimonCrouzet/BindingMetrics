"""The contract every registered predictor adapter meets, as reusable checks.

``test_contract.py`` runs each check below on every model in ``sorted(PARSERS)``. A check has
the signature ``check_<what>(model, workdir)`` and raises ``AssertionError`` with a message
that names the model when the adapter breaks the rule. The checks write their fixture with
the model's own writer, ``tests/predictors/synth_<model>.py::write_prediction`` (protocol in
``synth.py``); a registered model without that module fails ``check_class_attributes`` with a
message that says so, which is how a new adapter is made to bring its writer.

The rules, from ``binding_metrics.predictors.base`` and ``record``:

* ``check_class_attributes``: ``name`` equals the registry key, ``display_name`` and
  ``family`` are set and agree with the ``ParserSpec``.
* ``check_capabilities_declaration``: ``capabilities`` is None or a
  ``binding_metrics.capabilities.Capabilities`` (skipped while that class does not exist).
* ``check_load_valid_record``: a complete fixture loads into a record that passes
  ``validate(check_structure=True)``; pLDDT is per atom on 0-100 and equals the truth; PAE and
  PDE have the truth's orientation; ``chain_map`` is empty by default.
* ``check_sample_and_seed_selection``: ``seed_index`` and ``sample`` pick the right files and
  ``list_samples`` lists them in natural order.
* ``check_missing_files``: an empty or absent directory, or one absent file, gives NaN and a
  reason and never raises.
* ``check_corrupt_files``: a corrupt scores or arrays file raises a data error.
* ``check_scalars_parse_without_biotite``: loading the scalars needs no biotite and does not
  open the structure file; a per-atom array that needs the atoms may be None with a reason.
* ``check_chain_map``: ``chain_map`` renames the chains of ``atoms()``.
* ``check_completion``: ``complete`` returns the same record, does not raise, is idempotent and
  leaves a valid record whose ``chain_ptm`` and ``chain_pair_iptm`` are keyed by chain IDs of
  the structure file.
"""

import dataclasses
import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.predictors.base import FAMILIES, PredictionParser
from binding_metrics.predictors.registry import PARSERS, get_parser
from tests.predictors import synth

NAME = "cmplx"
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Exceptions that mean a bug in the adapter, not a corrupt input file.
PROGRAMMING_ERRORS = (
    AttributeError,
    NameError,
    ImportError,
    AssertionError,
    NotImplementedError,
    TypeError,
)

GARBAGE = b"\x00\x01 this is not a valid file \xff\xfe"


# ---------------------------------------------------------------------- the writer


def writer_module(model: str):
    """Import ``tests.predictors.synth_<model>``; a registered model must have one."""
    module_name = f"tests.predictors.synth_{model}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise AssertionError(
            f"parser '{model}' is registered but {module_name} does not exist: add it with a "
            "write_prediction(directory, name, complex_, *, seed_index=1, sample=1) function "
            "(protocol in tests/predictors/synth.py)"
        ) from None
    assert callable(getattr(module, "write_prediction", None)), (
        f"{module_name} has no write_prediction function"
    )
    return module


def _tolerances(module) -> dict[str, float]:
    return {
        "plddt": getattr(module, "PLDDT_ATOL", 0.05),
        "scalar": getattr(module, "SCALAR_ATOL", 0.01),
        "matrix": getattr(module, "MATRIX_ATOL", 0.01),
    }


def write_fixture(model: str, directory: Path, *, seed_index: int = 1, sample: int = 1, **kwargs):
    """Write the synthetic complex in the model's layout under ``directory``; return the truth."""
    truth = synth.synthetic_complex(**kwargs)
    Path(directory).mkdir(parents=True, exist_ok=True)
    writer_module(model).write_prediction(
        Path(directory), NAME, truth, seed_index=seed_index, sample=sample
    )
    return truth


def _load(model: str, directory: Path, **kwargs):
    return get_parser(model).load(directory, NAME, **kwargs)


def _load_without_raising(model: str, directory: Path, what: str):
    """Load ``directory``; an exception is a broken contract, reported as an AssertionError."""
    try:
        return _load(model, directory)
    except Exception as exc:  # noqa: BLE001 - reported as the broken rule "a missing file never raises"
        raise AssertionError(
            f"{model}: loading {what} raised {type(exc).__name__}: {exc}; a missing file "
            "must give NaN and a reason"
        ) from exc


# ---------------------------------------------------------------------- the checks


def check_class_attributes(model: str, workdir: Path) -> None:
    spec = PARSERS[model]
    cls = spec.load()
    assert issubclass(cls, PredictionParser), f"{model}: {spec.import_path} is no PredictionParser"
    assert cls.name == model, f"{model}: class attribute name is {cls.name!r}, not the registry key"
    assert isinstance(cls.display_name, str) and cls.display_name, f"{model}: display_name is empty"
    assert cls.family in FAMILIES, f"{model}: family {cls.family!r} is not one of {FAMILIES}"
    assert spec.display_name == cls.display_name, f"{model}: ParserSpec.display_name differs"
    assert spec.family == cls.family, f"{model}: ParserSpec.family differs"
    assert isinstance(get_parser(model), cls), f"{model}: get_parser returned another class"
    writer_module(model)  # a registered model brings its writer


def check_capabilities_declaration(model: str, workdir: Path) -> None:
    spec = PARSERS[model]
    declared = spec.load().capabilities
    assert spec.load_capabilities() is declared, f"{model}: load_capabilities() differs"
    if declared is None:
        return
    try:
        from binding_metrics.capabilities import Capabilities
    except ImportError:
        pytest.skip("binding_metrics.capabilities does not exist yet; its class cannot be checked")
    assert isinstance(declared, Capabilities), (
        f"{model}: capabilities must be None or a Capabilities, got {type(declared).__name__}"
    )


def check_load_valid_record(model: str, workdir: Path) -> None:
    truth = write_fixture(model, workdir)
    tol = _tolerances(writer_module(model))
    record = _load(model, workdir)

    assert record.model == model, f"{model}: record.model is {record.model!r}"
    assert record.name == NAME, f"{model}: record.name is {record.name!r}"
    assert (record.seed_index, record.sample) == (1, 1), f"{model}: seed_index/sample not echoed"
    assert record.structure_path is not None and Path(record.structure_path).exists(), (
        f"{model}: structure_path is missing or does not exist"
    )
    try:
        record.validate(check_structure=True)
    except ValueError as exc:
        raise AssertionError(f"{model}: {exc}") from exc

    assert np.isfinite(record.avg_plddt) or record.reasons, (
        f"{model}: avg_plddt is NaN without a reason"
    )
    assert abs(record.avg_plddt - truth.scalars["avg_plddt"]) <= tol["plddt"], (
        f"{model}: avg_plddt {record.avg_plddt} != truth {truth.scalars['avg_plddt']}"
    )
    assert record.plddt_per_atom is not None, f"{model}: no per-atom pLDDT"
    assert len(record.plddt_per_atom) == record.atoms().array_length() == truth.n_atoms, (
        f"{model}: pLDDT has {len(record.plddt_per_atom)} values for "
        f"{record.atoms().array_length()} atoms (truth {truth.n_atoms})"
    )
    np.testing.assert_allclose(
        record.plddt_per_atom,
        truth.plddt_per_atom,
        atol=tol["plddt"],
        err_msg=f"{model}: per-atom pLDDT differs from the truth (scale or order)",
    )
    for field in ("ptm", "iptm", "gpde", "ranking_score", "has_clash", "disorder"):
        value = getattr(record, field)
        if np.isfinite(value):
            assert abs(value - truth.scalars[field]) <= tol["scalar"], (
                f"{model}: {field} {value} != truth {truth.scalars[field]}"
            )
    for label in ("pae", "pde"):
        matrix = getattr(record, label)
        if matrix is None:
            continue
        expected = getattr(truth, label)
        assert matrix.shape == expected.shape, f"{model}: {label} shape {matrix.shape}"
        np.testing.assert_allclose(
            matrix,
            expected,
            atol=tol["matrix"],
            err_msg=f"{model}: {label} differs from the truth (orientation or scale)",
        )
    assert record.chain_map == {}, f"{model}: chain_map is not empty by default"
    assert isinstance(record.timing, dict) and isinstance(record.extras, dict)
    assert all(isinstance(r, str) and r for r in record.reasons), f"{model}: bad reasons"


def check_sample_and_seed_selection(model: str, workdir: Path) -> None:
    module = writer_module(model)
    tol = _tolerances(module)
    supports_seeds = getattr(module, "SUPPORTS_SEED_INDEX", True)
    cases = [(1, 1, 0.0), (1, 2, 10.0)] + ([(2, 1, 20.0)] if supports_seeds else [])
    for seed_index, sample, shift in cases:
        write_fixture(model, workdir, seed_index=seed_index, sample=sample, plddt_shift=shift)
    parser = get_parser(model)
    for seed_index, sample, shift in cases:
        record = parser.load(workdir, NAME, seed_index=seed_index, sample=sample)
        expected = synth.synthetic_complex(plddt_shift=shift).scalars["avg_plddt"]
        assert (record.seed_index, record.sample) == (seed_index, sample), (
            f"{model}: seed_index/sample not echoed for {(seed_index, sample)}"
        )
        assert abs(record.avg_plddt - expected) <= tol["plddt"], (
            f"{model}: seed_index={seed_index}, sample={sample} gave avg_plddt "
            f"{record.avg_plddt}, expected {expected}: the wrong files were read"
        )
    refs = parser.list_samples(workdir, NAME)
    assert [(r.seed_index, r.sample) for r in refs] == [(s, m) for s, m, _ in cases], (
        f"{model}: list_samples returned {[(r.seed_index, r.sample) for r in refs]}"
    )


def _assert_a_reason_and_no_data(record, model: str, what: str) -> None:
    assert np.isnan(record.avg_plddt), f"{model}: avg_plddt is not NaN for {what}"
    assert record.plddt_per_atom is None and record.pae is None and record.pde is None, (
        f"{model}: arrays are present for {what}"
    )
    assert record.reasons and all(isinstance(r, str) and r for r in record.reasons), (
        f"{model}: no reason given for {what}"
    )


def check_missing_files(model: str, workdir: Path) -> None:
    empty = workdir / "empty"
    empty.mkdir()
    for label, directory in (("an empty directory", empty), ("a missing directory", workdir / "x")):
        record = _load_without_raising(model, directory, label)
        assert record.structure_path is None, f"{model}: structure_path set for {label}"
        _assert_a_reason_and_no_data(record, model, label)

    write_fixture(model, workdir / "full")
    found = get_parser(model).find_files(workdir / "full", NAME).found()
    assert "structure" in found, f"{model}: find_files did not locate the structure"
    essential = [role for role in found if role not in ("structure", "timing")]
    assert essential, f"{model}: find_files located no scores or arrays file"
    for role in essential:
        directory = workdir / f"without_{role}"
        write_fixture(model, directory)
        get_parser(model).find_files(directory, NAME).found()[role].unlink()
        record = _load_without_raising(model, directory, f"a directory without the {role} file")
        assert record.reasons, f"{model}: no reason when the {role} file is missing"
        assert all(isinstance(r, str) and r for r in record.reasons)
        try:
            record.validate()
        except ValueError as exc:
            raise AssertionError(f"{model}: partial record invalid: {exc}") from exc

    directory = workdir / "without_structure"
    write_fixture(model, directory)
    get_parser(model).find_files(directory, NAME).found()["structure"].unlink()
    record = _load_without_raising(model, directory, "a directory without the structure file")
    assert record.structure_path is None, f"{model}: structure_path set for a missing file"
    with pytest.raises(ValueError, match="no structure file"):
        record.atoms()


def check_corrupt_files(model: str, workdir: Path) -> None:
    write_fixture(model, workdir / "probe")
    roles = [
        role
        for role in get_parser(model).find_files(workdir / "probe", NAME).found()
        if role not in ("structure", "timing")
    ]
    assert roles, f"{model}: find_files located no scores or arrays file"
    for role in roles:
        directory = workdir / f"corrupt_{role}"
        write_fixture(model, directory)
        get_parser(model).find_files(directory, NAME).found()[role].write_bytes(GARBAGE)
        try:
            _load(model, directory)
        except PROGRAMMING_ERRORS as exc:
            raise AssertionError(
                f"{model}: a corrupt {role} file raised {type(exc).__name__}, "
                f"which is a bug rather than a data error: {exc}"
            ) from exc
        except Exception:  # noqa: BLE001 - any data error satisfies the contract
            continue
        raise AssertionError(f"{model}: a corrupt {role} file was accepted without an error")


_NO_BIOTITE_SCRIPT = textwrap.dedent(
    """
    import json, math, sys
    sys.modules["biotite"] = None  # any import of biotite now fails
    from binding_metrics.predictors.registry import ParserSpec, get_parser, register_parser

    register_parser(ParserSpec(**json.loads(sys.argv[1])), replace=True)
    record = get_parser(sys.argv[2]).load(sys.argv[3], sys.argv[4])
    assert not math.isnan(record.avg_plddt), "avg_plddt is NaN: " + "; ".join(record.reasons)
    # a model whose pLDDT is per token or per residue needs the atoms to expand it: it may
    # leave the per-atom array out when the structure cannot be read, but it must say why
    assert record.plddt_per_atom is not None or record.reasons, "plddt_per_atom is None, no reason"
    assert sys.modules["biotite"] is None
    print("ok")
    """
)


def check_scalars_parse_without_biotite(model: str, workdir: Path) -> None:
    write_fixture(model, workdir)
    structure = get_parser(model).find_files(workdir, NAME).structure
    structure.write_text("# stub CIF\n", encoding="utf-8")  # proves the file is not opened
    spec = json.dumps(dataclasses.asdict(PARSERS[model]))
    paths = [str(REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths))
    done = subprocess.run(
        [sys.executable, "-c", _NO_BIOTITE_SCRIPT, spec, model, str(workdir), NAME],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
        env=env,
    )
    assert done.returncode == 0 and done.stdout.strip() == "ok", (
        f"{model}: loading with biotite blocked and a stub structure file failed:\n"
        f"{done.stderr[-1500:]}"
    )


def check_chain_map(model: str, workdir: Path) -> None:
    write_fixture(model, workdir)
    plain = _load(model, workdir).atoms()
    original = list(dict.fromkeys(str(c) for c in plain.chain_id))
    assert len(original) >= 2, f"{model}: the fixture has fewer than two chains"
    counts = {c: int((plain.chain_id == c).sum()) for c in original}

    renamed = {c: f"X{i}" for i, c in enumerate(original)}
    record = _load(model, workdir, chain_map=renamed)
    assert record.chain_map == renamed, f"{model}: chain_map not stored on the record"
    atoms = record.atoms()
    assert set(map(str, atoms.chain_id)) == set(renamed.values()), f"{model}: chains not renamed"
    for old, new in renamed.items():
        assert int((atoms.chain_id == new).sum()) == counts[old], f"{model}: atoms of {old} lost"
    record.validate(check_structure=True)

    partial = _load(model, workdir, chain_map={original[0]: "Q"}).atoms()
    assert set(map(str, partial.chain_id)) == {"Q", *original[1:]}, f"{model}: partial map"

    with pytest.raises(ValueError, match="same ID"):
        _load(model, workdir, chain_map={original[0]: "Z", original[1]: "Z"})


def check_completion(model: str, workdir: Path) -> None:
    write_fixture(model, workdir / "full")
    parser = get_parser(model)
    record = _load(model, workdir / "full")
    try:
        done = parser.complete(record)
        again = parser.complete(record)
    except Exception as exc:  # noqa: BLE001 - reported as the broken rule "complete never raises"
        raise AssertionError(
            f"{model}: complete raised {type(exc).__name__}: {exc}; a problem of the data is "
            "a sentence in record.reasons"
        ) from exc
    assert done is record and again is record, f"{model}: complete must return the same record"
    try:
        record.validate(check_structure=True)
    except ValueError as exc:
        raise AssertionError(f"{model}: record invalid after complete: {exc}") from exc
    reasons = list(record.reasons)
    parser.complete(record)
    assert record.reasons == reasons, f"{model}: complete is not idempotent (reasons grew)"

    # chain-keyed values are keyed by chain IDs of the structure file (before chain_map)
    chains = set(map(str, record.atoms().chain_id))
    keys = [*record.chain_ptm, *(p for k in record.chain_pair_iptm for p in re.findall(r"\w+", k))]
    stray = sorted(set(keys) - chains)
    assert not stray, (
        f"{model}: chain_ptm and chain_pair_iptm name {stray}, which are not chains of the "
        f"structure file {sorted(chains)}"
    )

    # a directory without output and a record without a structure are returned as they are
    empty = _load_without_raising(model, workdir / "nothing", "a missing directory")
    try:
        assert parser.complete(empty) is empty
    except Exception as exc:  # noqa: BLE001 - reported as the broken rule
        raise AssertionError(f"{model}: complete failed on a record without files: {exc}") from exc


CHECKS = [
    check_class_attributes,
    check_capabilities_declaration,
    check_load_valid_record,
    check_sample_and_seed_selection,
    check_missing_files,
    check_corrupt_files,
    check_scalars_parse_without_biotite,
    check_chain_map,
    check_completion,
]
