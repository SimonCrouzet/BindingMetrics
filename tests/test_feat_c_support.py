"""Helpers of the ``test_feat_c_*`` modules: a stand-in for the OpenFold3 run and its output.

No model runs. ``StubOpenFold`` replaces the three run functions of
``binding_metrics.metrics.openfold`` (the ``OpenFold3Runner`` calls them by name when a run
starts) and writes a synthetic OpenFold3 output for the input it is given, so the whole
step (store, session, adapter, metrics) runs for real and only the model is missing. The
synthetic prediction reproduces the coordinates of the input, which makes the EvoBind
adversarial check compare a structure with itself.
"""

from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics import openfold
from binding_metrics.metrics._common import load_structure
from binding_metrics.predictors.of3_runner import OpenFold3Runner
from tests.predictors import synth, synth_of3

biotite_structure = pytest.importorskip("biotite.structure")

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
PEPTIDE_CHAIN, RECEPTOR_CHAIN = "B", "A"  # 1YCR: p53 peptide (13 residues) and MDM2 (85)


def complex_from(input_path, plddt_low: float = 60.0, plddt_high: float = 95.0):
    """A ``SyntheticComplex`` with the atoms of ``input_path`` and made-up confidences.

    pLDDT rises linearly from ``plddt_low`` to ``plddt_high`` along the atoms; PAE and PDE
    are asymmetric matrices with one token per residue.
    """
    atoms = load_structure(input_path)
    atoms = atoms[biotite_structure.filter_amino_acids(atoms)]
    plddt = np.linspace(plddt_low, plddt_high, atoms.array_length())
    atoms.set_annotation("b_factor", plddt.copy())
    n_tokens = len(set(zip(atoms.chain_id.tolist(), atoms.res_id.tolist())))
    i = np.arange(n_tokens)[:, None]
    j = np.arange(n_tokens)[None, :]
    # OpenFold3 names the chains of its confidences after the chains of the query, which are the
    # chains of the input (a real run of 1CWA: "(A, C)"), so key them by the chains of the atoms
    chains = list(dict.fromkeys(atoms.chain_id.tolist()))
    first, second = chains[0], (chains[1] if len(chains) > 1 else "B")
    return synth.SyntheticComplex(
        atoms=atoms,
        plddt_per_atom=plddt,
        pae=1.0 + 0.05 * i + 0.02 * j,
        pde=0.5 + 0.03 * i + 0.01 * j,
        scalars={
            "avg_plddt": float(plddt.mean()),
            "ptm": 0.88,
            "iptm": 0.76,
            "gpde": 1.23,
            "ranking_score": 0.82,
            "has_clash": 0.0,
            "disorder": 0.12,
        },
        chain_ptm={first: 0.88, second: 0.80},
        chain_pair_iptm={f"{first}-{second}": 0.76, f"{second}-{first}": 0.74},
    )


def write_of3_output(directory, name, input_path, **kwargs):
    """Write the synthetic OpenFold3 output of ``name`` under ``directory``; return it."""
    synth_of3.write_prediction(Path(directory), name, complex_from(input_path, **kwargs))
    return Path(directory)


class StubOpenFold:
    """Replaces the OpenFold3 run functions and records each call.

    Attributes:
        calls: One dict per model start: ``kind`` (``scoring``, ``refolding`` or ``batched``),
            ``names`` (the queries it predicted) and ``kwargs``.
        error: When set, every start raises it instead of writing output.
    """

    def __init__(self, monkeypatch, *, error=None):
        self.calls: list[dict] = []
        self.error = error
        monkeypatch.setattr(OpenFold3Runner, "version", lambda runner: "0.5.0")
        monkeypatch.setattr(OpenFold3Runner, "is_available", lambda runner: True)
        monkeypatch.setattr(openfold, "run_openfold_scoring", self._single("scoring"))
        monkeypatch.setattr(openfold, "run_openfold_refolding", self._single("refolding"))
        monkeypatch.setattr(openfold, "run_openfold_batched", self._batched)

    @property
    def starts(self) -> int:
        """How many times the model was started."""
        return len(self.calls)

    @property
    def predicted(self) -> list[str]:
        """The query names of every prediction, in order."""
        return [name for call in self.calls for name in call["names"]]

    def _single(self, kind):
        def run(**kwargs):
            self.calls.append({"kind": kind, "names": [kwargs["query_name"]], "kwargs": kwargs})
            if self.error is not None:
                raise self.error
            predictions = Path(kwargs["output_dir"]) / "predictions"
            write_of3_output(predictions, kwargs["query_name"], kwargs["complex_structure_path"])
            return predictions

        return run

    def _batched(self, *, samples, output_dir, **kwargs):
        names = [sample.query_name for sample in samples]
        self.calls.append({"kind": "batched", "names": names, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        predictions = Path(output_dir) / "predictions"
        for sample in samples:
            write_of3_output(predictions, sample.query_name, sample.complex_structure_path)
        return predictions
