"""OpenFold3 as a prediction runner for the prediction store.

``OpenFold3Runner`` starts OpenFold3 through the run functions of
``binding_metrics.metrics.openfold`` (``run_openfold_scoring``, ``run_openfold_refolding``,
``run_openfold_batched`` and ``run_openfold``). It knows nothing about OpenFold3 that those
functions do not already encode; it adds the request, the store layout and the check that a run
wrote its output.

Public names and signatures (the module imports the standard library and the runner ABC; the
openfold module is imported when a run starts, never before)::

    class OpenFold3Runner(PredictionRunner):
        OpenFold3Runner(conda_env: Optional[str] = None)
        name = "of3"; capabilities = None
        .conda_env
        .make_request(input_path, *, name: str, binder_chain: Optional[str] = None,
            receptor_chain: Optional[str] = None, mode: str = "score",
            seeds: Optional[Sequence[int]] = None, num_samples: int = 5,
            presets: Optional[Sequence[str]] = None, use_msa_server: bool = True,
            num_model_seeds: int = 1, on_unmappable_residue: str = "error",
            extra_args: Sequence[str] = (), inference_ckpt_path: Optional[str | Path] = None,
            runner_yaml: Optional[str | Path] = None,
            template_cif_path: Optional[str | Path] = None) -> PredictionRequest
        .prepare(request, work_dir) -> Path
        .run(request, work_dir) -> Path                       # <work_dir>/predictions
        .supports_batch(request) -> bool
        .run_many(requests, work_dir) -> dict[str, Path | BaseException]
        .is_available() -> bool
        .version() -> Optional[str]

Requests. ``make_request`` builds the ``PredictionRequest`` of an OpenFold3 run with every
setting that changes the output written out (the defaults included), so two callers that mean the
same run get the same key. ``mode`` is ``"score"`` (``run_openfold_scoring``: the complex file
with both chains as templates), ``"refold"`` (``run_openfold_refolding``: the binder from its
sequence beside a templated receptor) or ``"predict"`` (``input_path`` is a ready OpenFold3 query
file for ``run_openfold``; files that the query names are not hashed). What goes into the key:

* the OpenFold3 version (``version()``; empty when it cannot be told), the seeds, the number of
  samples per seed, the chain roles and the content hash of the input file;
* ``options``: ``presets`` (``["predict", "low_mem"]`` by default, ``predict`` added when
  missing; None when ``runner_yaml`` replaces them), ``use_msa_server``, ``num_model_seeds``,
  ``on_unmappable_residue``, ``extra_args`` and ``inference_ckpt_path`` with the size of that file;
* the content of ``template_cif_path`` and ``runner_yaml`` when given, and of the user-default
  ``runner.yml`` that OpenFold3 merges under the toolkit's YAML (it can carry seeds, the MSA server
  URL and the structure format), when the file exists.

Not in the key: the conda environment (the version is), where the input file lives, and what an
MSA server returns (results from a remote server can change over time; ``use_msa_server`` is
recorded, the alignments are not).

A run calls the openfold functions with keyword arguments and leaves out every argument that has
its default value (as the pipeline does today), so the call is the same as a hand-written one.
The functions are looked up on ``binding_metrics.metrics.openfold`` when the run starts, so a test
that patches them is honoured. ``run`` returns ``<work_dir>/predictions`` and raises when the
directory holds no output of the query, so a run that OpenFold3 skipped is never stored as done;
the exception text of a failed run (``OpenFoldRunError``, ``OpenFoldQueryError``,
``UnmappableResidueError``, ``FileNotFoundError``) is what the store records as the reason.

Batches. Requests of mode ``score`` or ``refold`` without a template file can be predicted in one
``run_openfold_batched`` call (one model load). ``run_many`` copies each query's output into its
own folder (``<work_dir>/split/<key>``, with ``experiment_config.json``), returns an exception for
a query that has no output, and leaves the shared query files behind.

OpenFold3 licence: Apache 2.0; no weights or model code are read or shipped here.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.store import PredictionRequest

#: Defaults of the openfold run functions that are not exposed as module constants. A test
#: compares them with the signatures, so a change there cannot pass unnoticed.
_DEFAULT_NUM_SAMPLES = 5
_DEFAULT_NUM_MODEL_SEEDS = 1
_DEFAULT_USE_MSA_SERVER = True
_DEFAULT_ON_UNMAPPABLE = "error"

_ON_UNMAPPABLE_CHOICES = ("error", "x")


def _openfold() -> Any:
    """The module with the run functions; imported here so that the package stays light."""
    return importlib.import_module("binding_metrics.metrics.openfold")


def _run_module() -> Any:
    """The module with the defaults, the version probe and the failure readers."""
    return importlib.import_module("binding_metrics.metrics._openfold_run")


class OpenFold3Runner(PredictionRunner):
    """Runs ``run_openfold predict`` in the current environment or in a conda environment.

    Args:
        conda_env: Name of the conda environment that has OpenFold3 (``conda run -n <env>``);
            None uses the ``run_openfold`` on PATH.
    """

    name = "of3"

    def __init__(self, conda_env: Optional[str] = None):
        self.conda_env = conda_env
        self._version: Optional[str] = None
        self._version_probed = False

    # ------------------------------------------------------------------ the machine

    def version(self) -> Optional[str]:
        """The installed ``openfold3`` version, or None when it cannot be told.

        Read from the package metadata (in the conda environment when one is set, which starts a
        process); asked once per runner.
        """
        if not self._version_probed:
            python_cmd = (
                None if self.conda_env is None else ["conda", "run", "-n", self.conda_env, "python"]
            )
            self._version = _run_module().installed_openfold3_version(python_cmd)
            self._version_probed = True
        return self._version

    def is_available(self) -> bool:
        """True when ``run_openfold`` is on PATH, or the conda environment has ``openfold3``."""
        if self.conda_env is None:
            return shutil.which("run_openfold") is not None
        return self.version() is not None

    # ------------------------------------------------------------------ the request

    def make_request(
        self,
        input_path: str | Path,
        *,
        name: str,
        binder_chain: Optional[str] = None,
        receptor_chain: Optional[str] = None,
        mode: str = "score",
        seeds: Optional[Sequence[int]] = None,
        num_samples: int = _DEFAULT_NUM_SAMPLES,
        presets: Optional[Sequence[str]] = None,
        use_msa_server: bool = _DEFAULT_USE_MSA_SERVER,
        num_model_seeds: int = _DEFAULT_NUM_MODEL_SEEDS,
        on_unmappable_residue: str = _DEFAULT_ON_UNMAPPABLE,
        extra_args: Sequence[str] = (),
        inference_ckpt_path: Optional[str | Path] = None,
        runner_yaml: Optional[str | Path] = None,
        template_cif_path: Optional[str | Path] = None,
    ) -> PredictionRequest:
        """The store request of one OpenFold3 run (see the module docstring for what it holds).

        Args:
            input_path: The complex structure (``score``, ``refold``) or the query file
                (``predict``).
            name: Query name; the output files are named after it.
            binder_chain, receptor_chain: Chain roles; required for ``score`` and ``refold``.
            mode: ``"score"``, ``"refold"`` or ``"predict"``.
            seeds: Seed values of the query; None takes the toolkit's default.
            num_samples: Structures per seed (``--num_diffusion_samples``).
            presets: Model presets; None takes the toolkit's default. Ignored with
                ``runner_yaml``.
            use_msa_server: Use the ColabFold MSA server (sequences leave the machine).
            num_model_seeds: ``--num_model_seeds``.
            on_unmappable_residue: ``"error"`` or ``"x"`` (see ``run_openfold_scoring``).
            extra_args: Extra command-line arguments, passed verbatim.
            inference_ckpt_path: Checkpoint file; None uses OpenFold3's default.
            runner_yaml: Runner YAML that replaces ``presets``.
            template_cif_path: Pre-prepared complex or receptor template (``score``, ``refold``).

        Raises:
            ValueError: A missing chain role, an unknown mode or choice, or a template file with
                ``predict``.
        """
        if mode in ("score", "refold") and not (binder_chain and receptor_chain):
            raise ValueError(f"mode '{mode}' needs binder_chain and receptor_chain")
        if mode == "predict" and template_cif_path is not None:
            raise ValueError("a query file for mode 'predict' names its own templates")
        if on_unmappable_residue not in _ON_UNMAPPABLE_CHOICES:
            raise ValueError(
                f"on_unmappable_residue must be one of {_ON_UNMAPPABLE_CHOICES}, "
                f"got {on_unmappable_residue!r}"
            )
        run_module = _run_module()
        if runner_yaml is not None:
            presets_option = None
        else:
            presets_option = list(
                presets if presets is not None else run_module._DEFAULT_MODEL_PRESETS
            )
            if "predict" not in presets_option:
                presets_option.insert(0, "predict")

        checkpoint = None if inference_ckpt_path is None else Path(inference_ckpt_path)
        checkpoint_size = None
        if checkpoint is not None and checkpoint.is_file():
            checkpoint_size = checkpoint.stat().st_size
        extra_files: dict[str, Path] = {}
        if template_cif_path is not None:
            extra_files["template_cif"] = Path(template_cif_path)
        if runner_yaml is not None:
            extra_files["runner_yaml"] = Path(runner_yaml)
        user_default_yaml = run_module._user_default_runner_yaml()
        if user_default_yaml is not None:
            extra_files["user_default_runner_yaml"] = user_default_yaml

        return PredictionRequest(
            self.name,
            name,
            mode=mode,
            input_path=input_path,
            binder_chain=binder_chain,
            receptor_chain=receptor_chain,
            extra_files=extra_files,
            seeds=run_module._DEFAULT_QUERY_SEEDS if seeds is None else seeds,
            num_samples=num_samples,
            model_version=self.version() or "",
            options={
                "presets": presets_option,
                "use_msa_server": bool(use_msa_server),
                "num_model_seeds": int(num_model_seeds),
                "on_unmappable_residue": on_unmappable_residue,
                "extra_args": [str(argument) for argument in extra_args],
                "inference_ckpt_path": None if checkpoint is None else str(checkpoint),
                "inference_ckpt_size_bytes": checkpoint_size,
            },
        )

    # ------------------------------------------------------------------ running

    def prepare(self, request: PredictionRequest, work_dir: Path) -> Path:
        """Write the query JSON, templates and alignments below ``<work_dir>/query``.

        Returns the query file. Raises what ``run`` would raise about the input (an unmappable
        residue, a chain the structure lacks) without starting OpenFold3. In ``predict`` mode
        the request's own query file is returned and nothing is written.
        """
        self._check_request(request)
        if request.mode == "predict":
            return Path(request.input_path)
        openfold = _openfold()
        function = (
            openfold.prepare_scoring_query
            if request.mode == "score"
            else openfold.prepare_refolding_query
        )
        return Path(
            function(
                **self._structure_arguments(request, Path(work_dir) / "query"),
                **self._query_arguments(request),
            )
        )

    def run(self, request: PredictionRequest, work_dir: Path) -> Path:
        """Run OpenFold3 for ``request`` and return ``<work_dir>/predictions``.

        Raises:
            OpenFoldRunError: OpenFold3 exited non-zero.
            OpenFoldQueryError: OpenFold3 exited normally but every query failed.
            UnmappableResidueError: A residue cannot be sent (see ``on_unmappable_residue``).
            FileNotFoundError: ``run_openfold`` is not on PATH and no conda environment is set.
            RuntimeError: OpenFold3 finished but the directory has no output of the query.
        """
        self._check_request(request)
        openfold = _openfold()
        work_dir = Path(work_dir)
        if request.mode == "predict":
            predictions = openfold.run_openfold(
                query_json=request.input_path,
                output_dir=work_dir / "predictions",
                **self._run_arguments(request),
            )
        else:
            function = (
                openfold.run_openfold_scoring
                if request.mode == "score"
                else openfold.run_openfold_refolding
            )
            predictions = function(
                **self._structure_arguments(request, work_dir),
                **self._run_arguments(request),
                **self._query_arguments(request),
            )
        predictions = Path(predictions)
        self._require_output(request, predictions)
        return predictions

    def supports_batch(self, request: PredictionRequest) -> bool:
        """True for ``score`` and ``refold`` requests without a template file."""
        return request.mode in ("score", "refold") and "template_cif" not in request.extra_files

    def run_many(
        self, requests: Sequence[PredictionRequest], work_dir: Path
    ) -> dict[str, Union[Path, BaseException]]:
        """Predict several requests in one ``run_openfold_batched`` call.

        The requests must share their batch signature and have distinct names. Returns, for each
        request key, the folder that holds that query's output, or a ``RuntimeError`` (with the
        reason OpenFold3 logged, when it did) for a query that has no output.

        Raises:
            ValueError: The requests cannot share a batch.
            OpenFoldRunError, OpenFoldQueryError, UnmappableResidueError: As for ``run``; the
                batch fails as a whole.
        """
        from binding_metrics.predictors.of3 import OpenFold3Parser

        requests = list(requests)
        if not requests:
            return {}
        for request in requests:
            self._check_request(request)
            if not self.supports_batch(request):
                raise ValueError(f"the request for '{request.name}' cannot be batched")
        if len({request.batch_signature() for request in requests}) != 1:
            raise ValueError(
                "requests that differ in mode, seeds, options or files cannot share a batch"
            )
        names = [request.name for request in requests]
        if len(set(names)) != len(names):
            raise ValueError(f"the names of a batch must be distinct, got {names}")

        openfold = _openfold()
        work_dir = Path(work_dir)
        first = requests[0]
        samples = [
            openfold._BatchSample(
                query_name=request.name,
                complex_structure_path=request.input_path,
                receptor_chain=request.receptor_chain,
                binder_chain=request.binder_chain,
            )
            for request in requests
        ]
        predictions = Path(
            openfold.run_openfold_batched(
                samples=samples,
                output_dir=work_dir,
                mode=first.mode,
                **self._run_arguments(first),
                **self._query_arguments(first),
            )
        )

        parser = OpenFold3Parser()
        failures = _run_module()._failed_query_reasons(predictions)
        results: dict[str, Union[Path, BaseException]] = {}
        for request in requests:
            if not parser.find_files(predictions, request.name).has_output():
                results[request.key()] = RuntimeError(
                    _no_output_message(request, predictions, failures)
                )
                continue
            target = work_dir / "split" / request.key()
            shutil.copytree(predictions / request.name, target / request.name)
            config = predictions / "experiment_config.json"
            if config.is_file():
                shutil.copy2(config, target / config.name)  # names the checkpoint of the run
            results[request.key()] = target
        return results

    # ------------------------------------------------------------------ arguments

    def _check_request(self, request: PredictionRequest) -> None:
        if request.model != self.name:
            raise ValueError(f"the of3 runner cannot run a '{request.model}' request")
        if request.input_path is None:
            raise ValueError("an OpenFold3 request needs an input file")

    @staticmethod
    def _structure_arguments(request: PredictionRequest, output_dir: Path) -> dict[str, Any]:
        """Arguments shared by ``prepare_*_query`` and ``run_openfold_scoring/refolding``."""
        arguments: dict[str, Any] = {
            "complex_structure_path": request.input_path,
            "receptor_chain": request.receptor_chain,
            "binder_chain": request.binder_chain,
            "query_name": request.name,
            "output_dir": output_dir,
        }
        template = request.extra_files.get("template_cif")
        if template is not None:
            arguments["template_cif_path"] = template
        return arguments

    @staticmethod
    def _query_arguments(request: PredictionRequest) -> dict[str, Any]:
        """The arguments that go into the query file, when they differ from the defaults."""
        arguments: dict[str, Any] = {}
        if tuple(request.seeds) != tuple(_run_module()._DEFAULT_QUERY_SEEDS):
            arguments["seeds"] = tuple(request.seeds)
        residues = request.options.get("on_unmappable_residue", _DEFAULT_ON_UNMAPPABLE)
        if residues != _DEFAULT_ON_UNMAPPABLE:
            arguments["on_unmappable_residue"] = residues
        return arguments

    def _run_arguments(self, request: PredictionRequest) -> dict[str, Any]:
        """The arguments of ``run_openfold`` that differ from its defaults."""
        options = request.options
        arguments: dict[str, Any] = {}
        if self.conda_env is not None:
            arguments["conda_env"] = self.conda_env
        if request.num_samples != _DEFAULT_NUM_SAMPLES:
            arguments["num_diffusion_samples"] = request.num_samples
        if options.get("num_model_seeds", _DEFAULT_NUM_MODEL_SEEDS) != _DEFAULT_NUM_MODEL_SEEDS:
            arguments["num_model_seeds"] = int(options["num_model_seeds"])
        if options.get("use_msa_server", _DEFAULT_USE_MSA_SERVER) != _DEFAULT_USE_MSA_SERVER:
            arguments["use_msa_server"] = bool(options["use_msa_server"])
        presets = options.get("presets")
        if presets is not None and list(presets) != list(_run_module()._DEFAULT_MODEL_PRESETS):
            arguments["model_presets"] = list(presets)
        if options.get("extra_args"):
            arguments["extra_args"] = list(options["extra_args"])
        if options.get("inference_ckpt_path"):
            arguments["inference_ckpt_path"] = options["inference_ckpt_path"]
        if "runner_yaml" in request.extra_files:
            arguments["runner_yaml"] = request.extra_files["runner_yaml"]
        return arguments

    @staticmethod
    def _require_output(request: PredictionRequest, predictions: Path) -> None:
        """Raise when ``predictions`` holds no output of the query (see the module docstring)."""
        from binding_metrics.predictors.of3 import OpenFold3Parser

        if OpenFold3Parser().find_files(predictions, request.name).has_output():
            return
        failures = _run_module()._failed_query_reasons(predictions)
        raise RuntimeError(_no_output_message(request, predictions, failures))


def _no_output_message(
    request: PredictionRequest, predictions: Path, failures: dict[str, str]
) -> str:
    message = f"OpenFold3 wrote no output for query '{request.name}' in {predictions}"
    reason = failures.get(request.name)
    return f"{message}: {reason}" if reason else message
