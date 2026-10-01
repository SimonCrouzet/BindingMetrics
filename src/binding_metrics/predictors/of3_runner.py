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
        supports_custom_weights = True; weights_kind = "file"
        supported_modes = frozenset({"score", "refold"}); default_mode = "score"
        .conda_env
        .make_request(input_path, *, name: str, binder_chain: Optional[str] = None,
            receptor_chain: Optional[str] = None, mode: str = "score",
            seeds: Optional[Sequence[int]] = None, num_samples: int = 5,
            presets: Optional[Sequence[str]] = None, use_msa_server: bool = True,
            num_model_seeds: Optional[int] = None, on_unmappable_residue: str = "error",
            extra_args: Sequence[str] = (), inference_ckpt_path: Optional[str | Path] = None,
            runner_yaml: Optional[str | Path] = None,
            template_cif_path: Optional[str | Path] = None,
            binder_cyclic: bool | str = "auto",
            weights: Optional[str | Path | WeightsRef] = None) -> PredictionRequest
        .prepare(request, work_dir) -> Path
        .run(request, work_dir) -> Path                       # <work_dir>/predictions
        .supports_batch(request) -> bool
        .run_many(requests, work_dir) -> dict[str, Path | BaseException]
        .is_available() -> bool
        .version() -> Optional[str]

Requests. ``make_request`` builds the ``PredictionRequest`` of an OpenFold3 run with every
setting that changes the output written out (the defaults included), so two callers that mean the
same run get the same key. ``mode`` is ``"score"`` (``run_openfold_scoring``: the complex file
with each chain given its own structure as a template), ``"refold"`` (``run_openfold_refolding``:
the binder from its sequence beside a receptor given as template) or ``"predict"``
(``input_path`` is a ready OpenFold3 query file for ``run_openfold``; files that the query names
are not hashed). What goes into the key:

* the OpenFold3 version (``version()``; empty when it cannot be told), the number of samples per
  seed, the chain roles and the content hash of the input file;
* ``options``: ``presets`` (``["predict", "low_mem"]`` by default, ``predict`` added when
  missing; None when ``runner_yaml`` replaces them), ``use_msa_server``, ``num_model_seeds``,
  ``on_unmappable_residue``, ``binder_cyclic`` (``"auto"``, true or false; None for ``predict``,
  whose query file names its own chains), ``extra_args`` and ``inference_ckpt_path`` with the
  size of that file. ``"auto"`` writes ``cyclic: true`` on a head-to-tail binder when OpenFold3 is
  0.4.5 or later, which the structure (hashed) and the version (in the key) decide;
* the seeds that OpenFold3 samples with. ``seeds`` is the explicit list, ``[42]`` when the caller
  gives none, and empty when ``num_model_seeds`` asks OpenFold3 to generate them (the count is
  then in ``options``; None there means no generation). ``seeds`` and ``num_model_seeds`` cannot
  both be given, because OpenFold3 lets the generated seeds replace the explicit ones. A run
  writes the seeds to the runner YAML (``experiment_settings.seeds``): OpenFold3 0.5.0 does not
  read seeds from the query file;
* the custom weights, when given (``weights``: a checkpoint file, passed to OpenFold3 as
  ``--inference-ckpt-path``): the SHA-256 and size of the file, never its path, so a fine-tuned
  checkpoint has its own entries and a moved copy shares them. Without ``weights`` the key is
  what it was. ``inference_ckpt_path`` is the older way to name a checkpoint: it keeps its path
  and size in ``options`` and the two cannot be combined;
* the content of ``template_cif_path`` and ``runner_yaml`` when given, and of the user-default
  ``runner.yml`` that OpenFold3 merges under the toolkit's YAML (it can carry the MSA server URL
  and the structure format), when the file exists.

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
own folder (``<work_dir>/split/<key>``, with ``experiment_config.json`` and, when the run wrote
them, ``inference_query_set.json`` and ``template_accounting.json``), returns an exception for a
query that has no output, and leaves the shared query files behind.

OpenFold3 licence: Apache 2.0; no weights or model code are read or shipped here.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.store import PredictionRequest
from binding_metrics.predictors.weights import WeightsRef

#: Defaults of the openfold run functions that are not exposed as module constants. A test
#: compares them with the signatures, so a change there cannot pass unnoticed.
_DEFAULT_NUM_SAMPLES = 5
_DEFAULT_NUM_MODEL_SEEDS = None
_DEFAULT_USE_MSA_SERVER = True
_DEFAULT_ON_UNMAPPABLE = "error"
_DEFAULT_BINDER_CYCLIC = "auto"

_ON_UNMAPPABLE_CHOICES = ("error", "x")

#: Files below ``predictions/`` that belong to the run and not to one query; ``run_many`` copies
#: them next to the output of each query it splits off.
_RUN_FILES = ("experiment_config.json", "inference_query_set.json", "template_accounting.json")


def _lock_message() -> str:
    """Why OpenFold3 cannot run mode ``score-lock``: the reason its capabilities declare.

    The pre-flight check refuses such a request first; this is the stop for a caller that did
    not run it.
    """
    from binding_metrics.predictors.of3 import OpenFold3Parser

    return "OpenFold3 cannot run mode 'score-lock'. " + OpenFold3Parser.capabilities.reason_for(
        "modes", "score-lock"
    )


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
    #: OpenFold3 loads a checkpoint file given with ``--inference-ckpt-path``.
    supports_custom_weights = True
    weights_kind = "file"
    #: For a complex structure: ``score`` and ``refold``. Mode ``predict`` takes a query file that
    #: the caller wrote, and ``score-lock`` cannot be run (see ``make_request``). ``score`` is the
    #: default of the OpenFold3 step (``--openfold-mode``): every chain is given its own structure.
    supported_modes = frozenset({"score", "refold"})
    default_mode = "score"

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
        num_model_seeds: Optional[int] = _DEFAULT_NUM_MODEL_SEEDS,
        on_unmappable_residue: str = _DEFAULT_ON_UNMAPPABLE,
        extra_args: Sequence[str] = (),
        inference_ckpt_path: Optional[str | Path] = None,
        runner_yaml: Optional[str | Path] = None,
        template_cif_path: Optional[str | Path] = None,
        binder_cyclic: Union[bool, str] = _DEFAULT_BINDER_CYCLIC,
        weights: Optional[str | Path | WeightsRef] = None,
    ) -> PredictionRequest:
        """The store request of one OpenFold3 run (see the module docstring for what it holds).

        Args:
            input_path: The complex structure (``score``, ``refold``) or the query file
                (``predict``).
            name: Query name; the output files are named after it.
            binder_chain, receptor_chain: Chain roles; required for ``score`` and ``refold``.
            mode: ``"score"``, ``"refold"`` or ``"predict"``; ``"score-lock"`` raises, OpenFold3
                cannot pin the pose of the chains (``OpenFold3Parser.capabilities``).
            seeds: Seed values OpenFold3 samples with; None takes the toolkit's default
                (``[42]``), or none when ``num_model_seeds`` is given. Cannot be combined with
                ``num_model_seeds``.
            num_samples: Structures per seed (``--num_diffusion_samples``).
            presets: Model presets; None takes the toolkit's default. Ignored with
                ``runner_yaml``.
            use_msa_server: Use the ColabFold MSA server (sequences leave the machine).
            num_model_seeds: ``--num_model_seeds``, which makes OpenFold3 generate that many
                seeds; None (default) leaves it out.
            on_unmappable_residue: ``"error"`` or ``"x"`` (see ``run_openfold_scoring``).
            extra_args: Extra command-line arguments, passed verbatim.
            inference_ckpt_path: Checkpoint file; None uses OpenFold3's default. The key holds
                its path and size; ``weights`` identifies a checkpoint by content instead.
            runner_yaml: Runner YAML that replaces ``presets``.
            template_cif_path: Pre-prepared complex or receptor template (``score``, ``refold``).
            binder_cyclic: ``"auto"`` (default), ``True`` or ``False``; whether the binder chain
                of the query gets ``cyclic: true`` (see ``prepare_refolding_query``). Only for
                ``score`` and ``refold``: a query file of ``predict`` names its own chains.
            weights: A custom (fine-tuned) checkpoint file, or the ``WeightsRef`` that
                ``PredictionStore.weights_reference`` made for it. It goes to OpenFold3 as
                ``--inference-ckpt-path`` and its content is in the key. A path is hashed here
                without a cache; give a ``WeightsRef`` to use the store's.

        Raises:
            ValueError: Mode ``score-lock``, a missing chain role, an unknown mode or choice, a
                template file with ``predict``, both ``seeds`` and ``num_model_seeds``, a
                ``binder_cyclic`` that is not ``True``, ``False`` or ``"auto"``, or a value other
                than the default with ``predict``, or both ``weights`` and
                ``inference_ckpt_path``.
            FileNotFoundError: ``weights`` does not exist.
        """
        if weights is not None and inference_ckpt_path is not None:
            raise ValueError("give weights or inference_ckpt_path, not both")
        if mode == "score-lock":
            raise ValueError(_lock_message())
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
        run_module._check_binder_cyclic(binder_cyclic)
        if mode == "predict" and binder_cyclic != _DEFAULT_BINDER_CYCLIC:
            raise ValueError(
                "a query file for mode 'predict' names its own chains and cyclic flags"
            )
        seed_values, generated_seeds = run_module._resolve_run_seeds(
            seeds, num_model_seeds, extra_args
        )
        if seed_values is not None:
            request_seeds: tuple[int, ...] = tuple(seed_values)
        elif generated_seeds is not None:
            request_seeds = ()  # OpenFold3 generates them; the count is in the options
        else:
            request_seeds = tuple(run_module._DEFAULT_QUERY_SEEDS)
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
            weights=weights,
            seeds=request_seeds,
            num_samples=num_samples,
            model_version=self.version() or "",
            options={
                "presets": presets_option,
                "use_msa_server": bool(use_msa_server),
                "num_model_seeds": generated_seeds,
                "on_unmappable_residue": on_unmappable_residue,
                "binder_cyclic": None if mode == "predict" else binder_cyclic,
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
        # the environment that runs OpenFold3 is asked for its version when the flag is decided
        environment = {} if self.conda_env is None else {"conda_env": self.conda_env}
        return Path(
            function(
                **self._structure_arguments(request, Path(work_dir) / "query"),
                **self._query_arguments(request),
                **environment,
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
                results[request.key()] = RuntimeError(_no_output_message(request, failures))
                continue
            target = work_dir / "split" / request.key()
            shutil.copytree(predictions / request.name, target / request.name)
            # the files of the run that name the checkpoint and say what became of the templates
            # (each lists every query of the batch; a reader picks its own)
            for shared in _RUN_FILES:
                source = predictions / shared
                if source.is_file():
                    shutil.copy2(source, target / shared)
            results[request.key()] = target
        return results

    # ------------------------------------------------------------------ arguments

    def _check_request(self, request: PredictionRequest) -> None:
        if request.mode == "score-lock":
            raise ValueError(_lock_message())
        if request.model != self.name:
            raise ValueError(f"the of3 runner cannot run a '{request.model}' request")
        if request.input_path is None:
            raise ValueError("an OpenFold3 request needs an input file")
        self.check_weights(request)

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
        residues = request.options.get("on_unmappable_residue", _DEFAULT_ON_UNMAPPABLE)
        if residues != _DEFAULT_ON_UNMAPPABLE:
            arguments["on_unmappable_residue"] = residues
        binder_cyclic = request.options.get("binder_cyclic", _DEFAULT_BINDER_CYCLIC)
        if request.mode != "predict" and binder_cyclic not in (None, _DEFAULT_BINDER_CYCLIC):
            arguments["binder_cyclic"] = binder_cyclic
        return arguments

    def _run_arguments(self, request: PredictionRequest) -> dict[str, Any]:
        """The arguments of ``run_openfold`` that differ from its defaults."""
        options = request.options
        arguments: dict[str, Any] = {}
        if self.conda_env is not None:
            arguments["conda_env"] = self.conda_env
        if request.num_samples != _DEFAULT_NUM_SAMPLES:
            arguments["num_diffusion_samples"] = request.num_samples
        seeds = tuple(request.seeds)
        if seeds and seeds != tuple(_run_module()._DEFAULT_QUERY_SEEDS):
            arguments["seeds"] = seeds
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
        if request.weights is not None:
            arguments["inference_ckpt_path"] = str(request.weights.path)
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
        raise RuntimeError(_no_output_message(request, failures))


def _no_output_message(request: PredictionRequest, failures: dict[str, str]) -> str:
    """The reason of a query without output; it holds no path of the work directory.

    The store renames the temporary work directory when it keeps the run, so a path would point
    at nothing in the recorded reason: the folder is named relative to the stored entry.
    """
    message = (
        f"OpenFold3 wrote no output for query '{request.name}' in the predictions folder of the "
        "run (outputs/predictions of a stored entry)"
    )
    reason = failures.get(request.name)
    return f"{message}: {reason}" if reason else message
