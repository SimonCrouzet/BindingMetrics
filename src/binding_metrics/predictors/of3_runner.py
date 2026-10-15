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
        seed_dimension = True                    # one request carries all the seeds
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
            weights: Optional[str | Path | WeightsRef] = None,
            template_mode: str = "structure", msa_policy: str = "default",
            receptor_msa: Optional[str | Path] = None) -> PredictionRequest
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
  whose query file names its own chains), ``template_mode`` (``"structure"``, the default, or
  ``"alignment"``, how the templates reach OpenFold3; None for ``predict``),
  ``query_builder_version`` (the version of the query builders,
  ``binding_metrics.metrics.openfold.QUERY_BUILDER_VERSION``; None for ``predict``: an entry made
  by an older builder is not reused), ``extra_args`` and
  ``inference_ckpt_path`` with the
  size of that file. ``"auto"`` writes ``cyclic: true`` on a head-to-tail binder of standard
  residues when OpenFold3 is 0.4.5 or later, which the structure (hashed) and the version (in the
  key) decide;
* the seeds that OpenFold3 samples with. ``seeds`` is the explicit list, ``[42]`` when the caller
  gives none, and empty when ``num_model_seeds`` asks OpenFold3 to generate them (the count is
  then in ``options``; None there means no generation). ``seeds`` and ``num_model_seeds`` cannot
  both be given, because OpenFold3 lets the generated seeds replace the explicit ones. A run
  writes the seeds to the runner YAML (``experiment_settings.seeds``): OpenFold3 0.5.0 does not
  read seeds from the query file. One request carries all the seeds (``seed_dimension`` is True):
  ``make_requests(input_path, seeds=[42, 7, 11], ...)`` returns that one request, and its
  output holds the directories ``seed_7``, ``seed_11`` and ``seed_42``, which a session lists in
  that numeric order;
* the custom weights, when given (``weights``: a checkpoint file, passed to OpenFold3 as
  ``--inference-ckpt-path``): the SHA-256 and size of the file, never its path, so a fine-tuned
  checkpoint has its own entries and a moved copy shares them. Without ``weights`` the key is
  what it was. ``inference_ckpt_path`` is the older way to name a checkpoint: it keeps its path
  and size in ``options`` and the two cannot be combined;
* the content of ``template_cif_path`` and ``runner_yaml`` when given, and of the user-default
  ``runner.yml`` that OpenFold3 merges under the toolkit's YAML (it can carry the MSA server URL
  and the structure format), when the file exists;
* the MSA policy, only when it is not the default: ``options["msa_policy"]`` is
  ``"forbid-server"``, ``options["receptor_msa"]`` is ``"single"`` for a receptor that is predicted
  from its own sequence, and ``extra_files["receptor_msa"]`` is the receptor's alignment, hashed
  by content (another path to the same file is the same request). A request without these keeps
  the key it had.

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

MSA. With ``use_msa_server=False`` every chain of the query gets a dummy MSA that holds only its
sequence. ``receptor_msa`` gives the receptor chain an alignment instead: the file is copied to
``<query>/msas/<name>_<chain>/colabfold_main.a3m`` and named in ``main_msa_file_paths`` of the
chain, and the binder keeps the dummy MSA. OpenFold3 0.5.0 parses an MSA file only when its name is
a key of ``MSASettings.max_seq_counts`` (``colabfold_main`` is one) and takes the first row of the
file as the query row of the chain, without comparing it with the chain: the first sequence of the
file must be the receptor's sequence, which ``prepare`` and ``run`` check before anything is
written. With the MSA server on, OpenFold3 replaces the MSA files of the query, so a given
alignment needs ``use_msa_server=False``. ``receptor_msa="single"`` is the explicit choice of a
receptor without an alignment. ``msa_policy="forbid-server"`` requires ``use_msa_server=False``
and a ``receptor_msa`` (a file or ``"single"``), and ``prepare`` and ``run`` refuse a request
whose OpenFold3 command line would hold a server flag, ``extra_args`` included, before anything
starts (``binding_metrics.predictors._msa``). Neither applies to mode ``predict``, whose query
file names its own MSA files.

OpenFold3 licence: Apache 2.0; no weights or model code are read or shipped here.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from binding_metrics.capabilities import LOCK_MODES
from binding_metrics.predictors._msa import (
    DEFAULT_MSA_POLICY,
    FORBID_SERVER,
    RECEPTOR_MSA_ROLE,
    SINGLE_SEQUENCE,
    assert_no_server_flag,
    check_msa_of_request,
    check_msa_request,
)
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
_DEFAULT_TEMPLATE_MODE = "structure"

_ON_UNMAPPABLE_CHOICES = ("error", "x")

_SERVER_REPLACES_MSA = (
    "receptor_msa needs use_msa_server=False: with the ColabFold MSA server on, OpenFold3 "
    "replaces the MSA files of the query by the ones it gets from the server, so the alignment "
    "would not be used and the sequences would leave the machine"
)

#: Files below ``predictions/`` that belong to the run and not to one query; ``run_many`` copies
#: them next to the output of each query it splits off.
_RUN_FILES = ("experiment_config.json", "inference_query_set.json", "template_accounting.json")


def _lock_message(mode: str = "score-lock") -> str:
    """Why OpenFold3 cannot run the lock mode ``mode``: the reason its capabilities declare.

    The pre-flight check refuses such a request first; this is the stop for a caller that did
    not run it.
    """
    from binding_metrics.predictors.of3 import OpenFold3Parser

    return f"OpenFold3 cannot run mode '{mode}'. " + OpenFold3Parser.capabilities.reason_for(
        "modes", mode
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
    #: With the MSA server on OpenFold3 replaces the MSA files of the query, so ``make_request``
    #: refuses a ``receptor_msa`` with ``use_msa_server=True`` (``_SERVER_REPLACES_MSA``).
    msa_conflicts_with_server = True
    #: For a complex structure: ``score`` and ``refold``. Mode ``predict`` takes a query file that
    #: the caller wrote, and ``score-lock`` and ``refold-lock`` cannot be run (see
    #: ``make_request``). ``score`` is the default of the OpenFold3 step (``--openfold-mode``):
    #: every chain is given its own structure.
    supported_modes = frozenset({"score", "refold"})
    default_mode = "score"
    #: One OpenFold3 process samples every seed of the request and writes one ``seed_<value>``
    #: directory for each, so ``make_requests`` gives one request for a list of seeds. The
    #: attribute is not part of the key.
    seed_dimension = True

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

    def unavailable_reason(self) -> Optional[str]:
        """Why ``is_available()`` is False, naming the conda environment; None when it is True.

        The store's own message says only that the model cannot be started on this machine; this
        names what was looked for, so that a typo in ``--openfold-conda-env`` is found at once.
        """
        if self.is_available():
            return None
        if self.conda_env is None:
            return "run_openfold is not on PATH, and no conda environment was named"
        if shutil.which("conda") is None:
            return (
                f"conda is not on PATH, so the conda environment '{self.conda_env}' cannot be used"
            )
        return (
            f"the conda environment '{self.conda_env}' does not exist or has no openfold3 "
            f"(`conda run -n {self.conda_env} python -c 'import openfold3'` failed)"
        )

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
        template_mode: str = _DEFAULT_TEMPLATE_MODE,
        msa_policy: str = DEFAULT_MSA_POLICY,
        receptor_msa: Optional[str | Path] = None,
    ) -> PredictionRequest:
        """The store request of one OpenFold3 run (see the module docstring for what it holds).

        Args:
            input_path: The complex structure (``score``, ``refold``) or the query file
                (``predict``).
            name: Query name; the output files are named after it.
            binder_chain, receptor_chain: Chain roles; required for ``score`` and ``refold``.
            mode: ``"score"``, ``"refold"`` or ``"predict"``; ``"score-lock"`` and
                ``"refold-lock"`` raise, OpenFold3 cannot pin the pose of the chains or force a
                template (``OpenFold3Parser.capabilities``).
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
            template_mode: How the templates of ``score`` and ``refold`` reach OpenFold3:
                ``"structure"`` (default; the template CIFs in ``template_cif_paths``,
                OpenFold3's CIF Direct Template Mode, OpenFold3 0.4.2 or later, which the
                ColabFold MSA server does not overwrite) or ``"alignment"`` (an A3M
                self-alignment per chain, which the server overwrites, so with the server on
                there is no template). It is in the key, so the key of a default ``score`` or
                ``refold`` request is not the one it had when ``"alignment"`` was the default.
                Only for ``score`` and ``refold``: a query file of ``predict`` names its own
                templates.
            msa_policy: ``"default"`` or ``"forbid-server"``. The second requires
                ``use_msa_server=False`` and a ``receptor_msa``, and ``prepare`` and ``run``
                refuse a command line that holds a server flag. Only for ``score`` and
                ``refold``.
            receptor_msa: The receptor's alignment as an A3M file, or the word ``"single"`` for a
                receptor predicted from its own sequence. The file goes to the receptor chain as
                ``colabfold_main.a3m``; its first sequence must be the receptor's, and its
                content (not its path) is in the key. The binder is single-sequence in both
                cases. Needs ``use_msa_server=False``: with the server on OpenFold3 replaces the
                MSA files of the query. Only for ``score`` and ``refold``.

        Raises:
            ValueError: Mode ``score-lock`` or ``refold-lock``, a missing chain role, an unknown
                mode or choice, a template file with ``predict``, both ``seeds`` and
                ``num_model_seeds``, a ``binder_cyclic`` that is not ``True``, ``False`` or
                ``"auto"``, or a value other than the default with ``predict``, a
                ``template_mode`` that is not ``"structure"`` or ``"alignment"`` (or is
                ``"alignment"`` with ``predict``), both ``weights`` and ``inference_ckpt_path``, an
                unknown ``msa_policy``, ``forbid-server`` with ``use_msa_server`` true or without
                a ``receptor_msa``, a ``receptor_msa`` with ``use_msa_server`` true, or either
                MSA setting with ``predict``.
            FileNotFoundError: ``weights`` or the ``receptor_msa`` file does not exist.
        """
        if weights is not None and inference_ckpt_path is not None:
            raise ValueError("give weights or inference_ckpt_path, not both")
        if mode in LOCK_MODES:
            raise ValueError(_lock_message(mode))
        if mode in ("score", "refold") and not (binder_chain and receptor_chain):
            raise ValueError(f"mode '{mode}' needs binder_chain and receptor_chain")
        if mode == "predict" and template_cif_path is not None:
            raise ValueError("a query file for mode 'predict' names its own templates")
        if on_unmappable_residue not in _ON_UNMAPPABLE_CHOICES:
            raise ValueError(
                f"on_unmappable_residue must be one of {_ON_UNMAPPABLE_CHOICES}, "
                f"got {on_unmappable_residue!r}"
            )
        fragments = check_msa_request(
            self.name,
            use_msa_server=bool(use_msa_server),
            msa_policy=msa_policy,
            receptor_msa=receptor_msa,
            supports_given_msa=True,
            supports_single=True,
        )
        if mode == "predict" and (msa_policy != DEFAULT_MSA_POLICY or receptor_msa is not None):
            raise ValueError(
                "a query file for mode 'predict' names its own MSA files, so msa_policy and "
                "receptor_msa do not apply to it"
            )
        if receptor_msa is not None and use_msa_server:
            raise ValueError(_SERVER_REPLACES_MSA)
        run_module = _run_module()
        run_module._check_binder_cyclic(binder_cyclic)
        run_module._check_template_mode(template_mode)
        if mode == "predict" and binder_cyclic != _DEFAULT_BINDER_CYCLIC:
            raise ValueError(
                "a query file for mode 'predict' names its own chains and cyclic flags"
            )
        if mode == "predict" and template_mode != _DEFAULT_TEMPLATE_MODE:
            raise ValueError("a query file for mode 'predict' names its own templates")
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
        extra_files.update(fragments.extra_files)

        request = PredictionRequest(
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
                "template_mode": None if mode == "predict" else template_mode,
                # what the query builders write changes with their version; a query file of
                # ``predict`` is the caller's own
                "query_builder_version": (
                    None if mode == "predict" else run_module.QUERY_BUILDER_VERSION
                ),
                "extra_args": [str(argument) for argument in extra_args],
                "inference_ckpt_path": None if checkpoint is None else str(checkpoint),
                "inference_ckpt_size_bytes": checkpoint_size,
                **fragments.options,
            },
        )
        if msa_policy == FORBID_SERVER:
            # the arguments of the call (extra_args) can switch the server on again
            assert_no_server_flag(self._command_words(request), self.name)
        return request

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
        # a run without the MSA server needs the dummy MSA in the query; the run functions decide
        # that themselves from use_msa_server, a direct call of the query builder is told
        dummy_msa = {"dummy_msa": True} if not request.options.get("use_msa_server", True) else {}
        return Path(
            function(
                **self._structure_arguments(request, Path(work_dir) / "query"),
                **self._query_arguments(request),
                **dummy_msa,
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
        if request.mode in LOCK_MODES:
            raise ValueError(_lock_message(request.mode))
        if request.model != self.name:
            raise ValueError(f"the of3 runner cannot run a '{request.model}' request")
        if request.input_path is None:
            raise ValueError("an OpenFold3 request needs an input file")
        self.check_weights(request)
        self._check_msa(request)

    def _check_msa(self, request: PredictionRequest) -> None:
        """Refuse a request, built by hand or by an older version, that breaks its MSA setting.

        The rules of ``make_request`` are applied to the facts in the request. Under
        ``forbid-server`` the command line that the run would start is built and looked at for
        a server flag: ``run_openfold`` takes the underscore and the hyphen spelling of its
        options alike and the last one wins, so ``extra_args`` can switch the server on with
        either, and both are read as the underscore spelling here.
        """
        check_msa_of_request(self.name, request, supports_given_msa=True, supports_single=True)
        asked = (
            RECEPTOR_MSA_ROLE in request.extra_files
            or request.options.get(RECEPTOR_MSA_ROLE) == SINGLE_SEQUENCE
        )
        if asked and request.options.get("use_msa_server", _DEFAULT_USE_MSA_SERVER):
            raise ValueError(_SERVER_REPLACES_MSA)
        if request.options.get("msa_policy") == FORBID_SERVER:
            assert_no_server_flag(self._command_words(request), self.name)

    def _command_words(self, request: PredictionRequest) -> list[str]:
        """The words of the ``run_openfold`` command of ``request`` with the options spelled alike.

        The paths are placeholders: only the flags are of interest. Each option name is written
        with underscores, the spelling that ``assert_no_server_flag`` knows.
        """
        arguments = self._run_arguments(request)
        command = _run_module()._openfold_predict_command(
            "<query>",
            "<predictions>",
            "<runner_yaml>",
            use_msa_server=arguments.get("use_msa_server", _DEFAULT_USE_MSA_SERVER),
            num_model_seeds=arguments.get("num_model_seeds"),
            inference_ckpt_path=arguments.get("inference_ckpt_path"),
            extra_args=arguments.get("extra_args"),
            conda_env=self.conda_env,
        )
        words = []
        for word in command:
            name, equals, value = word.partition("=")
            if name.startswith("--"):
                name = "--" + name[2:].replace("-", "_")
            words.append(name + equals + value)
        return words

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
        template_mode = request.options.get("template_mode", _DEFAULT_TEMPLATE_MODE)
        if request.mode != "predict" and template_mode not in (None, _DEFAULT_TEMPLATE_MODE):
            arguments["template_mode"] = template_mode
        receptor_msa = request.extra_files.get(RECEPTOR_MSA_ROLE)
        if request.mode != "predict" and receptor_msa is not None:
            arguments["receptor_msa_path"] = receptor_msa
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
        raise RuntimeError(_no_output_message(request, failures, _output_keys(predictions)))


def _output_keys(predictions: Path) -> list[str]:
    """The queries that have output below ``predictions``: folders with a ``seed_*`` directory."""
    if not Path(predictions).is_dir():
        return []
    return sorted(
        folder.name
        for folder in Path(predictions).iterdir()
        if folder.is_dir() and any(seed.is_dir() for seed in folder.glob("seed_*"))
    )


def _no_output_message(
    request: PredictionRequest, failures: dict[str, str], found: Sequence[str] = ()
) -> str:
    """The reason of a query without output; it holds no path of the work directory.

    The store renames the temporary work directory when it keeps the run, so a path would point
    at nothing in the recorded reason: the folder is named relative to the stored entry. When the
    folder holds output under other keys (``found``), they are named: in mode ``predict`` OpenFold3
    names the output after the key of the query in the query file, and a request with another name
    finds nothing although the prediction exists.
    """
    message = (
        f"OpenFold3 wrote no output for query '{request.name}' in the predictions folder of the "
        "run (outputs/predictions of a stored entry)"
    )
    reason = failures.get(request.name)
    if reason:
        return f"{message}: {reason}"
    if found:
        keys = ", ".join(f"'{key}'" for key in found)
        message += f"; the folder holds output for {keys}"
        if request.mode == "predict":
            message += (
                ". In mode 'predict' OpenFold3 names the output after the key of the query in "
                "the query file, so the request must have that name"
            )
    return message
