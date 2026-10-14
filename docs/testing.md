# Running the tests

## The tiers

A plain `pytest` runs the fast tier: every test except those that take a second or more (marker `slow`), that run the real OpenMM and biotite stack (`integration`), that run the parsers of an installed OpenFold3 (`openfold`) and that need a GPU (`gpu`). Two options add the others:

```
pytest                # the fast tier
pytest -n 4           # the same on four processes
pytest --full         # adds slow, integration and openfold: every test but the GPU ones
pytest --gpu          # adds the GPU tests to the fast tier
pytest --full --gpu   # every test
```

The fast tier has 8,912 of the 9,324 cases that need no GPU. Recorded on 14 October 2026 (commit 1fe658d) on a 15-core machine whose load was 4 to 6 and 4 to 11 because another job used three of its cores, it took 2 min 42 s on four processes and 6 min 22 s on one process (the same command without `-n`). Earlier the same day, with 8,840 cases, four processes took 1 min 40 s at a load of 2 to 5 and between 2 min 28 s and 2 min 48 s at a load of 11 to 18. `-n` comes from pytest-xdist, which the `dev` extra installs (`pip install -e ".[dev]"`); nothing in the configuration sets it, so a plain `pytest` runs on one process and works without the plugin.

`pytest --full` is the gate before a pull request and before a signed release, and the run that CI makes. Its 9,324 cases took 7 min 31 s with `-n 4` at a load of 6 to 11 and about twenty minutes on one process. `pytest --gpu` adds the GPU tests, which skip on a machine without a CUDA device, so `pytest --full --gpu` is the run of a machine that has one.

Each run prints one line about its tier after the count of collected tests, with the number of tests it selected and how many each option adds:

```
tier: fast - 8912 of 9412 tests selected; --full adds 412 slow/integration/openfold tests, --gpu adds 88 GPU tests
```

With `-n` only the workers collect, so the controller prints the line at the end of the session, before the summary. The rules that decide what a run selects are in the docstring of `tests/_tiers.py`:

- A test file or a node id named on the command line runs whatever its tier: `pytest tests/test_geometry.py` runs every test of the file and `pytest "tests/test_geometry.py::TestOmegaPlanarity"` the tests it names. A directory (`pytest tests/poses`) and no argument at all obey the tier. `--tier-named` applies the tier to the named files as well, for a script that names the files to run (`scripts/pytest_for_diff.py` uses it).
- A `-m` expression that mentions `slow`, `integration`, `openfold` or `gpu` is a tier chosen by hand, so no tier is applied and the expression selects what it always did: `-m "not gpu"` runs every test but the GPU ones (the cases of `--full`), `-m slow` the slow tests and `-m gpu` the GPU tests. An expression about the areas of the code (`-m poses`, `-m "poses or preflight"`, see below) leaves the tier on and applies it to the tests of those areas, so `pytest --full -m poses` adds the slow ones. `-k` leaves the tier on as well.
- A test that carries `gpu` follows `--gpu` alone, whatever other speed marker it has.

A test of a second or more carries `slow` (`integration` and `openfold` mark the others). When a test grows slow it has to be tagged, and a section "untagged slow tests" at the end of a session lists the tests whose call took 2 s or more and carry none of `slow`, `integration`, `openfold` and `gpu`, which the fast tier therefore does not leave out. The run still passes unless `--strict-slow` is given, and `--slow-seconds` moves the threshold. A duration depends on the load of the machine, so the list asks for a look and does not prove a test slow. `python scripts/tag_slow_tests.py LIST` adds `@pytest.mark.slow` to the node ids of a file, one per line; `--dry-run` shows the edits and `--collected FILE` (the output of `pytest --collect-only -q --full --gpu`, which lists every case) lets it mark single cases of a literal parametrisation.

## Running a part of the suite

A change to one module needs fewer tests than even the fast tier, so every test carries a marker for each area of the code that its file is about, a coarse one and a finer one (table below): `pytest -m poses` runs the pose-layer tests of the fast tier, `pytest -m poses_frame` only those about the common frame, `pytest -m "poses or preflight"` runs two coarse areas, `pytest --full -m poses` adds the slow ones, and `pytest --collect-only -q -m cli` lists a part without running it. The markers come from the imports of each test file and the names it uses (`tests/_areas.py`), so a new test file is tagged when it is collected; a file takes the areas that hold a quarter of its references to the package, so a CLI test that also builds a structure is about the CLI. `python tests/_areas.py poses_frame` prints the files of one area without collecting anything, and `pytest --markers` describes each area. A fine area is named `<coarse>_<family>` and every test of the family carries the coarse marker too, so `-m metrics` runs all the metrics and `-m metrics_geometry` one family.

The markers say what a test is about, which is narrower than which tests a change can break. For that, `python scripts/pytest_for_diff.py` takes the files that differ from the merge-base with `origin/main` (committed, staged, unstaged and untracked) and selects the test files that import a changed module or a module that imports it, where at most one link of the chain is an import inside a function; a module that a table loads by name (a metric of the registry, a model adapter) also selects the files that import the code doing the loading, a package `__init__` selects the importers of its names, and a few cheap guard tests always run. `--explain` prints the reason for each file: `imports poses.score -> poses.frame` means that the test imports `poses.score`, which imports the changed `poses.frame`. `--working-tree` keeps to uncommitted work, `--base REF` moves the starting point, `--max-lazy N` changes the number of function-level imports a chain may contain, `--full` adds the tests that the fast tier leaves out (`slow`, `integration` and `openfold`; `--include-slow` is the earlier name), and `--dry-run` prints the pytest commands without running them. The chosen files run in the fast tier: they are named on the command line, which would run them whatever their tier, so the helper gives `--tier-named`. The guard files are named without it, in a command of their own, so every test of them runs.

Against the lines that each test executed (coverage with one context per test, recorded on commit b81b589 on 13 October 2026, 8,663 tests), the selection for a change to one module holds on average 99.6 % of the test files that run any line of it and never less than 93 %; the few files it misses reach the module through a subprocess or a table that no import statement shows. The median selection holds 41 % of the cases, against 60 % when the whole area of every affected module runs and 10 % for the files that really run the module, and the 90th percentile 74 %. A module such as `poses/frame.py` selects 13 % of the cases, whereas the modules that most of the code imports (`utils.py`, `core/residues.py`, `io/structures.py`, `metrics/_common.py`) select 87 to 98 %: an import-based rule cannot be narrow there. These figures belong to that snapshot and drift with the code, so they are an estimate; `pytest --full` is the gate before a pull request and before a signed release.

| Area | What its tests are about | Finer areas |
|---|---|---|
| `cli` | All command-line entry points: run, batch, check-env and the shared flags | `cli_batch`, `cli_env`, `cli_prediction`, `cli_run` |
| `core` | System preparation, force fields, simulation, GAFF, cyclic and non-standard residues | `core_cyclic`, `core_gaff`, `core_residues`, `core_system` |
| `io` | Reading and writing structures (load_structure, save_cif, chain detection) | none |
| `metrics` | The static and MD metrics and their registry | `metrics_common`, `metrics_comparison`, `metrics_energy`, `metrics_evobind`, `metrics_geometry`, `metrics_interface`, `metrics_receptor`, `metrics_registry`, `metrics_sasa` |
| `of3` | The OpenFold3 route: its run, templates, flags, adapter and runner | `of3_adapter`, `of3_metrics`, `of3_run`, `of3_templates` |
| `package` | The package root, constants, logging setup and provenance | `package_exports`, `package_support` |
| `poses` | The pose layer (kinds, reference, scoring, pose-set reports) and the testing package | `poses_agreement`, `poses_descriptors`, `poses_frame`, `poses_pose`, `poses_score` |
| `predictors` | Model adapters, runners, the prediction store and session (OpenFold3 aside) | `predictors_af2`, `predictors_base`, `predictors_boltz2`, `predictors_protenix`, `predictors_store` |
| `preflight` | What a model or metric accepts and the check of an input before a run | `preflight_capabilities`, `preflight_cli` |
| `protocols` | Relaxation, preparation, solvation, structural QC and the report | `protocols_qc`, `protocols_relaxation`, `protocols_report` |
| `docs` | The text of docs/, README.md and CHANGELOG.md, checked against the code | none |
| `golden` | Comparisons with stored reference output (tests/golden, OpenFold3 golden files) | none |
| `misc` | Tests that import nothing from the package | none |
