# Running the tests

The full suite takes about twenty minutes, too long to wait for after an edit to one module, so every test carries a marker for each area of the code that its file is about, a coarse one and a finer one (table below): `pytest -m poses` runs the pose-layer tests, `pytest -m poses_frame` only those about the common frame, `pytest -m "not gpu and (poses or preflight)"` runs two coarse areas, and `pytest --collect-only -q -m cli` lists a part without running it. For everyday work, `python scripts/pytest_for_diff.py` takes the files that differ from the merge-base with `origin/main` (committed, staged, unstaged and untracked), selects the test files that import the changed modules or a module that imports them, together with a few cheap guard tests (`--explain` says why each file was chosen); `--working-tree` keeps to uncommitted work, `--base REF` moves the starting point, `--include-slow` adds the tests marked `slow`, and `--dry-run` prints the pytest commands without running them. Because the selection follows imports, it misses a dependency that the code resolves by name at run time, such as the metric registry loading a function from its path, so the full suite (`pytest -m "not gpu"`) is the gate before a pull request and before a signed release.

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

The markers come from the imports of each test file and the names it uses (`tests/_areas.py`), so a new test file is tagged when it is collected; a file takes the areas that hold a quarter of its references to the package, so a CLI test that also builds a structure is about the CLI. `python tests/_areas.py poses_frame` prints the files of one area without collecting anything, and `pytest --markers` describes each area.
