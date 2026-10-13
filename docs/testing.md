# Running the tests

The full suite takes about twenty minutes, too long to wait for after an edit to one module, so every test carries a marker for each area of the code that its file imports (table below): `pytest -m poses` runs the pose-layer tests, `pytest -m "not gpu and (poses or preflight)"` runs two areas, and `pytest --collect-only -q -m cli` lists a part without running it. For everyday work, `python scripts/pytest_for_diff.py` takes the files that differ from the merge-base with `origin/main` (committed, staged, unstaged and untracked), finds the areas whose tests import them or import a module that does, and runs those areas together with a few cheap guard tests; `--working-tree` keeps to uncommitted work, `--base REF` moves the starting point, `--include-slow` adds the tests marked `slow`, and `--dry-run` prints the pytest commands without running them. Because the selection follows imports, it misses a dependency that the code resolves by name at run time, such as the metric registry loading a function from its path, so the full suite (`pytest -m "not gpu"`) is the gate before a pull request and before a signed release.

| Area | What its tests exercise |
|---|---|
| `cli` | The binding-metrics-run, -batch and -check-env entry points and their flags |
| `core` | System preparation, force fields, simulation, GAFF, cyclic and non-standard residues |
| `io` | Reading and writing structures (load_structure, save_cif, chain detection) |
| `metrics` | The static and MD metrics, their registry and the prediction confidence metrics |
| `of3` | The OpenFold3 route: metrics/openfold.py with its helpers, the of3 adapter and runner |
| `package` | The package root, constants, logging setup and provenance |
| `poses` | The pose layer (kinds, reference, scoring, pose-set reports) and the testing package |
| `predictors` | Model adapters, runners, the prediction store and session (OpenFold3 aside) |
| `preflight` | What a model or metric accepts and the check of an input before a run |
| `protocols` | Relaxation, preparation, solvation, structural QC and the report |
| `docs` | The text of docs/, README.md and CHANGELOG.md, checked against the code |
| `golden` | Comparisons with stored reference output (tests/golden, OpenFold3 golden files) |
| `misc` | Tests that import nothing from the package |

The markers come from the imports of each test file (`tests/_areas.py`), so a new test file is tagged when it is collected and a file carries the union of the areas of its imports; `python tests/_areas.py poses` prints the files of one area without collecting anything.
