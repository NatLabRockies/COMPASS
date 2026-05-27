![Greg Brockman: "evals are surprisingly often all you need"](image.png)

# Evals

Accuracy/quality evaluations of COMPASS extraction against real ordinance
documents. Each suite is a `test_run_<name>_evals.py` file that drives a
specific extractor end-to-end and writes a breakdown + metrics file to
`results/`. A regression gate (run by the test module's autouse fixture)
fails the run if the committed baseline gets worse.

## Run

```bash
pixi run evals date_extraction                 # Run as frequently as needed
pixi run evals date_extraction -- --held-out   # Only run once before a release
```

The `--` separator tells pixi to pass everything after it to pytest, so
you can also tack on flags like `-k Bartow` or `--maxfail=3` the same way.

Evals are deselected by default in regular `pytest` runs — they only
fire when the `-m evals` marker is explicitly selected (which the pixi
task does). `--held-out` is registered in `conftest.py` and toggles
which dataset the suite pulls cases from.

## `dev` vs `held-out` evals

Each dataset is split into a frequently-run **dev** set and a sacred
**held-out** set used as an unbiased measure of true performance:

| | `dev/` (default) | `held-out/` (`--held-out`) |
| --- | --- | --- |
| Purpose | iterate, tune prompts/logic, debug failures | unbiased estimate of true performance |
| Cadence | run frequently during development | run before a release |
| Regression gate | yes — fails on aggregate or per-row regression | no — unbiased read, just prints + writes JSON |
| Per-case breakdown | written + logged | hidden (no breakdown CSV, no per-case logs) |

The held-out set only gives an **honest** read if we *don't* tune against it:

- **Do not** run `--held-out` repeatedly while iterating — use the dev set for that.
- **Do not** inspect held-out failures to "fix" the extractor for those
  specific documents. The moment you optimize against the held-out set, it
  stops being held-out and its numbers become optimistic.
- Treat `--held-out` as a checkpoint you look at occasionally (e.g.
  before a release), not a development loop.

The harness helps enforce this: a `--held-out` run writes **only summary
metrics** (no per-case breakdown), per-case predictions are not logged,
and there is no regression gate — so there is nothing to eyeball or tune
against.

## Layout

```
test_run_<name>_evals.py   # one eval suite per extractor (e.g. test_run_date_extraction_evals.py)
conftest.py                # registers the --held-out pytest flag
utilities/                 # shared, eval-agnostic plumbing
  base.py                  #   Result schema, SUCCESS/FAILURE, classify, load_doc
  metrics.py               #   compute_metrics, wilson_ci (pure math, no I/O)
  reports.py               #   report_evals + load_baseline_failing + regressed_rows (I/O + formatting)
results/
  dev/<name>_evals.json              # committed baseline metrics (gate reads these)
  dev/<name>_evals_breakdown.csv     # committed per-case dev breakdown
  held_out/<name>_evals.json         # committed baseline held-out metrics (no per-case detail)
data/
  dev/<tech>/
    manifest.json5         # [{state, county, subdivision, jurisdiction_type, file, source, expected: {year, ...}}, ...]
    <documents>            # the ordinance PDFs/text files referenced by the manifest
  held-out/<tech>/
    manifest.json5
    <documents>
```

Datasets are organized by tech (`solar/` today; future additions like
`wind/`, `geothermal/` will be sibling directories). The `expected` block
nests per-field ground truth so additional fields (setbacks, max height,
etc.) can be added as keys alongside `year` without changing the manifest
shape. `expected.year: null` means the ground truth is "no enactment date
exists" — the extractor should return no year for that document.

## How a suite is wired

A `test_run_<name>_evals.py` file owns three pieces:

1. **`pytest_generate_tests(metafunc)`** reads `--held-out`, loads the
   right `manifest.json5`, and parametrizes the test's `case` argument.
   It also stamps each case with `case["fp"]` (the resolved document
   path), so the test body never has to know which dataset it came from.
2. **`@pytest.mark.evals` test function** runs the extractor on one
   case and appends a `Result` to the module-level `RESULTS` list.
3. **Module-scoped autouse teardown fixture** calls
   `report_evals(request, EVAL_NAME, RESULTS, results_dir,
   write_breakdown=not held_out)` to compute metrics, write the
   artifacts, and snapshot baselines. The returned dict's
   `baseline_failing` / `fails_now` / `regressed_rows` fields drive
   each suite's own gate -- the `reports` module does **not** decide
   what counts as a regression. (Held-out runs skip the gate entirely.)

## Adding an eval suite

1. Drop ground-truth docs + a `manifest.json5` under `data/{dev,held-out}/<tech>/`.
2. Copy `test_run_date_extraction_evals.py` as a starting point. Swap in
   your extractor function and the `expected.<feature>` key you compare
   against.
3. First run sets the baseline; commit the resulting
   `results/{dev,held_out}/<name>_evals.json` (and the dev breakdown `.csv`).
