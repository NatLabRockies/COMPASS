![Greg Brockman: "evals are surprisingly often all you need"](image.png)

# Evals

Accuracy/quality evaluations of COMPASS extraction against real ordinance
documents. Each suite is a `test_run_<name>_evals.py` file that drives a
specific extractor end-to-end and writes a breakdown + metrics file to
`results/`. A regression gate (run by the test module's autouse fixture)
fails the run if the committed baseline gets worse.

## Run

```bash
pixi run evals date_extraction              # dev dataset (live LLM calls, needs Azure creds)
pixi run evals date_extraction --held-out   # held-out dataset (release checkpoint, no gate)
```

Evals are deselected by default in regular `pytest` runs — they only fire
when the `-m evals` marker is explicitly selected (which the pixi task does).

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
against, and a held-out drop won't fail CI for you (since you're not
supposed to be running it in CI).

## Layout

```
test_run_<name>_evals.py   # one eval suite per extractor (e.g. test_run_date_extraction_evals.py)
conftest.py                # registers the --held-out pytest flag
utilities/
  base.py                  # Result schema, classify, load_doc
  metrics.py               # compute_metrics, wilson_ci (pure math, no I/O)
  reports.py               # report_evals + baseline-loading helpers (I/O + formatting)
results/
  dev/<name>_evals.json              # committed baseline metrics — list of per-feature dicts (gate reads these)
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

## Adding an eval suite

1. Drop ground-truth docs + a `manifest.json5` under `data/{dev,held-out}/<tech>/`.
2. Write `test_run_<name>_evals.py` with a single `@pytest.mark.evals`
   test function. Use `pytest_generate_tests` to parametrize cases from
   the dataset chosen by `--held-out`, and a module-scoped autouse
   fixture that calls `report_evals(...)` and enforces its own gate.
3. First run sets the baseline; commit the resulting
   `results/{dev,held_out}/<name>_evals.json` (and the dev breakdown `.csv`).
