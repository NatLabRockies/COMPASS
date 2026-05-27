![Greg Brockman: "evals are surprisingly often all you need"](image.png)

# Evals

Accuracy/quality evaluations of COMPASS extraction against real ordinance
documents. Each suite is a `test_run_<name>_evals.py` file that drives a
specific extractor end-to-end and writes a breakdown + metrics CSV to
`results/`. A regression gate (in `conftest.py`) fails the run if the
committed baseline gets worse.

## Run

```bash
pixi run evals date_extraction        # runs the dev set (live LLM calls, needs Azure creds)
```

Evals are deselected by default in regular `pytest` runs — they only fire
when their marker is explicitly selected.

## `dev` vs `held-out` evals

Each dataset is split into a frequently-run **dev** set and a sacred
 **held-out** set used as an unbiased measure of true performance:

| | `dev/` | `held-out/` |
| --- | --- | --- |
| Eval type | run frequently during development (`-m dev_evals`) | run before a release (`-m held_out_evals`) |
| Purpose | iterate, tune prompts/logic, debug failures | unbiased estimate of true performance |

The held-out set only gives an **honest** read if we *don't* tune against it:

- **Do not** run `held_out_evals` repeatedly while iterating — use `dev` for that.
- **Do not** inspect held-out failures to "fix" the extractor for those
  specific documents. The moment you optimize against the held-out set, it
  stops being held-out and its numbers become optimistic.
- Treat `held_out_evals` as a checkpoint you look at occasionally (e.g.
  before a release), not a development loop.

The harness helps enforce this: a `held_out_evals` run writes **only summary
metrics** (no per-case breakdown), and per-case predictions are not logged
— so there is nothing to eyeball or tune against. `dev` runs write the full
per-case breakdown.

## Layout

```
test_run_<name>_evals.py   # one eval suite per extractor (e.g. test_run_solar_evals.py)
conftest.py                # reporter + regression gate (writes into results/)
results/
  dev/<name>_evals.csv               # committed baseline metrics (regression gate reads these)
  dev/<name>_evals_breakdown.csv     # committed per-case dev breakdown
  held_out/<name>_evals.csv          # committed baseline held-out metrics (no per-case detail)
data/
  dev/<tech>/
    manifest.json5         # [{fips, jurisdiction, file, source, expected: {year, ...}}, ...]
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
2. Write `test_run_<name>_evals.py` with `@pytest.mark.dev_evals` and
   `@pytest.mark.held_out_evals` test functions.
3. First run sets the baseline; commit the resulting `results/{dev,held_out}/<name>_evals*.csv`.
