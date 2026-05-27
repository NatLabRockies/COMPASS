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

## Layout

```
test_run_<name>_evals.py   # one eval suite. Later we might have say test_run_solar_evals.py
conftest.py                # reporter + regression gate (writes results/*.csv)
data/                      # per-tech dev/ and held-out/ datasets (see data/README.md)
results/                   # committed baseline CSVs — the regression gate reads these
```

## Adding an eval suite

1. Drop ground-truth docs + a `manifest.json5` under `data/{dev,held-out}/<tech>/`
   (see `data/README.md` for the manifest schema).
2. Write `test_run_<name>_evals.py` with `@pytest.mark.dev_evals` and
   `@pytest.mark.held_out_evals` test functions.
3. First run sets the baseline; commit the resulting `results/<name>_*.csv`.
