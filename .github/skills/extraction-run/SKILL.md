---
name: extraction-run
description: Execute one-shot extraction with COMPASS, evaluate outputs, and iterate schema/config changes with minimal cost.
---

# Extraction Run Skill

**ONE-SHOT EXTRACTION ONLY.** This skill applies only to schema-driven extraction.
For legacy decision-tree extraction (solar, wind, small wind), consult COMPASS 
architecture docs.

Use this skill to run one-shot extraction in a repeatable, low-risk way,
then iterate quickly until you have stable structured outputs.

## When to use

- Schema exists and plugin config points to it.
- You are onboarding a new technology (diesel generator, geothermal, CHP, hydrogen).
- You need a reliable smoke-test workflow before scaling.
- You are NOT using legacy decision-tree extraction.

## Two-pipeline modes

COMPASS supports two distinct extraction pipelines. Choose one and do not mix
them for the same technology:

| Mode | Where code lives | Good for |
|---|---|---|
| **One-shot (schema-based)** | `examples/` → `compass/extraction/<tech>/` | New techs, no Python changes |
| **Legacy decision-tree** | Python code in `compass/extraction/<tech>/` | Existing solar, wind, small wind |

One-shot is the correct path for all new technology onboarding. It requires
only a schema JSON, a plugin YAML, and a run config — no Python source changes.

## Tech promotion lifecycle

New technology assets start in `examples/` and finish in `compass/extraction/`:

1. **Develop** — place all assets in `examples/one_shot_schema_extraction_<tech>/`
2. **Stabilize** — iterate schema/plugin until smoke and robustness gates pass
3. **Promote** — copy the three finalized files into `compass/extraction/<tech>/`:
   - `<tech>_schema.json`
   - `<tech>_plugin_config.yaml`
   - `<tech>_config.json5` (optional; useful as a reference run config)

The promoted extraction folder contains only config files — no Python code is
needed for one-shot techs.

## Required inputs

- Run config for `compass process`.
- Plugin config containing `schema`.
- API keys in environment (never hardcode in configs).
- A jurisdiction set sized to the current phase.

## Preflight checks (must pass before run)

- Jurisdiction CSV has headers `County,State`.
- `out_dir` is unique for this run.
- At least one acquisition step is enabled:
	`perform_se_search: true`, `perform_website_search: true`,
	`known_doc_urls`, or `known_local_docs`.
- If `heuristic_keywords` exists, all four required lists are present and
	non-empty.

## Naming convention

Use tech-first names for all one-shot assets:

- `<tech>_config*.json5`
- `<tech>_plugin_config.yaml`
- `<tech>_schema.json`
- `<tech>_jurisdictions*.csv`

The `tech` value in the run config must be a string that becomes the plugin
registry identifier. It must be unique, lowercase, and underscore-separated
(for example `concentrating_solar`, `geothermal_electricity`). COMPASS will
raise `Unknown tech input` if this key does not match any registered plugin.

## Canonical development pattern

For early development, start with the proven dynamic baseline, then fall back
to deterministic mode only when search infrastructure is unstable:

1. Use one small jurisdiction file (1-3 rows).
2. Use your preferred configured search engine.
3. Load `.env` into shell (`set -a && source .env && set +a`).
4. Run with verbose logs:
	 - `pixi run compass process -c config.json5 -p plugin.yaml -v`
5. Confirm output artifacts exist before tuning schema semantics.

Fallback mode when needed:

- Add `known_doc_urls` (or `known_local_docs`) in run config.
- Set `perform_se_search: false` and `perform_website_search: false`.

## Adaptation rule

When adapting this workflow for a new technology, keep the run structure
unchanged and swap only technology-specific inputs:

- `tech` in run config,
- schema file,
- plugin descriptor (`data_type_short_desc`),
- retrieval query/keyword vocabulary,
- known document URL set.

Change one axis per run unless debugging infrastructure failures.

## Canonical reference

- `examples/one_shot_schema_extraction/` — complete working examples
- `examples/one_shot_schema_extraction/README.rst` — general one-shot overview
- `examples/water_rights_demo/one-shot/` — multi-doc extraction example

## Environment setup

Load secrets from `.env` before running. Never commit key values in config files.

```bash
set -a && source .env && set +a   # no spaces around = in .env assignments
```

## Core command

```bash
pixi run compass process -c config.json5 -p path/to/plugin_config.yaml -v
```

## Phase-gated workflow

1. **Smoke test (1 jurisdiction)**
	 - Goal: verify wiring and output contract.
2. **Robustness (5 jurisdictions)**
	 - Goal: verify feature stability and edge-case handling.
3. **Scale (full set)**
	 - Goal: only after earlier phases pass acceptance gates.

## Validation checklist

Evaluate each run on:

- document relevance (exclude off-domain content),
- feature coverage vs expected ordinance topics,
- section/summary traceability,
- unit consistency,
- null discipline,
- **scope bleed** — check that no features appear in the output CSVs that
  fall outside the schema enum; generic land-use-code documents can cause
  unrelated provisions to leak through. Tighten `extraction_system_prompt`
  in plugin YAML to fix this.

## Expected output artifacts

A successful run produces these files under `out_dir`:

| Artifact | Meaning |
|---|---|
| `ordinance_files/*.pdf` | Downloaded source documents |
| `cleaned_text/*.txt` | Heuristic-filtered extracted text |
| `jurisdiction_dbs/*.csv` | Per-jurisdiction raw extraction rows |
| `quantitative_ordinances.csv` | Final compiled numeric features |
| `qualitative_ordinances.csv` | Final compiled qualitative features |
| `usage.json` | Per-jurisdiction LLM token and request counts |
| `meta.json` | Run metadata (cost, timing, version) |

Final CSV columns: `county`, `state`, `subdivision`, `jurisdiction_type`,
`FIPS`, `feature`, `value`, `units`, `adder`, `min_dist`, `max_dist`,
`summary`, `year`, `section`, `source`.

## Interpreting output status correctly

`cleaned_text` files can exist while `Number of documents found` is `0`.

This means acquisition/text collection worked, but no final structured ordinance
rows were emitted into consolidated DB outputs.

Check in order:

1. `outputs/*/cleaned_text/*.txt` (text extraction present)
2. `outputs/*/jurisdiction_dbs/*.csv` (per-jurisdiction parsed rows)
3. `outputs/*/quantitative_ordinances.csv` and
	 `outputs/*/qualitative_ordinances.csv` (final compiled results)

Treat the run as **failed for extraction quality** when either is true:
- `Number of jurisdictions with extracted data: 0`
- any configuration exception appears in logs (even if process exits 0)

Only treat a run as passing when both are true:
- at least one jurisdiction has extracted data
- at least one jurisdiction CSV in `jurisdiction_dbs/` has more than header row

## Root-cause triage

- **Wrong or noisy documents**
	- Tune query templates, URL keywords, and exclusions.
	- Prefer `known_doc_urls` while stabilizing.
- **Right documents, wrong fields**
	- Tune schema descriptions/examples and ambiguity rules.
	- Check `extraction_system_prompt` in plugin YAML — it is the primary
	  guard against scope bleed from generic legal documents.
- **Correct values, unstable formatting**
	- Tighten enums, unit vocabulary, and null behavior.
- **Nothing downloaded / unstable search**
	- Disable live search and use deterministic known URLs/local docs.
- **0 documents found for a jurisdiction during website crawl**
	- Expected for jurisdictions with few online ordinances. The website
	  crawl is a second acquisition pass after search-engine retrieval;
	  0 results there is not a pipeline failure.

## Acceptance gates

Do not advance phases until all are true:

- Output rows conform to required contract.
- High share of rows include useful `section` and `summary`.
- Feature names are stable and machine-consistent.
- Repeated runs on same sample show minimal drift.

## Cost and speed controls

- Keep sample size minimal while tuning.
- Change one variable per run.
- Archive run command, input set, and output path for each iteration.

## Workspace hygiene (important)

Keep one canonical working set per technology in `examples/`:

- one run config,
- one plugin config,
- one schema,
- one jurisdiction file,
- one known docs file.

Delete stale `_migrated`, `_smoke`, and duplicate output folders to avoid
configuration drift and debugging confusion.

## Known infrastructure issues

### Playwright timeouts

Web search via `rebrowser_playwright` may fail with 60s timeouts on
`Page.wait_for_selector`. Symptoms:
- `TimeoutError: Page.wait_for_selector: Timeout 60000ms exceeded`
- All search queries fail consistently
- Browser session crashes with `ProtocolError: Internal server error, session closed`

These errors during the **website crawl phase** (second acquisition pass) are
**non-fatal**. COMPASS logs them and continues. They do not block the
search-engine phase or extraction.

If search itself is failing, verify provider credentials are loaded and fall
back to deterministic mode.

**Workaround**: Use `known_local_docs` or `known_doc_urls` and disable
search/website steps while validating extraction logic.

### known_local_docs loading failures

`known_local_docs` may fail silently with `ERROR: Failed to read file` in
jurisdiction logs due to external loader behavior.

**Workaround**: Prefer `known_doc_urls` for deterministic smoke tests and
pre-validate local docs before pipeline runs.

