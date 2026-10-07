---
name: schema-fine-tuning
description: Fine-tune a COMPASS one-shot extraction schema after a wrong or missed extraction, using the Dead-Reckoning support tool. Use whenever a user asks to fine-tune, tune, calibrate, or debug an extraction schema (for example the transmission schema) against observed extraction errors.
---

# Schema Fine-Tuning Skill

Use this skill to fine-tune a COMPASS one-shot extraction schema (JSON5) in a
repeatable way, driven by the Dead-Reckoning support tool. The workflow diagnoses
observed extraction errors, locates the responsible schema construct, applies a
targeted fix, and verifies it.

## When to use

- A one-shot extraction produced a wrong, missed, or misclassified value and you
  want to adjust the schema to fix it.
- You are iterating on schema `$core_principles`, feature descriptions, `$scope`,
  or examples.

## Do not use

- Decision-tree extraction (solar, wind, small wind) feature engineering.
- Non-schema tasks (plugin config, retrieval tuning, docs-only updates).

## Step 1 — Ensure Dead-Reckoning is available

This skill drives the Dead-Reckoning CLI. Before doing anything else, confirm it
is installed in the active Pixi environment.

1. Check the CLI version:

   ```
   pixi run dead_reckoning --version
   ```

   Success looks like `dead_reckoning, version <X.Y.Z>`.

2. If the command is not found, install it as a PyPI git dependency and retry
   step 1:

   ```
   pixi add --pypi "dead-reckoning @ git+ssh://git@github.com/NatLabRockies/Dead-Reckoning.git@main"
   ```

Do not proceed to later steps until `pixi run dead_reckoning --version` succeeds.

## Step 2 — Run the current COMPASS extraction

Produce a fresh extraction with the schema as it stands today. This is the
baseline that fine-tuning will be measured against.

### Pre-run checks (do these before running)

Fine-tuning must iterate on a small, local, deterministic target set. Before
running, open the `config.json5` used for the run and verify:

1. **`jurisdiction_fp` points to a small target set.** Read the file it names
   (usually a CSV) and count its targets (rows excluding the header). The number
   of targets governs how to proceed:
   - 1-5 targets: acceptable, proceed.
   - More than 5 targets: STOP and ask the user to confirm this is what they want
     before running, since the run will be slower and more costly. Suggest
     pointing `jurisdiction_fp` at a smaller file.

2. **`known_local_docs` is defined and covers the target set.** Confirm the
   `known_local_docs` key is present and uncommented in `config.json5`, that the
   file it points to is valid JSON5, and that every jurisdiction in
   `jurisdiction_fp` has at least one document entry in `known_local_docs`.
   `known_local_docs` may list additional jurisdictions beyond those in
   `jurisdiction_fp`; those extra entries are fine. If `known_local_docs` is
   missing, commented out, invalid JSON5, or any `jurisdiction_fp` target lacks a
   corresponding document entry, STOP and report the gap rather than running.

3. **Labeled ground-truth data exists for validation.** Validation in Step 4
   compares the extraction against hand-labeled expectations, so confirm the
   labeled data is in place before spending a run. The labeled data is a directory
   of YAML files organized by state, where each YAML file describes the expected
   outputs for one jurisdiction (for example a `labeled/manual/` directory
   containing `<state>/<Jurisdiction>.yaml` files). Confirm the directory exists
   and is not empty, that it contains YAML files organized by state, and that the
   labeled jurisdictions cover the targets from `jurisdiction_fp` (note any target
   with no corresponding labeled YAML, since it cannot be validated). If no
   labeled-data directory is present, STOP and ask the user where the labeled
   ground truth lives rather than running.

Only run the extraction once all checks pass (or the user has confirmed an
oversized target set).

### Run

1. From the run directory that holds the extraction config, run:

   ```
   pixi run compass process -v -c config.json5
   ```

2. Let the run finish and note the output location it reports. These extracted
   outputs are the input to the diagnosis in the next step.

   Expect this to take a while — typically at least 10 minutes, and more than 15
   is not unusual. Let it run to completion rather than assuming it has hung; do
   not set a short timeout that would kill it. Waiting does not consume additional
   credits, since no model requests are made while the command runs.

Always run the current schema first; do not edit the schema before capturing
this baseline.

## Step 3 — Merge the extraction outputs

COMPASS writes the run results as two separate tables: a quantitative table and
a qualitative table. They live in the run's output directory, which is set by
`out_dir` in `config.json5` (COMPASS defaults to `./output` when the key is
absent). Do not hardcode `output/`; resolve `out_dir` from the config so the
merge follows the run wherever it wrote, then merge the two tables:

```
pixi run python -c "
import os, subprocess, pyjson5
cfg = pyjson5.load(open('config.json5'))
out = cfg.get('out_dir', './output')
quant = os.path.join(out, 'quantitative_ordinances.csv')
qual = os.path.join(out, 'qualitative_ordinances.csv')
combined = os.path.join(out, 'combined.csv')
subprocess.run(['dead_reckoning', 'qc', 'merge', quant, qual, '-o', combined], check=True)
print('merged ->', combined)
"
```

The resulting `combined.csv` (under `out_dir`) is the single artifact used for
diagnosis in the next step.

## Step 4 — Validate against labeled data

Validation compares the merged extraction (`combined.csv`) against the set of
hand-labeled ground-truth expectations confirmed in the Step 2 pre-run checks.

Run the validation with Dead-Reckoning, pointing `-t` at the labeled-data
directory and passing the merged `combined.csv`. Use `--format json` so the
report can be parsed programmatically, and redirect it to a file so the next
step can process it (the command has no output-file flag, so redirect stdout):

```
pixi run dead_reckoning qc validate -t <labeled_data_dir> --format json combined.csv > validation_report.json
```

Replace `<labeled_data_dir>` with the labeled directory verified in Step 2 and
`combined.csv` with the merged output from Step 3. The saved
`validation_report.json` has two top-level keys: `missing_locations` (labeled
jurisdictions absent from the run output) and `locations` (jurisdictions present
in the run, each with `missing_features` and per-field `feature_results`). A
`missing_locations` entry that is not in `jurisdiction_fp` is a benign coverage
gap in the labeled set, but one that *is* in `jurisdiction_fp` is critical: the
run was supposed to cover that jurisdiction and produced nothing (document not
found, parse failure, or run error), and it must be investigated before any
schema tuning. This report is the evidence used to diagnose and fine-tune the
schema.

## Step 5 — Find the most common feature with an issue

Use Dead-Reckoning's `qc triage` to rank the issues in the validation report and
recommend the highest-leverage target. Pass the `validation_report.json` from
Step 4 and restrict the analysis to the jurisdictions actually in scope for this
run by pointing `-s` at the same `jurisdiction_fp` CSV; triage drops the
out-of-scope labeled jurisdictions (labeled but never in `jurisdiction_fp`) so
they do not distort the ranking:

```
pixi run dead_reckoning qc triage validation_report.json -s <jurisdiction_fp.csv>
```

Replace `<jurisdiction_fp.csv>` with the same CSV named by `jurisdiction_fp` in
`config.json5`. Triage ranks each feature by how many in-scope jurisdictions show
an issue with it — a feature has an issue when it appears in a jurisdiction's
`missing_features` (the schema failed to extract it at all) or when its
`feature_results` entry has `passed: false` (extracted but a field check — value,
units, year, summary, or section — failed). The top of the output is the
recommended target, annotated with the `missing` versus `fail` split and the count
of affected jurisdictions. Add `--top N` to limit the list, or `-f json` to parse
the ranking programmatically.

Before trusting the ranking, check for a more serious failure that triage does not
rank: an in-scope jurisdiction that produced no output at all. Triage counts these
in `jurisdictions_in_scope` but excludes them from `jurisdictions_analyzed` (both
visible in the `-f json` `scope` block) and never names them, so a target that
yielded zero rows is invisible to the feature ranking. This is usually not a schema
problem but a document-not-found, parse-failure, or run error, and it is more
critical than any single feature miss. List any such jurisdiction by intersecting
the report's `missing_locations` with `jurisdiction_fp`:

```
pixi run python -c "
import csv, json
def k(s, c, su): return ((s or '').strip().lower(), (c or '').strip().lower(), (su or '').strip().lower())
scope={k(r.get('State'), r.get('County'), r.get('Subdivision')) for r in csv.DictReader(open('<jurisdiction_fp.csv>'))}
d=json.load(open('validation_report.json'))
crit=[m for m in d.get('missing_locations', []) if k(m.get('state'), m.get('county'), m.get('subdivision')) in scope]
print('CRITICAL in-scope jurisdictions missing from output:', len(crit))
for m in crit: print('  -', m.get('state'), m.get('county'), m.get('subdivision'), m.get('FIPS'))
"
```

Resolve any critical misses (or consciously set them aside) before spending effort
on the feature ranking.

The recommended target is the highest-leverage fix: correcting it resolves the
most jurisdictions at once. The `missing` versus `fail` split tells you which kind
of fix is needed — a `missing`-dominated feature usually means the schema is not
recognizing the provision (widen the feature description, `$scope`, or examples),
while a `fail`-dominated feature means it is recognized but a field is wrong
(tighten the value/units/section rules).

If only one or two jurisdictions are in scope, the ranking is not yet meaningful
(every feature ties at one). Widen `jurisdiction_fp` to more labeled jurisdictions
and re-run Steps 2-5 before drawing conclusions about which feature is truly most
common.

### 5.1 — Pick a target feature and a failing jurisdiction

Choose one feature from the top of the triage ranking (the recommended target, or
another high-ranking one; if several tie, pick any one, and prefer a `fail` over a
`missing` when you want richer report evidence to work from). From
`validation_report.json`, select one in-scope jurisdiction where that feature has an
issue — either it appears in that jurisdiction's `missing_features` or its
`feature_results` entry has `passed: false`. Note the jurisdiction's identity
(`state`, `county`, `subdivision`) and its `FIPS`, which the report carries on each
location. The FIPS code is the key used to find the source document in the next
sub-step.

### 5.2 — Identify the processed document via `known_local_docs`

The run extracted from a specific local document per jurisdiction, declared in the
`known_local_docs` file named by `config.json5`. That file maps each jurisdiction's
FIPS code to the document(s) that were processed, for example:

```
{
    "46039": [
        { "source_fp": "./pdf/Deuel_County_South_Dakota.pdf", "check_if_legal_doc": false },
    ],
}
```

Look up the failing jurisdiction's FIPS in `known_local_docs` to get its
`source_fp` — that PDF is the exact document the extraction saw. COMPASS also
writes the parsed plain text it actually fed to the model under the run's
`out_dir` (for example `output/cleaned_text/<Jurisdiction> Collected Text.txt`).
Prefer reading that collected-text file over the raw PDF, because it is what the
model saw — including any OCR degradation, which is itself a common root cause.

### 5.3 — Locate the relevant passage using the report

Do not read the whole document blindly. The report's checks for the failing
feature point straight at the passage. Use the `section` check keywords (for a
`missing` feature these are under `checks.section.keywords`; for a `fail` they are
in the `section` field's `expected`) and, when present, the `actual` section and
`summary` strings from the failing `feature_results`. Grep the collected-text file
for those keywords (section numbers such as `1248.01`, and phrases such as
`agricultural district` or `applicability`) to jump to the exact clause the label
was written against.

### 5.4 — Understand why the schema missed the feature

Read the located clause and compare what it says against the feature's definition
in the schema (its `description`, plus any relevant `$core_principles`, `$scope`,
and `$examples`). Classify the root cause:

- **Not recognized (missing).** The provision is present in the text but the schema
  told the model not to emit it — usually because the feature description is scoped
  too narrowly and excludes the phrasing the ordinance used. For example, the
  Deuel ordinance sets its voltage floor in an *applicability* clause
  ("the requirements of these regulations shall apply to overhead electrical
  transmission lines with a voltage greater than 345 kV"), but the `voltage
  threshold` description restricts extraction to a *definitional* statement of the
  term and explicitly excludes a voltage "used to impose a physical requirement
  rather than to define the term". The model followed the schema and omitted a
  value the label expected.
- **Recognized but wrong fields (fail).** The provision was extracted (summary and
  section populated) but `value`/`units` do not match. Check whether the schema's
  own `$examples` for that feature disagree with the label's expected `value`/
  `units` — a design-vs-label mismatch — versus the model simply failing to map the
  clause onto the field.
- **Document quality.** The clause did not survive PDF parsing (garbled or absent
  OCR text). This is not a schema fix; note it and, if needed, improve the source
  document or parsing rather than the schema.

### 5.5 — Propose a schema improvement

Write the smallest schema change that fixes the diagnosed cause without regressing
others, following the global-guard pattern used elsewhere in the schema:

- For a **narrow-scope miss**, widen the feature `description` to name the phrasing
  the ordinance used (for example, allow `voltage threshold` to come from an
  applicability or scope clause that sets which lines the ordinance governs, not
  only from a term definition), keep the existing exclusions that prevent false
  positives (per-requirement voltage tiers in tables), and add a short `$examples`
  entry demonstrating the newly-accepted phrasing with its expected `value`/`units`.
- For a **field-mapping fail**, tighten the value/units instruction for that feature
  and add an example that pins the expected `value`/`units`; if the mismatch is
  against the schema's own examples, reconcile the examples with the label
  convention first.
- Prefer editing a shared `$core_principles` guard when the same fix should apply to
  several features; prefer the single feature `description` when it is specific to
  one.

State the proposal (which construct changes, and the before/after wording) and,
after applying it, re-run Steps 2-5 on the same in-scope set to confirm the
targeted feature now passes and nothing else regressed.
