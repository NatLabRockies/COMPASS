---
name: iteration-development
description: Run → inspect → fix cycle for one-shot extraction after initial setup. Use whenever a user asks to diagnose poor output, reduce scope bleed, improve precision/recall, or scale from smoke tests.
---

# Iteration Development Skill

Use this skill after you have a working schema, plugin YAML, and run config
and want to improve extraction quality through systematic iteration.

## When to use

- First smoke run produced output that needs diagnosis or improvement.
- Feature values or units are wrong, missing, or inconsistent.
- Retrieval is returning off-target documents.
- Scaling from 3 jurisdictions to 10–25 or full production.

## Do not use

- First-time setup before any successful smoke run.
- Legacy decision-tree extraction development.

## Expected assistant output

When using this skill, return:

1. The observed failure class (retrieval, extraction scope, value/units, or null handling).
2. One concrete fix on a single axis.
3. The re-run command and pass/fail gate check.

## Canonical reference

- `examples/one_shot_schema_extraction/` — working examples
  to use as a baseline for comparing output quality.


## The run → inspect → fix loop

**Three Phases:** This skill guides you through three phases, all built into
example plugin configurations in the `examples/` directory.

Repeat this cycle once per iteration. Change exactly one axis per cycle.

```
Run → Inspect outputs → Identify failure → Fix one axis → Re-run same sample
```

**Never change multiple axes in the same iteration.** You will not know
which change caused the result.

**Phases encoded in plugin YAML comments:**

- **Phase 1 (Initial):** Includes query templates, website keywords, and
  basic heuristic filters to avoid obvious off-domain results.
  **This is ready to run immediately.**
- **Phase 2 (Optional Refinement):** Uncomment advanced heuristic tuning
  if Phase 1 retrieval produces off-target documents.
- **Phase 3 (Optional Refinement):** Uncomment extraction_system_prompt
  if Phase 1-2 retrieval works but extracted features are wrong (scope bleed).

Start with Phase 1. Only add Phase 2 / 3 if Phase 1 results need improvement.
See README.rst for the progression path.


## Step 1: Inspect output artifacts

After each run, check these locations inside `out_dir`:

| Artifact | What to look for |
|---|---|
| `ordinance_files/*.pdf` | Are these on-target documents? |
| `cleaned_text/*.txt` | Does page text contain target technology language? |
| `jurisdiction_dbs/*.csv` | Are feature rows present? Are values correct? |
| `quantitative_ordinances.csv` and `qualitative_ordinances.csv` | Final compiled output — check feature coverage and null rate |
| `logs/<jurisdiction>/*.log` | Error messages, 0-document warnings |

Minimum passing state for a smoke run:
- At least one `ordinance_files/` PDF per jurisdiction.
- At least one `cleaned_text/` file per jurisdiction.
- Compiled ordinance CSV outputs contain rows for most jurisdictions.

Immediate fail conditions (fix before any tuning):
- Jurisdiction CSV header mismatch (must include at least `County,State`).
- Plugin configuration exceptions in logs (for example missing required
  `heuristic_keywords` lists).
- `Number of jurisdictions with extracted data: 0`.


## Step 2: Classify the failure

Use this decision tree for any defect:

```
Is the right document being retrieved?
  └─ No → retrieval problem → fix query templates / heuristic_keywords
  └─ Yes
       Is the document text present in cleaned_text/?
         └─ No → text extraction problem → check PDF quality / OCR
         └─ Yes
              Are the right features being extracted?
                └─ No, wrong feature names → schema enum or description problem
                └─ No, off-domain features → scope bleed → fix extraction_system_prompt
                └─ Yes, but wrong values/units → schema description or units problem
                └─ Yes, but nulls where values should be → schema IGNORE clause too broad
```


## Step 3: Fix the right axis

### Retrieval problems (wrong or missing documents)

Fix in plugin YAML:
- Add more specific `query_templates` with legal code terms
  (e.g., `"filetype:pdf {jurisdiction} generator zoning code"`).
- Add target technology terms to `GOOD_TECH_KEYWORDS` and
  `GOOD_TECH_PHRASES`.
- Add adjacent-technology terms being confounded to `NOT_TECH_WORDS`.
- Increase `website_keywords` score for the most discriminating terms.

Required `heuristic_keywords` keys when present:
- `GOOD_TECH_KEYWORDS`
- `GOOD_TECH_PHRASES`
- `GOOD_TECH_ACRONYMS`
- `NOT_TECH_WORDS`

### Scope bleed (off-domain features extracted)

Fix in plugin YAML `extraction_system_prompt`:
- State explicitly what is excluded (e.g., "Do not extract requirements for
  residential portable generators").
- Add the same language to `$instructions.scope` in the schema for
  reinforcement.

### Wrong values or units

Fix in schema JSON, in the affected feature's `description`:
- Add or sharpen the `VALUE` rule.
- Expand the `UNITS` vocabulary list.
- Add a `IGNORE` clause for the near-miss case.

### Missing values (nulls where data exists)

Fix in schema JSON:
- Broaden the feature description to cover the phrasing used in source
  documents.
- Remove overly restrictive IGNORE clauses.
- Check that the feature ID is spelled exactly as it appears in the enum.

### Text extraction failures (blank cleaned_text)

- Verify the PDF is readable (not scanned without OCR).
- Add `from_ocr: true` to the doc entry in `known_local_docs`.
- Set `pytesseract_exe_fp` in run config if OCR is needed.


## Iteration hygiene

- Use a **unique `out_dir`** per iteration run. COMPASS aborts early if the
  output directory already contains results.
- Keep the same small jurisdiction sample across all iterations until
  quality gates pass.
- Record what changed and why in a short comment in the config file or
  a separate `CHANGELOG.md` in the example folder.
- Save schema versions as `<tech>_schema_v2.json` etc. to
  preserve a diff history. Point `schema:` in plugin YAML to the active
  version.


## Scale-up protocol

Only advance to the next phase when the current phase passes all gates.

| Phase | Jurisdictions | Gates |
|---|---|---|
| Smoke | 1–3 | Output rows exist; feature names match schema enum; section/summary present for most rows |
| Robustness | 10–25 | Feature value types are stable; null rate is explainable; no scope bleed |
| Production | Full national set | False positive/negative rates acceptable; repeated runs show minimal drift |

When advancing, keep the same config files. Only change the jurisdictions CSV.


## Diagnostic commands

```bash
# Check if cleaned text was produced
ls outputs/*/cleaned_text/

# Count output rows per jurisdiction
wc -l outputs/*/jurisdiction_dbs/*.csv

# Check for scope bleed — feature values that are off-domain
grep -v "diesel\|generator\|backup\|emergency" outputs/ordinances.csv | head -20

# View logs for a specific jurisdiction
cat outputs/logs/San\ Diego*/run.log | grep -i "error\|warning\|found 0"
```


## Common failure modes

| Symptom | Most likely cause | Fix axis |
|---|---|---|
| 0 documents for all jurisdictions | Credentials not loaded / search API down | Load `.env`; use `known_doc_urls` |
| Downloaded PDFs are from wrong domain | `query_templates` too generic | Narrow queries with `filetype:pdf` and legal code terms |
| `cleaned_text` present but no output CSV rows | Schema enum mismatch or extraction prompt failing | Check schema path in plugin YAML; verify `tech` value in run config |
| Off-domain feature names in output | Scope bleed from large land-use code | Add exclusion language to `extraction_system_prompt` |
| Correct features but wrong values | Feature description lacks VALUE rule | Add explicit VALUE rule to affected descriptions |
| Setback in wrong units | UNITS rule missing or implicit | Add explicit UNITS vocabulary to description |
| Null rows for features that are in the document | IGNORE clause too broad, or feature description doesn't match source phrasing | Broaden description; remove over-strict IGNORE clause |
| Playwright timeout errors in logs | Website crawl phase browser failure | Non-fatal; COMPASS continues. Use `known_doc_urls` while iterating |


## Acceptance criteria before promotion

A technology is ready to promote from `examples/` to
`compass/extraction/<tech>/` when all of the following are true on the
robustness run (10–25 jurisdictions):

- [ ] Output CSV rows conform to required schema contract.
- [ ] Feature IDs are stable and match the schema enum exactly.
- [ ] Most non-null rows include a useful `section` and `summary`.
- [ ] Repeated runs on the same sample show minimal drift.
- [ ] No scope bleed (off-domain features) is observed.
- [ ] Null rate for common features is explainable (jurisdiction has no rule).
