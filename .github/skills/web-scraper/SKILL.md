---
name: web-scraper
description: Build and tune one-shot plugin configs that search, rank, and collect ordinance documents with native COMPASS pipeline settings.
---

# Web Scraper Skill

Use this skill to improve retrieval precision/recall before extraction tuning.

## When to use

- Download step returns noisy sources.
- Ordinance recall is weak across jurisdictions.
- LLM filtering is compensating for poor search quality.

## Scope

- Query-template strategy.
- URL ranking and filtering patterns.
- Heuristic phrase controls before LLM validation.

## Example references (optional)

- `examples/one_shot_schema_extraction_geothermal/geothermal_plugin_config.yaml`
- `examples/one_shot_schema_extraction_geothermal/geothermal_config.json5`
- `examples/one_shot_schema_extraction_geothermal/geothermal_jurisdictions_one.csv`
- `examples/one_shot_schema_extraction_cst/cst_plugin_config.yaml`
- `examples/compass_tech_pipeline/README.md`

## Two retrieval phases

COMPASS runs two sequential acquisition passes per jurisdiction:

1. **Search-engine phase** — queries `SerpAPIGoogleSearch` (or configured
   engine) using `query_templates`. This phase is the primary source of
   ordinance documents.
2. **Website crawl phase** — crawls the jurisdiction's official website,
   ranking pages using `website_keywords`. This phase is a secondary pass
   and runs even if the SE phase found documents.

Key behaviors:
- Playwright browser errors during the website crawl phase are **non-fatal**.
  COMPASS logs the error and continues.
- `Found 0 potential documents` at the end of the crawl phase is **expected**
  for jurisdictions without relevant online ordinances.
- Disable the crawl phase with `perform_website_search: false` in run config
  when you want faster smoke tests or Playwright is unavailable.

## Key management

For SerpAPI-backed search, keep `api_key` out of committed config and provide
`SERPAPI_KEY` via environment (for example through `.env` loaded in shell).

Recommended shell setup:

```bash
set -a
source .env
set +a
```

Avoid spaces around `=` in `.env` assignments.

## Retrieval design pattern

1. Create 3-7 jurisdiction queries with `{jurisdiction}`.
2. Weight legal document indicators in URL keywords.
3. Apply exclusions for templates/reports/slides.
4. Add focused negative tech terms to reduce false positives.
5. Start with dynamic search, then switch to deterministic known URLs when
  search infrastructure is unstable.

For first-pass reliability, test retrieval with deterministic known URLs
before using live web search.

## Technology-specific retrieval controls (template)

- Include target-technology facility/deployment terms.
- Exclude adjacent and non-target terms (residential/HVAC/PV/etc as needed).
- Favor jurisdictional legal-code signals like `land use code`,
  `code of ordinances`, `use table`, and `special use permit`.

## Deterministic smoke-test mode

Use run-config controls to bypass flaky search while tuning:

- supply `known_doc_urls` or `known_local_docs`,
- set `perform_se_search: false`,
- set `perform_website_search: false`.

Then validate:

- download artifacts exist,
- cleaned text exists,
- ordinance DB rows are non-empty.

## Tuning loop

1. Run SE-search phase on small sample.
2. Inspect kept vs discarded PDFs (`ordinance_files/`).
3. Run heuristic filter and review false rejects/accepts (`cleaned_text/`).
4. Check website crawl phase independently if needed (enable, run, inspect logs).
5. Update one axis only:
	- query templates (affects SE phase),
	- URL weights (affects both phases),
	- include/exclude heuristic patterns (pre-LLM filter),
	- `not_tech_words` (upstream document rejection).
6. Re-run same sample and compare.

## Cross-tech onboarding

When reusing this workflow for any technology:

- keep legal retrieval tokens (`ordinance`, `zoning`, `code`),
- replace all technology terms in `query_templates`, `website_keywords`,
  and `heuristic_keywords`,
- seed `known_doc_urls` with authoritative regulatory documents for smoke
  testing,
- avoid copying negatives from previous technologies into the new tech config,
- verify `not_tech_words` excludes adjacent technologies for your domain.

## Phase gates

- **3 jurisdictions**: ensure major source classes are found.
- **10-25 jurisdictions**: verify stability across regions.
- **Full scale**: only once false positive/negative rates stabilize.

## Guardrails

- Keep feature extraction logic out of retrieval config.
- Do not overfit to one county's document style.
- Preserve auditable rationale for each retrieval change.
- Keep one canonical retrieval config per active technology.
- Ensure each run uses a unique `out_dir` to avoid COMPASS aborting early.
