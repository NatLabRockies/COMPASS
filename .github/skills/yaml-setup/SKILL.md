---
name: yaml-setup
description: Author and tune one-shot plugin YAML for COMPASS document discovery, filtering, and text collection. Use whenever a user asks to create, clean up, standardize, or troubleshoot one-shot plugin YAML for technology onboarding.
---

# YAML Setup Skill

**ONE-SHOT EXTRACTION ONLY.** This skill applies only to schema-driven extraction.
For legacy decision-tree extraction, consult COMPASS architecture docs.

Use this skill to create or tune one-shot plugin YAML that controls retrieval,
filtering, and text collection behavior.

## When to use

- New technology onboarding in one-shot extraction (NOT decision-tree extraction).
- Schema exists but source relevance is weak.
- You need reproducible config handoff across teams.

## Do not use

- Legacy decision-tree parser implementation changes.
- Schema feature semantics work that belongs in `<tech>_schema.json`.
- Run-result diagnosis after outputs are generated (use iteration loop skill).

## Expected assistant output

When using this skill, return:

1. The finalized plugin YAML content or exact diff.
2. Any required paired run-config changes.
3. A validation command and pass/fail checks for the edited YAML.

## Canonical reference

Consult the working examples in `examples/`:
- `examples/one_shot_schema_extraction/plugin_config.yaml` — standard working example
- `examples/water_rights_demo/one-shot/plugin_config.yaml` — multi-doc edge case

When creating new tech configs, `<tech>_plugin_config.yaml` is the recommended
naming convention (e.g. `geothermal_plugin_config.yaml`). The existing
`plugin_config.yaml` examples use a generic name; new tech-specific assets
should use the tech-first naming pattern.

Refer to any complete example in `examples/` that matches your retrieval goals.

## Naming convention

Use tech-first file names when creating new one-shot assets:
`<tech>_config*.json5`, `<tech>_plugin_config.yaml`,
`<tech>_schema.json`, `<tech>_jurisdictions*.csv`.

## Secret handling

Keep API keys in environment variables (for example `SERPAPI_KEY`,
`AZURE_OPENAI_API_KEY`) rather than in plugin or run config files.
Load them per shell session with `set -a && source .env && set +a`.
Avoid spaces around `=` in `.env` assignments.

## Required minimum

```yaml
schema: ./my_schema.json
```

## Non-negotiable runtime constraints

- Jurisdiction CSV headers are case-sensitive: use `County,State`.
- If `heuristic_keywords` is present, it must include all four lists and
  none may be empty.
- A run is not considered passing if logs show config errors or if
  extracted jurisdiction count is zero.

## Key plugin YAML fields

| Field | Type | Behavior |
|---|---|---|
| `schema` | string (path) | **Required.** Path to JSON schema file, relative to plugin YAML location. |
| `data_type_short_desc` | string | Short description used in LLM prompts (e.g. `utility-scale <tech> ordinance`). |
| `query_templates` | list | Search query templates; `{jurisdiction}` is replaced at runtime. |
| `website_keywords` | dict | Keyword → score map for URL ranking during website crawl. |
| `heuristic_keywords` | dict or `true` | Pre-LLM text filter. If `true`, LLM generates lists from schema. |
| `collection_prompts` | list or `true` | Text collection prompt(s). If **`true`**, LLM auto-generates from schema. |
| `text_extraction_prompts` | list or `true` | Text consolidation prompt(s). If **`true`**, LLM auto-generates from schema. |
| `extraction_system_prompt` | string | Overrides default LLM system prompt for the extraction step. Use this to scope extraction tightly to the target technology. |
| `cache_llm_generated_content` | bool | Cache LLM-generated `query_templates`, `website_keywords`, and `heuristic_keywords`. Set to `false` when iterating schema to see live changes. |

## Required `heuristic_keywords` shape

Use this exact structure when defining `heuristic_keywords`:

```yaml
heuristic_keywords:
  GOOD_TECH_KEYWORDS:
    - "<required single-word term>"
  GOOD_TECH_PHRASES:
    - "<required multi-word phrase>"
  GOOD_TECH_ACRONYMS:
    - "<required acronym or short token>"
  NOT_TECH_WORDS:
    - "<required exclusion term>"
```

Notes:
- Keys are normalized, but using canonical key names reduces mistakes.
- All four lists are required and must be non-empty.

### `collection_prompts: true` and `text_extraction_prompts: true`

Setting either flag to `true` (not a list) instructs COMPASS to use the LLM
to auto-generate the prompts from the schema content. This is the recommended
shortcut during development — do not write manual prompt lists until
auto-generated ones prove insufficient.

### `extraction_system_prompt`

This is the primary control for preventing scope bleed from generic land-use
code documents. Write it as a multi-line YAML literal block:

```yaml
extraction_system_prompt: |-
  You are a legal scholar extracting structured data from
  utility-scale <tech> ordinances.

  Extract only enacted requirements for utility-scale <tech> facilities.
  Exclude adjacent technologies and non-target use cases.
  Prefer explicit values. Use null for qualitative obligations.
```

See `compass/extraction/ghp/plugin_config.yaml` for a complete example.

## Progressive config path

1. **Minimal**
   - Confirm schema path and extraction invocation work.
2. **Simple**
   - Add `query_templates`, `heuristic_keywords`, and `cache_llm_generated_content`.
3. **Full**
   - Add `extraction_system_prompt` if scope bleed or off-domain extraction
     is observed.
   - Set `collection_prompts: true` and `text_extraction_prompts: true` to
     let the LLM auto-generate prompts from the schema.
   - Replace `heuristic_keywords: true` with an explicit list if precision
     is insufficient.

Use the same progression for any technology.

## Baseline YAML pattern

```yaml
schema: ./my_schema.json
data_type_short_desc: utility-scale <tech> ordinance
cache_llm_generated_content: true
query_templates:
  - "filetype:pdf {jurisdiction} <tech> ordinance"
  - "{jurisdiction} <tech> zoning ordinance"
  - "{jurisdiction} <tech> permitting requirements"
website_keywords:
  pdf: 92160
  <tech>: 46080
  ordinance: 23040
  zoning: 2880
  permit: 1440
heuristic_keywords:
  GOOD_TECH_KEYWORDS:
    - "<tech keyword 1>"
    - "<tech keyword 2>"
  GOOD_TECH_ACRONYMS:
    - "<tech acronym>"
  GOOD_TECH_PHRASES:
    - "<tech phrase 1>"
    - "<tech phrase 2>"
  NOT_TECH_WORDS:
    - "<adjacent technology term 1>"
    - "<adjacent technology term 2>"
```

Swap vocabulary for any technology while keeping the same structure.

## Stable development mode

Use run-config controls for deterministic smoke tests while iterating schema:

- `known_doc_urls` or `known_local_docs` — bypass live search
- `perform_se_search: false` — disable search-engine phase
- `perform_website_search: false` — disable website crawl phase

Re-enable search only after extraction quality is stable on known documents.

Recommended baseline: use dynamic search first, then use deterministic mode
if search infrastructure fails.

## Minimal run-config contract (to pair with plugin YAML)

Use this pattern and require users to provide their own model and client
values:

```json5
{
  out_dir: "./outputs_<tech>_<run_id>",
  tech: "<tech>",
  jurisdiction_fp: "./<tech>_jurisdictions.csv",
  perform_se_search: true,
  perform_website_search: false,
  model: [
    {
      name: "<PROVIDE-YOUR-MODEL-NAME>",
      llm_call_kwargs: { temperature: 0, timeout: 600 },
      client_kwargs: {
        api_version: "<PROVIDE-YOUR-API-VERSION>",
        azure_endpoint: "<PROVIDE-YOUR-AZURE-ENDPOINT>"
      }
    }
  ]
}
```

Do not hardcode model names in skills. Prompt the user to supply `name`.

## Acquisition phases

COMPASS acquisition runs in two sequential phases per jurisdiction:

1. **Search-engine phase** — uses `SerpAPIGoogleSearch` or similar; driven by
   `query_templates`.
2. **Website crawl phase** — crawls the jurisdiction's main website using
   `website_keywords` for ranking. Playwright browser errors during this
   phase are **non-fatal**; COMPASS logs them and moves on.

`perform_website_search: false` skips phase 2. Use it during smoke tests to
keep run time short and avoid Playwright dependency issues.

## Validation checklist

- Schema path resolves from runtime working directory.
- Query templates include `{jurisdiction}` consistently.
- URL weights favor legal and government documents.
- Heuristic exclusions are precise and not over-broad.
- Prompt overrides are only added when default behavior fails.

## Cross-tech adaptation checklist

When adapting to another technology:

- replace vocabulary in `query_templates` and `website_keywords`,
- keep legal-code terms (`ordinance`, `zoning`, `code of ordinances`),
- keep non-target exclusions explicit in `NOT_TECH_WORDS`,
- do not carry terms from a previous technology into new tech configs,
- write a technology-specific `extraction_system_prompt`.

## Run command

```bash
pixi run compass process -c config.json5 -p path/to/plugin_config.yaml -v
```

If running outside the tech folder, use absolute paths for `-c` and `-p`.

## Guardrails

- Retrieval behavior belongs in plugin YAML.
- Feature logic belongs in schema.
- Adjust one tuning axis per run for clean attribution.
- Keep one canonical plugin file per technology in the active example folder.

