---
name: schema-creation
description: Author and iterate one-shot extraction schemas that replace legacy decision-tree extraction logic in native COMPASS.
---

# Schema Creation Skill

**ONE-SHOT EXTRACTION ONLY.** This skill applies only to schema-driven extraction
(new technology onboarding with JSON schema + plugin YAML). For legacy decision-tree
extraction (existing solar/wind/small-wind in `compass/extraction/<tech>/`),
consult COMPASS architecture docs.

Use this skill to define what the LLM extracts and how it formats results.
The schema is the single most important config file for output quality.

## When to use

- Starting a new one-shot technology extraction (NOT decision-tree legacy extraction).
- Fixing inconsistent or incorrect extracted values in one-shot extraction.
- Adding new features to an existing one-shot extraction.

## Canonical reference

For complete examples, see the `examples/` directory:
- `examples/one_shot_schema_extraction/wind_schema.json`
- `examples/water_rights_demo/one-shot/water_rights_schema.json5`

Each follows the pattern: `<tech>_schema.json` or `<tech>_schema.json5`.

## Required output contract

Every schema must define `outputs` as an array. Each item must require
exactly these five fields and set `additionalProperties: false`:

```json
{
  "type": "object",
  "required": ["outputs"],
  "additionalProperties": false,
  "properties": {
    "outputs": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["feature", "value", "units", "section", "summary"],
        "additionalProperties": false,
        "properties": {
          "feature": { "type": "string", "enum": ["..."] },
          "value":   { "anyOf": [{"type": "number"}, {"type": "string"}, {"type": "boolean"}, {"type": "array", "items": {"type": "string"}}, {"type": "null"}] },
          "units":   { "type": ["string", "null"] },
          "section": { "type": ["string", "null"] },
          "summary": { "type": ["string", "null"] }
        }
      }
    }
  }
}
```

These five fields map directly to the output CSV columns. COMPASS adds
`county`, `state`, `FIPS`, and other metadata columns automatically.

## Build sequence

1. **Define the feature enum** — one stable lowercase ID per siting-relevant
   requirement. Group IDs by family (setbacks, noise, zoning, permitting).
2. **Define `value` and `units` rules per feature family** — in each
   feature's `description`, state the expected value type and accepted unit
   vocabulary explicitly.
3. **Add `$definitions`** — group related feature descriptions here to keep
   the `feature` enum block clean.
4. **Add `$instructions`** — encode global extraction policy (scope, null
   handling, one-row-per-feature contract, verbatim quote preference).
5. **Smoke-test on one jurisdiction** — validate all enum items appear in
   output and null rows are correctly populated for missing features.

## Feature definition template

Every feature description must answer four questions:

1. **What is this?** One sentence identifying the regulatory concept.
2. **VALUE rule:** What type is the value and what specific values/ranges are
   valid?
3. **UNITS rule:** What unit string is accepted, or `null` if not applicable?
4. **IGNORE / CLARIFICATION:** What near-miss concepts must NOT match this
   feature?

Example (abbreviated):

```json
"structure setback": {
  "description": "Minimum distance from the generator to an occupied building. VALUE: numerical distance. UNITS: 'feet' or 'meters'. IGNORE: setbacks from property lines or roads — those are separate features."
}
```

## Feature family taxonomy

Organize `$definitions` by these families:

| Family | Example features |
|---|---|
| Setbacks | `structure setback`, `property line setback`, `road setback` |
| Noise/Emissions | `noise limit`, `emissions standard`, `vibration limit` |
| Operational | `hours of operation` |
| Physical design | `screening requirement`, `enclosure requirement`, `exhaust stack height` |
| Zoning | `primary use districts`, `conditional use districts`, `prohibited use districts` |
| Permitting | `permit requirement`, `capacity threshold` |
| Compliance | `decommissioning`, `enactment date` |

## `$instructions` block

Always include a `$instructions` object at the top level with these keys:

```json
"$instructions": {
  "scope": "Describe exactly what to extract and what to ignore.",
  "null_handling": "Output every enum feature. Use null value and null summary when a feature is not found in the document. Do not omit features.",
  "one_row_per_feature": "Output exactly one row per feature. If multiple values apply, use the most restrictive and describe variants in summary.",
  "verbatim_quotes": "In summary fields, prefer verbatim quotes from the source. Enclose in double quotation marks.",
  "units_discipline": "Do not convert units. Record them exactly as they appear in the document."
}
```

## Scope bleed control

When COMPASS retrieves a large land-use code instead of a tech-specific
ordinance, the LLM may extract off-domain provisions.

Fix order (most powerful first):
1. `extraction_system_prompt` in plugin YAML — state explicitly what is in
   scope and what is excluded.
2. `$instructions.scope` in schema — reinforce with exclusion language.
3. `heuristic_keywords.NOT_TECH_WORDS` — reject documents upstream.

Do not expand the feature enum to absorb scope bleed. Narrow the prompt.

## Cross-technology adaptation checklist

When cloning this schema for a new technology:

- [ ] Replace all feature IDs with technology-specific names.
- [ ] Replace value/units rules in every feature description.
- [ ] Replace exclusion terms in `$instructions.scope` and feature IGNORE
      clauses.
- [ ] Replace `$definitions` group names to match new feature families.
- [ ] Smoke-test before widening to 10+ jurisdictions.

## Quality checklist

- [ ] Feature enum uses stable, lowercase, underscore-separated IDs.
- [ ] Every feature description contains VALUE, UNITS, and IGNORE clauses.
- [ ] `$instructions` block is present with all five keys.
- [ ] `additionalProperties: false` is set on the top-level object and on
      each item in the `outputs` array.
- [ ] Schema validates cleanly against a JSON Schema validator.
- [ ] A smoke run using this schema produces extracted rows (not just
   successful process exit logs).

## Anti-patterns to avoid

- Feature IDs that change names between iterations.
- Implicit unit assumptions not stated in description text.
- Missing IGNORE clauses for common near-miss features.
- Examples in descriptions that contradict field rules.
- Widening the enum to absorb scope bleed instead of tightening the prompt.
