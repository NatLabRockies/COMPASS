---
name: schema-creation
description: Author and iterate one-shot extraction schemas that replace legacy decision-tree extraction logic in native COMPASS.
---

# Schema Creation Skill

Use this skill to encode extraction logic in schema so behavior is repeatable
across jurisdictions and technologies.

## When to use

- Creating a new one-shot technology plugin.
- Migrating from decision-tree logic to schema-driven extraction.
- Stabilizing inconsistent model outputs.

## Example references (optional)

- `examples/one_shot_schema_extraction_geothermal/geothermal_schema.json`
- `examples/one_shot_schema_extraction_geothermal/README.rst`
- `examples/one_shot_schema_extraction_geothermal/geothermal_one_shot_guide.md`
- `docs/source/examples/one_shot_schema_extraction/wind_schema.json`

## Required output contract

Top-level object must define `outputs` and each item must require:

- `feature`
- `value`
- `units`
- `section`
- `summary`

```json
{
  "type": "object",
  "required": ["outputs"],
  "properties": {
    "outputs": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["feature", "value", "units", "section", "summary"],
        "additionalProperties": false
      }
    }
  }
}
```

## Build sequence

1. Copy baseline schema and rename for target tech.
2. Replace `feature` enum with target-tech IDs.
3. Define `value`/`units` rules per feature family.
4. Add `$definitions` for reusable decision logic.
5. Add `$examples` for top failure modes.
6. Add `$instructions` for global extraction policy.

For new technologies (for example CHP or CST), clone a working schema and
perform a strict vocabulary swap (features, units, exclusions) before adding
new logic.

## Output column mapping

Schema field names map directly to the final output CSV columns:

| Schema field | CSV column |
|---|---|
| `feature` | `feature` |
| `value` | `value` |
| `units` | `units` |
| `section` | `section` |
| `summary` | `summary` |

Additional columns added by COMPASS finalization: `county`, `state`,
`subdivision`, `jurisdiction_type`, `FIPS`, `adder`, `min_dist`, `max_dist`,
`year`, `source`. These do not need to appear in the schema.

## Scope bleed from generic legal documents

When COMPASS retrieves a large generic land-use code rather than a
technology-specific ordinance, the LLM may extract provisions that are
outside the schema enum. This is most visible when unfamiliar feature names
appear in the output CSV.

Primary controls:
- `extraction_system_prompt` in plugin YAML — this is the strongest signal.
  State explicitly what is in scope and what is out.
- `$instructions.scope` in schema — reinforce exclusion language here.
- `heuristic_keywords.not_tech_words` — filter documents upstream.

Do not widen the feature enum to accommodate scope bleed; narrow the prompt
and upstream filters instead.

## Technology adaptation guidance

When adapting a baseline schema to any new technology:

- Separate core utility-scale requirements from adjacent/non-target systems.
- Keep district/permit features distinct from numerical constraints.
- Encode jurisdiction/governance handling where relevant in summaries.
- Require explicit nulls when a feature is not enacted.

## Cross-technology adaptation checklist

Apply this for any new domain:

1. Define technology-specific `feature` enum with stable IDs.
2. Define allowed unit vocabulary for each feature family.
3. Add explicit exclusion language for adjacent-but-out-of-scope systems.
4. Ensure summaries preserve legal traceability (section + source-faithful text).
5. Validate on deterministic docs before tuning retrieval.
6. Consider including `enactment date` in the enum — COMPASS naturally surfaces it
   from documents and it provides important temporal context in outputs.

## Example specialization patterns (optional)

Use examples only to shape exclusion strategy:

- separate core utility-scale requirements from adjacent technologies,
- add explicit exclusion terms in `not_tech_words`,
- preserve legal traceability via `section` and `summary`.

## Reuse safeguards

- Keep tech-first file names consistent across assets:
  `<tech>_config*.json5`, `<tech>_plugin_config.yaml`,
  `<tech>_schema.json`, `<tech>_jurisdictions*.csv`.
- Keep credentials out of schema content and examples.
- Validate schema behavior with a small smoke run before scaling.

## High-value authoring patterns

- Put restrictive-value selection rules directly in descriptions.
- Explicitly define accepted unit vocabulary.
- Clarify near-miss terms that should not be treated as equivalent.
- State whether qualitative features should keep `value`/`units` null.

## Anti-patterns

- Retrieval instructions embedded in schema semantics.
- Feature IDs that change names across iterations.
- Implicit unit assumptions not declared in text.
- Examples that contradict field descriptions.
- Feature enums that include placeholders with no extraction logic.

## Quality checklist

- Enum matches target output columns.
- Every feature has deterministic extraction rules.
- `section` and `summary` preserve legal traceability.
- Repeated sample runs produce stable feature typing.

## Iteration loop

1. Run 3-jurisdiction smoke sample.
2. Catalog failure modes by feature.
3. Patch only affected descriptions/examples.
4. Re-run same sample before expanding scope.

Save iterated schema versions as `<tech>_schemav2.json`, `<tech>_schemav3.json`
etc. to preserve a diff history. The active version is what `schema:` in the
plugin YAML points to.

## Practical quality signal

Treat a schema as "working" when all are true on the smoke sample:

- final ordinance CSV outputs are non-empty,
- extracted rows include stable feature IDs,
- most non-null rows have useful `section` and `summary`,
- repeated runs do not shift feature semantics materially.
