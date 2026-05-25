# Eval datasets

These are the ground-truth datasets for the date-extraction evals. Both are
committed in plaintext and split from the same source labels.

## `dev/` vs `held-out/`

| | `dev/` | `held-out/` |
| --- | --- | --- |
| Cadence | run frequently during development (`-m dev_eval`) | run before a release (`-m held_out`) |
| Purpose | iterate, tune prompts/logic, debug failures | unbiased estimate of true performance |
| Size | ~70% of the labeled cases | ~30% of the labeled cases |

The split samples **30% from cases where a year exists** and **30% from cases
where it does not**, so both datasets keep the same mix of "has a date" and
"no date" cases.

## Philosophy (please read)

The held-out set is meant to give an **honest** read on how well date
extraction actually works. That only holds if we *don't* tune against it:

- **Do not** run `held_out` repeatedly while iterating — use `dev` for that.
- **Do not** inspect held-out failures to "fix" the extractor for those
  specific documents. The moment you optimize against the held-out set, it
  stops being held-out and its numbers become optimistic.
- Treat `held_out` as a checkpoint you look at occasionally (e.g. before a
  release), not a development loop.

The harness helps enforce this: a `held_out` run writes **only summary
metrics** (`held_out_eval_metrics.csv`) — no per-case breakdown, and per-case
predictions are not logged — so there is nothing to eyeball or tune against.
`dev` runs write the full per-case breakdown.

It is committed in plaintext (not encrypted/hidden) on purpose: we trust
developers to follow the above rather than adding friction. If we later find
this trust is being abused (too-frequent held-out runs, tuning against it),
we can revisit (e.g. move held-out behind encryption or a separate location).

## Layout

```
dev/solar_validation_files/
  manifest.json5      # [{fips, jurisdiction, file, source, expected_year}, ...]
  <documents>         # the ordinance PDFs/text files referenced by the manifest
held-out/solar_validation_files/
  manifest.json5
  <documents>
```

`expected_year: null` means the ground truth is "no enactment date exists" —
the extractor should return no year for that document.
