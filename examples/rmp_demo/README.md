# COMPASS BLM RMP Geothermal Restriction Extraction

This directory shows how to run COMPASS to extract geothermal leasing restrictions
from BLM Resource Management Plan (RMP) documents using the
[one-shot schema extraction approach](./one-shot/).

## Overview

RMPs are multi-document federal land-use plans issued by BLM field offices. Each
plan may contain closures, No Surface Occupancy (NSO) stipulations, seasonal timing
restrictions, wildlife/water buffers, and other conditions that directly constrain
geothermal leasing, exploration, and drilling.

This example targets the **Carson City Field Office Consolidated RMP** (Nevada) as
a demonstration, but the same plugin config and schema apply to any BLM RMP.

## Quick Start

```bash
compass process-c examples/rmp_demo/one-shot/config.json5
```

Or with pixi:

```bash
pixi run --manifest-path pixi.toml \
    compass process \
    -c examples/rmp_demo/one-shot/config.json5
```

## Files

| File | Description |
|------|-------------|
| `jurisdictions.csv` | List of jurisdictions (County, State) to process |
| `one-shot/config.json5` | Main COMPASS run configuration |
| `one-shot/local_docs.json5` | Template for pointing COMPASS at local RMP PDF files |

## Using Local PDFs

RMP documents are typically downloaded in advance rather than web-crawled. Edit
`one-shot/local_docs.json5` to point to your local PDF files and set
`known_local_docs` in `config.json5` to that file path.

See the [COMPASS documentation](https://nrel.github.io/COMPASS/) for more details
on local document configuration.
