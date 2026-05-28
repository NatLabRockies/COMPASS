"""Pull out websites for jurisdictions using Wikipedia"""  # noqa

import re
import sys
import asyncio
from asyncio import Semaphore
from datetime import datetime, UTC

import pandas as pd
import numpy as np
from docling.document_converter import DocumentConverter
from elm.web.search.dux import DuxDistributedGlobalSearch

from compass.utilities.jurisdictions import jurisdictions_from_df
from compass.utilities.finalize import _elapsed_time_as_str  # noqa


def _pull_out_website(markdown):
    for line in markdown.splitlines():
        if "Website" not in line:
            continue
        match = re.search(r"\|\s*Website\s*\|.*?\((https?://[^)]+)\)", line)
        if match:
            return match.group(1)
    return None


def _convert_link_to_website(link):
    converter = DocumentConverter()
    doc = converter.convert(link).document
    out = doc.export_to_markdown()
    return _pull_out_website(out)


async def _run_one_jurisdiction(jurisdiction, sem):
    async with sem:
        try:
            website = await _find_one(jurisdiction)
        except Exception as e:  # noqa
            print(
                f"Error processing {jurisdiction.full_name}: {e}", flush=True
            )
            website = None

    return jurisdiction.code, website


async def _find_one(jurisdiction):
    print(f"Processing {jurisdiction}...", flush=True)
    se = DuxDistributedGlobalSearch(region="us-en", timeout=10, verify=False)
    links = await se.results(f"site:wikipedia.org {jurisdiction.full_name}")
    links = links[0]
    if not links:
        print(f"{jurisdiction}:\n\t- No links found", flush=True)
        return None

    link = links[0]
    name = jurisdiction.subdivision_name
    if not name:
        name = jurisdiction.county

    if (
        name
        and name.replace(" ", "_").replace("-", "_").casefold()
        not in link.casefold()
    ):
        print(
            f"{jurisdiction}:\n\t- {link}\n\t- Not a match on name ({name})",
            flush=True,
        )
        return None

    if jurisdiction.state.casefold() not in link.casefold():
        print(
            f"{jurisdiction}:\n\t- {link}\n\t- Not a match on state",
            flush=True,
        )
        return None

    website = await asyncio.to_thread(_convert_link_to_website, link)
    print(f"{jurisdiction}:\n\t- {link}\n\t- {website}", flush=True)
    return website


async def _main(start_ind, end_ind):
    start_time = datetime.now(UTC)
    existing = pd.read_csv("jurisdictions.csv").replace({np.nan: None})
    subset = existing.iloc[int(start_ind) : int(end_ind)]
    subset = subset[subset["Website"].isna()]

    sem = Semaphore(200)
    tasks = [
        asyncio.create_task(_run_one_jurisdiction(jur, sem))
        for jur in jurisdictions_from_df(jurisdiction_info=subset)
    ]
    results = await asyncio.gather(*tasks)
    for fips, website in results:
        if not website:
            continue
        existing.loc[existing["FIPS"] == fips, "Website"] = website

    time_elapsed = datetime.now(UTC) - start_time
    msg = _elapsed_time_as_str(time_elapsed.seconds)
    print("Website search complete. Time elapsed: ", msg, flush=True)

    existing.to_csv("jurisdictions.csv", index=False)


if __name__ == "__main__":
    asyncio.run(_main(*sys.argv[1:]))
