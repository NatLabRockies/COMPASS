"""Core functionalities to validate CSV outputs with manual labels"""

from __future__ import annotations

import logging
from collections.abc import Generator

import polars as pl

from store import location_label

logger = logging.getLogger(__name__)


def match_labels(
    truth: dict[str, dict],
    lf: pl.LazyFrame,
) -> Generator[tuple[dict, pl.DataFrame], None, None]:
    """Pair ground-truth locations with matching run rows

    Iterates over each location in *truth*, builds a
    geographic filter (state, county, and subdivision when
    defined), applies it to *lf*, and collects the result.
    When the truth entry declares a FIPS code, the matched
    rows are checked for agreement; mismatches are reported
    via ``logging.error``.

    Parameters
    ----------
    truth : dict[str, dict]
        Ground-truth dict as returned by
        ``store.load_truth()``, keyed by normalised
        location string.
    lf : pl.LazyFrame
        Lazy representation of a run CSV, as produced by
        ``load_run(path).lazy()``.

    Yields
    ------
    tuple[dict, pl.DataFrame]
        A pair ``(loc_data, loc_df)`` for each location in
        *truth*:

        loc_data
            The truth dict for one location, containing
            state, county, subdivision, FIPS, and features
            with their check specs.
        loc_df
            Collected DataFrame with every run row that
            matches the location geographically.  May be
            empty when the run has no data for that
            location.
    """
    for _loc_key, loc_data in truth.items():
        mask = (
            (pl.col("county") == loc_data["county"])
            & (pl.col("state") == loc_data["state"])
        )
        subdiv = loc_data.get("subdivision")
        if subdiv:
            mask = mask & (pl.col("subdivision") == subdiv)
        else:
            mask = mask & pl.col("subdivision").is_null()

        loc_df = lf.filter(mask).collect()

        # Validate FIPS agreement
        expected_fips = loc_data.get("FIPS")
        if expected_fips is not None and not loc_df.is_empty():
            run_fips = loc_df["FIPS"].unique().to_list()
            mismatched = [
                f for f in run_fips
                if f is not None and f != expected_fips
            ]
            if mismatched:
                loc_lbl = location_label(loc_data)
                logger.error(
                    "FIPS mismatch for %s: truth declares %s, "
                    "run contains %s",
                    loc_lbl, expected_fips, mismatched,
                )

        yield loc_data, loc_df
