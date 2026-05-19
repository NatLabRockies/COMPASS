"""
core.py — Composable building blocks for matching and validation
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass

import polars as pl

from reference import location_label

logger = logging.getLogger(__name__)


# ── Data types ───────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Outcome of a single field-level check"""

    field: str
    mode: str
    passed: bool
    expected: str
    actual: str
    detail: str = ""


# ── Location-level functions ─────────────────────────────────────────


def extract_locations(
    lf: pl.LazyFrame,
) -> set[tuple[str, str, str | None]]:
    """
    Collect distinct locations from a LazyFrame

    Each location is a tuple of ``(state, county, subdivision)``
    where *subdivision* is ``None`` for county-level records.

    Parameters
    ----------
    lf : pl.LazyFrame
        Lazy representation of run data containing at least
        the columns ``state``, ``county``, and
        ``subdivision``.

    Returns
    -------
    set[tuple[str, str, str | None]]
        Unique location tuples found in the data.
    """
    rows = (
        lf.select("state", "county", "subdivision")
        .unique()
        .collect()
        .iter_rows()
    )
    return {(state, county, subdiv) for state, county, subdiv in rows}


def find_missing_locations(
    ref: dict[str, dict],
    lf: pl.LazyFrame,
) -> list[dict]:
    """
    Find reference locations absent from the target

    Compares the locations declared in *ref* against the
    distinct locations present in *lf*.  Returns the
    reference entries whose geographic key (state, county,
    subdivision) has no matching rows in the target.

    Parameters
    ----------
    ref : dict[str, dict]
        Reference dict as returned by
        ``reference.load_reference()``.
    lf : pl.LazyFrame
        Lazy representation of the target run data.

    Returns
    -------
    list[dict]
        The ``loc_data`` dicts for each reference location
        not found in the target, in the order they appear
        in *ref*.  Empty list when all reference locations
        are present.
    """
    target_locs = extract_locations(lf)

    missing = []
    for _loc_key, loc_data in ref.items():
        loc_tuple = (
            loc_data["state"],
            loc_data["county"],
            loc_data.get("subdivision"),
        )
        if loc_tuple not in target_locs:
            missing.append(loc_data)
    return missing


def match_labels(
    ref: dict[str, dict],
    lf: pl.LazyFrame,
) -> Generator[tuple[dict, pl.DataFrame], None, None]:
    """Pair ground-truth locations with matching run rows

    Iterates over each location in *ref*, builds a
    geographic filter (state, county, and subdivision when
    defined), applies it to *lf*, and collects the result.
    When the reference entry declares a FIPS code, the matched
    rows are checked for agreement; mismatches are reported
    via ``logging.error``.

    Parameters
    ----------
    ref : dict[str, dict]
        Reference dict as returned by
        ``reference.load_reference()``, keyed by normalised
        location string.
    lf : pl.LazyFrame
        Lazy representation of a run CSV, as produced by
        ``load_run(path).lazy()``.

    Yields
    ------
    tuple[dict, pl.DataFrame]
        A pair ``(loc_data, loc_df)`` for each location in
        *ref*:

        loc_data
            The reference dict for one location, containing
            state, county, subdivision, FIPS, and features
            with their check specs.
        loc_df
            Collected DataFrame with every run row that
            matches the location geographically.  May be
            empty when the run has no data for that
            location.
    """
    for _loc_key, loc_data in ref.items():
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
                    "FIPS mismatch for %s: reference declares %s, "
                    "run contains %s",
                    loc_lbl, expected_fips, mismatched,
                )

        yield loc_data, loc_df


def find_missing_features(
    loc_data: dict,
    lf: pl.LazyFrame,
) -> list[str]:
    """
    Find features declared in the reference but absent from the run.

    Compares the feature names listed in *loc_data* against
    the distinct ``feature`` values present in *lf*.  Any
    feature that appears in the reference but has no matching
    row in the run is considered missing.

    Designed to compose with :func:`match_labels`::

        for loc_data, loc_df in match_labels(ref, run_lf):
            missing = find_missing_features(
                loc_data, loc_df.lazy(),
            )

    Parameters
    ----------
    loc_data : dict
        Truth dict for a single location, as yielded by
        :func:`match_labels`.  Must contain a ``features``
        key mapping feature names to check specs (which
        may be empty dicts for presence-only features).
    lf : pl.LazyFrame
        Lazy representation of the run rows already scoped
        to this location.

    Returns
    -------
    list[str]
        Feature names present in the reference but not found
        in the run, in the order they appear in
        ``loc_data["features"]``.  Empty list when all
        features are present.
    """
    expected = set(loc_data.get("features", {}).keys())
    if not expected:
        return []

    present = set(
        lf.select("feature")
        .unique()
        .collect()
        .get_column("feature")
        .to_list()
    )

    return [f for f in loc_data["features"] if f not in present]
