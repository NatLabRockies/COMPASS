"""Ordinance jurisdiction info"""

import logging
from warnings import warn
from pathlib import Path

import numpy as np
import pandas as pd

from compass.exceptions import COMPASSValueError
from compass.warn import COMPASSWarning


logger = logging.getLogger(__name__)
_COUNTY_DATA_FP = (
    Path(__file__).parent.parent / "data" / "conus_jurisdictions.csv"
)


def load_all_jurisdiction_info():
    """Load DataFrame containing info for all jurisdictions

    Returns
    -------
    pandas.DataFrame
        DataFrame containing info like names, FIPS, websites, etc. for
        all jurisdictions.
    """
    return pd.read_csv(_COUNTY_DATA_FP).replace({np.nan: None})


def jurisdiction_websites(jurisdiction_info=None):
    """Load mapping of jurisdiction name and state to website

    Parameters
    ----------
    jurisdiction_info : pandas.DataFrame, optional
        DataFrame containing jurisdiction names and websites. If
        ``None``, this info is loaded using
        :func:`load_all_jurisdiction_info`.
        By default, ``None``.

    Returns
    -------
    dict
        Dictionary where keys are FIPS codes and values are the relevant
        website URL.
    """
    if jurisdiction_info is None:
        jurisdiction_info = load_all_jurisdiction_info()

    return {
        row["FIPS"]: row["Website"] for __, row in jurisdiction_info.iterrows()
    }


def load_jurisdictions_from_fp(jurisdiction_fp):
    """Load jurisdiction info based on jurisdictions in the input fp

    Parameters
    ----------
    jurisdiction_fp : path-like
        Path to csv file containing "County" and "State" columns that
        define the jurisdictions for which info should be loaded.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing jurisdiction info like names, FIPS,
        websites, etc. for all requested jurisdictions (that were
        found).
    """
    jurisdictions = pd.read_csv(jurisdiction_fp).replace({np.nan: None})
    jurisdictions = _validate_jurisdiction_input(jurisdictions)

    all_jurisdiction_info = load_all_jurisdiction_info()
    merge_cols = ["County", "State"]
    if "Subdivision" in jurisdictions:
        merge_cols += ["Subdivision", "Jurisdiction Type"]
    else:
        all_jurisdiction_info = all_jurisdiction_info[
            all_jurisdiction_info["Subdivision"].isna()
        ].reset_index(drop=True)

    jurisdictions = (  # remove dupes
        jurisdictions.groupby(merge_cols, dropna=False)
        .first()
        .reset_index()
        .drop(columns="Unnamed: 0", errors="ignore")
        .replace({np.nan: None})
    )
    jurisdictions["jur_merge"] = jurisdictions.apply(
        _build_merge_col, axis=1, merge_cols=merge_cols
    )
    all_jurisdiction_info["jur_merge"] = all_jurisdiction_info.apply(
        _build_merge_col, axis=1, merge_cols=merge_cols
    )
    jurisdictions = jurisdictions.merge(
        all_jurisdiction_info,
        on="jur_merge",
        how="left",
        suffixes=["_user", ""],
    )

    jurisdictions = _filter_not_found_jurisdictions(jurisdictions, merge_cols)
    return _format_jurisdiction_df_for_output(jurisdictions)


def _validate_jurisdiction_input(jurisdictions):
    """Throw error if user is missing required columns"""
    if "State" not in jurisdictions:
        msg = "The jurisdiction input must have at least a 'State' column!"
        raise COMPASSValueError(msg)

    jurisdictions["State"] = jurisdictions["State"].str.strip()
    if "County" not in jurisdictions:
        jurisdictions["County"] = None
    else:
        jurisdictions["County"] = jurisdictions["County"].str.strip()

    if "Subdivision" in jurisdictions:
        if "Jurisdiction Type" not in jurisdictions:
            msg = (
                "The jurisdiction input must have a 'Jurisdiction Type' "
                "column if a 'Subdivision' column is provided (this helps "
                "avoid name clashes for certain subdivisions)!"
            )
            raise COMPASSValueError(msg)

        jurisdictions["Subdivision"] = jurisdictions["Subdivision"].str.strip()
        jurisdictions["Jurisdiction Type"] = (
            jurisdictions["Jurisdiction Type"].str.casefold().str.strip()
        )

    return jurisdictions


def _filter_not_found_jurisdictions(df, merge_cols):
    """Filter out jurisdictions with null FIPS codes"""
    _warn_about_missing_jurisdictions(df, merge_cols)
    return df[~df["FIPS"].isna()].copy()


def _warn_about_missing_jurisdictions(df, merge_cols):
    """Throw warning about jurisdictions that were not in the list"""
    not_found_jurisdictions = df[df["FIPS"].isna()]
    if len(not_found_jurisdictions):
        out_cols = {f"{col}_user": col for col in merge_cols}
        not_found_jurisdictions = not_found_jurisdictions[
            list(out_cols)
        ].rename(columns=out_cols)
        not_found_jurisdictions_str = not_found_jurisdictions[
            merge_cols
            # cspell: disable-next-line
        ].to_markdown(index=False, tablefmt="psql")
        msg = (
            "The following jurisdictions were not found! Please make sure to "
            "use proper spelling and capitalization.\n"
            f"{not_found_jurisdictions_str}"
        )
        warn(msg, COMPASSWarning)


def _format_jurisdiction_df_for_output(df):
    """Format jurisdiction DataFrame for output"""
    out_cols = [
        "County",
        "State",
        "Subdivision",
        "Jurisdiction Type",
        "FIPS",
        "Website",
    ]
    df["FIPS"] = df["FIPS"].astype(int)
    return df[out_cols].replace({np.nan: None}).reset_index(drop=True)


def _build_merge_col(row, merge_cols):
    """Build column to merge jurisdiction DataFrames on"""
    return " ".join(str(row[c]).casefold() for c in merge_cols)
