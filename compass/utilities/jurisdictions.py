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
    pd.DataFrame
        DataFrame containing info like names, FIPS, websites, etc. for
        all jurisdictions.
    """
    jurisdiction_info = pd.read_csv(_COUNTY_DATA_FP).replace({np.nan: None})
    jurisdiction_info = _convert_to_title(jurisdiction_info, "State")
    jurisdiction_info = _convert_to_title(jurisdiction_info, "County")
    return _convert_to_title(jurisdiction_info, "Subdivision")


def jurisdiction_websites(jurisdiction_info=None):
    """Load mapping of jurisdiction name and state to website

    Parameters
    ----------
    jurisdiction_info : pd.DataFrame, optional
        DataFrame containing jurisdiction names and websites. If
        ``None``, this info is loaded using
        :func:`load_jurisdiction_info`.
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
    pd.DataFrame
        DataFrame containing jurisdiction info like names, FIPS,
        websites, etc. for all requested jurisdictions (that were
        found).
    """
    jurisdictions = pd.read_csv(jurisdiction_fp)
    jurisdictions = _validate_jurisdiction_input(jurisdictions)

    all_jurisdiction_info = load_all_jurisdiction_info()
    merge_cols = ["County", "State"]
    if "Subdivision" in jurisdictions:
        merge_cols += ["Subdivision", "Jurisdiction Type"]
    else:
        all_jurisdiction_info = all_jurisdiction_info[
            all_jurisdiction_info["Subdivision"].isna()
        ].reset_index(drop=True)

    jurisdictions = jurisdictions.merge(
        all_jurisdiction_info, on=merge_cols, how="left"
    )

    jurisdictions = _filter_not_found_jurisdictions(jurisdictions)
    return _format_jurisdiction_df_for_output(jurisdictions)


def _validate_jurisdiction_input(jurisdictions):
    """Throw error if user is missing required columns"""
    if "State" not in jurisdictions:
        msg = "The jurisdiction input must have at least a 'State' column!"
        raise COMPASSValueError(msg)

    jurisdictions = _convert_to_title(jurisdictions, "State")

    if "County" not in jurisdictions:
        jurisdictions["County"] = None
    else:
        jurisdictions = _convert_to_title(jurisdictions, "County")

    if "Subdivision" in jurisdictions:
        if "Jurisdiction Type" not in jurisdictions:
            msg = (
                "The jurisdiction input must have a 'Jurisdiction Type' "
                "column if a 'Subdivision' column is provided (this helps "
                "avoid name clashes for certain subdivisions)!"
            )
            raise COMPASSValueError(msg)

        jurisdictions = _convert_to_title(jurisdictions, "Subdivision")
        jurisdictions["Jurisdiction Type"] = jurisdictions[
            "Jurisdiction Type"
        ].str.casefold()

    return jurisdictions


def _filter_not_found_jurisdictions(df):
    """Filter out jurisdictions with null FIPS codes"""
    _warn_about_missing_jurisdictions(df)
    return df[~df["FIPS"].isna()].copy()


def _warn_about_missing_jurisdictions(df):
    """Throw warning about jurisdictions that were not in the list"""
    not_found_jurisdictions = df[df["FIPS"].isna()]
    if len(not_found_jurisdictions):
        not_found_jurisdictions_str = not_found_jurisdictions[
            ["State", "County", "Subdivision", "Jurisdiction Type"]
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


def _convert_to_title(df, column):
    """Convert the values of a DataFrame column to titles"""
    df[column] = df[column].str.strip().str.casefold().str.title()
    return df
