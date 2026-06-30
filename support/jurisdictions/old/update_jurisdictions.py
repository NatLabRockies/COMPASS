"""Script to update known jurisdictions

This adds the AHJ to already known states and counties.
"""

import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

OUT_COLS = [
    "State",
    "County",
    "Subdivision",
    "Jurisdiction Type",
    "FIPS",
    "Website",
]
FIPS_CODE_LEN = 5


def _compile_states(ord_areas):
    states = []
    for state, group in ord_areas.groupby("state_name"):
        fips = group["state_fips"].unique()
        assert len(fips) == 1
        fips = fips[0]
        states.append({"State": state, "FIPS": int(fips) * 1000})

    states = pd.DataFrame(states)
    states["Jurisdiction Type"] = "state"
    states["County"] = None
    states["Subdivision"] = None
    states["Website"] = None
    return states[OUT_COLS]


def _main(ahj_data):
    working_dir = Path(__file__).parent
    og_data = working_dir / "conus_counties_compiled_2024.csv"
    counties = pd.read_csv(og_data)
    counties = counties[
        ["County", "State", "FIPS", "County Type", "Website"]
    ].rename(columns={"County Type": "Jurisdiction Type"})

    ord_areas = gpd.read_file(ahj_data)

    print("AHJ subdivisions:")
    print(ord_areas.subd_type.value_counts())
    print(f"\nTotal number of AHJ: {len(ord_areas):,}\n")

    states = _compile_states(ord_areas)

    ord_areas["FIPS"] = ord_areas.subd_fips.astype(int)
    ord_areas = ord_areas.merge(
        counties[["FIPS", "Website"]], how="left", on="FIPS"
    )

    missed_counties = counties[~counties.FIPS.isin(ord_areas.FIPS)].copy()
    missed_counties["Subdivision"] = None
    missed_counties = missed_counties[OUT_COLS]

    ord_areas = ord_areas.drop(
        columns=["geometry", "FIPS", "state_fips", "county_fips"]
    )
    ord_areas = ord_areas.rename(
        columns={
            "county_name": "County",
            "state_name": "State",
            "subd_name": "Subdivision",
            "subd_type": "Jurisdiction Type",
            "subd_fips": "FIPS",
        }
    )[OUT_COLS]

    out_data = pd.concat([ord_areas, missed_counties, states])
    out_data["FIPS"] = out_data["FIPS"].astype(int).apply(lambda x: f"{x:05d}")

    assert (out_data["FIPS"].apply(len) >= FIPS_CODE_LEN).all()

    city_fips = out_data["FIPS"].apply(len) > FIPS_CODE_LEN
    assert (~out_data[city_fips]["Jurisdiction Type"].isna()).all()
    assert (
        ~out_data[city_fips]["Jurisdiction Type"].isin({"state", "county"})
    ).all()

    out_data = out_data.sort_values(by="FIPS")

    print("Final jurisdiction breakdown:")
    print(out_data["Jurisdiction Type"].value_counts())
    print(f"\nFinal number of jurisdiction: {len(out_data):,}")

    out_data.to_csv(
        working_dir.parent / "compass" / "data" / "conus_jurisdictions.csv",
        index=False,
    )


if __name__ == "__main__":
    _main(*sys.argv[1:])
