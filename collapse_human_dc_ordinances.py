"""Collapse duplicate ordinance keys using the agreed feature-specific rules."""

import argparse
import csv
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

# Features the extraction schema records as text rather than a number
QUALITATIVE = {
    "parking",
    "equipment screening/enclosure",
    "fencing/security screening",
    "backup generator operating restrictions",
    "cooling technology",
}

# Most restrictive value picked by hand from each multi-value row, keyed by
# (jurisdiction, feature). None marks a row with no single value to pick,
# such as a berm-only buffer or a setback that only scales with height.
MOST_RESTRICTIVE = {
    ("Phoenix", "buffer/landscaping"): ("30", "feet"),
    ("Elk Grove Village", "Minimum Lot Size"): ("4", "acres"),
    ("Elk Grove Village", "Maximum Height"): ("100", "feet"),
    ("Elk Grove Village", "Setback - Property Line"): ("50", "feet"),
    ("Elk Grove Village", "Light Illumination Limits"): ("0.5", "footcandles"),
    ("Elk Grove Village", "Maximum Lot Coverage"): ("50", "percent"),
    ("Tippecanoe", "Vibration"): ("0.002", "g peak"),
    ("Tippecanoe", "Setback - Property Line"): ("40", "feet"),
    ("Tippecanoe", "Setback - Roads"): ("60", "feet"),
    ("Tippecanoe", "Minimum Lot Size"): ("10000", "sq ft"),
    ("Tippecanoe", "buffer/landscaping"): None,
    ("Lancaster City", "buffer/landscaping"): ("100", "feet"),
    ("West Pennsboro Township", "Minimum Lot Size"): ("25", "acres"),
    ("West Pennsboro Township", "Setback - Property Line"): ("150", "feet"),
    ("West Pennsboro Township", "Maximum Lot Coverage"): ("70", "percent"),
    ("Loudoun", "Setback - Residential"): ("200", "feet"),
    ("Loudoun", "buffer/landscaping"): None,
    ("Prince William", "noise (low-frequency/tonal)"): ("68", "dB(C)"),
    ("Prince William", "buffer/landscaping"): None,
    ("Manassas City", "Setback - Property Line"): ("25", "feet"),
    ("Manassas City", "Setback - Residential"): None,
    ("Manassas City", "buffer/landscaping"): ("25", "feet"),
    ("Albemarle", "noise mitigation/acoustic barriers"): ("70", "dBA"),
}

# Units recorded with measurement detail, mapped to the unit alone. The
# detail moves into the summary, matching the extraction schema.
PLAIN_UNITS = {
    "% max building coverage; no FAR standard": "percent",
    "% minimum vegetative cover of lot": "percent",
    "FAR (max)": "FAR",
    "feet (max)": "feet",
    "feet (up to)": "feet",
    "feet from all lot lines": "feet",
    "feet from listed receptors; measured property line to property line": "feet",
    "feet from major/minor arterial; at least one arterial access required": "feet",
    "feet from residential/community lot lines and residentially zoned or PO districts": "feet",
    "feet from the Rural Areas zoning district": "feet",
    "feet max building height": "feet",
    "footcandles in residential or RE zone": "footcandles",
    "percent annual cap": "percent",
    "percent lot coverage": "percent",
    "percent minimum reclaimed-water offset of total water use (director may set another percentage based on proximity to the reclaimed system)": "percent",
    "sq ft GFA": "sq ft",
    "sq ft/floor": "sq ft",
    "°F maximum increase at zone boundary line": "°F",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("human_extracted_dc_ordinances.csv"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input.with_name(
        "human_extracted_dc_ordinances_collapsed.csv"
    )
    if output.resolve() == args.input.resolve():
        raise ValueError("Use a separate output file to preserve the original CSV.")
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        original = list(reader)
    groups = defaultdict(list)
    for row in original:
        groups[row_key(row)].append(row)
    result = []
    for rows in groups.values():
        result.extend(collapse(rows) if len(rows) > 1 else rows)
    validate(result, groups)
    before = [row.copy() for row in result]
    noise = select_lowest_noise_limits(result)
    validate_noise(before, result)
    before = [row.copy() for row in result]
    reduced = select_most_restrictive_values(result)
    validate_most_restrictive(before, result)
    before = [row.copy() for row in result]
    qualitative = use_summary_for_qualitative(result)
    validate_qualitative(before, result)
    before = [row.copy() for row in result]
    filled = fill_value_from_summary(result)
    validate_fill(before, result)
    before = [row.copy() for row in result]
    plain = use_plain_units(result)
    validate_units(before, result)
    before = [row.copy() for row in result]
    standard = use_standard_types(result)
    validate_types(before, result)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result)
    with output.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == columns
        assert list(reader) == result
    duplicates = sum(len(rows) > 1 for rows in groups.values())
    print(
        f"{len(original)} -> {len(result)} rows; {duplicates} duplicate groups resolved."
    )
    print(f"{noise} noise rows reduced to their lowest listed limit.")
    print(f"{reduced} multi-value rows reduced to their most restrictive value.")
    print(f"{qualitative} qualitative rows now carry their summary as value.")
    print(f"{filled} rows had an empty value; copied their summary into value.")
    print(f"{plain} rows had unit detail moved into their summary.")
    print(
        f"{standard} rows had a local class name mapped to a standard type."
    )
    print(
        f"Validated unique keys, unchanged singleton rows, integer years, and CSV round-trip: {output}"
    )


def collapse(rows):
    feature = rows[0]["feature"]
    if feature in {"equipment screening/enclosure", "Cooling Technology"}:
        row = merge_metadata(rows)
        row.update(value=row["summary"], units="str")
        return [row]
    if feature == "Maximum Building Size":
        assert len({r["units"] for r in rows}) == 1, (
            "Size limits must use comparable units."
        )
        row = merge_metadata(rows)
        row["value"] = str(min(Decimal(r["value"]) for r in rows))
        row["summary"] = (
            f"Selected most restrictive listed maximum: {row['value']} {row['units']}. "
            + row["summary"]
        )
        return [row]
    if feature == "Noise":
        # keep every listed limit; select_lowest_noise_limits picks one later
        row = merge_metadata(rows)
        row["value"] = "; ".join(r["value"].strip() for r in rows)
        row["units"] = "; ".join(r["units"].strip() for r in rows)
        return [row]
    if feature == "definition":
        return [collapse_definition(rows)]
    raise ValueError(f"No collapse rule for duplicate feature: {feature}")


def select_lowest_noise_limits(rows):
    """Keep the lowest listed limit as each Noise row's value.

    This matches the extraction schema, which records the single most
    restrictive noise limit and leaves the other limits to the summary.
    """
    selected = 0
    for row in rows:
        if row["feature"] != "Noise":
            continue
        limit = min(listed_noise_limits(row["value"]))
        row["summary"] = (
            f"Selected most restrictive listed limit: {limit} dBA "
            f"(listed: {row['value']} {row['units']}). " + row["summary"]
        )
        row.update(value=str(limit), units="dBA")
        selected += 1
    return selected


def listed_noise_limits(value):
    """Numbers in a noise value, skipping '+N' exception allowances.

    A segment such as '+10 for 15 min' raises the limit for a short
    duration rather than setting one, so none of its numbers are limits.
    """
    return [
        Decimal(number)
        for segment in value.split(";")
        if not segment.strip().startswith("+")
        for number in re.findall(r"\d+(?:\.\d+)?", segment)
    ]


def validate_noise(before, after):
    """Only Noise rows changed, each to the lowest of its listed limits."""
    assert len(before) == len(after)
    for old, new in zip(before, after, strict=True):
        if old["feature"] != "Noise":
            assert new == old, "A non-noise row was changed."
            continue
        assert Decimal(new["value"]) == min(listed_noise_limits(old["value"]))
        assert new["units"] == "dBA"
        assert new["summary"].endswith(old["summary"])
        rewritten = {"value", "units", "summary"}
        assert all(new[k] == v for k, v in old.items() if k not in rewritten)


def collapse_definition(rows):
    """Keep the adopted definition primary and label denied text as historical."""
    adopted = [row for row in rows if "ADOPTED/CURRENT." in row["explanation"]]
    assert len(adopted) == 1, "A single adopted definition must be identified."
    row = adopted[0].copy()
    historical = [r for r in rows if r is not adopted[0]]
    assert all("PROPOSED/DENIED." in r["explanation"] for r in historical)
    row["summary"] = (
        "Uses the adopted July 7, 2026 moratorium Resolution definition, which draws no "
        "size distinctions and applies for the moratorium period through March 30, 2027, "
        "as recorded in the workbook. The denied 2026 draft definition is retained in "
        "explanation as historical text and is not part of the selected definition."
    )
    notes = ["Original adopted-definition notes:\n" + row["explanation"].strip()]
    for old in historical:
        notes.append(
            "Historical proposed/denied definition (not operative):\n"
            + old["value"]
            + f"\nSection: {old['section']}\nYear: {old['year']}\nSource: {old['source']}"
            + "\nOriginal denied-draft notes:\n"
            + old["explanation"].strip()
        )
    row.update(explanation="\n\n".join(notes), units="str")
    return row


def select_most_restrictive_values(rows):
    """Reduce each multi-value row to the value picked in MOST_RESTRICTIVE.

    The listed values are kept at the front of the summary, matching the
    extraction schema, which records one value and leaves the rest to the
    summary.
    """
    matched = set()
    for row in rows:
        key = (row["subdivision"].strip() or row["county"].strip(), row["feature"])
        if key not in MOST_RESTRICTIVE:
            continue
        assert key not in matched, f"More than one row for {key}"
        matched.add(key)
        listed = f"(listed: {row['value']} {row['units']})"
        picked = MOST_RESTRICTIVE[key]
        if picked is None:
            row["summary"] = f"No single most restrictive value {listed}. " + row["summary"]
            row.update(value=row["summary"], units="str")
        else:
            value, units = picked
            row["summary"] = (
                f"Selected most restrictive listed value: {value} {units} {listed}. "
                + row["summary"]
            )
            row.update(value=value, units=units)
    assert matched == set(MOST_RESTRICTIVE), set(MOST_RESTRICTIVE) - matched
    return len(matched)


def validate_most_restrictive(before, after):
    """Only picked rows changed, each to a number its old value listed."""
    assert len(before) == len(after)
    for old, new in zip(before, after, strict=True):
        if new == old:
            continue
        key = (old["subdivision"].strip() or old["county"].strip(), old["feature"])
        assert key in MOST_RESTRICTIVE, f"An unpicked row was changed: {key}"
        assert new["summary"].endswith(old["summary"])
        if MOST_RESTRICTIVE[key] is not None:
            assert new["value"] in old["value"].replace(",", "")
        rewritten = {"value", "units", "summary"}
        assert all(new[k] == v for k, v in old.items() if k not in rewritten)


def use_plain_units(rows):
    """Keep only the unit in units and move its detail into the summary."""
    changed = 0
    for row in rows:
        units = row["units"].strip()
        if units not in PLAIN_UNITS:
            continue
        row["summary"] = f"Units detail: {units}. " + row["summary"]
        row["units"] = PLAIN_UNITS[units]
        changed += 1
    return changed


def validate_units(before, after):
    """Only detailed units changed, and every numeric row has a plain unit."""
    assert len(before) == len(after)
    for old, new in zip(before, after, strict=True):
        if new == old:
            continue
        units = old["units"].strip()
        assert new == {
            **old,
            "units": PLAIN_UNITS[units],
            "summary": f"Units detail: {units}. " + old["summary"],
        }
    allowed = set(PLAIN_UNITS.values()) | {"str", "acres", "dBA", "dB(C)", "g peak"}
    assert all(row["units"] in allowed for row in after), {
        row["units"] for row in after
    } - allowed


def use_summary_for_qualitative(rows):
    """Carry the summary as value for features the schema extracts as text.

    This follows the convention finalize applies to qualitative features.
    """
    changed = 0
    for row in rows:
        if row["feature"].casefold() not in QUALITATIVE:
            continue
        summary = row["summary"].strip()
        if summary and (row["value"], row["units"]) != (summary, "str"):
            row.update(value=summary, units="str")
            changed += 1
    return changed


def validate_qualitative(before, after):
    """Only qualitative rows changed, and only their value and units."""
    assert len(before) == len(after)
    for old, new in zip(before, after, strict=True):
        if new == old:
            continue
        assert old["feature"].casefold() in QUALITATIVE
        assert new == {**old, "value": old["summary"].strip(), "units": "str"}


def fill_value_from_summary(rows):
    """Use the summary as the value when the value is empty but a summary exists.

    Free-text answers are sometimes recorded only in ``summary``; an empty
    ``value`` would otherwise read as "nothing found" downstream.
    """
    filled = 0
    for row in rows:
        if not row["value"].strip() and row["summary"].strip():
            row["value"] = row["summary"].strip()
            filled += 1
    return filled


def validate_fill(before, after):
    """Only empty values with a summary changed, and only their value."""
    assert len(before) == len(after)
    for old, new in zip(before, after, strict=True):
        if old["value"].strip() or not old["summary"].strip():
            assert new == old, "A row with a value (or no summary) was changed."
        else:
            assert new == {**old, "value": old["summary"].strip()}
    assert all(row["value"].strip() for row in after if row["summary"].strip())


def merge_metadata(rows):
    row = rows[0].copy()
    assert len({r["year"] for r in rows}) == 1
    for field in ("section", "summary", "explanation", "source"):
        separator = " " if field == "summary" else "\n\n"
        row[field] = separator.join(
            dict.fromkeys(r[field].strip() for r in rows if r[field].strip())
        )
    return row


# The fixed list the extraction schema allows for data_center_type, keyed
# by the class names the workbook uses. An empty type is a generic one.
STANDARD_TYPES = {
    "": "",
    "data processing center": "",
    "hyperscale data center": "Hyperscale Data Center",
    "large data center": "Large Data Center",
    "large-scale data center": "Large Data Center",
    "major": "Large Data Center",
    "medium data center": "Medium Data Center",
    "small urban data center": "Small Data Center",
    "minor": "Small Data Center",
    "micro data center": "Micro Data Center",
    "accessory data center": "Accessory Data Center",
    "ancillary": "Accessory Data Center",
    "associated data room or server room": "Accessory Data Center",
    "campus": "Data Center Campus",
    "large quantity water user": "Other",
    # Albemarle defines this apart from data centers: staffed data entry
    "data processing facility": "Other",
}


def use_standard_types(rows):
    """Map each local class name to a standard type, keeping it in summary.

    Runs last: the steps above tell rows apart by their local class name.
    """
    changed = 0
    for row in rows:
        local = row["data_center_type"].strip()
        standard = STANDARD_TYPES[local.casefold()]
        if standard == local:
            continue
        row["data_center_type"] = standard
        row["summary"] = f"{local}: {row['summary']}"
        changed += 1
    return changed


def validate_types(before, after):
    """Only the type and summary changed, and every type is standard."""
    allowed = set(STANDARD_TYPES.values())
    keys = Counter(row_key(row) for row in after)
    assert max(keys.values()) == 1, "Two rows share a standard type and feature."
    for old, new in zip(before, after, strict=True):
        assert new["data_center_type"] in allowed
        if new == old:
            continue
        local = old["data_center_type"].strip()
        assert new["summary"] == f"{local}: {old['summary']}"
        rewritten = {"data_center_type", "summary"}
        assert all(new[k] == v for k, v in old.items() if k not in rewritten)


def validate(result, groups):
    counts = Counter(row_key(row) for row in result)
    assert all(count == 1 for count in counts.values())
    assert all(not row["year"] or row["year"].isdigit() for row in result)
    assert all(
        row["data_center_type"].lower() not in {"all", "data center"} for row in result
    )
    by_key = {row_key(row): row for row in result}
    for key, rows in groups.items():
        if len(rows) == 1:
            assert by_key[key] == rows[0], "A singleton row was changed."
        elif rows[0]["feature"] in {
            "equipment screening/enclosure",
            "Cooling Technology",
        }:
            merged = by_key[key]
            assert merged["units"] == "str" and merged["value"]
            assert all(row["summary"].strip() in merged["value"] for row in rows)
    assert len(result) == len(groups)


def row_key(row):
    fields = (
        "state",
        "county",
        "subdivision",
        "jurisdiction_type",
        "data_center_type",
        "feature",
    )
    return tuple(row[field].strip().casefold() for field in fields)


if __name__ == "__main__":
    main()
