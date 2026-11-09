import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import openpyxl


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    parser.add_argument("--workbook", type=Path, default=base / "COMPASS DC Ordinance Research_2026.08.16.xlsx")
    parser.add_argument("--reference", type=Path, default=base / "examples/one_shot_schema_extraction_dc/outputs/ordinances.csv")
    parser.add_argument("--output", type=Path, default=base / "human_extracted_dc_ordinances.csv")
    args = parser.parse_args()
    workbook = openpyxl.load_workbook(args.workbook, data_only=True)
    ordinances = sheet_rows(workbook["DC Ordinances"])
    definitions = sheet_rows(workbook["Definitions"])
    records = [ordinance_record(row) for row in ordinances]
    records.extend(definition_record(row) for row in definitions)
    assign_jurisdiction_metadata(records, base / "compass/data/conus_jurisdictions.csv")
    with args.reference.open(encoding="utf-8-sig", newline="") as stream:
        fields = [
            field for field in next(csv.reader(stream))
            if field in records[0] and field != "FIPS"
        ]
    fields.append("FIPS")
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    verify_export(args.output, fields, records, ordinances, definitions)
    print(f"Wrote {len(records)} rows: {len(ordinances)} ordinance features and {len(definitions)} definitions.")
    print("Data center types:", dict(Counter(record["data_center_type"] for record in records)))
    for sheet in workbook["DC Ordinances"], workbook["Definitions"]:
        report_link_mismatches(sheet)


def assign_jurisdiction_metadata(records, catalog_path):
    """Resolve catalog types and FIPS using names and explicit name suffixes."""
    locations = defaultdict(list)
    with catalog_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            locations[(row["State"].casefold(), row["Subdivision"].casefold())].append(row)

    for record in records:
        state = record["state"].strip().casefold()
        subdivision = (record["subdivision"] or "").strip().casefold()
        county = (record["county"] or "").strip().casefold()
        candidates = locations[(state, subdivision)]
        if not candidates:
            suffix = re.fullmatch(r"(.+) (city|town|township|village|borough)", subdivision)
            if suffix:
                candidates = [
                    row for row in locations[(state, suffix[1])]
                    if row["Jurisdiction Type"].casefold() == suffix[2]
                ]
        if not subdivision:
            candidates = [row for row in candidates if row["County"].casefold() == county]
        elif county:
            # Some catalog municipalities have no county recorded.
            candidates = [row for row in candidates if row["County"].casefold() in {"", county}]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one catalog match for {(state, county, subdivision)}, "
                f"found {len(candidates)}"
            )
        record["jurisdiction_type"] = candidates[0]["Jurisdiction Type"].title()
        record["FIPS"] = candidates[0]["FIPS"]


def ordinance_record(row):
    record = common_record(row)
    record.update(
        feature=normalize_feature(row["Feature"].value),
        data_center_type=normalize_type(row["Data Center Type"].value),
        value=row["Value"].value,
        units="str" if row["Feature Data Type"].value == "Qualitative" else row["Units"].value,
        summary=row["Summary"].value,
    )
    return record


def definition_record(row):
    record = common_record(row)
    term = row["Defined Term"].value
    record.update(
        feature="definition",
        data_center_type=normalize_type(term),
        value=row["Definition (verbatim)"].value,
        units="str",
        summary=None,
        explanation=f"Defined term: {term}.\n{record['explanation']}".rstrip(),
    )
    return record


def common_record(row):
    mapping = {
        "county": "County",
        "state": "State",
        "subdivision": "City / Subdivision",
        "jurisdiction_type": "Jurisdiction Type",
        "section": "Section (Pin Cite)",
    }
    record = {key: row[column].value for key, column in mapping.items()}
    raw_year = row["Ordinance Year"].value
    year = int(re.match(r"^\s*(\d{4})\b", str(raw_year)).group(1))
    notes = row["Notes"].value or ""
    # Keep adoption, repeal, and proposal annotations outside the numeric year.
    if str(raw_year).strip() != str(year):
        notes = f"{notes}\nOrdinance Year in workbook: {raw_year}".strip()
    pdf = row.get("Direct PDF URL")
    selected = pdf if pdf is not None and (pdf.value or pdf.hyperlink) else row["Source URL"]
    record.update(year=year, explanation=notes, source=cell_url(selected))
    return record


def normalize_feature(value):
    aliases = {
        'buffer / landscaping': 'buffer/landscaping',
        'equipment screening / enclosure': 'equipment screening/enclosure',
        'fencing / security screening': 'fencing/security screening',
        'heat / thermal impact': 'heat/thermal impact',
        'noise (low-frequency / tonal)': 'noise (low-frequency/tonal)',
        'noise mitigation / acoustic barriers': 'noise mitigation/acoustic barriers',
        'prohibitions / moratorium': 'prohibitions/moratorium',
        'setback - parks / natural areas': 'setback - parks/natural areas',
        'water reuse / reclaimed': 'water reuse/reclaimed',
    }
    return aliases.get(value.strip().casefold(), value)


def normalize_type(value):
    if value is None:
        return ""
    term = value.strip()
    # Parenthetical annotations describe the instrument, not a facility category.
    if re.fullmatch(r"data centers?(?:\s*\(.*\))?", term, re.IGNORECASE):
        return ""
    if term.casefold() in {"all", "all types", "all data center types", "hyperscale / medium / micro"}:
        return ""
    aliases = {
        "hyperscale": "Hyperscale Data Center",
        "medium": "Medium Data Center",
        "micro": "Micro Data Center",
        "accessory": "Accessory Data Center",
        "data center, hyperscale": "Hyperscale Data Center",
        "data center, medium": "Medium Data Center",
        "data center, micro": "Micro Data Center",
        "data center, accessory": "Accessory Data Center",
        "accessory use": "Accessory Data Center",
        "data center as an accessory use": "Accessory Data Center",
        "data center, accessory use": "Accessory Data Center",
        "data center, ancillary": "Ancillary",
        "data center, campus": "Campus",
        "data center, major": "Major",
        "data center, minor": "Minor",
    }
    return aliases.get(term.casefold(), term)


def verify_export(path, fields, records, ordinances, definitions):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        saved = list(reader)
        assert reader.fieldnames == fields
    assert len(saved) == len(ordinances) + len(definitions)
    for actual, expected in zip(saved, records):
        assert all(actual[key] == ("" if expected[key] is None else str(expected[key])) for key in fields)
        assert type(expected["year"]) is int
        assert re.fullmatch(r"\d{4}", actual["year"])
        assert actual["source"].startswith(("http://", "https://"))
        assert not re.fullmatch(r"data center(?:\s*\(.*\))?", actual["data_center_type"], re.IGNORECASE)
    for actual, row in zip(saved, ordinances):
        if row["Feature Data Type"].value == "Qualitative":
            assert actual["units"] == "str"
        else:
            units = row["Units"].value
            assert actual["units"] == ("" if units is None else str(units))
    for actual, row in zip(saved[len(ordinances):], definitions):
        assert actual["feature"] == "definition" and actual["units"] == "str"
        assert actual["value"] == row["Definition (verbatim)"].value
    print("Verified column order, all exported values, numeric years, qualitative units, and definition text.")


def sheet_rows(sheet):
    headers = [cell.value for cell in sheet[1]]
    return [
        dict(zip(headers, row))
        for row in sheet.iter_rows(min_row=2)
        if any(cell.value is not None for cell in row)
    ]


def cell_url(cell):
    return cell.hyperlink.target if cell.hyperlink and cell.hyperlink.target else cell.value


def report_link_mismatches(sheet):
    for row in sheet_rows(sheet):
        cell = row["Source URL"]
        visible = str(cell.value or "")
        target = cell_url(cell)
        if cell.hyperlink and visible.startswith("http") and urlsplit(visible).netloc != urlsplit(target).netloc:
            print(f"Hyperlink host mismatch at {sheet.title}!{cell.coordinate}: {visible!r} -> {target!r}")


if __name__ == "__main__":
    main()
