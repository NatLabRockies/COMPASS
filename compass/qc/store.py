"""
truth_store.py — Load, merge, and validate ground-truth YAML files.

Handles single files, directories (recursive), and duplicate detection
both within and across files.

Location keys support two granularities:

  County level  : "County, State"                → e.g. "Power, Idaho"
  Township level: "Subdivision, County, State"    → e.g. "Springfield, Power, Idaho"

load_truth() returns a dict grouped by location rather than a flat list,
so downstream code can process and summarise per-location.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# ── Field definitions ────────────────────────────────────────────────────────
# These define which CSV columns the ground-truth format understands.

# Columns where exact string match is the natural comparison
EXACT_FIELDS = ["value", "units", "adder", "min_dist", "max_dist", "year"]

# Columns where text / keyword / not-null matching makes more sense
TEXT_FIELDS = ["summary", "section", "source"]

ALL_CHECK_FIELDS = EXACT_FIELDS + TEXT_FIELDS

# ── Exceptions ───────────────────────────────────────────────────────────────


class DuplicateLocationError(Exception):
    """Raised when the same location key appears in more than one place."""


# ── Location key parsing ────────────────────────────────────────────────────


def parse_location_key(key: str) -> dict[str, str | None]:
    """
    Parse a YAML location key into its component parts.

    Supports:
      "County, State"                → county-level
      "Subdivision, County, State"   → township-level

    Returns a dict with keys: state, county, subdivision (None if county-level).
    """
    parts = [p.strip() for p in key.split(",")]

    if len(parts) == 2:
        return {
            "state": parts[1].lower(),
            "county": parts[0].lower(),
            "subdivision": None,
        }
    elif len(parts) == 3:
        return {
            "state": parts[2].lower(),
            "county": parts[1].lower(),
            "subdivision": parts[0].lower(),
        }
    else:
        raise ValueError(
            f"Location key must have 2 parts (County, State) or "
            f"3 parts (Subdivision, County, State), got {len(parts)}: '{key}'"
        )


def location_label(loc: dict[str, str | None]) -> str:
    """
    Build a human-readable label from parsed location components.

    Returns "County, State" or "Subdivision, County, State".
    """
    parts = []
    if loc.get("subdivision"):
        parts.append(loc["subdivision"].title())
    parts.append(loc["county"].title())
    parts.append(loc["state"].title())
    return ", ".join(parts)


# ── File collection ──────────────────────────────────────────────────────────


def collect_truth_files(path: str | Path) -> list[Path]:
    """
    Walk *path* and return every .yaml / .yml file found.

    If *path* is a single file, return it in a one-element list.
    If *path* is a directory, recurse into all sub-folders (sorted for
    deterministic ordering).
    """
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.rglob("*.yaml")) + sorted(p.rglob("*.yml"))
        # rglob may return .yml files that also matched .yaml; deduplicate
        seen: set[Path] = set()
        unique: list[Path] = []
        for f in files:
            resolved = f.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(f)
        return unique
    raise FileNotFoundError(f"Ground-truth path not found: {p}")


# ── Merging & duplicate detection ────────────────────────────────────────────


def merge_truth_dicts(files: list[Path]) -> dict:
    """
    Load every YAML file and merge into one dict.

    Raises DuplicateLocationError if a top-level location key (e.g.
    "Power, Idaho") appears more than once — whether across different
    files or duplicated inside the same file.
    """
    merged: dict = {}
    # Track where each key was first seen for the error message
    origin: dict[str, Path] = {}

    for fpath in files:
        raw = yaml.safe_load(fpath.read_text())
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise ValueError(
                f"Expected a YAML mapping at the top level of {fpath}, "
                f"got {type(raw).__name__}"
            )

        # Check for intra-file duplicates.  PyYAML silently keeps the last
        # occurrence when a key is repeated, so we do a quick text-level
        # scan to catch that case before it's swallowed.
        _check_intra_file_duplicates(fpath)

        for key in raw:
            norm = _normalise_location_key(key)
            if norm in origin:
                raise DuplicateLocationError(
                    f"Duplicate location '{key}' — already defined in "
                    f"{origin[norm]}, found again in {fpath}"
                )
            origin[norm] = fpath
            merged[key] = raw[key]

    return merged


def _normalise_location_key(key: str) -> str:
    """Lowercase + strip so 'Power, Idaho' and ' power , idaho ' collide."""
    return ", ".join(p.strip().lower() for p in key.split(","))


def _check_intra_file_duplicates(fpath: Path) -> None:
    """
    Detect duplicate top-level keys inside a single YAML file.

    PyYAML's safe_load silently drops all-but-the-last duplicate key,
    so we scan the raw text for top-level keys (lines that start at
    column 0 and end with ':') and flag repeats.
    """
    seen: dict[str, int] = {}
    for lineno, line in enumerate(fpath.read_text().splitlines(), start=1):
        stripped = line.rstrip()
        # Skip blank lines, comments, and indented lines
        if not stripped or stripped.startswith("#") or line[0] in (" ", "\t"):
            continue
        # A top-level key line looks like  `"Power, Idaho":` or `Power, Idaho:`
        if stripped.endswith(":"):
            raw_key = stripped[:-1].strip().strip('"').strip("'")
            norm = _normalise_location_key(raw_key)
            if norm in seen:
                raise DuplicateLocationError(
                    f"Duplicate location '{raw_key}' inside {fpath} "
                    f"(lines {seen[norm]} and {lineno})"
                )
            seen[norm] = lineno


# ── Check-spec builder (internal) ───────────────────────────────────────────


def _build_checks(field_checks: dict) -> dict[str, dict]:
    """
    Convert a raw YAML feature block into a checks dict.

    Each key is a field name, each value is a dict describing the match mode.
    """
    checks: dict[str, dict] = {}
    for fld, spec in field_checks.items():
        if fld not in ALL_CHECK_FIELDS:
            continue
        if isinstance(spec, dict) and "keywords" in spec:
            checks[fld] = {
                "mode": "keywords",
                "keywords": [str(k).lower() for k in spec["keywords"]],
            }
        elif spec == "not_null":
            checks[fld] = {"mode": "not_null"}
        elif spec == "absent":
            checks[fld] = {"mode": "absent"}
        else:
            checks[fld] = {"mode": "exact", "expected": str(spec).strip().lower()}
    return checks


# ── Main loader ──────────────────────────────────────────────────────────────


def load_truth(path: str | Path) -> dict[str, dict]:
    """
    Parse ground-truth YAML(s) into a dict grouped by location.

    *path* can be:
      - a single .yaml / .yml file
      - a directory — every .yaml / .yml underneath is collected and merged

    Raises DuplicateLocationError if any location key appears more than once.

    Returns a dict keyed by normalised location string::

        {
            "power, idaho": {
                "state": "idaho",
                "county": "power",
                "subdivision": None,
                "FIPS": "16077",
                "features": {
                    "residential buildings": {
                        "value": {"mode": "exact", "expected": "1500"},
                        "units": {"mode": "exact", "expected": "feet"},
                        "summary": {"mode": "keywords", "keywords": [...]},
                        ...
                    },
                    "property lines": { ... },
                }
            },
            "springfield, power, idaho": {
                "state": "idaho",
                "county": "power",
                "subdivision": "springfield",
                ...
            },
        }
    """
    files = collect_truth_files(path)
    if not files:
        raise FileNotFoundError(f"No .yaml / .yml files found under {path}")

    raw = merge_truth_dicts(files)
    result: dict[str, dict] = {}

    for location_key, loc_data in raw.items():
        loc = parse_location_key(location_key)
        norm_key = _normalise_location_key(location_key)

        fips = loc_data.get("FIPS")
        raw_features = loc_data.get("features", {})

        parsed_features: dict[str, dict] = {}
        for feat_name, field_checks in raw_features.items():
            if field_checks is None:
                continue
            checks = _build_checks(field_checks)
            if checks:
                parsed_features[feat_name.lower()] = checks

        result[norm_key] = {
            **loc,
            "FIPS": str(fips) if fips is not None else None,
            "features": parsed_features,
        }

    return result
