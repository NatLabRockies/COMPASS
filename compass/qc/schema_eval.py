"""
schema_eval.py — CLI entry point for extraction evaluation

Subcommands
-----------
  init      Scaffold a reference YAML from an existing CSV run.
  validate  Score one CSV run against reference.
  compare   Diff two CSV runs; optionally score both against reference.

Examples
--------
  python schema_eval.py init run1.csv -o reference.yaml
  python schema_eval.py validate run1.csv -t reference.yaml
  python schema_eval.py compare run1.csv run2.csv -t reference.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import polars as pl

from core import (
    CheckResult,
    extract_locations,
    find_feature_row,
    find_missing_features,
    find_missing_locations,
    load_run,
    match_labels,
    run_checks,
    score_run,
    validate_formated,
)
from reference import (
    ALL_CHECK_FIELDS,
    EXACT_FIELDS,
    TEXT_FIELDS,
    load_reference,
    location_label,
)

# ── Constants ────────────────────────────────────────────────────────

KEY_COLS = ["county", "state", "subdivision", "feature"]

# ── ANSI helpers ─────────────────────────────────────────────────────


class C:
    """Tiny ANSI colour helpers"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def ok(s: str)   -> str: return f"{C.GREEN}{s}{C.RESET}"
    @staticmethod
    def fail(s: str) -> str: return f"{C.RED}{s}{C.RESET}"
    @staticmethod
    def warn(s: str) -> str: return f"{C.YELLOW}{s}{C.RESET}"
    @staticmethod
    def bold(s: str) -> str: return f"{C.BOLD}{s}{C.RESET}"


# ── Formatting helpers ───────────────────────────────────────────────


def _truncate(s: str | None, n: int) -> str:
    """Shorten a string for display"""
    if s is None:
        return "(null)"
    return s[:n] + "…" if len(s) > n else s


def _pct_color(pct: float) -> callable:
    """Pick a colour function based on percentage thresholds"""
    if pct >= 90:
        return C.ok
    return C.warn if pct >= 70 else C.fail


def _sortable_key(t: tuple) -> tuple:
    """Replace None with '' so tuples are sortable"""
    return tuple(v if v is not None else "" for v in t)


# ── Validate subcommand ──────────────────────────────────────────────


def cmd_validate(
    run_path: str,
    ref_path: str,
    verbose: bool = False,
    output_format: str = "text",
):
    """Validate a run against reference and return formatted output

    Parameters
    ----------
    run_path : str
        Path to the CSV run file to validate.
    ref_path : str
        Path to the reference YAML file or directory.
    verbose : bool, default=False
        Include passing checks in text output. By default, False.
    output_format : str, default="text"
        Output format to render. Supported values are ``"text"``
        and ``"json"``. By default, text.

    Returns
    -------
    str
        Rendered validation report as text or JSON string.
    """
    lf = load_run(run_path)
    ref = load_reference(ref_path)

    return validate_formated(
        ref,
        lf,
        run_path,
        ref_path,
        output_format=output_format,
        verbose=verbose,
        style=C,
    )


# ── Compare subcommand ───────────────────────────────────────────────


def cmd_compare(
    run_a_path: str,
    run_b_path: str,
    ref_path: str | None = None,
    verbose: bool = False,
):
    df_a = load_run(run_a_path).collect()
    df_b = load_run(run_b_path).collect()

    label_a = Path(run_a_path).stem
    label_b = Path(run_b_path).stem

    print(C.bold(f"\n{'=' * 70}"))
    print(C.bold(f"  Comparison: {label_a}  vs  {label_b}"))
    print(C.bold(f"{'=' * 70}\n"))

    # Build key sets — tuples of (county, state, subdivision, feature)
    def key_set(df: pl.DataFrame) -> set[tuple]:
        return set(df.select(KEY_COLS).unique().iter_rows())

    keys_a = key_set(df_a)
    keys_b = key_set(df_b)

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    common = keys_a & keys_b

    # ── Row-presence diff ────────────────────────────────────────
    if only_a or only_b:
        print(C.bold("  Row presence changes:"))
        if only_a:
            print(
                f"\n  {C.fail(f'Removed in {label_b}')}"
                f" ({len(only_a)}):"
            )
            for c, s, sd, f in sorted(only_a, key=_sortable_key):
                loc = location_label(
                    {"county": c, "state": s, "subdivision": sd}
                )
                print(f"    − {loc} → {f}")
        if only_b:
            print(
                f"\n  {C.ok(f'Added in {label_b}')}"
                f" ({len(only_b)}):"
            )
            for c, s, sd, f in sorted(only_b, key=_sortable_key):
                loc = location_label(
                    {"county": c, "state": s, "subdivision": sd}
                )
                print(f"    + {loc} → {f}")
        print()

    # ── Field-level diff on shared rows ──────────────────────────
    compare_fields = [
        f for f in ALL_CHECK_FIELDS
        if f in df_a.columns and f in df_b.columns
    ]
    n_changed = 0
    n_unchanged = 0

    for key in sorted(common, key=_sortable_key):
        county, state, subdiv, feature = key

        def _filter(df, c, s, sd, f):
            mask = (
                (pl.col("county") == c)
                & (pl.col("state") == s)
                & (pl.col("feature") == f)
            )
            if sd:
                mask &= pl.col("subdivision") == sd
            else:
                mask &= pl.col("subdivision").is_null()
            return df.filter(mask)

        row_a = _filter(
            df_a, county, state, subdiv, feature
        ).row(0, named=True)
        row_b = _filter(
            df_b, county, state, subdiv, feature
        ).row(0, named=True)

        diffs: list[tuple[str, str | None, str | None]] = []
        for fld in compare_fields:
            va = row_a.get(fld)
            vb = row_b.get(fld)
            na = va.strip().lower() if va else None
            nb = vb.strip().lower() if vb else None
            if na != nb:
                diffs.append((fld, va, vb))

        loc = location_label(
            {"county": county, "state": state, "subdivision": subdiv}
        )
        label = f"{loc} → {feature}"

        if diffs:
            n_changed += 1
            print(f"  {C.warn('CHANGED')}  {label}")
            for fld, va, vb in diffs:
                va_d = _truncate(va, 40) if va else "(null)"
                vb_d = _truncate(vb, 40) if vb else "(null)"
                print(
                    f"           {C.DIM}├─{C.RESET} {fld}:"
                    f" {C.fail(va_d)} → {C.ok(vb_d)}"
                )
        elif verbose:
            n_unchanged += 1
            print(f"  {C.DIM}SAME{C.RESET}     {label}")
        else:
            n_unchanged += 1

    # ── Comparison summary ───────────────────────────────────────
    print(C.bold(f"\n{'─' * 70}"))
    print(C.bold("  Comparison summary"))
    print(f"{'─' * 70}")
    print(f"  Rows only in {label_a}: {len(only_a)}")
    print(f"  Rows only in {label_b}: {len(only_b)}")
    print(f"  Shared rows, changed : {C.warn(str(n_changed))}")
    print(f"  Shared rows, same    : {n_unchanged}")
    print()

    # ── Optional: score both against reference ───────────────────
    if ref_path:
        _print_ref_scoring(
            ref_path, df_a, df_b, label_a, label_b,
        )


def _print_ref_scoring(ref_path, df_a, df_b, label_a, label_b):
    """Score both runs against reference and show divergences"""
    ref = load_reference(ref_path)

    print(C.bold(f"{'─' * 70}"))
    print(C.bold("  Reference scoring"))
    print(f"{'─' * 70}\n")

    for label, df in [(label_a, df_a), (label_b, df_b)]:
        passed, total = score_run(ref, df.lazy())
        pct = (passed / total * 100) if total else 0
        clr = _pct_color(pct)
        print(f"  {label:.<40s} {clr(f'{passed}/{total}')} ({pct:.1f}%)")

    print()

    divergences = _find_divergences(ref, df_a, df_b)
    if divergences:
        print(f"  {C.bold('Divergent reference results')}:\n")
        for d in divergences:
            print(f"    {d['location']}  ·  {d['field']}")
            sa = C.ok("✓") if d["a_pass"] else C.fail("✗")
            sb = C.ok("✓") if d["b_pass"] else C.fail("✗")
            print(
                f"      {label_a}: {sa}  {label_b}: {sb}"
                f"  — {d['detail']}"
            )
        print()


def _find_divergences(
    ref: dict[str, dict],
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
) -> list[dict]:
    """Find checks where two runs disagree against the reference"""
    slices_a = {
        location_label(loc_data): loc_df
        for loc_data, loc_df in match_labels(ref, df_a.lazy())
    }

    divs = []
    for loc_data, loc_df_b in match_labels(ref, df_b.lazy()):
        loc_lbl = location_label(loc_data)
        loc_df_a = slices_a.get(loc_lbl, pl.DataFrame())

        for feat_name, checks in loc_data["features"].items():
            feat_label = f"{loc_lbl} → {feat_name}"
            row_a = find_feature_row(loc_df_a, feat_name)
            row_b = find_feature_row(loc_df_b, feat_name)

            for fld, check in checks.items():
                res_a = (
                    run_checks(row_a, {fld: check}) if row_a
                    else [CheckResult(
                        fld, check["mode"], False,
                        "", "(missing)", "row missing",
                    )]
                )
                res_b = (
                    run_checks(row_b, {fld: check}) if row_b
                    else [CheckResult(
                        fld, check["mode"], False,
                        "", "(missing)", "row missing",
                    )]
                )
                if res_a[0].passed != res_b[0].passed:
                    divs.append({
                        "location": feat_label,
                        "field": fld,
                        "a_pass": res_a[0].passed,
                        "b_pass": res_b[0].passed,
                        "detail": (
                            f"A: {res_a[0].actual[:50]}"
                            f"  B: {res_b[0].actual[:50]}"
                        ),
                    })
    return divs


# ── Init subcommand ──────────────────────────────────────────────────


def cmd_init(run_path: str, output_path: str):
    """Generate a reference YAML template from an existing CSV run"""
    df = load_run(run_path).collect()

    grouped: dict[str, dict[str, Any]] = {}
    for row in df.iter_rows(named=True):
        county = row["county"] or "unknown"
        state = row["state"] or "unknown"
        subdiv = row.get("subdivision")
        feature = row["feature"] or "unknown"
        fips = row.get("FIPS", "")

        loc_key = location_label({
            "county": county, "state": state,
            "subdivision": subdiv,
        })

        if loc_key not in grouped:
            grouped[loc_key] = {"FIPS": fips, "features": {}}

        feat_entry: dict[str, Any] = {}
        for fld in EXACT_FIELDS:
            v = row.get(fld)
            if v:
                feat_entry[fld] = v
        for fld in TEXT_FIELDS:
            v = row.get(fld)
            if v:
                feat_entry[fld] = "not_null"

        grouped[loc_key]["features"][feature] = (
            feat_entry or None
        )

    out = Path(output_path)
    lines = [
        "# Reference template — generated from: "
        + Path(run_path).name,
        "# Review each entry and adjust match modes:",
        '#   exact value  →  value: "1500"',
        "#   keywords     →  summary:",
        "#                     keywords: [word1, word2]",
        "#   not_null     →  section: not_null",
        "#   absent       →  adder: absent",
        "#   remove line  →  field won't be checked",
        "#",
        "# Location keys:",
        '#   County level   →  "County, State"',
        '#   Township level →  "Subdivision, County, State"',
        "",
    ]

    for loc_key in sorted(grouped):
        data = grouped[loc_key]
        lines.append(f'"{loc_key}":')
        if data["FIPS"]:
            lines.append(f'  FIPS: "{data["FIPS"]}"')
        lines.append("  features:")
        for feat_name in sorted(data["features"]):
            lines.append("")
            lines.append(f"    {feat_name}:")
            feat = data["features"][feat_name]
            if feat is None:
                lines.append("      # (no fields extracted)")
                continue
            for fld, val in feat.items():
                if val == "not_null":
                    lines.append(f"      {fld}: not_null")
                else:
                    lines.append(f'      {fld}: "{val}"')
        lines.append("")

    out.write_text("\n".join(lines))
    print(f"\n  {C.ok('✓')} Template written to {C.bold(str(out))}")
    print(
        f"  {C.DIM}Edit the file to set expected values"
        f" and match modes.{C.RESET}\n"
    )


# ── CLI ──────────────────────────────────────────────────────────────


@click.group(
    epilog=__doc__,
    context_settings={"help_option_names": ["-h", "--help"]},
)
def main():
    """Evaluate and compare LLM extraction runs against reference"""


@main.command("init")
@click.argument("run")
@click.option(
    "-o",
    "--output",
    "output_path",
    default="ground_truth.yaml",
    show_default=True,
    help="Output YAML path",
)
def init_command(run: str, output_path: str):
    """Scaffold reference YAML from a CSV run"""
    cmd_init(run, output_path)


@main.command("validate")
@click.argument("run")
@click.option(
    "-t",
    "--ref",
    "ref_path",
    required=True,
    help="Path to reference YAML file or directory",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Show passing checks too",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format for validation report",
)
def validate_command(
    run: str,
    ref_path: str,
    verbose: bool,
    output_format: str,
):
    """Validate a CSV run against reference"""
    print(cmd_validate(run, ref_path, verbose, output_format))


@main.command("compare")
@click.argument("run_a")
@click.argument("run_b")
@click.option(
    "-t",
    "--ref",
    "ref_path",
    default=None,
    help="Optional reference YAML file or directory",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Show unchanged rows too",
)
def compare_command(
    run_a: str,
    run_b: str,
    ref_path: str | None,
    verbose: bool,
):
    """Compare two CSV runs"""
    cmd_compare(run_a, run_b, ref_path, verbose)


if __name__ == "__main__":
    main()
