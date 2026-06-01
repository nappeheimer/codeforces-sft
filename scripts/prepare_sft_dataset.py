#!/usr/bin/env python3
"""
Export SFT JSONL from mega_dataset parquet.

User: problem statement only (no traces).
Assistant: <think>...</think> + <code>...</code>
  - distill both reasoning and implementation (loss on full assistant turn).

Hard (>1700) wrong-path policy:
  - n_wp=2 -> export k=2 (paths 1..2 + s3_narrative)
  - n_wp=4 -> export k=4 (all paths + s3_narrative)
  - n_wp=3 -> tiered 25% drop of path 3 (hash-stable): export k=2 or k=3

Easy (<=1700):
  - small_thinking_trace only in thinking (no wrong paths)
  - s4_template in code
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Tiered P(drop path 3) for n_wp=3 rows; weighted ~25% on train pool.
P3_DROP_TIERS: list[tuple[int, int, float]] = [
    (1800, 2100, 0.40),
    (2100, 2400, 0.31),
    (2400, 2700, 0.23),
    (2700, 100_000, 0.10),
]

P3_DROP_HASH_SALT = "p3drop_v1"

SYSTEM_PROMPT = """\
You are an expert competitive programming assistant for Codeforces-style problems.

The user message contains only the problem statement, optional time/memory limits, and sample \
tests. Do not ask for clarifications; solve the problem as stated.

Respond with exactly two blocks and nothing else:

1. <think>...</think>
   - Reason step by step in natural language.
   - Derive the correct algorithm, check constraints and edge cases, and walk through samples \
if helpful.
   - Do not write final C++ inside the thinking block.

2. <code>...</code>
   - A complete, submission-ready C++17 solution.
   - Use the standard competitive template: includes, main(), and solve() (or equivalent \
structure used in the dataset).
   - No debug prints, no text outside the tags.

Output must be valid for the judge: correct I/O format, handle all constraints, and avoid \
off-by-one and overflow mistakes."""


def _clean(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("", "none", "nan") else s


def _hash_u(problem_id: str, salt: str = P3_DROP_HASH_SALT) -> float:
    h = int(hashlib.md5(f"{problem_id}|{salt}".encode()).hexdigest()[:8], 16)
    return h / 2**32


def p3_drop_probability(rating: int) -> float:
    for lo, hi, p in P3_DROP_TIERS:
        if lo <= rating < hi:
            return p
    return P3_DROP_TIERS[-1][2]


def count_stored_wrong_paths(row: pd.Series) -> int:
    n = 0
    for i in range(1, 5):
        wp = row.get(f"wrong_path_{i}")
        if isinstance(wp, dict) and _clean(wp.get("exploration")):
            n += 1
    return n


def assign_export_k(row: pd.Series, *, apply_p3_drop: bool) -> int:
    """How many wrong-path prefixes to include (always 1..k)."""
    rating = int(float(row.get("rating") or 0))
    if rating <= 1700:
        return 0

    n_wp = count_stored_wrong_paths(row)
    if n_wp <= 0:
        return 0
    if n_wp == 2:
        return 2
    if n_wp >= 4:
        return min(4, n_wp)

    # n_wp == 3
    if not apply_p3_drop:
        return 3
    pid = _clean(row.get("problem_id"))
    if _hash_u(pid) < p3_drop_probability(rating):
        return 2
    return 3


def _get(row: dict, *keys: str, default: str = "N/A") -> str:
    for key in keys:
        val = row.get(key)
        if val is not None:
            s = str(val).strip()
            if s and s.lower() not in ("nan", "none"):
                return s
    return default


def _interaction_section(row: dict) -> str:
    val = _get(row, "interaction_format", default="")
    return f"\nInteraction protocol:\n{val}\n" if val and val != "N/A" else ""


def _note_section(row: dict) -> str:
    val = _get(row, "note", default="")
    return f"\nNote:\n{val}\n" if val and val != "N/A" else ""


def _limits_section(row: dict) -> str:
    lines: list[str] = []
    tl = _fmt_limit(row.get("time_limit"), "s")
    ml = _fmt_limit(row.get("memory_limit"), "MB")
    if tl != "N/A":
        lines.append(f"Time limit: {tl}")
    if ml != "N/A":
        lines.append(f"Memory limit: {ml}")
    if not lines:
        return ""
    return "\n" + "\n".join(lines) + "\n"


def _fmt_examples(examples_raw: Any) -> str:
    if examples_raw is None:
        return "N/A"
    try:
        if isinstance(examples_raw, np.ndarray):
            examples_raw = examples_raw.tolist()
    except Exception:
        pass
    if isinstance(examples_raw, str):
        return examples_raw.strip() or "N/A"
    if isinstance(examples_raw, list):
        lines = []
        for i, ex in enumerate(examples_raw, 1):
            if isinstance(ex, dict):
                inp = str(ex.get("input", "") or "").strip()
                outp = str(ex.get("output", "") or "").strip()
                ib = "\n".join("    " + ln for ln in inp.splitlines()) if inp else "    (empty)"
                ob = "\n".join("    " + ln for ln in outp.splitlines()) if outp else "    (empty)"
                lines.append(f"  Example {i}:\n    Input:\n{ib}\n    Output:\n{ob}")
            else:
                lines.append(f"  Example {i}: {ex}")
        return "\n\n".join(lines) if lines else "N/A"
    return str(examples_raw)


def _fmt_limit(val: object, suffix: str) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    try:
        f = float(val)
        if f == int(f):
            return f"{int(f)}{suffix}"
        return f"{f}{suffix}"
    except (TypeError, ValueError):
        s = str(val).strip()
        return s if s else "N/A"


def build_user_content(row: pd.Series) -> str:
    r = row.to_dict()
    return f"""[PROBLEM]

[STATEMENT]
{_get(r, "description")}
{_limits_section(r)}
Input:
{_get(r, "input_format")}

Output:
{_get(r, "output_format")}
{_interaction_section(r)}{_note_section(r)}Examples:
{_fmt_examples(row.get("examples"))}
"""


def _wrong_path_block(wp: object) -> str:
    if not isinstance(wp, dict):
        return ""
    name = _clean(wp.get("name")) or "Wrong approach"
    exploration = _clean(wp.get("exploration"))
    if not exploration:
        return ""
    return f"[WRONG ATTEMPT — {name}]\n{exploration}"


def _strip_code_fences(code: str) -> str:
    m = re.search(r"```(?:cpp|c\+\+)?\s*\n([\s\S]*?)```", code, re.IGNORECASE)
    return m.group(1).strip() if m else code


def build_thinking(row: pd.Series, export_k: int) -> str | None:
    rating = float(row.get("rating") or 0)
    parts: list[str] = []

    if rating <= 1700:
        thinking = _clean(row.get("small_thinking_trace"))
        return thinking or None

    for i in range(1, export_k + 1):
        block = _wrong_path_block(row.get(f"wrong_path_{i}"))
        if block:
            parts.append(block)

    narrative = _clean(row.get("s3_narrative"))
    if narrative:
        parts.append(narrative)

    return "\n\n".join(parts) if parts else None


def build_assistant_content(row: pd.Series, export_k: int) -> str | None:
    thinking = build_thinking(row, export_k)
    code = _strip_code_fences(_clean(row.get("s4_template")))
    if not thinking or not code:
        return None
    return (
        f"<think>\n{thinking}\n</think>\n"
        f"<code>\n{code}\n</code>"
    )


def row_to_record(
    row: pd.Series,
    *,
    apply_p3_drop: bool,
    include_system: bool,
) -> tuple[dict | None, dict | None]:
    if not _clean(row.get("description")):
        return None, None

    rating = float(row.get("rating") or 0)
    export_k = assign_export_k(row, apply_p3_drop=apply_p3_drop)
    n_wp = count_stored_wrong_paths(row)

    assistant = build_assistant_content(row, export_k)
    if not assistant:
        return None, None

    user = build_user_content(row)
    messages: list[dict[str, str]] = []
    if include_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})

    meta = {
        "problem_id": _clean(row.get("problem_id")),
        "rating": rating,
        "tier": "easy" if rating <= 1700 else "hard",
        "n_wp": n_wp,
        "export_k": export_k,
        "p3_dropped": bool(rating > 1700 and n_wp == 3 and export_k == 2),
        "user_chars": len(user),
        "assistant_chars": len(assistant),
    }
    return {"messages": messages, **meta}, meta


# Zone A/B/C: (name, fraction of dataset, target P(easy))
# Targets calibrated so ~3957 easy rows suffice (85% / 42% / 25% ≈ 1736+1715+510).
CURRICULUM_RAMP_ZONES: list[tuple[str, float, float]] = [
    ("A", 0.25, 0.85),
    ("B", 0.50, 0.42),
    ("C", 0.25, 0.25),
]

# Zone A only: no very high-rated problems in the warm-up block.
DEFAULT_ZONE_A_MAX_RATING = 2200


def _easy_sort_key(r: dict) -> tuple:
    return (float(r.get("rating", 0)), r.get("problem_id", ""))


def _hard_sort_key(r: dict) -> tuple:
    return (int(r.get("export_k", 0)), float(r.get("rating", 0)), r.get("problem_id", ""))


def _fill_zone_from_pools(
    zone_n: int,
    p_easy_target: float,
    easy_pool: list[dict],
    hard_pool: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Take rows from pool fronts; return (zone_rows, easy_remainder, hard_remainder)."""
    n_easy_want = int(round(zone_n * p_easy_target))
    n_hard_want = zone_n - n_easy_want
    zone_easy: list[dict] = []
    zone_hard: list[dict] = []
    ei = 0
    hi = 0

    while len(zone_easy) < n_easy_want and ei < len(easy_pool):
        zone_easy.append(easy_pool[ei])
        ei += 1
    while len(zone_hard) < n_hard_want and hi < len(hard_pool):
        zone_hard.append(hard_pool[hi])
        hi += 1
    while len(zone_easy) + len(zone_hard) < zone_n and ei < len(easy_pool):
        zone_easy.append(easy_pool[ei])
        ei += 1
    while len(zone_easy) + len(zone_hard) < zone_n and hi < len(hard_pool):
        zone_hard.append(hard_pool[hi])
        hi += 1

    zone_rows = zone_easy + zone_hard
    return zone_rows, easy_pool[ei:], hard_pool[hi:]


def _zone_report_row(
    zone_name: str,
    zone_rows: list[dict],
    *,
    target_frac: float,
    target_p_easy: float,
    rating_cap: int | None = None,
) -> dict:
    n_easy = sum(1 for r in zone_rows if r.get("tier") == "easy")
    hard_rows = [r for r in zone_rows if r.get("tier") == "hard"]
    rep: dict[str, Any] = {
        "zone": zone_name,
        "target_frac": target_frac,
        "target_p_easy": target_p_easy,
        "n": len(zone_rows),
        "n_easy": n_easy,
        "n_hard": len(zone_rows) - n_easy,
        "actual_p_easy": n_easy / len(zone_rows) if zone_rows else 0.0,
        "rating_mean": sum(float(r["rating"]) for r in zone_rows) / len(zone_rows) if zone_rows else 0.0,
        "rating_min": min((float(r["rating"]) for r in zone_rows), default=0.0),
        "rating_max": max((float(r["rating"]) for r in zone_rows), default=0.0),
        "hard_k2": sum(1 for r in hard_rows if r.get("export_k") == 2),
        "hard_k3": sum(1 for r in hard_rows if r.get("export_k") == 3),
        "hard_k4": sum(1 for r in hard_rows if r.get("export_k") == 4),
    }
    if rating_cap is not None:
        rep["rating_cap"] = rating_cap
    return rep


def curriculum_ramp_order(
    records: list[dict],
    *,
    zone_a_max_rating: int = DEFAULT_ZONE_A_MAX_RATING,
) -> tuple[list[dict], list[dict]]:
    """Zones A→B→C. Zone A draws only from rows with rating <= zone_a_max_rating."""
    n = len(records)
    easy_all = sorted((r for r in records if r.get("tier") == "easy"), key=_easy_sort_key)
    hard_all = sorted((r for r in records if r.get("tier") == "hard"), key=_hard_sort_key)

    easy_a = [r for r in easy_all if float(r["rating"]) <= zone_a_max_rating]
    easy_hi = [r for r in easy_all if float(r["rating"]) > zone_a_max_rating]
    hard_a = [r for r in hard_all if float(r["rating"]) <= zone_a_max_rating]
    hard_hi = [r for r in hard_all if float(r["rating"]) > zone_a_max_rating]

    out: list[dict] = []
    zone_reports: list[dict] = []

    # --- Zone A (capped rating) ---
    name_a, frac_a, p_easy_a = CURRICULUM_RAMP_ZONES[0]
    zone_n_a = int(round(n * frac_a))
    zone_a, easy_a_rem, hard_a_rem = _fill_zone_from_pools(zone_n_a, p_easy_a, easy_a, hard_a)
    for r in zone_a:
        r["curriculum_zone"] = name_a
    out.extend(zone_a)
    zone_reports.append(
        _zone_report_row(name_a, zone_a, target_frac=frac_a, target_p_easy=p_easy_a, rating_cap=zone_a_max_rating)
    )

    # B/C pools: capped leftovers + all above-cap rows (re-sorted)
    easy_bc = sorted(easy_a_rem + easy_hi, key=_easy_sort_key)
    hard_bc = sorted(hard_a_rem + hard_hi, key=_hard_sort_key)

    for zi in range(1, len(CURRICULUM_RAMP_ZONES)):
        name, frac, p_easy = CURRICULUM_RAMP_ZONES[zi]
        if zi == len(CURRICULUM_RAMP_ZONES) - 1:
            zone_n = n - len(out)
        else:
            zone_n = int(round(n * frac))

        zone_rows, easy_bc, hard_bc = _fill_zone_from_pools(zone_n, p_easy, easy_bc, hard_bc)
        for r in zone_rows:
            r["curriculum_zone"] = name
        out.extend(zone_rows)
        zone_reports.append(_zone_report_row(name, zone_rows, target_frac=frac, target_p_easy=p_easy))

    if easy_bc or hard_bc:
        for r in easy_bc + hard_bc:
            r["curriculum_zone"] = "tail"
        out.extend(easy_bc + hard_bc)
        zone_reports.append(
            {
                "zone": "tail",
                "n": len(easy_bc) + len(hard_bc),
                "n_easy": len(easy_bc),
                "n_hard": len(hard_bc),
            }
        )

    return out, zone_reports


def cumulative_zone_report(records: list[dict], n_bins: int = 10) -> list[dict]:
    """P(easy) and mean rating in each contiguous decile of the ordered file."""
    n = len(records)
    bins: list[dict] = []
    for b in range(n_bins):
        lo = b * n // n_bins
        hi = (b + 1) * n // n_bins
        chunk = records[lo:hi]
        if not chunk:
            continue
        n_easy = sum(1 for r in chunk if r.get("tier") == "easy")
        bins.append(
            {
                "bin": b + 1,
                "row_lo": lo,
                "row_hi": hi - 1,
                "n": len(chunk),
                "p_easy": n_easy / len(chunk),
                "rating_mean": sum(float(r["rating"]) for r in chunk) / len(chunk),
                "export_k_mean": sum(int(r.get("export_k", 0)) for r in chunk) / len(chunk),
            }
        )
    return bins


def print_curriculum_report(records: list[dict], zone_reports: list[dict]) -> None:
    n = len(records)
    print("\n=== Curriculum ramp zones (A → B → C) ===")
    print(f"Total rows: {n:,}\n")
    print(
        f"{'Zone':<6} {'Rows':>6} {'Target':>8} {'p_easy':>8} {'actual':>8} "
        f"{'rating':>14} {'hard k2/k3/k4':>16}"
    )
    print("-" * 72)
    for z in zone_reports:
        if z.get("zone") == "tail":
            print(f"{'tail':<6} {z['n']:>6}   (remainder pools)")
            continue
        rk = f"{z.get('hard_k2',0)}/{z.get('hard_k3',0)}/{z.get('hard_k4',0)}"
        cap = z.get("rating_cap")
        cap_s = f" cap≤{cap}" if cap is not None else ""
        rspan = f"{z['rating_min']:.0f}–{z['rating_max']:.0f} (μ={z['rating_mean']:.0f})"
        print(
            f"{z['zone']:<6} {z['n']:>6} {z['target_p_easy']:>7.0%} "
            f"{z['target_p_easy']:>7.0%} {z['actual_p_easy']:>7.1%} {rspan:>14}{cap_s} {rk:>16}"
        )

    print("\n=== Cumulative deciles (position in JSONL) ===")
    print(f"{'Bin':>4} {'rows':>12} {'p_easy':>8} {'μ_rating':>10} {'μ_export_k':>12}")
    print("-" * 50)
    for b in cumulative_zone_report(records):
        rows = f"{b['row_lo']}–{b['row_hi']}"
        print(
            f"{b['bin']:>4} {rows:>12} {b['p_easy']:>7.1%} "
            f"{b['rating_mean']:>10.0f} {b['export_k_mean']:>12.2f}"
        )

    # Global compare
    p_easy_all = sum(1 for r in records if r.get("tier") == "easy") / n
    first_q = records[: n // 4]
    last_q = records[-n // 4 :]
    print("\n=== Sanity ===")
    print(f"  Dataset P(easy) overall: {p_easy_all:.1%}")
    print(
        f"  First 25% of file: P(easy)={sum(1 for r in first_q if r['tier']=='easy')/len(first_q):.1%}, "
        f"μ_rating={sum(float(r['rating']) for r in first_q)/len(first_q):.0f}"
    )
    print(
        f"  Last 25% of file:  P(easy)={sum(1 for r in last_q if r['tier']=='easy')/len(last_q):.1%}, "
        f"μ_rating={sum(float(r['rating']) for r in last_q)/len(last_q):.0f}"
    )


def sort_records(records: list[dict], order: str, *, zone_a_max_rating: int = DEFAULT_ZONE_A_MAX_RATING) -> list[dict]:
    if order == "none":
        return records
    if order == "rating_asc":
        return sorted(records, key=lambda r: (r.get("rating", 0), r.get("problem_id", "")))
    if order == "rating_desc":
        return sorted(records, key=lambda r: (-r.get("rating", 0), r.get("problem_id", "")))
    if order == "curriculum_v1":
        def key(r: dict) -> tuple:
            rating = float(r.get("rating", 0))
            tier = 0 if rating <= 1700 else 1
            export_k = int(r.get("export_k", 0))
            return (tier, export_k, rating, r.get("problem_id", ""))

        return sorted(records, key=key)
    if order == "curriculum_ramp":
        ordered, _ = curriculum_ramp_order(records, zone_a_max_rating=zone_a_max_rating)
        return ordered
    raise ValueError(f"unknown order: {order}")


def summarize_manifest(rows: list[dict]) -> dict:
    hard = [r for r in rows if r.get("tier") == "hard"]
    p3 = [r for r in hard if r.get("n_wp") == 3]
    dropped = [r for r in p3 if r.get("p3_dropped")]
    out: dict[str, Any] = {
        "n_rows": len(rows),
        "n_easy": sum(1 for r in rows if r.get("tier") == "easy"),
        "n_hard": len(hard),
        "hard_k2": sum(1 for r in hard if r.get("export_k") == 2),
        "hard_k3": sum(1 for r in hard if r.get("export_k") == 3),
        "hard_k4": sum(1 for r in hard if r.get("export_k") == 4),
        "p3_pool": len(p3),
        "p3_dropped": len(dropped),
        "p3_drop_rate": len(dropped) / len(p3) if p3 else 0.0,
    }
    return out


def build_all_records(
    df: pd.DataFrame,
    *,
    apply_p3_drop: bool,
    include_system: bool,
) -> tuple[list[dict], int, list[dict]]:
    records: list[dict] = []
    manifest_rows: list[dict] = []
    skipped = 0
    for _, row in df.iterrows():
        rec, meta = row_to_record(
            row,
            apply_p3_drop=apply_p3_drop,
            include_system=include_system,
        )
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
        manifest_rows.append(meta)
    return records, skipped, manifest_rows


def export_split(
    df: pd.DataFrame,
    out_path: Path,
    *,
    apply_p3_drop: bool,
    include_system: bool,
    order: str,
    zone_a_max_rating: int = DEFAULT_ZONE_A_MAX_RATING,
) -> tuple[int, int, list[dict]]:
    records, skipped, manifest_rows = build_all_records(
        df,
        apply_p3_drop=apply_p3_drop,
        include_system=include_system,
    )

    records = sort_records(records, order, zone_a_max_rating=zone_a_max_rating)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps({"messages": rec["messages"]}, ensure_ascii=False) + "\n")

    return len(records), skipped, manifest_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Export SFT JSONL with thinking + code assistant.")
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None, help="defaults to <output>.manifest.json")
    p.add_argument(
        "--apply-p3-drop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="tiered 25%% path-3 drop on n_wp=3 hard rows (train)",
    )
    p.add_argument("--no-system", action="store_true", help="omit system message")
    p.add_argument(
        "--order",
        choices=("none", "rating_asc", "rating_desc", "curriculum_v1", "curriculum_ramp"),
        default="none",
        help="line order in JSONL; curriculum_ramp = zones A/B/C with rising P(hard)",
    )
    p.add_argument(
        "--curriculum-report-only",
        action="store_true",
        help="print zone/decile stats for curriculum_ramp; do not write JSONL",
    )
    p.add_argument(
        "--zone-a-max-rating",
        type=int,
        default=DEFAULT_ZONE_A_MAX_RATING,
        help=f"zone A only includes rating <= this (default {DEFAULT_ZONE_A_MAX_RATING})",
    )
    args = p.parse_args()

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")

    print(f"Loading {args.parquet}...", flush=True)
    df = pd.read_parquet(args.parquet)
    print(f"  rows: {len(df):,}", flush=True)

    if args.curriculum_report_only:
        records, n_skip, _ = build_all_records(
            df,
            apply_p3_drop=args.apply_p3_drop,
            include_system=not args.no_system,
        )
        ordered, zone_reports = curriculum_ramp_order(
            records, zone_a_max_rating=args.zone_a_max_rating
        )
        print(f"Built {len(ordered):,} records, skipped={n_skip}")
        print(f"Zone A max rating: {args.zone_a_max_rating}")
        print_curriculum_report(ordered, zone_reports)
        return

    n_ok, n_skip, manifest_rows = export_split(
        df,
        args.output,
        apply_p3_drop=args.apply_p3_drop,
        include_system=not args.no_system,
        order=args.order,
        zone_a_max_rating=args.zone_a_max_rating,
    )

    summary = summarize_manifest(manifest_rows)
    summary.update(
        {
            "parquet": str(args.parquet),
            "output": str(args.output),
            "apply_p3_drop": args.apply_p3_drop,
            "order": args.order,
            "zone_a_max_rating": args.zone_a_max_rating,
            "skipped": n_skip,
            "p3_drop_tiers": P3_DROP_TIERS,
        }
    )
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output} ({n_ok:,} lines, {mb:.1f} MB), skipped={n_skip}")
    print(f"Manifest: {manifest_path}")
    print(
        f"  hard k=2/3/4: {summary['hard_k2']}/{summary['hard_k3']}/{summary['hard_k4']}  "
        f"p3_drop: {summary['p3_dropped']}/{summary['p3_pool']} ({summary['p3_drop_rate']:.1%})"
    )


if __name__ == "__main__":
    main()
