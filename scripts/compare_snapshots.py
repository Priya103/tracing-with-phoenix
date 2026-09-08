"""Diff two snapshots produced by scripts/run_evaluation.py --json ...

Shows how pass rates moved between two runs and flags newly-failing
cases. This is the DIY regression-tracking flow (no external service):

  1. Run the eval every so often with `--json snapshots/<date>.json`
  2. Diff any two snapshots with this script
  3. Investigate anything that regressed

Usage:
    python scripts/compare_snapshots.py snapshots/2026-09-01.json snapshots/2026-09-08.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("baseline", type=Path, help="Earlier snapshot (JSON).")
    p.add_argument("current", type=Path, help="Later snapshot (JSON).")
    return p.parse_args()


def _delta(old: float, new: float) -> str:
    diff = new - old
    if abs(diff) < 1e-9:
        return "     ="
    arrow = "▲" if diff > 0 else "▼"
    return f"{arrow} {diff:+.0%}"


def run(args: argparse.Namespace) -> int:
    baseline = json.loads(args.baseline.read_text())
    current = json.loads(args.current.read_text())

    print(f"Baseline: {args.baseline}  ({baseline['timestamp']})")
    print(f"Current : {args.current}   ({current['timestamp']})")
    if baseline["recommender_model"] != current["recommender_model"]:
        print(
            f"\n⚠  Model changed: {baseline['recommender_model']} -> {current['recommender_model']}"
        )

    print(
        f"\nOverall: {baseline['pass_rate']:.0%} -> {current['pass_rate']:.0%}   "
        f"{_delta(baseline['pass_rate'], current['pass_rate'])}"
    )

    print("\nPer-metric:")
    all_metrics = sorted(set(baseline["per_metric"]) | set(current["per_metric"]))
    for name in all_metrics:
        b = baseline["per_metric"].get(name, {}).get("rate", 0.0)
        c = current["per_metric"].get(name, {}).get("rate", 0.0)
        print(f"  {name:<25s} {b:.0%} -> {c:.0%}   {_delta(b, c)}")

    print("\nPer-category:")
    all_categories = sorted(set(baseline["per_category"]) | set(current["per_category"]))
    for name in all_categories:
        b = baseline["per_category"].get(name, {}).get("rate", 0.0)
        c = current["per_category"].get(name, {}).get("rate", 0.0)
        print(f"  {name:<25s} {b:.0%} -> {c:.0%}   {_delta(b, c)}")

    # Newly failing / newly passing cases.
    baseline_by_id = {c["id"]: c for c in baseline["per_case"]}
    current_by_id = {c["id"]: c for c in current["per_case"]}
    common = set(baseline_by_id) & set(current_by_id)

    regressed = sorted(
        cid for cid in common if baseline_by_id[cid]["success"] and not current_by_id[cid]["success"]
    )
    recovered = sorted(
        cid for cid in common if not baseline_by_id[cid]["success"] and current_by_id[cid]["success"]
    )

    if regressed:
        print(f"\n🔴 Regressed ({len(regressed)} case(s)):")
        for cid in regressed:
            print(f"  - {cid}  ({current_by_id[cid]['subcategory']})")

    if recovered:
        print(f"\n🟢 Recovered ({len(recovered)} case(s)):")
        for cid in recovered:
            print(f"  - {cid}  ({current_by_id[cid]['subcategory']})")

    if not regressed and not recovered:
        print("\nNo per-case status changes.")

    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))
