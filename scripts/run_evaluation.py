"""Quality-monitoring runner for the movie recommender.

Runs the full golden dataset through the recommender, scores every case
with the same metrics the pytest suite uses, and prints an aggregate
report. Unlike `pytest tests/`, this:

  - Returns a structured aggregate (overall, per-metric, per-category
    pass rates) instead of just per-case pass/fail.
  - Runs each case's metrics in parallel via asyncio.
  - Can dump a JSON snapshot for regression tracking across runs; pair
    with `scripts/compare_snapshots.py` to diff two runs.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --category failure
    python scripts/run_evaluation.py --limit 5             # smoke test
    python scripts/run_evaluation.py --json snapshots/2026-09-08.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.test_case import LLMTestCase

from evaluation import GOLDEN_DATASET, GoldenCase, metrics_for
from recommender import MovieRecommender


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--category",
        choices=["normal", "edge", "failure"],
        help="Only run cases from this bucket.",
    )
    p.add_argument(
        "--limit",
        type=int,
        help="Only run the first N cases (after --category filter). Useful for smoke tests.",
    )
    p.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="Write a JSON snapshot of the run to PATH.",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Async concurrency for judge calls (default: 10).",
    )
    return p.parse_args()


def _normalize_metric_name(name: str) -> str:
    # rubric metrics are per-case (`rubric[fail-04]`); collapse for aggregation.
    return "rubric" if name.startswith("rubric[") else name


def run(args: argparse.Namespace) -> int:
    cases: list[GoldenCase] = list(GOLDEN_DATASET)
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("No cases match the filter.", file=sys.stderr)
        return 2

    print(f"Running {len(cases)} case(s) through the recommender...\n")
    recommender = MovieRecommender()
    per_case: list[tuple[GoldenCase, object]] = []

    for i, case in enumerate(cases, 1):
        print(f"  [{i:>3}/{len(cases)}] {case.id:<10} ({case.category})", flush=True)
        rec = recommender.recommend(**case.inputs)
        tc = LLMTestCase(input=rec.input, actual_output=rec.output)
        result = evaluate(
            test_cases=[tc],
            metrics=metrics_for(case),
            async_config=AsyncConfig(run_async=True, max_concurrent=args.max_concurrent),
            display_config=DisplayConfig(print_results=False, show_indicator=False),
        )
        for r in result.test_results:
            per_case.append((case, r))

    # ---- aggregate ----
    total = len(per_case)
    passed = sum(1 for _, r in per_case if r.success)

    per_metric: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for case, r in per_case:
        per_category[case.category][1] += 1
        if r.success:
            per_category[case.category][0] += 1
        for m in r.metrics_data:
            name = _normalize_metric_name(m.name)
            per_metric[name][1] += 1
            if m.success:
                per_metric[name][0] += 1

    print(f"\nOverall: {passed}/{total} ({passed / total:.0%})")

    print("\nPer-metric pass rate:")
    for name, (p, t) in sorted(per_metric.items()):
        print(f"  {name:<25s} {p:>3d}/{t:<3d} ({p / t:.0%})")

    print("\nPer-category pass rate:")
    for name, (p, t) in sorted(per_category.items()):
        print(f"  {name:<25s} {p:>3d}/{t:<3d} ({p / t:.0%})")

    # ---- snapshot ----
    if args.json:
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommender_model": recommender.model,
            "case_count": total,
            "passed": passed,
            "pass_rate": passed / total,
            "per_metric": {
                name: {"passed": p, "total": t, "rate": p / t}
                for name, (p, t) in per_metric.items()
            },
            "per_category": {
                name: {"passed": p, "total": t, "rate": p / t}
                for name, (p, t) in per_category.items()
            },
            "per_case": [
                {
                    "id": case.id,
                    "category": case.category,
                    "subcategory": case.subcategory,
                    "success": r.success,
                    "metric_scores": {
                        _normalize_metric_name(m.name): m.score for m in r.metrics_data
                    },
                }
                for case, r in per_case
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(snapshot, indent=2))
        print(f"\nSnapshot written to {args.json}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run(parse_args()))
