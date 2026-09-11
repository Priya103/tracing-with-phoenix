"""Push DeepEval per-case scores back onto their originating spans.

Called at the end of a run. Groups rows by metric name and uploads one
dataframe per metric via Phoenix's span-annotations API. In the Phoenix
UI, each recommender span then shows the score/label/explanation inline
and becomes filterable (e.g. `rubric.score < 0.7`).

No-ops when tracing is off (PHOENIX_ENABLED != 1) so this file is safe
to import unconditionally.
"""

from __future__ import annotations

import os
from typing import Any

# One accumulated row per (metric, span). Filled by `record()`, flushed
# by `flush()`. Kept module-global so the eval runner can just record
# and flush without threading extra state through its loop.
_ROWS: dict[str, list[dict[str, Any]]] = {}


def enabled() -> bool:
    return os.environ.get("PHOENIX_ENABLED") == "1"


def _normalize_metric_name(name: str) -> str:
    # rubric metrics are per-case (`rubric[fail-04]`); collapse for logging.
    return "rubric" if name.startswith("rubric[") else name


def record(
    span_id: str | None,
    case_id: str,
    metric_name: str,
    score: float | None,
    success: bool,
    explanation: str | None = None,
) -> None:
    """Buffer one metric result for later upload."""
    if not enabled() or not span_id:
        return
    normalized = _normalize_metric_name(metric_name)
    _ROWS.setdefault(normalized, []).append(
        {
            "span_id": span_id,
            "score": float(score) if score is not None else None,
            "label": "pass" if success else "fail",
            "explanation": (explanation or "")[:2000],  # trim to keep payload small
            "case_id": case_id,
        }
    )


def flush(annotator_kind: str = "LLM") -> int:
    """Upload buffered rows. Returns the total row count uploaded."""
    if not enabled() or not _ROWS:
        return 0
    try:
        import pandas as pd
        from phoenix.client import Client
    except ImportError as e:
        raise RuntimeError(
            "Annotation upload needs `pandas` and `arize-phoenix-client`. "
            "Install with: pip install -e '.[tracing]'"
        ) from e

    client = Client()
    total = 0
    for metric_name, rows in _ROWS.items():
        df = pd.DataFrame(rows).set_index("span_id")
        client.spans.log_span_annotations_dataframe(
            dataframe=df,
            annotation_name=metric_name,
            annotator_kind=annotator_kind,
        )
        total += len(rows)
    _ROWS.clear()
    return total
