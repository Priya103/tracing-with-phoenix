"""Arize Phoenix OTLP tracing setup — enabled only when PHOENIX_ENABLED=1."""

# Deliberately no `from __future__ import annotations` here: it would
# bind `annotations` as a `_Feature` in this namespace and shadow the
# `from . import annotations` submodule re-export below.

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace

from . import annotations  # re-exported so callers can `from tracing import annotations`

_INSTRUMENTED = False
PHOENIX_ENDPOINT = "http://127.0.0.1:6006/v1/traces"

_tracer = trace.get_tracer("recommender.eval")

# Cached OpenInference semconv constants; None if the package isn't
# available (tracing extras not installed).
try:
    from openinference.semconv.trace import (
        OpenInferenceSpanKindValues,
        SpanAttributes,
    )

    _SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
    _INPUT_VALUE = SpanAttributes.INPUT_VALUE
    _INPUT_MIME = SpanAttributes.INPUT_MIME_TYPE
    _OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
    _OUTPUT_MIME = SpanAttributes.OUTPUT_MIME_TYPE
    _CHAIN_KIND = OpenInferenceSpanKindValues.CHAIN.value
except ImportError:  # pragma: no cover
    _SPAN_KIND = _INPUT_VALUE = _INPUT_MIME = None  # type: ignore[assignment]
    _OUTPUT_VALUE = _OUTPUT_MIME = None  # type: ignore[assignment]
    _CHAIN_KIND = ""


def setup_tracing(
    project_name: str = "tracing-movie-recommendation",
    instrument_fastapi: bool = False,
) -> None:
    """Wire OpenTelemetry to Phoenix.

    Reads two optional env vars:
      - PHOENIX_COLLECTOR_ENDPOINT: overrides the default localhost endpoint
        (use this to point at Phoenix Cloud or a shared collector).
      - : float in [0.0, 1.0]. When set, only that
        fraction of root traces are sampled (parent-based, so child
        spans inherit). Unset = 100% (fine for dev / evals).
    """
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return
    if os.environ.get("PHOENIX_ENABLED") != "1":
        return

    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from phoenix.otel import register
    except ImportError as e:
        raise RuntimeError(
            "PHOENIX_ENABLED=1 is set but tracing dependencies are missing. "
            "Install with: pip install -e '.[tracing]'"
        ) from e

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", PHOENIX_ENDPOINT)
    register_kwargs: dict[str, Any] = {
        "project_name": project_name,
        "endpoint": endpoint,
        "batch": True,  # cheap in prod; single-span exports would murder throughput
    }
    ratio = os.environ.get("PHOENIX_SAMPLE_RATIO")
    if ratio:
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

        register_kwargs["sampler"] = ParentBased(TraceIdRatioBased(float(ratio)))

    register(**register_kwargs)
    AnthropicInstrumentor().instrument()
    OpenAIInstrumentor().instrument()

    if instrument_fastapi:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor().instrument()
        except ImportError as e:
            raise RuntimeError(
                "instrument_fastapi=True but opentelemetry-instrumentation-fastapi is missing. "
                "Install with: pip install -e '.[server]'"
            ) from e

    _INSTRUMENTED = True
    sample_note = f", sampling={ratio}" if ratio else ""
    print(f"[tracing] Phoenix enabled -> {endpoint} (project: {project_name}{sample_note})")


@contextmanager
def eval_case_span(
    case_id: str,
    category: str,
    subcategory: str,
    inputs: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Open a CHAIN span for one golden case.

    Nests the recommender's `recommend` span and any DeepEval judge
    calls under a single parent, so a failing case in Phoenix links
    to exactly the LLM + tool + judge spans it produced.
    """
    with _tracer.start_as_current_span(f"eval_case: {case_id}") as span:
        if _SPAN_KIND:
            span.set_attribute(_SPAN_KIND, _CHAIN_KIND)
        span.set_attribute("case.id", case_id)
        span.set_attribute("case.category", category)
        span.set_attribute("case.subcategory", subcategory)
        if _INPUT_VALUE and inputs is not None:
            span.set_attribute(_INPUT_VALUE, json.dumps(inputs, ensure_ascii=False, default=str))
            span.set_attribute(_INPUT_MIME, "application/json")
        yield span
