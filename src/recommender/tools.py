"""Anthropic tool schemas and a dispatcher for the recommender.

Three tools:
  - search_movies: query the local catalog for candidates.
  - get_movie_details: fetch a full record before recommending.
  - submit_recommendations: structured-output tool that terminates the loop.

Each dispatch is wrapped in an OpenInference TOOL span so Phoenix shows
the tool name, inputs, and outputs alongside the model's LLM spans.
"""

from __future__ import annotations

import json
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from . import catalog

_tracer = trace.get_tracer("recommender.tools")

# Cached OpenInference semconv constants; None if the package isn't
# installed (tracing disabled), in which case we just skip setting them.
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
    _TOOL_NAME = SpanAttributes.TOOL_NAME
    _TOOL_KIND = OpenInferenceSpanKindValues.TOOL.value
    _CHAIN_KIND = OpenInferenceSpanKindValues.CHAIN.value
except ImportError:  # pragma: no cover - tracing extras not installed
    _SPAN_KIND = _INPUT_VALUE = _INPUT_MIME = None  # type: ignore[assignment]
    _OUTPUT_VALUE = _OUTPUT_MIME = _TOOL_NAME = None  # type: ignore[assignment]
    _TOOL_KIND = _CHAIN_KIND = ""  # type: ignore[assignment]


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_movies",
        "description": (
            "Search the movie catalog for candidates matching a free-text query. "
            "Query can mention genres (e.g. 'sci-fi thriller'), tags (e.g. 'slow-burn'), "
            "directors, or descriptive words. Returns up to `limit` matches with brief info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search query."},
                "limit": {
                    "type": "integer",
                    "description": "Max matches to return (default 5).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_movie_details",
        "description": (
            "Fetch the full catalog record for a specific movie by exact title. "
            "Use this to verify a movie exists and inspect its genres/tags/summary "
            "before including it in the final recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Exact movie title."},
            },
            "required": ["title"],
        },
    },
    {
        "name": "submit_recommendations",
        "description": (
            "Submit the final list of 3 movie recommendations. Calling this ends "
            "the recommendation flow — the caller uses this output directly. "
            "Each entry must include title, year, and a one-sentence reason tying "
            "it to the user's stated preferences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "year": {"type": "integer"},
                            "reason": {
                                "type": "string",
                                "description": "One sentence tying the pick to the user's preferences.",
                            },
                        },
                        "required": ["title", "year", "reason"],
                    },
                },
            },
            "required": ["recommendations"],
        },
    },
]


def _set_span_kind(span, kind: str) -> None:
    if _SPAN_KIND and kind:
        span.set_attribute(_SPAN_KIND, kind)


def _set_io(span, input_obj: Any, output_obj: Any) -> None:
    if _INPUT_VALUE:
        span.set_attribute(_INPUT_VALUE, json.dumps(input_obj, ensure_ascii=False))
        span.set_attribute(_INPUT_MIME, "application/json")
    if _OUTPUT_VALUE:
        span.set_attribute(_OUTPUT_VALUE, json.dumps(output_obj, ensure_ascii=False))
        span.set_attribute(_OUTPUT_MIME, "application/json")


def _run_search(inputs: dict[str, Any]) -> dict[str, Any]:
    query = inputs.get("query", "")
    limit = int(inputs.get("limit", 5))
    hits = catalog.search(query, limit=limit)
    return {
        "matches": [
            {"title": m.title, "year": m.year, "genres": list(m.genres), "tags": list(m.tags)}
            for m in hits
        ]
    }


def _run_details(inputs: dict[str, Any]) -> dict[str, Any]:
    title = inputs.get("title", "")
    movie = catalog.get(title)
    if movie is None:
        return {"found": False, "title": title}
    return {"found": True, **catalog.as_dict(movie)}


def _run_submit(inputs: dict[str, Any]) -> dict[str, Any]:
    # Passthrough — the loop uses the input directly. Returning it as
    # the tool_result keeps the trace symmetric (input == output).
    return {"accepted": True, "recommendations": inputs.get("recommendations", [])}


_DISPATCH = {
    "search_movies": _run_search,
    "get_movie_details": _run_details,
    "submit_recommendations": _run_submit,
}


def dispatch(name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool call inside its own OpenInference TOOL span."""
    with _tracer.start_as_current_span(f"tool.{name}") as span:
        _set_span_kind(span, _TOOL_KIND)
        if _TOOL_NAME:
            span.set_attribute(_TOOL_NAME, name)
        handler = _DISPATCH.get(name)
        if handler is None:
            err = {"error": f"unknown tool: {name}"}
            _set_io(span, inputs, err)
            span.set_status(Status(StatusCode.ERROR, f"unknown tool: {name}"))
            return err
        try:
            result = handler(inputs)
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        _set_io(span, inputs, result)
        return result


# Re-export so agent.py can start CHAIN spans without importing semconv.
CHAIN_KIND = _CHAIN_KIND
SPAN_KIND_ATTR = _SPAN_KIND
INPUT_VALUE_ATTR = _INPUT_VALUE
INPUT_MIME_ATTR = _INPUT_MIME
OUTPUT_VALUE_ATTR = _OUTPUT_VALUE
OUTPUT_MIME_ATTR = _OUTPUT_MIME
