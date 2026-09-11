from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from opentelemetry import trace

_log = logging.getLogger(__name__)

# Sentinel returned when the tool loop can't produce a real answer. Kept
# non-empty so downstream metrics (which reject empty actual_output) can
# still score it — they'll rightfully fail the case, and the sentinel
# tells you *why* when you read the trace/snapshot.
EMPTY_OUTPUT_SENTINEL = "[recommender produced no output]"

from . import tools
from .prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv(override=True)

DEFAULT_MODEL = os.getenv("RECOMMENDER_MODEL", "claude-sonnet-4-6")

# Bounds the tool loop. Each iteration is one round-trip to Claude; a
# well-behaved run needs 2–4 (search → maybe details → submit).
MAX_ITERATIONS = 8

# The Anthropic SDK appends "/v1/messages" to base_url, and OpenRouter's
# Anthropic-compatible endpoint lives at /api/v1/messages — so the base_url
# must stop at /api (not /api/v1), otherwise requests hit /api/v1/v1/messages.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"

_tracer = trace.get_tracer("recommender.agent")


def _default_client() -> Anthropic:
    # Route through OpenRouter when its key is set; otherwise use the native
    # Anthropic API. The SDK sends the key as `x-api-key`, but OpenRouter
    # authenticates via `Authorization: Bearer` — inject that header so the
    # otherwise-Anthropic-shaped request is accepted.
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return Anthropic(
            base_url=OPENROUTER_BASE_URL,
            api_key=openrouter_key,
            default_headers={"Authorization": f"Bearer {openrouter_key}"},
        )
    return Anthropic()


@dataclass
class Recommendation:
    """A single recommender invocation, kept together so evaluators
    can inspect both what the user asked and what the model returned."""

    input: str
    output: str
    model: str
    span_id: str | None = None  # hex span_id of the `recommend` CHAIN span, if tracing is on


def _format_recommendations(picks: list[dict[str, Any]]) -> str:
    lines = []
    for i, p in enumerate(picks, 1):
        title = p.get("title", "?")
        year = p.get("year", "?")
        reason = p.get("reason", "").strip()
        lines.append(f"{i}. {title} ({year}) — {reason}")
    return "\n".join(lines)


def _set_span_kind(span, kind: str) -> None:
    if tools.SPAN_KIND_ATTR and kind:
        span.set_attribute(tools.SPAN_KIND_ATTR, kind)


def _set_span_io(span, input_obj: Any, output_obj: Any) -> None:
    if tools.INPUT_VALUE_ATTR:
        span.set_attribute(tools.INPUT_VALUE_ATTR, json.dumps(input_obj, ensure_ascii=False))
        span.set_attribute(tools.INPUT_MIME_ATTR, "application/json")
    if tools.OUTPUT_VALUE_ATTR:
        span.set_attribute(tools.OUTPUT_VALUE_ATTR, json.dumps(output_obj, ensure_ascii=False))
        span.set_attribute(tools.OUTPUT_MIME_ATTR, "application/json")


class MovieRecommender:
    def __init__(self, model: str = DEFAULT_MODEL, client: Anthropic | None = None):
        self.model = model
        self.client = client or _default_client()

    def recommend(
        self,
        favorite_genres: list[str] | None = None,
        liked_movies: list[str] | None = None,
        mood: str | None = None,
        avoid: list[str] | None = None,
        max_tokens: int = 1024,
    ) -> Recommendation:
        user_prompt = build_user_prompt(
            favorite_genres=favorite_genres,
            liked_movies=liked_movies,
            mood=mood,
            avoid=avoid,
        )

        with _tracer.start_as_current_span("recommend") as span:
            _set_span_kind(span, tools.CHAIN_KIND)
            span.set_attribute("recommender.model", self.model)
            output_text = self._run_tool_loop(user_prompt, max_tokens=max_tokens)
            _set_span_io(span, {"prompt": user_prompt}, {"output": output_text})
            ctx = span.get_span_context()
            span_id = format(ctx.span_id, "016x") if ctx and ctx.span_id else None

        return Recommendation(
            input=user_prompt, output=output_text, model=self.model, span_id=span_id
        )

    def _run_tool_loop(self, user_prompt: str, max_tokens: int) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]
        system = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        for _ in range(MAX_ITERATIONS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                tools=tools.TOOL_SCHEMAS,
                messages=messages,
            )

            # Persist the assistant turn so the model sees its own tool_uses
            # on the next round-trip.
            messages.append(
                {"role": "assistant", "content": [b.model_dump() for b in response.content]}
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            # Check for the terminating tool first — if the model submitted
            # its final answer, we stop here regardless of what else it did.
            for block in tool_uses:
                if block.name == "submit_recommendations":
                    picks = block.input.get("recommendations", []) if isinstance(block.input, dict) else []
                    formatted = _format_recommendations(picks)
                    if formatted:
                        return formatted
                    _log.warning("submit_recommendations returned no picks; falling back")

            if response.stop_reason == "tool_use" and tool_uses:
                tool_results = []
                for block in tool_uses:
                    result = tools.dispatch(block.name, block.input or {})
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                continue

            # No tools invoked — model produced a plain text response.
            # Return it directly (preserves the pre-tools contract).
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if text:
                return text
            break

        # Loop exhausted (or model returned nothing usable). Return a
        # sentinel so downstream metrics don't crash the whole batch.
        _log.warning("tool loop produced no output after %d iterations", MAX_ITERATIONS)
        return EMPTY_OUTPUT_SENTINEL
