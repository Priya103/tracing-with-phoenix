from __future__ import annotations

import os
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv(override=True)

DEFAULT_MODEL = os.getenv("RECOMMENDER_MODEL", "claude-sonnet-4-6")

# The Anthropic SDK appends "/v1/messages" to base_url, and OpenRouter's
# Anthropic-compatible endpoint lives at /api/v1/messages — so the base_url
# must stop at /api (not /api/v1), otherwise requests hit /api/v1/v1/messages.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"


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
        max_tokens: int = 512,
    ) -> Recommendation:
        user_prompt = build_user_prompt(
            favorite_genres=favorite_genres,
            liked_movies=liked_movies,
            mood=mood,
            avoid=avoid,
        )


        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )

        output = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return Recommendation(input=user_prompt, output=output, model=self.model)
