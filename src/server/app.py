"""Production FastAPI surface for the movie recommender.

Companion to the eval harness. Same recommender internals, but the
trace parent is an HTTP request instead of an `eval_case:` chain,
and per-request identity (session, user, tenant) is attached via
OpenInference context managers so Phoenix can group traces by
conversation or filter by user.

Run:
    export PHOENIX_ENABLED=1
    export PHOENIX_SAMPLE_RATIO=0.1        # optional: sample 10% of traces
    export PHOENIX_COLLECTOR_ENDPOINT=...  # optional: point at Phoenix Cloud
    uvicorn server.app:app --reload --port 8000

Then:
    curl -sS -X POST http://127.0.0.1:8000/recommend \\
        -H 'content-type: application/json' \\
        -H 'x-session-id: s-abc' -H 'x-user-id: u-42' \\
        -d '{"favorite_genres":["thriller"],"mood":"slow-burn"}'
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, Header
from openinference.instrumentation import using_metadata, using_session, using_user
from pydantic import BaseModel, Field

from recommender import MovieRecommender
from tracing import setup_tracing

# instrument_fastapi=True registers OTel's FastAPI middleware, which
# creates the HTTP `SERVER` span that everything downstream nests under.
setup_tracing(instrument_fastapi=True)

app = FastAPI(title="Movie Recommender", version="0.1.0")
_recommender = MovieRecommender()


class RecommendRequest(BaseModel):
    favorite_genres: Optional[list[str]] = None
    liked_movies: Optional[list[str]] = None
    mood: Optional[str] = None
    avoid: Optional[list[str]] = None


class RecommendResponse(BaseModel):
    output: str
    model: str
    request_id: str
    trace_span_id: Optional[str] = Field(
        default=None,
        description="Hex span_id of the `recommend` chain span — useful for support tickets.",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(
    body: RecommendRequest,
    x_session_id: Optional[str] = Header(default=None, alias="x-session-id"),
    x_user_id: Optional[str] = Header(default=None, alias="x-user-id"),
    x_tenant_id: Optional[str] = Header(default=None, alias="x-tenant-id"),
    x_request_id: Optional[str] = Header(default=None, alias="x-request-id"),
) -> RecommendResponse:
    # Fall back to fresh IDs so every trace still has *something* to
    # group by even if the caller didn't send headers.
    session_id = x_session_id or f"anon-{uuid.uuid4().hex[:8]}"
    user_id = x_user_id or "anonymous"
    request_id = x_request_id or uuid.uuid4().hex
    metadata = {
        "request_id": request_id,
        "tenant_id": x_tenant_id or "default",
    }

    with using_session(session_id), using_user(user_id), using_metadata(metadata):
        rec = _recommender.recommend(**body.model_dump(exclude_none=True))

    return RecommendResponse(
        output=rec.output,
        model=rec.model,
        request_id=request_id,
        trace_span_id=rec.span_id,
    )
