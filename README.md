# tracing-with-phoenix

A movie recommender wired up three ways:

- **DeepEval** evaluates a 58-case golden dataset (per-case pytest gate
  and an aggregate report with JSON snapshots for regression tracking).
- **Arize Phoenix** captures OpenInference traces for every recommender
  call — the tool loop, each `search_movies` / `get_movie_details` /
  `submit_recommendations` invocation, the LLM turn, and (in the eval
  runner) the wrapping `eval_case` chain. DeepEval scores are pushed
  back onto those spans as annotations.
- A **FastAPI** surface exposes the same recommender over HTTP with
  per-request session / user / tenant metadata attached to the trace.

The recommender itself is a Claude-driven agent with a tool loop over a
small hand-curated movie catalog.

## Layout

```
.
├── src/
│   ├── recommender/           # Claude-powered agent
│   │   ├── agent.py           # MovieRecommender.recommend(...) + tool loop
│   │   ├── prompts.py         # System + user prompt templates
│   │   ├── tools.py           # Anthropic tool schemas + OpenInference TOOL spans
│   │   └── catalog.py         # In-memory movie catalog (~50 titles)
│   ├── tracing/
│   │   ├── __init__.py        # setup_tracing() + eval_case_span()
│   │   └── annotations.py     # Push DeepEval scores onto Phoenix spans
│   └── server/
│       └── app.py             # FastAPI /recommend endpoint
├── evaluation/
│   ├── datasets.py            # 58 goldens: 20 normal / 15 edge / 23 failure
│   └── metrics.py             # Judge model (gpt-4o-mini) + DeepEval metrics
├── tests/
│   └── test_recommender.py    # pytest + deepeval assert_test
├── scripts/
│   ├── run_evaluation.py      # aggregate evaluate() runner + JSON snapshots
│   └── compare_snapshots.py   # diff two snapshots, flag regressions
├── conftest.py                # setup_tracing() for pytest
├── main.py                    # Manual demo entry point
├── pyproject.toml
└── .env.example
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,tracing,server]"     # pick the extras you need
cp .env.example .env                       # then fill in the keys below
```

Extras:

- `dev` — pytest.
- `tracing` — `arize-phoenix-otel`, `arize-phoenix-client`, OpenInference
  instrumentors for Anthropic + OpenAI, pandas (for annotation upload).
  Required whenever `PHOENIX_ENABLED=1`.
- `server` — FastAPI, uvicorn, OTel's FastAPI instrumentor.

### Keys

- `OPENROUTER_API_KEY` — the recommender calls Claude via OpenRouter's
  Anthropic-compatible endpoint, **and** the DeepEval judge
  (`openai/gpt-4o-mini`) is routed through OpenRouter too. One key wires
  up both sides. If you'd rather call Anthropic directly, set
  `ANTHROPIC_API_KEY` instead — but the judge configured in
  `evaluation/metrics.py` still needs OpenRouter.
- `RECOMMENDER_MODEL` (optional) — override the recommender model,
  e.g. `anthropic/claude-sonnet-4.5` when routing through OpenRouter,
  or `claude-sonnet-4-6` for the native API. Defaults to
  `claude-sonnet-4-6`.

### Phoenix env vars (all optional)

- `PHOENIX_ENABLED=1` — turn tracing on. Off by default so tests and
  the demo don't require Phoenix to be running.
- `PHOENIX_COLLECTOR_ENDPOINT` — override the OTLP endpoint. Defaults
  to `http://127.0.0.1:6006/v1/traces` (local Phoenix). Point this at
  Phoenix Cloud or a shared collector if you have one.
- `PHOENIX_SAMPLE_RATIO` — float in `[0.0, 1.0]`, parent-based sampling.
  Unset = 100% (fine for dev / evals; consider lowering in production).

## Run the demo

```bash
python main.py
```

One-shot recommendation against a hard-coded prompt. Useful smoke test
that your keys are wired up.

## Run the recommender as an HTTP service

```bash
export PHOENIX_ENABLED=1                          # optional: emit traces
uvicorn server.app:app --reload --port 8000

curl -sS -X POST http://127.0.0.1:8000/recommend \
    -H 'content-type: application/json' \
    -H 'x-session-id: s-abc' -H 'x-user-id: u-42' \
    -d '{"favorite_genres":["thriller"],"mood":"slow-burn"}'
```

The server attaches `x-session-id` / `x-user-id` / `x-tenant-id` (with
sensible fallbacks) to each trace via OpenInference context managers,
so Phoenix can group by conversation or filter by user. The response
includes the `trace_span_id` of the `recommend` chain span — handy for
correlating a support ticket with a specific trace.

## Run the evaluation (pytest / `assert_test`)

Fast per-case pass/fail — this is the CI gate.

```bash
pytest                              # run all 58 goldens
pytest tests/ -k "fail-"            # only failure cases
pytest tests/ -k "edge-08"          # one specific case
deepeval test run tests/            # same suite via the DeepEval CLI
```

Each case shows up as `test_movie_agent[normal-01]`,
`test_movie_agent[fail-04]`, etc. If `PHOENIX_ENABLED=1`, every case
also emits an `eval_case:<id>` chain span that parents the
recommender's LLM + tool spans.

## Run the aggregate report (`evaluate()`)

Structured overall / per-metric / per-category pass rates. Parallelizes
judge calls via `AsyncConfig`, and (when tracing is on) pushes each
metric result back onto its originating span so failing goldens are
filterable in the Phoenix UI (e.g. `rubric.score < 0.7`).

```bash
python scripts/run_evaluation.py                             # full dataset
python scripts/run_evaluation.py --category failure          # one bucket
python scripts/run_evaluation.py --cases fail-04,normal-15   # specific ids
python scripts/run_evaluation.py --limit 5                   # smoke test
python scripts/run_evaluation.py --max-concurrent 20         # bump parallelism
```

### Regression tracking (snapshots)

Snapshot a run and diff it against a previous snapshot to catch quality
regressions across model or prompt changes:

```bash
python scripts/run_evaluation.py --json snapshots/2026-09-01.json
# ...make a change, run again...
python scripts/run_evaluation.py --json snapshots/2026-09-08.json
python scripts/compare_snapshots.py snapshots/2026-09-01.json snapshots/2026-09-08.json
```

The diff shows overall / per-metric / per-category deltas and lists
newly-regressed and newly-recovered case IDs. Exits non-zero if any
case regressed — safe to gate CI on.

## Phoenix tracing

Turn tracing on by exporting `PHOENIX_ENABLED=1` and (usually) running
Phoenix locally on port 6006:

```bash
pip install arize-phoenix          # if you want the local UI too
phoenix serve                      # http://127.0.0.1:6006
```

Once on, every recommender call produces a trace shaped like:

```
recommend                                    (CHAIN)
├── ChatAnthropic ...                        (LLM)
├── tool.search_movies                       (TOOL)
├── ChatAnthropic ...                        (LLM)
├── tool.get_movie_details                   (TOOL)
├── ChatAnthropic ...                        (LLM)
└── tool.submit_recommendations              (TOOL)
```

Under `scripts/run_evaluation.py` and `pytest`, each of those trees is
additionally wrapped in an `eval_case:<id>` chain span carrying
`case.id`, `case.category`, and `case.subcategory` attributes, and any
judge LLM calls (`openai/gpt-4o-mini`) nest under the same parent — so
one failing golden links to exactly the LLM + tool + judge spans that
produced it. DeepEval per-metric scores/labels/explanations are then
uploaded as span annotations (`rubric`, `Answer Relevancy`, etc.) via
`src/tracing/annotations.py`.

Everything in `src/tracing/` is a no-op when `PHOENIX_ENABLED != 1`,
and the `tracing` extras are only required when tracing is actually
enabled — so pytest and the demo work without them.

## Extending

- Add cases to `evaluation/datasets.py` — one entry per user scenario.
  Bump the invariant assertions at the bottom of the file.
- Add or tune metrics in `evaluation/metrics.py` (e.g.
  `FaithfulnessMetric`, another `GEval` criterion) and re-export from
  `evaluation/__init__.py`.
- Grow the catalog in `src/recommender/catalog.py` and/or add new tools
  in `src/recommender/tools.py` — the dispatcher wraps every tool call
  in its own OpenInference TOOL span automatically.
- Swap the recommender model with the `RECOMMENDER_MODEL` env var.
- Swap the judge model by editing `JUDGE = ...` in
  `evaluation/metrics.py`.

For the full walkthrough of design decisions, see [TUTORIAL.md](TUTORIAL.md).
