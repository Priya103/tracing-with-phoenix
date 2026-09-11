# Tracing: Phoenix + OpenTelemetry

How observability is wired up in this project — Phoenix setup, the
OpenTelemetry ↔ Phoenix data flow, and how span attributes / metadata /
annotations are populated. Everything below points at the actual
implementation so you can read the code alongside it.

---

## 1. Setting Phoenix up

Phoenix has three moving parts here:

1. **The Phoenix server** — the UI and the OTLP receiver. Runs locally
   at `http://127.0.0.1:6006`. Anything sent to `/v1/traces` on that
   port shows up in the UI.
2. **The OTLP exporter** in this process — configured by
   `phoenix.otel.register(...)` in `src/tracing/__init__.py:82`. This
   is what actually ships spans over HTTP to the server.
3. **The OpenInference instrumentors** — auto-patch the Anthropic and
   OpenAI SDKs so every `messages.create(...)` / `chat.completions`
   call becomes an LLM span with prompt, completion, token counts,
   and model name populated for free.

### Install

```bash
pip install -e ".[tracing]"     # OTLP exporter + instrumentors + client
pip install arize-phoenix       # only if you want the local UI/server
```

The `[tracing]` extra (defined in `pyproject.toml:16-22`) pulls in:

- `arize-phoenix-otel` — thin wrapper around the OTel SDK; provides
  the `register()` helper that configures the tracer provider and OTLP
  exporter in one call.
- `arize-phoenix-client` — needed for uploading span annotations
  (`log_span_annotations_dataframe`, see §4).
- `openinference-instrumentation-anthropic` — auto-instruments the
  `anthropic` SDK. Every call the recommender makes becomes an LLM
  span.
- `openinference-instrumentation-openai` — same, for the DeepEval
  judge (`openai/gpt-4o-mini` via OpenRouter).
- `pandas` — the annotation upload API takes a DataFrame.

### Run Phoenix locally

```bash
phoenix serve                           # http://127.0.0.1:6006
```

### Turn tracing on for this process

```bash
export PHOENIX_ENABLED=1
```

That's the master switch. `setup_tracing()` (`src/tracing/__init__.py:41`)
is called unconditionally from `conftest.py`, `scripts/run_evaluation.py`,
and `src/server/app.py`, but it's a no-op unless `PHOENIX_ENABLED=1`
(`src/tracing/__init__.py:57`). This means:

- Tests and the demo work fine without Phoenix installed or running.
- The `[tracing]` extras are only actually needed when you flip the
  switch. If they're missing when you flip it, you get a clear
  RuntimeError telling you to install them
  (`src/tracing/__init__.py:65-68`).

### Optional env vars

| Var                          | Default                              | Effect |
|------------------------------|--------------------------------------|--------|
| `PHOENIX_ENABLED`            | unset (off)                          | Master switch. `1` = enable. |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://127.0.0.1:6006/v1/traces`    | OTLP endpoint. Point at Phoenix Cloud or a shared collector. |
| `PHOENIX_SAMPLE_RATIO`       | unset (100%)                         | Float in `[0.0, 1.0]`, parent-based. Lower this in production. |

Sampling is `ParentBased(TraceIdRatioBased(ratio))`
(`src/tracing/__init__.py:78-80`) — a decision is made once per root
trace and inherited by children. Never `1/N` random dropping mid-trace.

---

## 2. How OpenTelemetry and Phoenix work together

Phoenix is an **OTLP receiver** — it speaks the same wire protocol as
any other OpenTelemetry backend. OpenTelemetry is what actually creates,
enriches, and ships the spans. Phoenix is what stores, indexes, and
renders them.

### The pipeline

```
your code
    │  otel API (start_as_current_span, set_attribute, using_session, ...)
    ▼
Tracer Provider  ──►  BatchSpanProcessor  ──►  OTLP HTTP Exporter
    ▲                                                │
    │                                                │  POST /v1/traces
    │                                                ▼
    │                                          Phoenix server
    │                                          (indexes + UI)
    │
    └── auto-instrumentation of Anthropic / OpenAI / FastAPI wraps
        library calls to emit LLM / HTTP spans automatically
```

Everything above the exporter is vendor-neutral OpenTelemetry. Only the
last hop is Phoenix-specific, and only because Phoenix has chosen the
`/v1/traces` OTLP path.

### `setup_tracing()` step by step

Located at `src/tracing/__init__.py:41-99`. When
`PHOENIX_ENABLED=1`, it does exactly this:

1. **Import lazily.** Instrumentors and `phoenix.otel.register` are
   imported inside the function so the module is safe to import
   without the extras installed
   (`src/tracing/__init__.py:60-68`).
2. **Register a tracer provider.** `register(project_name=..., endpoint=..., batch=True)`
   creates an OTel `TracerProvider`, attaches a `BatchSpanProcessor`
   (batched OTLP export — single-span export would murder throughput),
   and installs it as the global provider. Optionally attaches a
   parent-based sampler when `PHOENIX_SAMPLE_RATIO` is set
   (`src/tracing/__init__.py:71-82`).
3. **Auto-instrument the SDKs.**
   `AnthropicInstrumentor().instrument()` and
   `OpenAIInstrumentor().instrument()` monkey-patch the two client
   libraries (`src/tracing/__init__.py:83-84`). From that point on,
   every `client.messages.create(...)` and every judge chat completion
   emits an LLM span with prompt / completion / token counts /
   model name populated for free, following the OpenInference semantic
   conventions.
4. **Optionally instrument FastAPI.** When called with
   `instrument_fastapi=True` (from `src/server/app.py:36`), OTel's
   FastAPI middleware is installed, so each HTTP request produces a
   root SERVER span that everything downstream nests under
   (`src/tracing/__init__.py:86-95`).
5. **Guard against double-init.** The `_INSTRUMENTED` flag makes
   `setup_tracing()` idempotent (`src/tracing/__init__.py:54-56`) —
   pytest calls it in every worker, and the server calls it at import
   time; both are safe.

### The trace tree we build

The instrumentors give us LLM spans for free. The code adds two more
layers on top, both using plain OTel APIs:

- **`recommend` CHAIN span** (`src/recommender/agent.py:110-120`) —
  wraps one full recommender invocation.
- **`tool.<name>` TOOL spans** (`src/recommender/tools.py:167-184`) —
  one per tool dispatch, wraps each `search_movies` /
  `get_movie_details` / `submit_recommendations` call.

When the eval runner is driving:

- **`eval_case:<id>` CHAIN span** (`src/tracing/__init__.py:102-124`,
  used from `scripts/run_evaluation.py:107` and
  `tests/test_recommender.py:21`) — wraps recommender + judge for one
  golden case.

Result:

```
eval_case:fail-04                            (CHAIN)
└── recommend                                (CHAIN)
    ├── ChatAnthropic messages.create        (LLM, auto)
    ├── tool.search_movies                   (TOOL)
    ├── ChatAnthropic messages.create        (LLM, auto)
    ├── tool.get_movie_details               (TOOL)
    ├── ChatAnthropic messages.create        (LLM, auto)
    └── tool.submit_recommendations          (TOOL)
    ...then the DeepEval judge fires OpenAI spans, also nested here
```

Under FastAPI, the outermost span is the HTTP SERVER span, and
`recommend` nests below it — but the trace tree is otherwise identical.

### Why OpenInference?

Phoenix knows how to render LLM interactions specifically (prompts,
completions, token usage, tool calls) because the spans follow the
**OpenInference semantic conventions** — a set of well-known
attribute names on top of OTel. This project uses them in two ways:

- Indirectly, via the instrumentors, which emit LLM spans in the
  right shape automatically.
- Directly, by setting semconv attributes on the CHAIN and TOOL spans
  we create by hand (see §3).

---

## 3. Attributes / labels on spans

There are **no "labels" as a first-class concept in OTel** — everything
that looks like a label in the Phoenix UI is a span attribute (a
key/value pair set on the span). This project sets three flavors:

### 3.1 OpenInference semantic-convention attributes

These are what make Phoenix render our custom spans correctly (as a
CHAIN or TOOL, with input/output blobs displayed cleanly).

Constants are imported once at module scope, guarded by `try/except
ImportError` so the file still loads if the `[tracing]` extra isn't
installed (`src/recommender/tools.py:26-43`,
`src/tracing/__init__.py:23-38`):

| Attribute (constant)                    | Set at                                              | Purpose |
|-----------------------------------------|-----------------------------------------------------|---------|
| `SpanAttributes.OPENINFERENCE_SPAN_KIND`| every custom span                                   | `CHAIN` or `TOOL` — controls how Phoenix renders the span. |
| `SpanAttributes.INPUT_VALUE`            | `recommend` chain, `tool.*`, `eval_case:*`          | JSON blob shown in the "input" panel. |
| `SpanAttributes.INPUT_MIME_TYPE`        | same                                                | Always `application/json` here. |
| `SpanAttributes.OUTPUT_VALUE`           | `recommend` chain, `tool.*`                         | JSON blob shown in the "output" panel. |
| `SpanAttributes.OUTPUT_MIME_TYPE`       | same                                                | `application/json`. |
| `SpanAttributes.TOOL_NAME`              | `tool.*` dispatch                                   | Tool name — makes Phoenix's tools filter work. |

Set via the tiny helpers `_set_span_kind` / `_set_span_io` in
`src/recommender/tools.py:118-129` and `src/recommender/agent.py:76-87`.
When the semconv package isn't installed, the constants are `None` and
the helpers no-op — the CHAIN/TOOL spans still get created, they just
render as unattributed spans.

### 3.2 Project-specific attributes

Plain OTel attributes we invented for filtering in the UI:

| Attribute            | Set at                                    | Value             |
|----------------------|-------------------------------------------|-------------------|
| `recommender.model`  | `recommend` chain (`agent.py:112`)        | e.g. `claude-sonnet-4-6` |
| `case.id`            | `eval_case:*` (`tracing/__init__.py:118`) | e.g. `fail-04`   |
| `case.category`      | `eval_case:*` (`tracing/__init__.py:119`) | `normal` / `edge` / `failure` |
| `case.subcategory`   | `eval_case:*` (`tracing/__init__.py:120`) | free-form        |

These become filterable in Phoenix: e.g. `case.category = "failure"`
to see only failure-bucket traces.

### 3.3 Request-scoped metadata (server only)

For the FastAPI surface, we don't set attributes on spans directly —
instead we use OpenInference **context managers** that attach
metadata to *every span opened inside them* automatically
(`src/server/app.py:82`):

```python
with using_session(session_id), using_user(user_id), using_metadata(metadata):
    rec = _recommender.recommend(...)
```

- `using_session(session_id)` → Phoenix groups traces by session
  (great for multi-turn conversations).
- `using_user(user_id)` → filter by user in the UI.
- `using_metadata({...})` → arbitrary bag; here we store
  `request_id` and `tenant_id` for correlation.

Headers are read straight off the request (`x-session-id`,
`x-user-id`, `x-tenant-id`, `x-request-id`) with UUID fallbacks so
every trace still has *something* to group by even for anonymous
callers (`src/server/app.py:65-80`). The chosen `request_id` and the
hex `span_id` of the `recommend` chain span are echoed back in the
response body for support-ticket correlation
(`src/server/app.py:85-90`).

---

## 4. Annotations (DeepEval scores → Phoenix)

Annotations are Phoenix's mechanism for **attaching a score/label to
an existing span after the fact** — distinct from attributes, which
are set at span creation. In this project they're how DeepEval's
per-metric verdicts land on the recommender span that produced them,
so a rubric-failing case in the UI links straight to the LLM + tool
spans that caused it.

All annotation logic lives in `src/tracing/annotations.py` (~80 lines).

### The pipeline

```
per case:
  recommender.recommend() ── returns Recommendation(span_id=<hex>)
      │
      │  span_id captured from `recommend` CHAIN span
      │
  deepeval evaluate() ── returns per-metric result (score, success, reason)
      │
      ▼
  annotations.record(span_id, case_id, metric_name, score, success, reason)
      │
      ▼
  buffered in _ROWS[metric_name]   (module-global dict[str, list[dict]])
                                    ─────────────────────────
                                     (accumulated across all cases)

at end of run:
  annotations.flush()  ── one DataFrame per metric
      │
      ▼
  phoenix.client.Client().spans.log_span_annotations_dataframe(...)
      │
      ▼
  Phoenix server ── attaches scores to spans by span_id
```

### `record()` — buffer per case

`src/tracing/annotations.py:32-52`. Called from
`scripts/run_evaluation.py:122-131` for every metric of every case:

```python
annotations.record(
    span_id=rec.span_id,          # from Recommendation dataclass
    case_id=case.id,
    metric_name=m.name,           # e.g. "Answer Relevancy", "rubric[fail-04]"
    score=m.score,                # 0.0..1.0
    success=m.success,            # bool
    explanation=getattr(m, "reason", None) or getattr(m, "explanation", None),
)
```

Two important behaviors:

- **No-op when tracing is off** or when `span_id` is `None`
  (`annotations.py:41`). Safe to leave in place; costs nothing when
  disabled.
- **Rubric metric normalization** (`annotations.py:27-29`,
  also `scripts/run_evaluation.py:75-77`): DeepEval names per-case
  GEval rubrics `rubric[fail-04]`, `rubric[normal-15]`, etc. — one
  per case, since each case gets its own criteria. For aggregation
  and annotation, these are collapsed to a single `rubric` bucket so
  the UI shows one annotation type across the whole run rather than
  58 distinct ones.

The row shape written per metric is:

```python
{
    "span_id":     "<16-hex-chars>",
    "score":       float | None,
    "label":       "pass" | "fail",
    "explanation": str (trimmed to 2000 chars),   # keeps payloads sane
    "case_id":     "<case id>",                    # extra column, informational
}
```

### `flush()` — one DataFrame per metric

`src/tracing/annotations.py:55-79`. Called once at the end of
`scripts/run_evaluation.py:201-204`:

```python
if annotations.enabled():
    uploaded = annotations.flush()
    if uploaded:
        print(f"\nUploaded {uploaded} span annotations to Phoenix.")
```

Under the hood, `flush()` groups the buffered rows by metric name and
sends one upload per group:

```python
df = pd.DataFrame(rows).set_index("span_id")     # span_id is the join key
client.spans.log_span_annotations_dataframe(
    dataframe=df,
    annotation_name=metric_name,                  # "rubric", "Answer Relevancy", ...
    annotator_kind="LLM",                         # judge is an LLM
)
```

Result in the Phoenix UI: every recommender span shows an
**Annotations** panel with entries like
`rubric: score=0.42 label=fail` and `Answer Relevancy: score=0.91
label=pass`, each with the judge's explanation attached. Those
annotations are also filterable — `rubric.score < 0.7` returns every
run whose rubric metric failed, across all snapshots.

### Why annotations instead of attributes?

Attributes are set when a span is created; they belong to the span
itself. Scores from an offline evaluator arrive **after** the span has
closed, potentially from a completely different process or a much
later run. Annotations are the right primitive for that: they're a
side-channel that references a span by ID and can be uploaded at any
time, in bulk.

---

## 5. Quick reference: file map

| File                                  | Role |
|---------------------------------------|------|
| `src/tracing/__init__.py`             | `setup_tracing()`, `eval_case_span()`, semconv constants |
| `src/tracing/annotations.py`          | `record()` / `flush()` — DeepEval scores → Phoenix |
| `src/recommender/agent.py`            | Opens the `recommend` CHAIN span, captures `span_id` |
| `src/recommender/tools.py`            | Opens `tool.<name>` TOOL spans, sets `TOOL_NAME` |
| `src/server/app.py`                   | FastAPI + `using_session` / `using_user` / `using_metadata` |
| `scripts/run_evaluation.py`           | Wraps each case in `eval_case_span`, records + flushes annotations |
| `tests/test_recommender.py`           | Wraps each pytest case in `eval_case_span` |
| `conftest.py`                         | Calls `setup_tracing()` before any test runs |
| `pyproject.toml` (`[tracing]` extra)  | Declares the tracing dependency set |
