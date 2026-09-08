# tracing-with-phoenix

A movie recommender evaluated with DeepEval, to be instrumented with
[Arize Phoenix](https://docs.arize.com/phoenix) for LLM tracing.

**Status:** baseline copied from the DeepEval evaluation lab. Phoenix
integration (span capture for recommender + judge calls, local UI,
trace-linked eval results) is the next step.

## What's here now (baseline)

Everything below is the DeepEval evaluation harness: golden dataset,
metrics, pytest suite, and an aggregate `evaluate()` runner with JSON
snapshot diffing. Anthropic Claude generates recommendations; DeepEval
scores them.

## Phoenix roadmap

Planned additions on top of the baseline:

1. Add `arize-phoenix` + `openinference-instrumentation-anthropic` to deps.
2. `px.launch_app()` in a small `scripts/phoenix.py` to run the local UI.
3. `register()` + Anthropic instrumentation so recommender and judge
   calls both emit spans.
4. (Optional) log DeepEval per-case results back onto the trace so a
   failing golden links straight to the LLM call that produced it.

## Layout

```
.
├── src/recommender/       # Claude-powered recommender
│   ├── agent.py           # MovieRecommender.recommend(...)
│   └── prompts.py         # System + user prompt templates
├── evaluation/
│   ├── datasets.py        # Golden test cases (58: normal/edge/failure)
│   └── metrics.py         # Judge model + DeepEval metric config
├── tests/
│   └── test_recommender.py  # pytest + deepeval assert_test
├── scripts/
│   ├── run_evaluation.py    # aggregate evaluate() runner + JSON snapshots
│   └── compare_snapshots.py # diff two snapshots, flag regressions
├── main.py                # Manual demo entry point
├── pyproject.toml
└── .env.example
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env       # then fill in the keys below
```

Two keys are needed:

- `OPENROUTER_API_KEY` — the recommender calls Claude via OpenRouter's
  Anthropic-compatible endpoint, **and** the DeepEval judge
  (`openai/gpt-4o-mini`) is routed through OpenRouter too. One key wires
  up both sides. If you'd rather call Anthropic directly, set
  `ANTHROPIC_API_KEY` instead — but the judge configured in
  `evaluation/metrics.py` still needs OpenRouter.
- `RECOMMENDER_MODEL` (optional) — override the recommender model,
  e.g. `anthropic/claude-sonnet-4.5` when routing through OpenRouter.

## Run the demo

```bash
python main.py
```

One-shot recommendation against a hard-coded prompt. Useful smoke test
that your keys are wired up.

## Run the evaluation (pytest / `assert_test`)

Fast per-case pass/fail — this is the CI gate.

```bash
pytest                              # run all 58 goldens
pytest tests/ -k "fail-"            # only failure cases
pytest tests/ -k "edge-08"          # one specific case
deepeval test run tests/            # same suite via the DeepEval CLI
```

Each case shows up as `test_movie_agent[normal-01]`, `test_movie_agent[fail-04]`, etc.

## Run the aggregate report (`evaluate()`)

Structured overall / per-metric / per-category pass rates. Parallelizes
judge calls via `AsyncConfig`.

```bash
python scripts/run_evaluation.py                             # full dataset
python scripts/run_evaluation.py --category failure          # one bucket
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

## Extending

- Add cases to `evaluation/datasets.py` — one entry per user scenario.
  Bump the invariant assertions at the bottom of the file.
- Add or tune metrics in `evaluation/metrics.py` (e.g.
  `FaithfulnessMetric`, another `GEval` criterion) and re-export from
  `evaluation/__init__.py`.
- Swap the recommender model with the `RECOMMENDER_MODEL` env var.
- Swap the judge model by editing `JUDGE = ...` in `evaluation/metrics.py`.

For the full walkthrough of design decisions, see [TUTORIAL.md](TUTORIAL.md).
