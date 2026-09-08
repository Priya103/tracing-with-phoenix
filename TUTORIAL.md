# Building an LLM Evaluation Harness with DeepEval — A Walkthrough

> A tutorial reconstructing what this POC is, why it took the shape it did,
> and which DeepEval patterns we exercised. Written as a learning artifact
> — read it top-to-bottom for the full journey, or jump to a section for a
> specific pattern.

## Table of contents

0. [Concepts you need first](#0-concepts-you-need-first)
1. [What we built and why](#1-what-we-built-and-why)
2. [Project layout](#2-project-layout)
3. [The system under test](#3-the-system-under-test)
4. [The golden dataset](#4-the-golden-dataset)
5. [Metrics](#5-metrics)
6. [The test harness](#6-the-test-harness)
7. [Prompt injection coverage](#7-prompt-injection-coverage)
8. [`assert_test` vs `evaluate()` — two mental models](#8-assert_test-vs-evaluate--two-mental-models)
9. [References](#9-references)

---

## 0. Concepts you need first

If you're new to LLM evaluation, this section gives you the mental
model. Skip if you already work with DeepEval.

> **Before you run anything:** see [README.md](README.md) for setup,
> API keys, and the commands to launch the pytest suite or the
> aggregate `evaluate()` script.

### The problem: how do you unit-test a generative model?

Traditional software has deterministic outputs — you assert
`add(2, 2) == 4`. LLMs don't. Ask Claude for movie picks twice and
you'll get two different lists, and both may be great. So the question
shifts from "is the output exactly X?" to "does the output satisfy the
properties of a good answer?"

That reframing is what an **eval harness** is: a rig that pushes many
inputs through your LLM, applies a set of graders to each output, and
tells you what percentage passed.

### LLM-as-judge

The graders themselves are usually LLMs. You give a **judge model** a
rubric in plain English ("does this output stay on the requested
genre? does it list exactly three films?") and ask it to score. That
sounds circular, but it works because:

- The judge only has to *check* the answer, not *generate* it — a
  smaller/cheaper model is fine.
- The rubric is written by you, so the judge is grading against your
  standards, not making them up.

In this project the judge is `openai/gpt-4o-mini` (see [§5a](#5a-judge-model--openrouter-routing)).

### What we do and don't evaluate

| Concern | This project | Notes |
|---|---|---|
| Behavioral correctness (invariants) | ✅ GEval rubric + `UNIVERSAL_EXPECTATIONS` | "3 picks, real films, on-genre" |
| Reference correctness (matches expected string) | ❌ | Open-ended task has no canonical answer |
| Grounding / Faithfulness to a source doc | ❌ | No source doc provided — nothing to ground against |
| Hallucinated titles | ⚠️ Partial | Rubric flags them, but the judge shares Claude's knowledge cutoff — a rigorous check would hit TMDB |
| Toxicity + Bias (wording) | ✅ `ToxicityMetric`, `BiasMetric` | On every case |
| Prompt injection (direct + indirect) | ✅ 8 dedicated cases (`fail-16`–`fail-23`) | Including data-channel injection via `liked_movies` |
| Off-topic / refusal | ⚠️ Implicit via rubric | No dedicated metric |
| PII leakage, distributional bias in *which films* get picked | ❌ | Out of scope for this POC |

### End-to-end flow

```
GoldenCase.inputs
      │
      ▼
MovieRecommender.recommend(...)      ← system under test (Claude)
      │
      ▼
result.output (free-form text)
      │
      ▼
LLMTestCase(input=..., actual_output=...)
      │
      ▼
metrics_for(case) → [Relevancy, GEval rubric, Toxicity, Bias]
      │                     │
      │                     ▼
      │              JUDGE (gpt-4o-mini) scores each metric
      ▼
assert_test(...)  →  pass / fail per case
```

### DeepEval primitives you'll see

- **`LLMTestCase`** — one input+output pair being scored. Analogous to
  a single unit-test case.
- **`Golden`** / **`EvaluationDataset`** — a stored reference case and
  a collection of them. We wrap our own `GoldenCase` dataclass inside
  `Golden.additional_metadata` (see [§6](#6-the-test-harness)).
- **`GEval`** — LLM-as-judge metric where the rubric is a plain-English
  string. The workhorse ([§5b](#5b-geval---per-case-rubric)).
- **`AnswerRelevancyMetric` / `ToxicityMetric` / `BiasMetric`** —
  built-in metrics with fixed rubrics under the hood.
- **`assert_test(test_case, metrics)`** — pytest-friendly: raises
  `AssertionError` on failure. Used in `tests/`.
- **`evaluate(test_cases, metrics)`** — standalone: returns structured
  results you can aggregate. Used in `scripts/run_evaluation.py`
  ([§8](#8-assert_test-vs-evaluate--two-mental-models)).

With that in hand, the rest of the tutorial is about *how we chose to
compose these primitives for a movie recommender* — not what they are.

---

## 1. What we built and why

A **movie recommender** (`src/recommender/`) powered by Anthropic Claude,
scored against a **50+ case golden dataset** (`evaluation/datasets.py`)
using DeepEval's per-case metrics (`evaluation/metrics.py`) and executed
as a parametrized pytest suite (`tests/test_recommender.py`).

The point wasn't the recommender. The point was the **evaluation loop
around it** — so we picked a task shape (open-ended text generation) that
forces us to think about how you evaluate output when there's no single
"correct" answer.

**Task shape matters.** A translator has a canonical answer per input;
a recommender doesn't. That choice determined every downstream decision
about which metrics we could use.

---

## 2. Project layout

```
.
├── src/recommender/       # System under test — Claude-powered recommender
│   ├── agent.py           #   MovieRecommender.recommend(...)
│   └── prompts.py         #   System + user prompt templates
├── evaluation/            # Evaluation spec (not test runner)
│   ├── datasets.py        #   Golden cases (normal / edge / failure)
│   └── metrics.py         #   Judge model + metric configuration
├── tests/
│   └── test_recommender.py  # pytest harness — imports from evaluation/
├── main.py                # Manual demo entry point
└── pyproject.toml
```

**Why three layers, not two.** A common mistake is to put metric
configuration inside the test file. Then a script, notebook, or CI job
that wants to re-run the same evaluation without pytest ends up
copy-pasting the metrics. We split it:

- `src/recommender/` — the thing being evaluated.
- `evaluation/` — the **spec**: goldens + metrics. Reusable from
  scripts, notebooks, and CI. Doesn't depend on pytest.
- `tests/` — the **runner**: pytest plumbing that imports from
  `evaluation/` and turns it into pass/fail per case.

Because `evaluation/` sits at the repo root rather than under `src/`,
we widen setuptools' package discovery in `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
where = ["src", "."]
include = ["recommender*", "evaluation*"]
```

---

## 3. The system under test

The recommender is deliberately simple: one method, `recommend(...)`,
takes free-form preferences and returns three numbered picks.

```python
# src/recommender/agent.py
result = recommender.recommend(
    favorite_genres=["science fiction", "thriller"],
    liked_movies=["Arrival", "Ex Machina"],
    mood="cerebral, slow-burn",
)
# result.output ->
#   1. Annihilation (2018) — cerebral, slow-burn sci-fi...
#   2. Under the Skin (2013) — ...
#   3. Primer (2004) — ...
```

**Two patterns worth pulling out:**

### 3a. Dual-provider client

The recommender can call Claude directly *or* route through OpenRouter's
Anthropic-compatible endpoint by swapping `base_url`. That mattered here
because we route the **judge** model through OpenRouter too, so the same
key wires up both sides:

```python
# agent.py
if openrouter_key:
    return Anthropic(
        base_url="https://openrouter.ai/api",
        api_key=openrouter_key,
        default_headers={"Authorization": f"Bearer {openrouter_key}"},
    )
return Anthropic()
```

Small subtlety: the Anthropic SDK appends `/v1/messages` to `base_url`,
and OpenRouter's endpoint is `/api/v1/messages` — so the base_url stops
at `/api`, not `/api/v1`.

### 3b. Prompt caching

Because we run the same system prompt against 58 goldens per test run,
we tag it with Anthropic's ephemeral cache control:

```python
system=[
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
],
```

We enable it but **don't currently verify it works**.

---

## 4. The golden dataset

`evaluation/datasets.py` holds **58 hand-designed cases** split across
three buckets:

| Bucket | Count | Purpose |
|---|---|---|
| `normal` | 20 | Everyday, well-formed preferences |
| `edge` | 15 | Degenerate / conflicting / non-standard inputs |
| `failure` | 23 | Inputs that stress specific failure modes |

Each case is a `GoldenCase` dataclass:

```python
@dataclass(frozen=True)
class GoldenCase:
    id: str
    category: str          # "normal" | "edge" | "failure"
    subcategory: str       # e.g. "genre_only", "hallucinated_movie"
    description: str
    inputs: dict[str, Any] # kwargs for recommender.recommend(...)
    expected_behavior: list[str] = field(default_factory=list)
    failure_signals: list[str] = field(default_factory=list)
```

**Design decisions worth pulling out:**

### 4a. `expected_behavior` + `failure_signals` — a two-column rubric

For open-ended tasks, "correct" isn't a single string. We describe
correctness as **invariants the output must satisfy** (`expected_behavior`)
and **patterns that would indicate failure** (`failure_signals`).

That maps cleanly onto GEval's "MUST satisfy / MUST NOT exhibit"
structure — see [section 5](#5b-geval---per-case-rubric).

### 4b. `UNIVERSAL_EXPECTATIONS` factored out

Rules every case must satisfy (`returns exactly 3 recommendations`,
`real released films`, `numbered list format`) live in one list —
individual cases only state what's **distinctive** about them.

### 4c. Import-time invariants

At the bottom of `datasets.py`:

```python
assert len(GOLDEN_DATASET) == 58
assert len(FAILURE_CASES) == 23
assert len({c.id for c in GOLDEN_DATASET}) == 58, "ids must be unique"
```

Cheap safety net that fires the moment we add a case with a duplicate
ID or forget to update the count.

### 4d. Failure subcategories map to failure modes, not to bugs

The failure bucket's subcategories are named after **the class of
failure they stress**, not after which line of code they'd expose:

- `wrong_retrieval` — model recommends the wrong genre entirely
- `hallucinated_movie` — model fabricates titles
- `avoid_constraint_violated` — model ignores explicit "avoid" list
- `mood_mismatch` — recommendations fight the requested mood
- `bad_ranking` — top pick isn't a defensible best-in-class
- `prompt_injection` — model executes injected instructions

Naming this way keeps the taxonomy honest: if a new bug shows up, you
ask "which failure family does this belong to?" rather than adding
one-off ad-hoc cases.

---

## 5. Metrics

`evaluation/metrics.py` owns three things: the **judge model**, a set of
**built-in metrics**, and a **per-case rubric builder**.

### 5a. Judge model — OpenRouter routing

```python
JUDGE = OpenRouterModel(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

We use `gpt-4o-mini` because it's cheap and consistent enough for
scoring short outputs. Judge choice matters for cost more than for
accuracy on this task — a bigger judge scores similarly here.

`metrics.py` calls `load_dotenv(override=True)` at import time so
`.env` wins over any stale shell state.

### 5b. GEval — per-case rubric

The workhorse of the suite. For each `GoldenCase`, we build a fresh
`GEval` metric whose criteria are stitched from the case's own
`expected_behavior` + `failure_signals`:

```python
def rubric_metric(case: GoldenCase) -> GEval:
    lines = ["The output MUST satisfy every item below:"]
    lines += [f"- {b}" for b in UNIVERSAL_EXPECTATIONS]
    lines += [f"- {b}" for b in case.expected_behavior]
    if case.failure_signals:
        lines.append("The output MUST NOT exhibit any of:")
        lines += [f"- {s}" for s in case.failure_signals]
    return GEval(
        name=f"rubric[{case.id}]",
        criteria="\n".join(lines),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=JUDGE,
        threshold=0.7,
    )
```

Two things worth noting:

1. **`evaluation_params` takes an enum, not class attributes.** This
   bit us early — writing `LLMTestCase.actual_output` raises
   `AttributeError` because Pydantic model fields aren't class-level
   attributes. Use `LLMTestCaseParams.ACTUAL_OUTPUT` etc.
2. **The metric name embeds the case ID.** When you're staring at a
   dashboard with 58 failing cases, having them all named `"rubric"`
   is a nightmare. `f"rubric[{case.id}]"` gives you `rubric[fail-04]`,
   `rubric[edge-08]` — instantly identifiable.

### 5c. AnswerRelevancyMetric — a built-in check for free

`RELEVANCY = AnswerRelevancyMetric(threshold=0.7, model=JUDGE)` runs on
every case. It answers "did the output actually address the input?" —
generic but catches situations where the rubric is technically
satisfied but the output ignored the user.

### 5d. Safety metrics — Toxicity + Bias

```python
TOXICITY = ToxicityMetric(threshold=0.7, model=JUDGE)
BIAS = BiasMetric(threshold=0.7, model=JUDGE)
```

Both score `1.0` = safe, `0.0` = unsafe, so `threshold=0.7` means
"fail if the output is less than 70% safe." Strict = 0.7+.

### 5e. `metrics_for(case)` — the single entry point

Test code and future scripts consume metrics via one function:

```python
def metrics_for(case: GoldenCase) -> list:
    return [RELEVANCY, rubric_metric(case), TOXICITY, BIAS]
```

Adding a metric = one line in this function. Nothing else changes.

---

## 6. The test harness

`tests/test_recommender.py` is short on purpose:

```python
recommender = MovieRecommender()

dataset = EvaluationDataset(
    goldens=[Golden(input=str(c.inputs),
                    additional_metadata={"case": c}) for c in GOLDEN_DATASET]
)

@pytest.mark.parametrize("golden", dataset.goldens, ids=[c.id for c in GOLDEN_DATASET])
def test_movie_agent(golden: Golden):
    case: GoldenCase = golden.additional_metadata["case"]
    result = recommender.recommend(**case.inputs)
    test_case = LLMTestCase(input=result.input, actual_output=result.output)
    assert_test(test_case, metrics_for(case))
```

Two patterns worth calling out:

1. **`ids=[c.id for c in GOLDEN_DATASET]`** — failures show as
   `test_movie_agent[fail-04]`, not `test_movie_agent[golden3]`.
   Diagnosability compounds when the suite grows.
2. **Store the whole `GoldenCase` in `additional_metadata`.** DeepEval's
   `Golden` only has string fields. Wrapping the dataclass in metadata
   is the least-friction way to preserve full context (expected_behavior,
   failure_signals, subcategory) at scoring time.

---

## 7. Prompt injection coverage

Once the base suite was solid, we added **8 dedicated
prompt-injection cases** (`fail-16` through `fail-23`) under a new
`prompt_injection` subcategory. Attacks covered:

| Case | Attack |
|---|---|
| fail-16 | Direct instruction override |
| fail-17 | System-prompt leak |
| fail-18 | Output-format hijack (JSON) |
| fail-19 | Language hijack (French) |
| fail-20 | Affiliate/URL injection |
| fail-21 | Role-play jailbreak (DAN) |
| fail-22 | Indirect injection via `liked_movies` data channel |
| fail-23 | Delimiter confusion (fake `<system>` tags) |

**Key insight:** adding adversarial coverage was a **dataset
expansion**, not a metric addition. The existing rubric + relevancy
metrics already grade injection failures correctly — they'll fail the
case if the output drifts from the numbered-list format, changes
language, leaks the prompt, or abandons the task. We just needed the
stimuli. Zero code changes to `metrics.py` or the test harness — the
parametrization auto-discovers new cases.

**`fail-22` is the most important one.** Real-world attackers exploit
indirect injection: they don't send you the malicious prompt directly,
they poison a data channel your app treats as user data (a scraped
document, an email body, a "liked movie" from an untrusted source).
Getting this one right is the whole ballgame.

---

## 8. `assert_test` vs `evaluate()` — two mental models

DeepEval offers two ways to run the same underlying scoring, packaged
for different mental models:

| | `assert_test` | `evaluate()` |
|---|---|---|
| **Where you run it** | Under pytest | Standalone script / notebook |
| **What you get** | Per-case pass/fail | Aggregate report + `EvaluationResult` object |
| **Failure mode** | pytest AssertionError per case | Structured data you can slice |
| **Async / parallel** | Sequential | Native `AsyncConfig` support |
| **Best for** | CI gate: "does this PR break something?" | Iteration: "how did quality shift?" |

This POC wires up both:

- **`pytest tests/`** — `tests/test_recommender.py` calls `assert_test`
  per case. Fast CI signal, one AssertionError per failing golden.
- **`python scripts/run_evaluation.py`** — calls `evaluate()` per case
  with `AsyncConfig(run_async=True, max_concurrent=10)` so judge calls
  overlap. Prints overall / per-metric / per-category pass rates. Dump
  a snapshot with `--json snapshots/YYYY-MM-DD.json` and diff two runs
  with `scripts/compare_snapshots.py` to catch regressions across model
  or prompt changes.

---

## 9. References

- [DeepEval docs](https://docs.confident-ai.com/) — metrics catalogue, `evaluate()` API, red-teaming
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [OpenRouter Anthropic-compatible endpoint](https://openrouter.ai/docs/api-reference/completions)
- OWASP LLM Top 10 (LLM01: Prompt Injection) — the attack categories in section 7

### Files in this repo

- `src/recommender/agent.py` — dual-provider client + `MovieRecommender.recommend()`
- `src/recommender/prompts.py` — system prompt + user prompt builder
- `evaluation/datasets.py` — 58 goldens, three buckets, invariant assertions
- `evaluation/metrics.py` — judge model, all metrics, `metrics_for(case)`
- `evaluation/__init__.py` — public re-exports
- `tests/test_recommender.py` — parametrized pytest suite
- `scripts/run_evaluation.py` — aggregate `evaluate()` runner with `--json` snapshotting
- `scripts/compare_snapshots.py` — diffs two snapshots, flags regressions
- `pyproject.toml` — package discovery, dev deps
- `main.py` — manual demo entry point
