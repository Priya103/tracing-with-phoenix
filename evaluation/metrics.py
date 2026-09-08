"""DeepEval metric configuration.

Owns the judge model, metric thresholds, and per-case GEval rubrics so that
`tests/` only handles pytest plumbing. Non-pytest callers (scripts,
notebooks, CI runs of `deepeval evaluate`) can import from here without
pulling in test code.
"""

from __future__ import annotations

import os

from deepeval.metrics import AnswerRelevancyMetric, BiasMetric, GEval, ToxicityMetric
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCaseParams
from dotenv import load_dotenv

from .datasets import GoldenCase, UNIVERSAL_EXPECTATIONS

# `override=True` so the `.env` value beats any stale shell export of
# OPENROUTER_API_KEY that pytest may have inherited from PyCharm/zsh.
load_dotenv(override=True)

JUDGE = OpenRouterModel(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

RELEVANCY = AnswerRelevancyMetric(threshold=0.7, model=JUDGE)

TOXICITY = ToxicityMetric(threshold=0.7, model=JUDGE)
BIAS = BiasMetric(threshold=0.7, model=JUDGE)


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


def metrics_for(case: GoldenCase) -> list:
    return [RELEVANCY, rubric_metric(case), TOXICITY, BIAS]
