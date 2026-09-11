import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase

from evaluation import GOLDEN_DATASET, GoldenCase, metrics_for
from recommender import MovieRecommender
from tracing import eval_case_span

recommender = MovieRecommender()

dataset = EvaluationDataset(
    goldens=[Golden(input=str(c.inputs),
                    additional_metadata={"case": c}) for c in GOLDEN_DATASET]
)


@pytest.mark.parametrize("golden", dataset.goldens, ids=[c.id for c in GOLDEN_DATASET])
def test_movie_agent(golden: Golden):
    case: GoldenCase = golden.additional_metadata["case"]
    with eval_case_span(case.id, case.category, case.subcategory, case.inputs):
        result = recommender.recommend(**case.inputs)
        test_case = LLMTestCase(input=result.input, actual_output=result.output)
        assert_test(test_case, metrics_for(case))

