from deepeval.dataset import EvaluationDataset
from deepeval.metrics import ContextualRecallMetric


metric = ContextualRecallMetric(threshold=0.5)

dataset = EvaluationDataset(test_cases=[...])
dataset.evaluate(metrics=[metric])


from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[
        "pdfs/health-and-safety-risk-management.pdf",
        "pdfs/ECI-SHE7-SHE-Guide.pdf",
        "pdfs/hsg129.pdf",
        "pdfs/l153.pdf",
        "pdfs/the-business-case-for-engineering-in-health-and-safety.pdf",
    ],
    max_goldens_per_context=15,
    include_expected_output=True,
)

from deepeval.metrics import GEval, ContextualRecallMetric
from deepeval.test_case import LLMTestCaseParams

contextual_recall = ContextualRecallMetric(threshold=0.6)
correctness = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.6,
)

"""Run the benchmark by first converting the goldens generated in Step 1 to test cases that are reday for evaluation"""
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

...

test_cases = []
for golden in goldens:
    query = golden.input
    llm_output, retrieved_nodes = run_chatbot_qa(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=llm_output,
        retrieval_context=retrieved_nodes,
        expected_output=golden.expected_output,
    )

test_cases.append(test_case)

dataset = EvaluationDataset(test_cases=test_cases)
dataset.evaluate(metrics=[correctness, contextual_recall])
