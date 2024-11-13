from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

first_test_case = LLMTestCase(input="W", actual_output="...")
second_test_case = LLMTestCase(input="...", actual_output="...")

test_cases = [first_test_case, second_test_case]

dataset = EvaluationDataset(test_cases=test_cases)
