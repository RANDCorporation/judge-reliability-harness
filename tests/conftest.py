import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
	sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def autograder_config():
	return {
		"score_type_map": {"single_judge": "numerical"},
		"models_without_temperature": [],
	}


@pytest.fixture
def autograder_sample_row():
	return {
		"response": "This is a test response.",
		"question": "What is the capital of France?",
	}


@pytest.fixture
def autograder_dataset():
	return pd.DataFrame(
		[
			{"response": "Paris", "question": "Capital of France?"},
			{"response": "Berlin", "question": "Capital of Germany?"},
		]
	)


@pytest.fixture
def evaluator_config():
	return {
		"judge_under_test": {
			"model": "openai/gpt-4o-mini",
			"eval_mode": "single_judge",
		}
	}


@pytest.fixture
def evaluator_test_df():
	return pd.DataFrame(
		{
			"idx": [1, 2],
			"instruction": ["Instruction A", "Instruction B"],
			"response": ["Response A", "Response B"],
			"rubric": ["Rubric A", "Rubric B"],
			"request": ["Request A", "Request B"],
		}
	)


@pytest.fixture
def evaluator_compare_df():
	return pd.DataFrame(
		{
			"idx": [1, 2],
			"instruction": ["Instruction C", "Instruction D"],
			"response": ["Response C", "Response D"],
			"rubric": ["Rubric C", "Rubric D"],
			"request": ["Request C", "Request D"],
		}
	)


@pytest.fixture
def harness_config():
	return {
		"judge_under_test": {
			"model": "openai/gpt-4o-mini",
			"eval_mode": "single_judge",
			"default_params_path": {
				"instruction": "./data/rubrics/harmbench_binary_instruction.txt",
				"rubric": "./data/rubrics/harmbench_binary_rubric.txt",
			},
		},
		"dataset": {
			"path": "mock_dataset.csv",
			"columns": {
				"request": "test_case",
				"response": "generation",
				"expected": "human_consensus",
			},
		},
		"tests": ["label_flip"],
		"overwrite_perturbations": False,
		"reporting": {"output_dir": "output/reports"},
		"perturbation_review": {"output_dir": "output/review"},
	}


@pytest.fixture
def harness_dataframe():
	return pd.DataFrame(
		{
			"test_case": ["Q1", "Q2"],
			"generation": ["A1", "A2"],
			"human_consensus": [1, 0],
		}
	)


@pytest.fixture
def completion_response_builder():
	def _builder(content: str):
		from unittest.mock import MagicMock

		message = MagicMock()
		message.content = content
		choice = MagicMock()
		choice.message = message
		response = MagicMock()
		response.choices = [choice]
		return response

	return _builder


@pytest.fixture
def perturbations_config():
	return {"perturbations": {"model": "mock-model"}}


@pytest.fixture
def perturbations_dataset_true_false():
	return pd.DataFrame(
		[
			{
				"id": "true-row",
				"expected": True,
				"response": "Original response.",
				"request": "Original request.",
				"context": "Context",
			},
			{
				"id": "false-row",
				"expected": False,
				"response": "Should stay.",
				"request": "Original request.",
			},
		]
	)
