# tests/reliability_tests/test_evaluator.py

from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from reliability_tests import Evaluator


@pytest.fixture
def dummy_test_df():
	return pd.DataFrame(
		{"perturbed_idx": [1, 2], "original_request": ["Req A", "Req B"], "generation_response": ["Resp A", "Resp B"]}
	)


@pytest.fixture
def dummy_judge_df():
	return pd.DataFrame({"perturbed_idx": [1, 2], "score": [1, 0], "reasoning": ["Reason A", "Reason B"]})


@pytest.fixture
def dummy_config(tmp_path):
	mock_config = MagicMock()
	mock_config.autograder_model_config.model = "openai/gpt-4o-mini"
	mock_config.autograder_model_config.model_copy.return_value = mock_config.autograder_model_config
	mock_config.autograder_model_config.template = "some_template"
	mock_config.autograder_model_config.rate_limit = {}
	mock_config.max_workers = 1
	mock_config.output_dir = tmp_path
	mock_config.overwrite_results = True
	mock_config.output_file_format = "xlsx"
	return mock_config


def test_grade_single(dummy_test_df, dummy_judge_df, dummy_config):
	evaluator = Evaluator(dummy_config)

	class DummyResponse:
		def __init__(self, score, reasoning):
			self._score = score
			self._reasoning = reasoning

		def model_dump(self):
			return {"score": self._score, "reasoning": self._reasoning}

	with patch("reliability_tests.evaluator.LLMClient") as mock_client:
		mock_instance = MagicMock()
		mock_instance.call.side_effect = [
			DummyResponse(row.score, row.reasoning) for _, row in dummy_judge_df.iterrows()
		]
		mock_client.return_value = mock_instance

		result = evaluator._grade_single(dummy_test_df)

		# Check result matches expected
		result_sorted = result.sort_values("perturbed_idx").reset_index(drop=True)
		expected_sorted = dummy_judge_df.sort_values("perturbed_idx").reset_index(drop=True)

		assert isinstance(result, pd.DataFrame)
		assert list(result_sorted["score"]) == list(expected_sorted["score"])
		assert list(result_sorted["reasoning"]) == list(expected_sorted["reasoning"])

		assert mock_instance.call.call_count == len(dummy_test_df)
		mock_instance.call.assert_any_call(ANY, ANY)


def test_grade_synthetic_data_creates_file(dummy_test_df, dummy_config, dummy_judge_df):
	"""Test that grade_synthetic_data generates results for new data."""
	evaluator = Evaluator(dummy_config)

	with patch.object(Evaluator, "_grade_single", return_value=dummy_judge_df) as mock_grade:
		result = evaluator.grade_synthetic_data("test_run", dummy_test_df)

		# Columns are renamed in the output
		expected_df = dummy_judge_df.rename(
			columns={
				"score": "autograder_score",
				"reasoning": "autograder_reasoning",
			}
		)

		assert isinstance(result, pd.DataFrame)
		assert result.equals(expected_df)

		mock_grade.assert_called_once()


@pytest.mark.parametrize("file_format", ["xlsx", "csv"])
def test_grade_synthetic_data_uses_cached(dummy_test_df, dummy_config, dummy_judge_df, file_format):
	"""Test that grade_synthetic_data uses cached results when overwrite_results=False."""
	evaluator = Evaluator(dummy_config)

	# Prepare cached results
	model_name = dummy_config.autograder_model_config.model.replace("/", "_")
	if file_format == "xlsx":
		cached_path = dummy_config.output_dir / f"test_run_results_{model_name}.xlsx"
		dummy_judge_df.to_excel(cached_path, index=False)
	else:
		cached_path = dummy_config.output_dir / f"test_run_results_{model_name}.csv"
		dummy_judge_df.to_csv(cached_path)

	dummy_config.output_file_format = file_format
	dummy_config.overwrite_results = False

	with patch.object(Evaluator, "_grade_single") as mock_grade:
		result = evaluator.grade_synthetic_data("test_run", dummy_test_df)

		# Should load cached data, not re-run grading
		assert isinstance(result, pd.DataFrame)
		assert list(result["perturbed_idx"]) == [1, 2]
		mock_grade.assert_not_called()
