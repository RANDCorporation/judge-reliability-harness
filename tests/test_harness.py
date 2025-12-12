# tests/test_harness.py

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.harness import Harness


class MockSyntheticDataAdapter:
	def __init__(self):
		self.called_with = []

	def get_validated_test_names(self):
		return ["test1", "test2"]

	def load_perturbations(self, test_name):
		return pd.DataFrame({"request": ["x"], "response": ["y"], "original_idx": [1]})

	def run(self, original_dataset, test_name):
		# mimic generating perturbations
		self.called_with.append((test_name, len(original_dataset)))
		return pd.DataFrame({"perturbed_request": ["p1"], "perturbed_response": ["r1"]})

	def get_meta_data(self, test_name):
		# return dummy metadata to match expected structure
		return {"meta": f"metadata for {test_name}"}


class MockEvaluator:
	def __init__(self, config):
		pass

	def grade_original_preprocess(self, df):
		df_copy = df.copy()
		df_copy["score"] = [1] * len(df_copy)
		df_copy["reasoning"] = ["ok"] * len(df_copy)
		return df_copy

	def grade_synthetic_data(self, test_name, df):
		df_copy = df.copy()
		df_copy["score"] = [1] * len(df_copy)
		df_copy["reasoning"] = ["ok"] * len(df_copy)
		return df_copy

	def save_evaluated_output(self, test_name, graded_df):
		# pretend to save, but do nothing
		pass


class MockAdminConfig:
	class dataset_config:
		dataset_path = Path("mock.csv")
		use_original_data_as_expected = True

	class perturbation_config:
		preprocess_columns_map = {"request": "request", "response": "response", "expected": "expected"}
		output_file_format = "csv"

	class evaluation_config:
		tests_to_evaluate = ["test1", "test2"]
		output_file_format = "csv"

	output_file_format = "csv"


@pytest.fixture
def harness_instance():
	admin_config = MockAdminConfig()
	synth_service = MockSyntheticDataAdapter()
	with patch("src.harness.Evaluator", MockEvaluator):
		return Harness(admin_config, synth_service)


@pytest.fixture
def dummy_dataset():
	return pd.DataFrame({"request": ["q1", "q2"], "response": ["a1", "a2"], "expected": [0, 1]})


def test_load_and_preprocess_data_creates_original_idx(harness_instance):
	df = pd.DataFrame({"request": ["q"], "response": ["a"]})
	with patch("pandas.read_csv", return_value=df):
		result = harness_instance._load_and_preprocess_data()
	assert "original_idx" in result.columns
	assert result["request"].iloc[0] == "q"


def test_load_and_preprocess_data_missing_columns(harness_instance):
	df = pd.DataFrame({"response": ["a"]})  # missing 'request'
	with patch("pandas.read_csv", return_value=df):
		result = harness_instance._load_and_preprocess_data()
	assert result.empty


def test_run_eval_against_original_data_with_expected(harness_instance, dummy_dataset):
	# If 'expected' exists, dataset returned as-is
	result = harness_instance._run_eval_against_original_data(dummy_dataset)
	pd.testing.assert_frame_equal(result, dummy_dataset)


def test_run_eval_against_original_data_generate_expected(harness_instance):
	df = pd.DataFrame({"request": ["q"], "response": ["a"]})
	harness_instance.admin_config.dataset_config.use_original_data_as_expected = True
	result = harness_instance._run_eval_against_original_data(df)
	assert "expected" in result.columns
	assert "reasoning" not in result.columns


def test_run_pipeline_calls_all_steps(harness_instance, dummy_dataset):
	with (
		patch.object(harness_instance, "_load_and_preprocess_data", return_value=dummy_dataset),
		patch.object(harness_instance, "_run_eval_against_original_data", return_value=dummy_dataset),
		patch.object(harness_instance.eval_metrics, "calculate_metrics", return_value={"accuracy": 1}),
		patch.object(harness_instance.eval_metrics, "generate_report") as mock_report,
	):
		harness_instance.run()
	mock_report.assert_called_once()
