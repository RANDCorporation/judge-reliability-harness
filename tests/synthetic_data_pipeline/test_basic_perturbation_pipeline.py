# tests/synthetic_data_pipeline/test_basic_perturbation_pipeline.py

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from src.synthetic_data_pipeline.basic_perturbation_pipeline import BasicPerturbationPipeline
from schemas import SyntheticDataParams, SavedItem, BasicLLMResponseStr, LLMClientConfig


@pytest.fixture
def dummy_seed_df():
	return pd.DataFrame(
		[
			{"original_idx": "1", "request": "Request 1", "response": "Response 1", "expected": 0},
			{"original_idx": "2", "request": "Request 2", "response": "Response 2", "expected": 1},
		]
	)


@pytest.fixture
def llm_config():
	return LLMClientConfig(
		model="openai/gpt-4o-mini", template="mock-template", temperature=0.0, max_tokens=100, test_debug_mode=True
	)


@pytest.fixture
def dummy_config(llm_config, tmp_path):
	return SyntheticDataParams(
		output_dir=tmp_path,
		generation_model_config=llm_config,
		validation_model_config=llm_config,
		max_tokens_generation=100,
		max_tokens_validation=50,
		use_similarity_filter=False,
		sample_num_from_orig=2,
		target_num_per_bucket=2,
		similarity_threshold=0.8,
		initial_temp=0.5,
		num_seed_examples_per_generation=1,
		temp_increment=0.1,
		max_temp_cap=1.0,
		max_consecutive_failures=2,
		seed=42,
	)


@pytest.fixture
def dummy_data_registry():
	registry = MagicMock()
	registry.get_data.return_value = pd.DataFrame(columns=["original_idx", "question"])
	registry.append = MagicMock()
	registry.close = MagicMock()
	return registry


@pytest.fixture
def dummy_generator_judge():
	mock_judge = MagicMock()
	mock_judge.call.return_value = BasicLLMResponseStr(score="1", reasoning="ok")
	return mock_judge


@pytest.fixture
def pipeline(dummy_config, dummy_seed_df, dummy_data_registry, dummy_generator_judge):
	with (
		patch("src.synthetic_data_pipeline.base_pipeline.LLMClient") as mock_llm,
		patch("src.synthetic_data_pipeline.basic_perturbation_pipeline.get_response_schema") as mock_schema,
	):
		mock_schema.return_value = MagicMock()
		mock_llm.return_value = dummy_generator_judge
		bp = BasicPerturbationPipeline("basic_test", dummy_config, dummy_seed_df, dummy_data_registry)
		bp.generator_judge = dummy_generator_judge
		bp.validator_judge = dummy_generator_judge
		yield bp


def test_load_state_empty_registry_returns_sample(pipeline, dummy_data_registry):
	dummy_data_registry.get_data.return_value = pd.DataFrame(columns=["original_idx", "question"])
	sample = pipeline.load_state()
	assert not sample.empty
	assert len(sample) <= pipeline.config.target_num_per_bucket


def test_load_state_with_data_returns_sample(pipeline, dummy_data_registry, dummy_seed_df):
	# registry already has one row
	dummy_data_registry.get_data.return_value = pd.DataFrame([{"original_idx": 1, "question": "Q1"}])
	sample = pipeline.load_state()
	assert not sample.empty
	assert all(col in sample.columns for col in dummy_seed_df.columns)


def test_generate_item_returns_saved_item(pipeline, dummy_seed_df):
	row = dummy_seed_df.iloc[0].to_dict()
	item = pipeline._generate_item(row)
	assert isinstance(item, SavedItem)

	dumped = item.model_dump()
	assert "generation_response" in dumped
	assert "generation_reasoning" in dumped


def test_run_returns_dataframe(pipeline, dummy_seed_df):
	# patch threaded_executor to return generator of (index, item)
	with patch("src.synthetic_data_pipeline.basic_perturbation_pipeline.threaded_executor") as mock_exec:
		mock_item = MagicMock(spec=SavedItem)
		mock_item.model_dump.return_value = {"score": "1", "reasoning": "ok"}
		mock_exec.return_value = [(0, mock_item), (1, mock_item)]

		df = pipeline.run(None)
		assert isinstance(df, pd.DataFrame)
		assert not df.empty
		# Each row should have keys from model_dump
		for row in df.to_dict(orient="records"):
			assert "score" in row
			assert "reasoning" in row


def test_run_with_item_callback_calls_callback(pipeline, dummy_seed_df):
	with patch("src.synthetic_data_pipeline.basic_perturbation_pipeline.threaded_executor") as mock_exec:
		mock_item = MagicMock(spec=SavedItem)
		mock_item.model_dump.return_value = {"score": "1", "reasoning": "ok"}
		mock_exec.return_value = [(0, mock_item)]

		callback = MagicMock()
		pipeline.run(callback)
		callback.assert_called_once_with(mock_item)
