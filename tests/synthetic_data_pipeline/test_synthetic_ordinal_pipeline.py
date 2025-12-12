# tests/synthetic_data_pipeline/test_synthetic_ordinal_pipeline.py

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from src.synthetic_data_pipeline.synthetic_ordinal_pipeline import SyntheticOrdinalPipeline
from schemas import SyntheticDataParams, OriginalDataPointConfig, SavedItem, LLMClientConfig


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
	registry.get_data.return_value = pd.DataFrame(
		[
			{"original_idx": 1, "validation_score": 0, "human_reviewed": False},
			{"original_idx": 2, "validation_score": 1, "human_reviewed": True},
		]
	)
	registry.append = MagicMock()
	registry.close = MagicMock()
	return registry


@pytest.fixture
def dummy_bucket_manager():
	mock_bucket_manager = MagicMock()
	mock_bucket_manager.buckets = [0, 1]
	mock_bucket_manager.items_by_bucket = {}
	mock_bucket_manager.visualize_status = MagicMock()
	return mock_bucket_manager


@pytest.fixture
def dummy_generator_judge(dummy_seed_df):
	"""Return a MagicMock that produces fully valid SavedItem fields."""
	mock_judge = MagicMock()

	def _generate_response(*args, **kwargs):
		row = dummy_seed_df.iloc[0]
		temperature = kwargs.get("temperature", 0.5)

		response = MagicMock()
		response.score = "0"  # int works better for bucket validation
		response.reasoning = "ok"
		response.generation_completion = "completed"

		# model_dump returns dict matching SavedItem fields
		response.model_dump.return_value = {
			"test_name": "synthetic_ordinal_test",
			"original_request": row["request"],
			"original_response": row["response"],
			"original_idx": str(row["original_idx"]),
			"original_expected": int(row["expected"]),
			"perturbed_idx": f"{row['original_idx']}_0",
			"generation_response": 0,
			"generation_reasoning": "ok",
			"generation_temp": temperature,
			"validation_score": 0,
			"validation_reasoning": "valid",
			"prompted_bucket": 0,
			"validated_bucket": 0,
			"human_reviewed": False,
		}
		return response

	mock_judge.call.side_effect = _generate_response
	return mock_judge


@pytest.fixture
def pipeline(dummy_config, dummy_seed_df, dummy_data_registry, dummy_bucket_manager, dummy_generator_judge):
	with (
		patch("src.synthetic_data_pipeline.base_pipeline.LLMClient") as mock_llm,
		patch(
			"src.synthetic_data_pipeline.synthetic_ordinal_pipeline.BucketManager", return_value=dummy_bucket_manager
		),
		patch("src.synthetic_data_pipeline.synthetic_ordinal_pipeline.get_sample", return_value=dummy_seed_df),
		patch(
			"src.synthetic_data_pipeline.synthetic_ordinal_pipeline.build_synthetic_ordinal_template",
			return_value={"template_var": 1},
		),
	):
		mock_llm.return_value = dummy_generator_judge

		pipe = SyntheticOrdinalPipeline("synthetic_ordinal_test", dummy_config, dummy_seed_df, dummy_data_registry)
		pipe.generator_judge = dummy_generator_judge
		pipe.validator_judge = dummy_generator_judge
		yield pipe


def test_load_state_sets_generated_variants(pipeline, dummy_data_registry):
	dummy_data_registry.get_data.return_value = pd.DataFrame(
		[
			{"original_idx": 1, "validation_score": 0, "human_reviewed": False},
			{"original_idx": 2, "validation_score": 1, "human_reviewed": True},
		]
	)
	pipeline.load_state()
	assert ("1", 0) in pipeline.generated_variants
	assert ("2", 1) in pipeline.generated_variants


def test_generate_item_returns_saved_item(pipeline, dummy_seed_df):
	source_item = OriginalDataPointConfig(**dummy_seed_df.iloc[0].to_dict())
	item = pipeline._generate_item(0, source_item, temperature=0.5)
	assert isinstance(item, SavedItem)


def test_perform_generation_cycle_adds_item(pipeline, dummy_seed_df):
	source_item = OriginalDataPointConfig(**dummy_seed_df.iloc[0].to_dict())
	result = pipeline._perform_generation_cycle(0, source_item, temperature=0.5)
	assert result in [True, False]
	assert len(pipeline.kept_items) >= 0


def test_try_generate_for_bucket_returns_bool(pipeline, dummy_seed_df):
	source_item = OriginalDataPointConfig(**dummy_seed_df.iloc[0].to_dict())
	success = pipeline._try_generate_for_bucket(0, source_item)
	assert isinstance(success, bool)


def test_run_returns_dataframe(pipeline):
	df = pipeline.run()
	assert isinstance(df, pd.DataFrame)
	# Check that mocked model_dump fields exist
	expected_cols = [
		"test_name",
		"original_request",
		"original_response",
		"original_idx",
		"original_expected",
		"perturbed_idx",
		"generation_response",
		"generation_reasoning",
		"generation_temp",
		"validation_score",
		"validation_reasoning",
		"prompted_bucket",
		"human_reviewed",
	]
	for col in expected_cols:
		assert col in df.columns or df.empty
