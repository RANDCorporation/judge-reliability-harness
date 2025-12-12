# tests/synthetic_data_pipeline/test_synthetic_data_adapter.py

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from synthetic_data_pipeline.synthetic_data_adapter import SyntheticDataAdapter

# -------------------------------------------------------------------------
# FIXTURES
# -------------------------------------------------------------------------


@pytest.fixture
def dummy_config(tmp_path):
	"""Fake PerturbationConfig with minimal attributes."""
	mock_config = MagicMock()
	mock_config.seed = 42
	mock_config.use_HITL_process = False
	mock_config.output_dir = tmp_path
	mock_config.output_file_format = "csv"
	mock_config.tests_to_run = []
	return mock_config


@pytest.fixture
def dummy_dataset():
	return pd.DataFrame({"request": ["Q1", "Q2"], "response": ["A1", "A2"]})


@pytest.fixture
def dummy_entry():
	"""Simulate a TestRegistryEntry object."""
	entry = MagicMock()
	config_mock = MagicMock()
	config_mock.seed = 42
	entry.config = config_mock
	entry.description = "Test description"
	return entry


@pytest.fixture
def validated_test_list(dummy_entry):
	return {
		"basic_test": dummy_entry,
		"synthetic_ordinal": dummy_entry,
		"stochastic_stability": dummy_entry,
		"agent_perturbation": dummy_entry,
		"agent_positives": dummy_entry,
	}


# -------------------------------------------------------------------------
# BASIC FUNCTIONAL TESTS
# -------------------------------------------------------------------------


def test_get_validated_test_names(dummy_config, validated_test_list):
	adapter = SyntheticDataAdapter(dummy_config, validated_test_list)
	names = adapter.get_validated_test_names()
	assert sorted(names) == sorted(list(validated_test_list.keys()))


def test_load_perturbations_calls_data_registry(dummy_config, validated_test_list, tmp_path):
	adapter = SyntheticDataAdapter(dummy_config, validated_test_list)
	fake_df = pd.DataFrame({"x": [1]})

	test_name = "basic_test"
	with patch("synthetic_data_pipeline.synthetic_data_adapter.DataRegistry") as mock_registry:
		mock_instance = MagicMock()
		mock_instance.get_data.return_value = fake_df
		mock_registry.return_value = mock_instance

		dummy_config.output_dir = tmp_path
		result = adapter.load_perturbations(test_name)

		expected_path = tmp_path / f"synthetic_{test_name}.csv"
		mock_registry.assert_called_once_with(expected_path)
		mock_instance.get_data.assert_called_once()
		assert result.equals(fake_df)


def test_load_perturbations_respects_xlsx_format(dummy_config, validated_test_list, tmp_path):
	adapter = SyntheticDataAdapter(dummy_config, validated_test_list)
	fake_df = pd.DataFrame({"x": [1]})

	test_name = "basic_test"
	with patch("synthetic_data_pipeline.synthetic_data_adapter.DataRegistry") as mock_registry:
		mock_instance = MagicMock()
		mock_instance.get_data.return_value = fake_df
		mock_registry.return_value = mock_instance

		dummy_config.output_dir = tmp_path
		dummy_config.output_file_format = "xlsx"
		result = adapter.load_perturbations(test_name)

		expected_path = tmp_path / f"synthetic_{test_name}.xlsx"
		mock_registry.assert_called_once_with(expected_path)
		mock_instance.get_data.assert_called_once()
		assert result.equals(fake_df)


# -------------------------------------------------------------------------
# RUN METHOD TESTS
# -------------------------------------------------------------------------


def test_run_returns_empty_if_original_df_empty(dummy_config, validated_test_list):
	adapter = SyntheticDataAdapter(dummy_config, validated_test_list)
	empty_df = pd.DataFrame()
	result = adapter.run(empty_df, "basic_test")
	assert result.empty


def test_run_returns_empty_if_test_not_validated(dummy_config):
	adapter = SyntheticDataAdapter(dummy_config, validated_test_list={})
	df = pd.DataFrame({"request": ["x"]})
	result = adapter.run(df, "nonexistent_test")
	assert result.empty


def test_run_stochastic_stability_path(dummy_config, validated_test_list, dummy_dataset):
	adapter = SyntheticDataAdapter(dummy_config, validated_test_list)

	with patch("synthetic_data_pipeline.synthetic_data_adapter.generate_stochastic_stability") as mock_gen:
		mock_df = pd.DataFrame({"out": [123]})
		mock_gen.return_value = mock_df

		result = adapter.run(dummy_dataset, "stochastic_stability")

		mock_gen.assert_called_once()
		assert result.equals(mock_df)


def test_run_agent_perturbation_path(dummy_config, validated_test_list, dummy_dataset):
	adapter = SyntheticDataAdapter(dummy_config, validated_test_list)

	with patch("synthetic_data_pipeline.synthetic_data_adapter.generate_agent_judge_perturbation") as mock_gen:
		mock_df = pd.DataFrame({"out": [999]})
		mock_gen.return_value = mock_df

		result = adapter.run(dummy_dataset, "agent_perturbation")

		mock_gen.assert_called_once()
		assert mock_gen.call_args.kwargs["test_name"] == "agent_perturbation"
		assert result.equals(mock_df)


def test_basic_pipeline_run():
	# Minimal config
	config = MagicMock()
	config.use_HITL_process = False

	# Minimal test entry with seed
	test_entry = MagicMock()
	test_entry.config.seed = 42
	test_entry.description = "Basic test"
	validated_test_list = {"basic_test": test_entry}

	adapter = SyntheticDataAdapter(config, validated_test_list)

	# Patch DataRegistry to return non-empty data
	with (
		patch("synthetic_data_pipeline.synthetic_data_adapter.DataRegistry") as mock_registry,
		patch("synthetic_data_pipeline.synthetic_data_adapter.BasicPerturbationPipeline") as mock_pipeline,
		patch("synthetic_data_pipeline.synthetic_data_adapter.ReviewServerManager"),
	):
		mock_registry.return_value.get_data.return_value = pd.DataFrame({"request": [1], "response": [2]})

		fake_pipeline = MagicMock()
		fake_pipeline.run.return_value = pd.DataFrame({"perturbed": [1]})
		mock_pipeline.return_value = fake_pipeline

		dummy_dataset = pd.DataFrame({"request": ["Q1"], "response": ["A1"]})
		adapter.run(dummy_dataset, "basic_test")

		mock_pipeline.assert_called_once()
		fake_pipeline.run.assert_called_once()


def test_run_with_HITL_enabled_uses_review_server(dummy_config, validated_test_list, dummy_dataset):
	dummy_config.use_HITL_process = True


def test_load_perturbations_filters_combined_modes(dummy_config, validated_test_list):
	dummy_config.tests_to_run = ["agent_perturbation", "agent_positives"]
	adapter = SyntheticDataAdapter(dummy_config, validated_test_list)
	fake_df = pd.DataFrame(
		{
			"test_name": ["agent_perturbation", "agent_positives"],
			"value": [1, 2],
		}
	)

	with patch("synthetic_data_pipeline.synthetic_data_adapter.DataRegistry") as mock_registry:
		mock_instance = MagicMock()
		mock_instance.get_data.return_value = fake_df
		mock_registry.return_value = mock_instance

		result = adapter.load_perturbations("agent_positives")

		assert set(result["test_name"]) == {"agent_positives"}


def test_combined_agent_modes_run_once(dummy_config, validated_test_list, dummy_dataset):
	dummy_config.use_HITL_process = True
	dummy_config.tests_to_run = ["agent_perturbation", "agent_positives"]
	adapter = SyntheticDataAdapter(dummy_config, validated_test_list)

	with (
		patch("synthetic_data_pipeline.synthetic_data_adapter.DataRegistry") as mock_registry,
		patch("synthetic_data_pipeline.synthetic_data_adapter.ReviewServerManager") as mock_manager,
		patch("synthetic_data_pipeline.synthetic_data_adapter.generate_agent_judge_perturbation") as mock_gen,
	):
		mock_registry.return_value = MagicMock()

		def _fake_run_agentic_session(batch_id, runner):
			return runner(lambda _: None)

		manager_instance = MagicMock()
		manager_instance.run_agentic_session.side_effect = _fake_run_agentic_session
		mock_manager.return_value = manager_instance

		mock_gen.side_effect = [
			pd.DataFrame({"test_name": ["agent_perturbation"], "perturbed_idx": ["p1"]}),
			pd.DataFrame({"test_name": ["agent_positives"], "perturbed_idx": ["p2"]}),
		]

		result = adapter.run(dummy_dataset, "agent_perturbation")
		assert set(result["test_name"]) == {"agent_perturbation"}

		result_positive = adapter.run(dummy_dataset, "agent_positives")
		assert set(result_positive["test_name"]) == {"agent_positives"}

		assert manager_instance.run_agentic_session.call_count == 1
		assert mock_gen.call_count == 2

	with (
		patch("synthetic_data_pipeline.synthetic_data_adapter.DataRegistry") as mock_registry,
		patch("synthetic_data_pipeline.synthetic_data_adapter.ReviewServerManager") as mock_review_mgr,
		patch("synthetic_data_pipeline.synthetic_data_adapter.BasicPerturbationPipeline"),
	):
		fake_df = pd.DataFrame({"perturbed": [5]})
		mock_review_mgr_instance = MagicMock()
		mock_review_mgr_instance.run_HITL_server.return_value = fake_df
		mock_review_mgr.return_value = mock_review_mgr_instance

		mock_registry.return_value = MagicMock()
		patch("synthetic_data_pipeline.synthetic_data_adapter.BasicPerturbationPipeline").start()

		result = adapter.run(dummy_dataset, "basic_test")

		mock_review_mgr_instance.run_HITL_server.assert_called_once()
		assert result.equals(fake_df.reset_index(drop=True))


def test_run_uses_synthetic_ordinal_pipeline(dummy_config, validated_test_list, dummy_dataset):
	dummy_config.use_HITL_process = False
	adapter = SyntheticDataAdapter(dummy_config, validated_test_list)

	with (
		patch("synthetic_data_pipeline.synthetic_data_adapter.DataRegistry") as mock_registry,
		patch("synthetic_data_pipeline.synthetic_data_adapter.SyntheticOrdinalPipeline") as mock_pipeline,
		patch("synthetic_data_pipeline.synthetic_data_adapter.ReviewServerManager"),
	):
		mock_registry_instance = MagicMock()
		mock_registry_instance.get_data.return_value = pd.DataFrame({"request": ["x"], "response": ["y"]})
		mock_registry.return_value = mock_registry_instance

		pipeline_instance = MagicMock()
		pipeline_instance.run.return_value = pd.DataFrame({"result": [42]})
		mock_pipeline.return_value = pipeline_instance

		result = adapter.run(dummy_dataset, "synthetic_ordinal")

		mock_pipeline.assert_called_once()
		pipeline_instance.run.assert_called_once()
		assert "result" in result.columns
