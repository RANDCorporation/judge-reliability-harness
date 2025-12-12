# src/synthetic_data_pipeline/synthetic_data_adapter.py

import random
import time as time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core import console
from schemas import PerturbationConfig, SyntheticDataParams, TestRegistryEntry

from .agent_perturbation import generate_agent_judge_perturbation
from .basic_perturbation_pipeline import BasicPerturbationPipeline
from .data_registry import DataRegistry
from .review_server_manager import ReviewServerManager
from .stochastic_stability import generate_stochastic_stability
from .synthetic_ordinal_pipeline import SyntheticOrdinalPipeline


class SyntheticDataAdapter:
	"""
	Primary manager of the synthetic data pipeline.
	"""

	def __init__(self, perturbation_config: PerturbationConfig, validated_test_list: Dict[str, TestRegistryEntry]):
		self.perturbation_config = perturbation_config
		self.validated_test_list = validated_test_list
		self._agent_mode_results: Optional[Dict[str, pd.DataFrame]] = None

	def get_validated_test_names(self) -> List[str]:
		"""Returns list of validated test names"""
		return list(self.validated_test_list.keys())

	def get_meta_data(self, test_name: str) -> Dict[str, str]:
		"""Returns relevant meta data for available tests."""
		if test_name in self.validated_test_list:
			entry = self.validated_test_list[test_name]
			if isinstance(entry.config, SyntheticDataParams):
				return {
					"generation_name": entry.config.generation_model_config.model,
					"validation_name": entry.config.validation_model_config.model,
					"description": entry.description,
				}
			elif test_name == "stochastic_stability":
				return {
					"generation_name": "N/A",
					"validation_name": "N/A",
					"description": "Evaluation should remain constant for stochastic stability.",
				}
			elif test_name == "agent_perturbation":
				return {
					"generation_name": "N/A",
					"validation_name": "N/A",
					"description": "This is the agent perturbation test.",
				}
		return {"generation_name": "N/A", "validation_name": "N/A", "description": "N/A"}

	def _get_registry_path(self, test_name: str) -> Path:
		extension = self.perturbation_config.output_file_format
		effective_name = test_name
		if self._should_combine_agent_modes() and test_name == "agent_positives":
			effective_name = "agent_perturbation"
		return self.perturbation_config.output_dir / f"synthetic_{effective_name}.{extension}"

	def _should_combine_agent_modes(self) -> bool:
		tests = set(self.perturbation_config.tests_to_run or [])
		return {"agent_perturbation", "agent_positives"}.issubset(tests)

	def _filter_agent_rows(self, df: pd.DataFrame, test_name: str) -> pd.DataFrame:
		if (
			not df.empty
			and test_name in {"agent_perturbation", "agent_positives"}
			and "test_name" in df.columns
			and self._should_combine_agent_modes()
		):
			return df[df["test_name"] == test_name].reset_index(drop=True)
		return df

	def load_perturbations(self, test_name: str) -> pd.DataFrame:
		"""Loads saved perturbations (if any) as-is, without any processing."""
		data_registry_path = self._get_registry_path(test_name)
		data_registry = DataRegistry(data_registry_path)
		df = data_registry.get_data()
		return self._filter_agent_rows(df, test_name)

	def run(self, original_dataset: pd.DataFrame, test_name: str) -> pd.DataFrame:
		"""
		Generates perturbations for a specific test on the dataset.
		Uses cached perturbations if they already exist and generates only new ones.

		Args:
		    original_dataset (pd.DataFrame): The unperturbed dataset.
		    test_name (str): The name of the perturbation test to run.

		Returns:
		    pd.DataFrame: The perturbed dataset. Empty DataFrame if no perturbations generated.
		"""
		entry = self.validated_test_list.get(test_name)
		if not entry:
			console.print(f"[yellow]Test '{test_name}' is not a validated test. Skipping.[/yellow]")
			return pd.DataFrame()

		if getattr(entry, "requires_dataset", True) and original_dataset.empty:
			console.print("[WARNING] Original dataset is empty. Skipping synthetic generation process.")
			return pd.DataFrame()

		if test_name == "stochastic_stability":
			return generate_stochastic_stability(entry.config, original_dataset)

		if test_name in ("agent_perturbation", "agent_positives"):
			return self._run_agent_modes(test_name, entry)

		random.seed(entry.config.seed)
		np.random.seed(entry.config.seed)

		data_registry_path = self._get_registry_path(test_name)
		data_registry = DataRegistry(data_registry_path)
		self.review_server_manager = ReviewServerManager(self.perturbation_config, data_registry)

		if test_name == "synthetic_ordinal":
			pipeline = SyntheticOrdinalPipeline(test_name, entry.config, original_dataset, data_registry)
		else:
			pipeline = BasicPerturbationPipeline(test_name, entry.config, original_dataset, data_registry)

		if self.perturbation_config.use_HITL_process:
			perturbed_df = self.review_server_manager.run_HITL_server(test_name, pipeline)
		else:
			perturbed_df = pipeline.run(None)

		data_registry.close()

		if perturbed_df.empty:
			print(f"[yellow]Perturbations for {test_name} returned no data.[/]")

		return perturbed_df.reset_index(drop=True)

	def _run_agent_modes(self, test_name: str, entry: TestRegistryEntry) -> pd.DataFrame:
		if self._should_combine_agent_modes():
			if self._agent_mode_results is None:
				self._agent_mode_results = self._generate_combined_agent_modes()
			result = self._agent_mode_results.get(test_name, pd.DataFrame()).copy()
			if result.empty:
				console.print(f"[yellow]Perturbations for {test_name} returned no data.[/]")
			return result.reset_index(drop=True)
		return self._run_single_agent_mode(test_name, entry)

	def _run_single_agent_mode(self, test_name: str, entry: TestRegistryEntry) -> pd.DataFrame:
		data_registry_path = self._get_registry_path(test_name)
		data_registry = DataRegistry(data_registry_path)
		self.review_server_manager = ReviewServerManager(self.perturbation_config, data_registry)

		def _runner(callback):
			return generate_agent_judge_perturbation(entry.config, progress_callback=callback, test_name=test_name)

		try:
			if self.perturbation_config.use_HITL_process:
				perturbed_df = self.review_server_manager.run_agentic_session(test_name, _runner)
			else:

				def _append_only(record):
					data_registry.append(record)

				perturbed_df = _runner(_append_only)
		finally:
			data_registry.close()

		if perturbed_df.empty:
			print(f"[yellow]Perturbations for {test_name} returned no data.[/]")

		return self._filter_agent_rows(perturbed_df.reset_index(drop=True), test_name)

	def _generate_combined_agent_modes(self) -> Dict[str, pd.DataFrame]:
		data_registry_path = self._get_registry_path("agent_perturbation")
		data_registry = DataRegistry(data_registry_path)
		self.review_server_manager = ReviewServerManager(self.perturbation_config, data_registry)

		def _run_generators(callback):
			results = []
			for mode in ("agent_perturbation", "agent_positives"):
				entry = self.validated_test_list.get(mode)
				if not entry:
					continue
				df = generate_agent_judge_perturbation(entry.config, progress_callback=callback, test_name=mode)
				if df.empty:
					console.print(f"[yellow]Perturbations for {mode} returned no data.[/]")
				results.append(df)
			if not results:
				return pd.DataFrame()
			return pd.concat(results, ignore_index=True)

		try:
			if self.perturbation_config.use_HITL_process:
				combined_df = self.review_server_manager.run_agentic_session("agent_combined", _run_generators)
			else:

				def _append_only(record):
					data_registry.append(record)

				_run_generators(_append_only)
				combined_df = data_registry.get_data()
		finally:
			data_registry.close()

		if combined_df.empty:
			console.print("[yellow]Combined agent modes returned no data.[/]")

		return {mode: self._filter_agent_rows(combined_df, mode) for mode in ("agent_perturbation", "agent_positives")}
