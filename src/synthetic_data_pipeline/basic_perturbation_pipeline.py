# src/synthetic_data_pipeline/basic_perturbation_pipeline.py

import pandas as pd
from typing import (
	Optional,
	Callable,
	Any,
	Dict,
)
import time as time

from schemas import (
	SyntheticDataParams,
	OriginalDataPointConfig,
	SavedItem,
	BasicLLMResponseStr,
)
from core import console, get_response_schema, threaded_executor
from .base_pipeline import BasePipeline
from .data_registry import DataRegistry
from .utils import get_sample


class BasicPerturbationPipeline(BasePipeline):
	def __init__(
		self,
		test_name: str,
		config: SyntheticDataParams,
		seed_df: pd.DataFrame,
		data_registry: DataRegistry,
	):
		super().__init__(test_name, config, seed_df, data_registry)

		self.validation_response_schema = get_response_schema(self.config.validation_model_config.template)

	def _generate_item(self, row: Dict[str, Any]) -> SavedItem:
		"""Generates a single item."""
		source_item = OriginalDataPointConfig(**row)

		# generation stage
		generated_response = self.generator_judge.call(source_item.model_dump(), BasicLLMResponseStr, None, None)

		# validation stage
		generated_item = source_item.model_copy(update={"response": generated_response.score})
		validated_response = self.validator_judge.call(
			generated_item.model_dump(), self.validation_response_schema, None, None
		)

		# Return the item
		return self._format_output(
			generated_response,
			validated_response,
			source_item,
			self.config.generation_model_config.temperature,
		)

	def load_state(self) -> Optional[pd.DataFrame]:
		"""
		Determines the state of the existing perturbation file, including how many new synthetic data points to
		generate, and which data currently need to be reviewed, from the original file.
		"""
		cached = self.data_registry.get_data()

		# assume only a single bucket for basic perturbations
		num_to_perturb = max(0, self.config.target_num_per_bucket - len(cached))

		if cached.empty:
			sample_df = get_sample(self.seed_df, num_to_perturb)
			return sample_df

		# TODO: Uncomment this if we want to restrict generation based on existing samples
		# processed_ids = set(cached["original_idx"]) if "original_idx" in cached else set()
		# to_process = self.seed_df[~self.seed_df["original_idx"].isin(processed_ids)]
		# if to_process.empty:
		#     console.print(f"[yellow]Original dataset already contained in cached synthetic data set. Skipping generation.[/]")
		#     return pd.DataFrame()
		to_process = self.seed_df

		if self._data_has_missing_cols(to_process):
			return pd.DataFrame()

		sample_df = get_sample(to_process, num_to_perturb, with_replacement=True)

		return sample_df

	def run(self, item_callback: Optional[Callable[[SavedItem], None]]) -> pd.DataFrame:
		"""Main runner for basic_perturbation pipeline"""
		console.print(f"\nRunning {self.__class__.__name__}...")

		dataset = self.load_state()

		if dataset.empty:
			console.print("[bold red]Original dataset is empty. Skipping test.[/]")
			return pd.DataFrame()

		results = []
		for _, result in threaded_executor(dataset, self._generate_item, self.config.max_workers):
			self.data_registry.append(result)
			results.append(result.model_dump())
			if item_callback:
				item_callback(result)

		return pd.DataFrame(results)
