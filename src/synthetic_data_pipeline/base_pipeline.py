# src/synthetic_data_pipeline/base_pipeline.py

import pandas as pd
import uuid
from typing import Type, Optional, Callable, Set
import time as time
from pydantic import BaseModel

from schemas import (
	SyntheticDataParams,
	OriginalDataPointConfig,
	BasicLLMResponseStr,
	SavedItem,
)
from core import console, LLMClient
from .utils import SimilarityFilter
from .data_registry import DataRegistry
from abc import ABC, abstractmethod


class BasePipeline(ABC):
	"""A base class for synthetic data generation pipelines."""

	def __init__(
		self,
		test_name: str,
		config: SyntheticDataParams,
		seed_df: pd.DataFrame,
		data_registry: DataRegistry,
	):
		self.test_name = test_name
		self.config = config
		self.seed_df = seed_df
		self.data_registry = data_registry
		self.item_callback = None

		if self.config.use_similarity_filter:
			self.similarity_filter = SimilarityFilter(threshold=self.config.similarity_threshold)
			self.similarity_filter.load_seed_data(self.seed_df.get("response", pd.Series([])).tolist())

		self.generator_judge = LLMClient(self.config.generation_model_config)
		self.validator_judge = LLMClient(self.config.validation_model_config)

	def get_data_to_review(self):
		"""Retrieves synthetic data that still needs to be reviewed."""
		cached = self.data_registry.get_data()
		data_to_be_reviewed = cached[~cached["human_reviewed"]]
		return data_to_be_reviewed

	@abstractmethod
	def load_state(self) -> Optional[pd.DataFrame]:
		"""
		Abstract method for determining which data from the original file should be processed. Should only be run by subclass.
		"""
		pass

	def _data_has_missing_cols(self, dataset: pd.DataFrame) -> Set[str]:
		"""Validation function for original dataset."""
		required_cols = {"original_idx", "request", "response", "expected"}
		missing = required_cols - set(dataset.columns)
		if missing:
			console.print(f"[yellow]Missing required columns: {', '.join(missing)}. Skipping generation.[/]")
			return missing
		return set()

	def _format_output(
		self,
		generated_data: BasicLLMResponseStr,
		validation_result: Type[BaseModel],
		source_item: OriginalDataPointConfig,
		temperature: float,
		prompted_bucket: Optional[int] = None,
	) -> SavedItem:
		"""Reformats data generated from the pipeline into a SavedItem object."""
		perturbed_idx = f"syn_{uuid.uuid4().hex[:8]}"
		saved_item = SavedItem(
			# original item
			test_name=self.test_name,
			original_request=source_item.request,
			original_response=source_item.response,
			original_idx=source_item.original_idx,
			original_expected=source_item.expected,
			# generated data
			generation_response=generated_data.score,
			generation_reasoning=generated_data.reasoning,
			perturbed_idx=perturbed_idx,
			generation_temp=temperature,
			# validation data
			validation_score=validation_result.score,
			validation_reasoning=validation_result.reasoning,
			# synthetic_ordinal
			prompted_bucket=str(prompted_bucket) if prompted_bucket is not None else "N/A",
			# HTIL part
			human_reviewed=False,
		)
		return saved_item

	@abstractmethod
	def _generate_item(self, **args) -> SavedItem:
		"""Abstract method for generating an individual item. Should only be run by subclass."""
		pass

	@abstractmethod
	def run(self, item_callback: Optional[Callable[[SavedItem], None]]) -> pd.DataFrame:
		"""Main runner for synthetic_ordinal pipeline. Should only be run by subclass."""
		pass
