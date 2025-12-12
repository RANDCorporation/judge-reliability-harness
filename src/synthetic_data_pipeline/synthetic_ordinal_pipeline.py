# src/synthetic_data_pipeline/synthetic_ordinal.py

import time as time
from typing import (
	Callable,
	List,
	Optional,
	Set,
	Tuple,
)

import pandas as pd

from core import console, get_response_schema
from schemas import BasicLLMResponseStr, OriginalDataPointConfig, SavedItem, SyntheticDataParams

from .base_pipeline import BasePipeline
from .data_registry import DataRegistry
from .utils import BucketManager, build_synthetic_ordinal_template, get_sample


class SyntheticOrdinalPipeline(BasePipeline):
	"""A base class for synthetic data generation pipelines."""

	def __init__(self, test_name: str, config: SyntheticDataParams, seed_df: pd.DataFrame, data_registry: DataRegistry):
		super().__init__(test_name, config, seed_df, data_registry)

		self.bucket_manager = BucketManager(self.seed_df, seed=config.seed)
		console.print(f"\nTarget buckets: {self.bucket_manager.buckets}")
		self.kept_items: List[SavedItem] = []
		self.generated_variants: Set[Tuple[str, int]] = set()

		self.validation_response_schema = get_response_schema(self.config.validation_model_config.template)

	@staticmethod
	def _coerce_bucket_value(raw_value) -> Optional[int]:
		"""
		Best-effort conversion of cached bucket/score values into ints.
		"""
		if pd.isna(raw_value):
			return None
		try:
			return int(raw_value)
		except (TypeError, ValueError):
			return None

	def load_state(self) -> Optional[pd.DataFrame]:
		"""
		Loads state based on existing saved file.
		"""
		cached = self.data_registry.get_data()
		self.generated_variants = set()
		if cached.empty:
			return cached

		for item in cached.itertuples():
			source_id = str(getattr(item, "original_idx", ""))
			validated_bucket = self._coerce_bucket_value(getattr(item, "validation_score", None))

			if source_id:
				if validated_bucket is not None:
					self.generated_variants.add((source_id, validated_bucket))
		return cached

	def _generate_item(
		self, target_bucket: int, source_item: OriginalDataPointConfig, temperature: float
	) -> Optional[SavedItem]:
		"""Performs a single generate-validate-filter cycle. Returns SavedItem on success."""
		# generation stage
		template_vars = build_synthetic_ordinal_template(
			self.seed_df,
			source_item,
			target_bucket,
			self.config.num_seed_examples_per_generation,
			self.bucket_manager.items_by_bucket,
		)
		if template_vars is None:
			return None

		generated_response = self.generator_judge.call(template_vars, BasicLLMResponseStr, temperature, None)

		# validation stage
		generated_item = source_item.model_copy(update={"response": generated_response.score})
		validation_result = self.validator_judge.call(
			generated_item.model_dump(), self.validation_response_schema, None, None
		)

		# Return the item
		return self._format_output(generated_response, validation_result, source_item, temperature, target_bucket)

	def _perform_generation_cycle(
		self, target_bucket: int, source_item: OriginalDataPointConfig, temperature: float
	) -> bool:
		"""
		Performs a single generate-validate-filter cycle. Returns True on success.
		Incorporates logic to save the generated item.
		"""
		# 1. Generate the item
		saved_item = self._generate_item(target_bucket, source_item, temperature)
		if saved_item is None:
			return None

		# 2. Check if we already have this validation_score for this source (prevent duplicates)
		source_id = str(source_item.original_idx)
		validated_bucket = self._coerce_bucket_value(saved_item.validation_score)
		if validated_bucket is not None and (source_id, validated_bucket) in self.generated_variants:
			console.print(
				f"Slot full: {target_bucket} → {saved_item.validation_score} (bucket {saved_item.validation_score} already filled for {source_id})"
			)
			return False

		# 3. Apply similarity filter (if eligible)
		if self.config.use_similarity_filter and not self.similarity_filter.check(saved_item.generation_response):
			return False

		# 4. Save the item (whether it matches target or not)
		self.kept_items.append(saved_item)
		self.data_registry.append(saved_item)

		if self.item_callback:
			try:
				self.item_callback(saved_item)
			except Exception as exc:
				console.print(f"[red]Item callback failed: {exc}[/]")

		# Track by prompted_bucket (what we tried to generate) for resume logic
		if validated_bucket is not None:
			self.generated_variants.add((str(saved_item.original_idx), validated_bucket))

		match_symbol = "✓" if saved_item.prompted_bucket == saved_item.validation_score else "↗"
		console.print(
			f"\n    {match_symbol} Saved {saved_item.perturbed_idx}:{saved_item.prompted_bucket} → {saved_item.validation_score} (temp: {saved_item.generation_temp:.2f})"
		)

		# 5. Return True only if we got exactly what we asked for
		return saved_item.validation_score == target_bucket

	def _try_generate_for_bucket(self, target_bucket: int, source_item: OriginalDataPointConfig) -> bool:
		"""
		Tries to generate a variant for a given source item and bucket,
		with retries and temperature scaling.
		"""
		console.print(f"   Targeting bucket: {target_bucket}")
		current_temp = self.config.initial_temp

		while current_temp <= self.config.max_temp_cap:
			for attempt in range(self.config.max_consecutive_failures):
				console.print(
					f"Attempt {attempt + 1}/{self.config.max_consecutive_failures} (temp: {current_temp:.2f})"
				)
				success = self._perform_generation_cycle(target_bucket, source_item, current_temp)
				if success:
					return True

			console.print(
				f"Attempt failed. Increasing temperature: {current_temp:.2f} → {current_temp + self.config.temp_increment:.2f}"
			)
			current_temp += self.config.temp_increment

		return False

	def run(self, item_callback: Optional[Callable[[SavedItem], None]] = None) -> pd.DataFrame:
		"""Main runner for synthetic_ordinal pipeline"""
		# initialize runner
		console.print(f"\nRunning {self.__class__.__name__}...")

		self.item_callback = item_callback
		dataset = get_sample(self.seed_df, self.config.sample_num_from_orig, with_replacement=True)

		if self._data_has_missing_cols(dataset):
			return pd.DataFrame()

		self.load_state()

		# iteratue through dataset
		for idx, row in enumerate(dataset.to_dict(orient="records"), 1):
			source_item = OriginalDataPointConfig(**row)

			# visualize status
			console.print(f"\n[{idx}/{len(dataset)}] Processing source: {source_item.original_idx}")
			self.bucket_manager.visualize_status(source_item.original_idx, self.generated_variants)

			# run target bucket loop
			source_id = str(source_item.original_idx)
			for target_bucket in self.bucket_manager.buckets:
				if (source_id, target_bucket) in self.generated_variants:
					console.print(f"   Skipping bucket {target_bucket}: validator already supplied coverage.")
					continue

				success = self._try_generate_for_bucket(target_bucket, source_item)

				if not success:
					console.print(f"      Failed to generate valid item for bucket {target_bucket} (max temp reached)")

			# visualize status
			self.bucket_manager.visualize_status(source_item.original_idx, self.generated_variants)

		final_output = pd.DataFrame([item.model_dump() for item in self.kept_items])
		return final_output
