# src/core/schemas/configs.py

from pathlib import Path
from typing import (
	Any,
	Dict,
	List,
	Literal,
	Optional,
)

from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, field_validator


class LLMClientConfig(BaseModel):
	model: Optional[str] = Field(None, description="Model identifier (e.g., openai/gpt-4o-mini).")
	template: str = Field(..., description="Template to use.")

	default_params: Optional[Dict[str, str]] = Field(
		default_factory=dict, description="Default parameters for template filling."
	)

	temperature: float = Field(0.0, description="Temperature to parameterize LLM model.")
	test_debug_mode: bool = Field(True, description="Use debugging mode to reduce API calls.")
	rate_limit: Dict[str, int] = Field(
		default_factory=dict, description="Rate limit to impose on LLM (e.g., calls per minute)."
	)
	retries: int = Field(3, description="Max number of retries to use.")
	max_tokens: int = Field(1200, gt=0, description="Maximum tokens for LLMs.")


class JudgeConfig(BaseModel):
	"""Configuration for a single judge in multi-judge mode."""

	name: str = Field(..., description="Unique name identifier for this judge.")
	model: str = Field(..., description="Model identifier (e.g., openai/gpt-4o-mini).")
	template: Optional[str] = Field(
		None, description="Template to use for this judge (defaults to evaluation_config.template)."
	)
	preprocess_columns_map: Dict[str, str] = Field(
		default_factory=dict,
		description="Column mapping for this judge (request, response, expected columns).",
	)


class EvaluationConfig(BaseModel):
	template: str = Field(..., description="Template to use for aiautograder")
	autograder_model_name: str = Field(..., description="Model name for autograding.")
	autograder_model_config: Optional[LLMClientConfig] = Field(
		None, description="LLM Model config to use for Evaluation stage."
	)

	# Multiple judges mode
	judges: Optional[List[JudgeConfig]] = Field(
		None,
		description="List of judges for multi-judge evaluation. If provided, each judge evaluates the same perturbed samples.",
	)

	overwrite_results: bool = Field(True, description="Should overwrite existing results in storage.")
	max_workers: int = Field(10, description="Max number of workers to use when deploying in parallel.")

	output_dir: Optional[DirectoryPath] = Field(None, description="Resolved output directory.")
	tests_to_evaluate: Optional[List[str]] = Field(default_factory=list, description="List of tests to evaluate.")

	metric: str = Field(..., description="sklearn metric to employ for evaluation")
	bootstrap_size: Optional[float] = Field(None, description="Size for bootstrap evaluation.")
	bootstrap_repetitions: int = Field(10, description="Number of repetitions to get bootstapped metric.")
	get_cost_curves: bool = Field(False, description="Whether to run the cost curves module at reporting stage.")
	output_file_format: Literal["csv", "xlsx"] = Field("csv", description="File format for evaluator outputs.")

	# Judge aggregation (for multi-judge mode)
	aggregation_method: Optional[str] = Field(
		None,
		description="Method to aggregate multiple judge scores. Options: 'majority_vote' (or None to skip aggregation).",
	)
	aggregation_reference_column: Optional[str] = Field(
		None,
		description="Column name in original dataset to compare aggregated result against (e.g., 'majority_vote'). Only used if aggregation_method is set.",
	)

	@field_validator("tests_to_evaluate", mode="before")
	def handle_none(cls, v):
		return v or []

	@field_validator("judges", mode="before")
	def handle_judges(cls, v):
		"""Ensure judges is a list if provided."""
		if v is None:
			return None
		if isinstance(v, list):
			return v
		return [v]  # Convert single judge dict to list

	def is_multi_judge_mode(self) -> bool:
		"""Check if this config is using multiple judges."""
		return self.judges is not None and len(self.judges) > 0


class DatasetConfig(BaseModel):
	dataset_name: str = Field(..., description="CSV dataset filename in ./data/inputs/{module_name}")
	dataset_path: Optional[Path] = Field(None, description="Resolved dataset path.")
	default_params: dict = Field(default_factory=dict, description="Default parameters to include for all tests.")
	use_original_data_as_expected: bool = Field(
		False, description="Preprocess by calling the aiautograder on original data."
	)


class TestRegistryEntry(BaseModel):
	test_name: str = Field(..., description="Name identifier of the test, matching AdminConfig attribute.")
	config: Optional[Any] = Field(
		None, description="Validated config instance specific to the test (e.g., TestLabelFlipConfig)."
	)
	description: str = Field(..., description="Human-readable description of the test's purpose.")
	requires_dataset: bool = Field(True, description="Whether this test requires the original dataset input.")


class SyntheticDataParams(BaseModel):
	model_config = ConfigDict(extra="ignore")  # Allows and ignores extra fields
	output_dir: Optional[DirectoryPath] = Field(None, description="Resolved output directory.")

	generation_model_config: LLMClientConfig = Field(..., description="LLM Model config to use for Generation stage.")
	validation_model_config: LLMClientConfig = Field(..., description="LLM Model config to use for Validation stage.")
	max_tokens_generation: int = Field(..., gt=0, description="Maximum tokens for generation.")
	max_tokens_validation: int = Field(..., gt=0, description="Maximum tokens for validation.")
	max_workers: int = Field(10, description="Max number of workers to use when deploying in parallel.")

	use_similarity_filter: bool = Field(
		True, description="Flag for whether to use similarity filter in synthetic data pipeline."
	)
	sample_num_from_orig: Optional[int] = Field(None, description="Number of samples to draw from original dataset.")
	target_num_per_bucket: int = Field(int, description="Number of examples to generate per bucket")
	similarity_threshold: float = Field(..., ge=0, le=1, description="Minimum similarity for re-use or acceptance.")
	initial_temp: float = Field(..., description="Initial temperature for generation.")
	num_seed_examples_per_generation: int = Field(..., gt=0, description="Number of seed examples used per generation.")
	temp_increment: float = Field(..., description="Amount to increment temperature on failure.")
	max_temp_cap: float = Field(..., description="Maximum allowable generation temperature.")
	max_consecutive_failures: int = Field(..., description="Abort pipeline after this many retries.")
	seed: int = Field(..., description="Random seed for reproducibility.")


class PerturbationConfig(BaseModel):
	preprocess_columns_map: dict = Field(default_factory=dict, description="Map for relabelling columns upon input.")
	output_dir: Optional[DirectoryPath] = Field(None, description="Resolved output directory.")
	tests_to_run: Optional[List[str]] = Field(default_factory=list, description="List of test names to run.")
	output_file_format: Literal["csv", "xlsx"] = Field("csv", description="File format for synthetic data outputs.")

	# preview config items
	use_HITL_process: bool = Field(False, description="Whether review UI for perturbations is enabled.")
	review_log_dir: Optional[DirectoryPath] = Field(None, description="Resolved output directory.")

	@field_validator("tests_to_run", mode="before")
	def handle_none(cls, v):
		return v or []


class TestStochasticStabilityConfig(BaseModel):
	sample_num_from_orig: int = Field(0, description="Number of examples to evaluate.")
	number_of_seeds: int = Field(10, description="Number of seeds to generate for stability sequence.")
	repetitions: int = Field(1, description="Number of repetitions per element in stability sequence.")
	seed: int = Field(..., description="Random seed for reproducibility.")


class AdminConfig(BaseModel):
	model_config = ConfigDict(extra="ignore")

	base_module_name: str = Field(..., description="Base module without timestamp.")
	module_name: str = Field(..., description="Base name of the module being run.")
	time_stamp: Optional[str] = Field(None, description="Optional timestamp string. If None, current time is used.")
	test_debug_mode: bool = Field(True, description="Use debugging mode to reduce API calls.")
	output_dir: Optional[DirectoryPath] = Field(None, description="Resolved output directory.")
	output_file_format: Literal["csv", "xlsx"] = Field("csv", description="File format for outputs.")

	dataset_config: DatasetConfig
	perturbation_config: PerturbationConfig
	evaluation_config: EvaluationConfig
