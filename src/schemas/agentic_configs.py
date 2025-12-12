# src/core/schemas/agentic_configs.py

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, Field, field_validator


class AgentLLMStageConfig(BaseModel):
	model: str = Field(..., description="Model identifier for the agent stage.")
	prompt_path: Path = Field(..., description="Path to the prompt template used for this stage.")
	temperature: float = Field(0.0, description="Sampling temperature.")
	system_prompt: Optional[str] = Field(None, description="Optional system prompt.")

	@field_validator("prompt_path", mode="before")
	@classmethod
	def _coerce_prompt_path(cls, value: Union[str, Path]) -> Path:
		path = Path(value)
		if not path.is_file():
			raise FileNotFoundError(f"Prompt template not found at '{path}'.")
		return path.resolve()


class AgentOutputConfig(BaseModel):
	dir: Path = Field(default_factory=lambda: Path("./outputs/agent_perturbation"))
	write_jsonl: bool = Field(True, description="Whether to write perturbation records to JSONL.")
	overwrite: bool = Field(True, description="Whether to overwrite existing artifacts.")

	@field_validator("dir", mode="before")
	@classmethod
	def _coerce_dir(cls, value: Union[str, Path]) -> Path:
		return Path(value).resolve()


class TestAgentPerturbationConfig(BaseModel):
	input_log_path: Path = Field(..., description="Path to an Inspect-AI `.eval` archive containing agent runs.")
	rubric_path: Path = Field(..., description="Path to the rubric JSON used for agent perturbations.")

	planner: Optional[AgentLLMStageConfig] = Field(
		None, description="Configuration for drafting a coherent perturbation plan."
	)
	editor: AgentLLMStageConfig = Field(..., description="Configuration for generating single edits.")
	summary: Optional[AgentLLMStageConfig] = Field(
		None, description="Optional configuration for conversation summarisation."
	)
	verifier: Optional[AgentLLMStageConfig] = Field(
		None, description="Optional configuration for instruction verification."
	)

	max_summary_messages: int = Field(20, ge=1, description="Maximum number of turns retained in the running summary.")
	max_edit_rounds: int = Field(3, ge=1, description="Maximum number of edit rounds attempted per run.")
	trace_messages: bool = Field(False, description="Log planner decisions for debugging.")
	sample_num_from_orig: Optional[int] = Field(
		None, ge=1, description="Optional limit on number of Inspect runs to process."
	)
	sampling_seed: int = Field(8234, description="Seed used when sampling Inspect runs.")

	target_rubric_ids: List[str] = Field(
		default_factory=list, description="Subset of rubric instruction IDs to target. Uses all if empty."
	)

	transcript_preprocessors: List[Callable[..., Any]] = Field(
		default_factory=list,
		description=(
			"Optional sequence of callables that can transform each normalized agent run after ingestion. Entries may be dotted import path strings or callables."
		),
		validation_alias=AliasChoices("transcript_preprocessors", "transcript_preprocessor"),
	)

	autograder_template: str = Field(
		"agent_judge",
		description="Name of the autograder prompt template to use when generating evaluation rows.",
	)
	autograder_default_params: Dict[str, str] = Field(
		default_factory=dict,
		description="Template variables to seed the autograder prompt (e.g., lowest/highest score).",
	)
	score_targets: Optional[List[int]] = Field(
		None,
		description="Optional subset of ordinal scores to target when the autograder template expects them.",
	)

	objective: Literal["fail", "pass"] = Field(
		"fail",
		description="Whether to induce failure (negative) or ensure satisfaction (positive) for the rubric.",
	)
	pass_required: bool = Field(
		True,
		description="When objective='pass', require verifier PASS; otherwise skip the item.",
	)

	output: AgentOutputConfig = Field(default_factory=AgentOutputConfig, description="Output settings for artifacts.")

	@field_validator("input_log_path", "rubric_path", mode="before")
	@classmethod
	def _coerce_paths(cls, value: Union[str, Path]) -> Path:
		path = Path(value)
		if not path.exists():
			raise FileNotFoundError(f"Agent configuration path not found: '{path}'")
		return path.resolve()

	@staticmethod
	def _coerce_transcript_preprocessor(value: Any) -> Callable[..., Any]:
		if callable(value):
			return value
		if isinstance(value, str):
			module_path, _, attr = value.rpartition(".")
			if not module_path or not attr:
				raise ValueError(
					"transcript_preprocessors entries must be dotted paths like 'package.module:function'."
				)
			module = importlib.import_module(module_path)
			try:
				target = getattr(module, attr)
			except AttributeError as exc:
				raise ValueError(
					f"transcript_preprocessors target '{attr}' not found in module '{module_path}'."
				) from exc
			if not callable(target):
				raise TypeError(f"transcript_preprocessors target '{module_path}.{attr}' is not callable.")
			return target
		raise TypeError("transcript_preprocessors entries must be callables or dotted import path strings.")

	@field_validator("transcript_preprocessors", mode="before")
	@classmethod
	def _resolve_transcript_preprocessors(cls, value: Any) -> List[Callable[..., Any]]:
		if value in (None, "", []):
			return []
		if isinstance(value, (list, tuple)):
			return [cls._coerce_transcript_preprocessor(item) for item in value]
		return [cls._coerce_transcript_preprocessor(value)]

	@field_validator("score_targets", mode="before")
	@classmethod
	def _coerce_score_targets(cls, value: Any) -> Optional[List[int]]:
		if value in (None, "", []):
			return None
		if isinstance(value, str):
			parts = [part.strip() for part in value.split(",") if part.strip()]
			return [int(part) for part in parts] if parts else None
		if isinstance(value, (list, tuple)):
			return [int(item) for item in value]
		raise TypeError("score_targets must be a sequence of integers or a comma-separated string.")
