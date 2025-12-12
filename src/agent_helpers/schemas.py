"""Shared data schemas for the agent perturbation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field

NormalizedMessage = Dict[str, Any]


@dataclass
class RubricInstruction:
	"""Structured representation of a single rubric instruction w/optional ordinal scale."""

	id: str
	instructions: str
	score_levels: List["RubricScoreLevel"] = field(default_factory=list)

	@property
	def has_ordinal_scale(self) -> bool:
		return bool(self.score_levels)

	@property
	def lowest_score(self) -> Optional[int]:
		if not self.score_levels:
			return None
		return self.score_levels[0].score

	@property
	def highest_score(self) -> Optional[int]:
		if not self.score_levels:
			return None
		return self.score_levels[-1].score

	def descriptor_for_score(self, score: int) -> Optional[str]:
		for level in self.score_levels:
			if level.score == score:
				return level.label
		return None

	def to_score_table(self) -> List[Dict[str, str]]:
		"""
		Return a list of dictionaries suitable for templating score guidance.

		Each entry contains stringified `score` and `label` keys to keep template
		rendering straightforward.
		"""
		return [{"score": str(level.score), "label": level.label} for level in self.score_levels]


@dataclass
class RubricScoreLevel:
	"""Ordinal score anchor within a rubric instruction."""

	score: int
	label: str


@dataclass
class PerturbationConfig:
	"""Runtime settings for the multi-edit perturbation engine."""

	max_summary_messages: int = 20
	max_edit_rounds: int = 3
	trace_messages: bool = False
	planner_model: Optional[str] = None
	planner_prompt_path: Optional[Path] = None
	planner_temperature: float = 0.0
	planner_system_prompt: Optional[str] = None
	summary_model: Optional[str] = None
	summary_prompt_path: Optional[Path] = None
	summary_temperature: float = 0.0
	summary_system_prompt: Optional[str] = None
	verifier_model: Optional[str] = None
	verifier_prompt_path: Optional[Path] = None
	verifier_temperature: float = 0.0
	verifier_system_prompt: Optional[str] = None


@dataclass
class EditProposal:
	"""Structured representation of the LLM's proposed edit."""

	new_content: str
	reasoning: str
	raw_response: str


@dataclass
class PerturbationStep:
	"""Record of an applied edit within a perturbation run."""

	index: int
	message_id: str
	original_content: str
	edited_content: str
	selector_reason: str
	editor_reason: str
	editor_raw_response: str
	summary_snapshot: str
	round_index: int


@dataclass
class PerturbationOutcome:
	"""Final result of a multi-edit perturbation pass."""

	perturbed_messages: List[NormalizedMessage]
	steps: List[PerturbationStep]
	final_summary: str
	failure_confirmed: bool
	failure_rationale: str = ""
	plan_thesis: str = ""
	planned_steps: List["PlannerStepPlan"] = field(default_factory=list)


@dataclass
class PlannerSelection:
	"""Planner decision describing which assistant message should be edited."""

	index: int
	message_id: str
	reason: str
	summary_snapshot: str
	message: NormalizedMessage = field(repr=False)
	round_index: int = 0
	target_score: Optional[int] = None
	score_descriptor: str = ""
	score_levels_table: str = ""


@dataclass
class PlannerStepPlan:
	"""Structured target describing a single planned edit."""

	index: int
	message_id: str
	goal: str
	rationale: str


@dataclass
class PlannerPlan:
	"""High-level plan describing how to induce rubric failure."""

	thesis: str
	steps: List[PlannerStepPlan]


class PlannerStepResponseModel(BaseModel):
	"""Structured planner step returned by the LLM."""

	message_index: int
	goal: str
	rationale: str = ""


class PlannerResponseModel(BaseModel):
	"""Planner response encapsulating failure thesis plus ordered steps."""

	thesis: str = Field("", description="High-level description of the induced failure.")
	steps: List[PlannerStepResponseModel] = Field(default_factory=list)


class EditProposalResponseModel(BaseModel):
	"""LLM payload describing a single edit proposal."""

	new_content: str
	reason: str = ""

	@property
	def explanation(self) -> str:
		return self.reason.strip()


class SummaryResponseModel(BaseModel):
	"""Structured summary of the conversation for progress tracking."""

	summary: str
	user_intent: str = ""
	assistant_progress: List[str] = []
	outstanding_items: List[str] = []
	risks: List[str] = []


class InstructionVerdictResponseModel(BaseModel):
	"""LLM evaluation indicating whether the rubric failure succeeded."""

	verdict: Literal["PASS", "FAIL"]
	rationale: str = ""
