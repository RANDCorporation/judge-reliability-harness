"""LLM-backed planning utilities for coherent agent perturbations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import instructor
from pydantic import ValidationError

from .ingestion import NormalizedAgentRun, NormalizedMessage
from .schemas import PlannerPlan, PlannerResponseModel, PlannerStepPlan, RubricInstruction
from .text_utils import load_prompt_text, stringify_content

logger = logging.getLogger(__name__)


class PlannerError(RuntimeError):
	"""Raised when planner execution fails."""


PlannerCallable = Callable[
	[
		NormalizedAgentRun,
		RubricInstruction,
		int,
		Optional[int],
		str,
		str,
	],
	Optional[PlannerPlan],
]


def make_llm_planner(
	model: str,
	prompt_path: Union[str, Path],
	temperature: float = 0.0,
	system_prompt: Optional[str] = None,
) -> PlannerCallable:
	"""
	Construct a planner callable that emits a coherent multi-edit plan.

	The planner sees the full transcript and rubric instructions, then returns a
	high-level failure thesis accompanied by ordered edit targets.
	"""
	template = load_prompt_text(prompt_path)
	if "/" in model:
		provider, model_name = model.split("/", 1)
	else:
		provider, model_name = "openai", model
	provider_model = f"{provider}/{model_name}"
	client = instructor.from_provider(provider_model)

	def _planner(
		run: NormalizedAgentRun,
		rubric: RubricInstruction,
		max_steps: int,
		target_score: Optional[int] = None,
		score_descriptor: str = "",
		score_levels_table: str = "",
	) -> Optional[PlannerPlan]:
		messages = list(run.get("messages", []))
		if not _has_assistant_turn(messages):
			return None

		assistant_transcript = _render_assistant_transcript(messages)
		ordinal_guidance = _render_ordinal_guidance(
			target_score=target_score,
			score_descriptor=score_descriptor,
			score_levels_table=score_levels_table,
		)
		filled_prompt = template.format(
			rubric_id=rubric.id,
			rubric_instructions=rubric.instructions,
			assistant_transcript=assistant_transcript,
			max_steps=max_steps,
			ordinal_guidance=ordinal_guidance,
		)

		rubric_token = rubric.id or "rubric"
		max_token = max_steps
		debug_dump = [
			f"[planner-{rubric_token}-{max_token}]",
			f"run_id={run.get('run_id', '')}",
			f"model={model}",
			f"prompt_path={prompt_path}",
			f"rubric_id={rubric.id}",
			f"max_steps={max_steps}",
			f"target_score={target_score}",
			filled_prompt,
			"",
		]
		debug_path = Path("agent_llm_debug.txt")
		try:
			with debug_path.open("a", encoding="utf-8") as handle:
				handle.write("\n".join(debug_dump) + "\n")
		except OSError:
			logger.debug("Failed to write planner debug payload to %s", debug_path)

		request_messages = []
		if system_prompt:
			request_messages.append({"role": "system", "content": system_prompt})
		request_messages.append({"role": "user", "content": filled_prompt})

		try:
			response = client.chat.completions.create(
				model=model_name,
				messages=request_messages,
				temperature=temperature,
				response_model=PlannerResponseModel,
			)
		except Exception as exc:
			logger.warning("Planner LLM call failed: %s", exc)
			return None

		try:
			plan = _build_plan_from_response(response, messages, max_steps)
		except PlannerError as exc:
			logger.info("Planner response discarded: %s", exc)
			return None
		except ValidationError as exc:
			logger.info("Planner response validation failed: %s", exc)
			return None

		return plan

	return _planner


def _build_plan_from_response(
	response: PlannerResponseModel,
	messages: Sequence[NormalizedMessage],
	max_steps: int,
) -> PlannerPlan:
	"""
	Translate structured LLM response into a validated `PlannerPlan` object for downstream use.

	This is a helper function that enforces several core features of the planner response (thesis,
	max_steps, and step validity). Also serves to filter out invalid steps so downstream code only
	sees edits that refer to assistant turns. Invalid entries will be skipped rather than causing
	errors, and an empty step list will trigger the`PlannerError`.
	"""
	thesis = response.thesis.strip()
	if not thesis:
		raise PlannerError("Planner response did not include a thesis.")

	if max_steps <= 0:
		raise PlannerError("Planner requested with non-positive max_steps.")

	steps: List[PlannerStepPlan] = []
	seen_indices: set[int] = set()

	for item in response.steps:
		if len(steps) >= max_steps:
			break
		index = int(item.message_index)
		if index < 0 or index >= len(messages):
			continue
		if index in seen_indices:
			continue
		message = messages[index]
		if str(message.get("role", "")).lower() != "assistant":
			continue

		goal = item.goal.strip()
		rationale = item.rationale.strip()
		message_id = str(message.get("id", index))

		steps.append(
			PlannerStepPlan(
				index=index,
				message_id=message_id,
				goal=goal or "Induce rubric failure via this turn.",
				rationale=rationale,
			)
		)
		seen_indices.add(index)

	if not steps:
		raise PlannerError("Planner response did not yield any valid steps.")

	return PlannerPlan(thesis=thesis, steps=steps)


def _render_assistant_transcript(messages: Sequence[NormalizedMessage]) -> str:
	lines: List[str] = []
	for idx, message in enumerate(messages):
		if str(message.get("role", "")).lower() != "assistant":
			continue
		message_id = str(message.get("id", idx))
		content = stringify_content(message.get("content", ""))
		lines.append(f"[{idx}] {message_id}: {content}" if message_id else f"[{idx}] {content}")
	return "\n".join(lines)


def _has_assistant_turn(messages: Sequence[NormalizedMessage]) -> bool:
	return any(str(message.get("role", "")).lower() == "assistant" for message in messages)


def _render_ordinal_guidance(
	*,
	target_score: Optional[int],
	score_descriptor: str,
	score_levels_table: str,
) -> str:
	"""Format ordinal targeting cues for injection into the planner prompt."""
	if target_score is None:
		return ""

	lines: List[str] = []
	lines.append("Ordinal Target Details:")
	lines.append(f"- Desired score: {target_score}")
	if score_descriptor:
		lines.append(f"- Descriptor: {score_descriptor.strip()}")
	if score_levels_table:
		lines.append("")
		lines.append("Score Scale Reference:")
		lines.append(score_levels_table.strip())
	return "\n".join(lines).strip()
