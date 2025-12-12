"""Context-aware planning utilities for multi-edit agent perturbations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import instructor
import truststore

from . import single_edit
from .ingestion import NormalizedAgentRun, NormalizedMessage
from .instruction_verifier import (
	InstructionVerifier,
)
from .planner import PlannerCallable
from .schemas import (
	PerturbationConfig,
	PerturbationOutcome,
	PerturbationStep,
	PlannerPlan,
	PlannerSelection,
	PlannerStepPlan,
	RubricInstruction,
	SummaryResponseModel,
)
from .text_utils import render_transcript, stringify_content

truststore.inject_into_ssl()

logger = logging.getLogger(__name__)


class ConversationSummarizer:
	"""Maintains a concise LLM summary of the evolving conversation."""

	def __init__(
		self,
		max_messages: int,
		model: Optional[str] = None,
		prompt_path: Optional[Path] = None,
		temperature: float = 0.0,
		system_prompt: Optional[str] = None,
	):
		if not prompt_path or not model:
			raise ValueError("ConversationSummarizer requires both a summary model and prompt.")

		self.max_messages = max_messages
		self._turns: List[Tuple[str, str, bool]] = []
		self._summary_cache: str = ""
		self._summary_temperature = temperature
		self._summary_system_prompt = system_prompt

		prompt_path_obj = Path(prompt_path)
		self._summary_prompt = prompt_path_obj.read_text(encoding="utf-8")

		if "/" in model:
			provider, model_name = model.split("/", 1)
		else:
			provider, model_name = "openai", model
		provider_model = f"{provider}/{model_name}"

		self._summary_client = instructor.from_provider(provider_model)
		self._summary_model_name = model_name
		self._dirty = True
		self._base_index = 0

	def snapshot(self, edits: Optional[Sequence[PerturbationStep]] = None) -> str:
		"""Return the latest LLM summary, optionally appending edit notes."""
		base = self._generate_if_needed()
		if not edits:
			return base

		notes = [
			f"- Message {step.index} ({step.message_id}): {step.editor_reason or 'Edited to induce failure.'}"
			for step in edits
		]
		if not notes:
			return base
		return f"{base}\n\nApplied edits so far:\n" + "\n".join(notes)

	def push(self, role: str, content: str, *, edited: bool = False) -> None:
		"""Append a conversation turn and invalidate the cached summary."""
		normalized_role = role.upper() if role else "UNKNOWN"
		text = " ".join(content.strip().split())
		self._turns.append((normalized_role, text, edited))
		if self.max_messages and len(self._turns) > self.max_messages:
			excess = len(self._turns) - self.max_messages
			self._turns = self._turns[-self.max_messages :]
			self._base_index += excess
		self._dirty = True

	def load_transcript(self, messages: Sequence[NormalizedMessage]) -> None:
		"""Seed the summarizer with the entire conversation."""
		turns: List[Tuple[str, str, bool]] = []
		for message in messages:
			role = str(message.get("role", "")).upper() if message else "UNKNOWN"
			content = stringify_content(message.get("content", ""))
			text = " ".join(content.strip().split())
			turns.append((role, text, False))
		if self.max_messages and len(turns) > self.max_messages:
			self._base_index = len(turns) - self.max_messages
			turns = turns[-self.max_messages :]
		else:
			self._base_index = 0
		self._turns = turns
		self._dirty = True

	def update_turn(self, index: int, role: str, content: str, *, edited: bool) -> None:
		"""Update an existing turn to reflect new content."""
		relative = index - self._base_index
		if relative < 0:
			return
		normalized_role = role.upper() if role else "UNKNOWN"
		text = " ".join(content.strip().split())
		if relative >= len(self._turns):
			self._turns.append((normalized_role, text, edited))
			if self.max_messages and len(self._turns) > self.max_messages:
				excess = len(self._turns) - self.max_messages
				self._turns = self._turns[-self.max_messages :]
				self._base_index += excess
		else:
			self._turns[relative] = (normalized_role, text, edited)
		self._dirty = True

	def _generate_if_needed(self) -> str:
		if not self._turns:
			return ""
		if not self._dirty:
			return self._summary_cache
		self._summary_cache = self._request_summary()
		self._dirty = False
		return self._summary_cache

	def _request_summary(self) -> str:
		history_text = self._render_history_for_prompt()

		filled_prompt = self._summary_prompt.format(
			chronological_log=history_text,
		)

		messages = []
		if self._summary_system_prompt:
			messages.append({"role": "system", "content": self._summary_system_prompt})
		messages.append({"role": "user", "content": filled_prompt})

		response = self._summary_client.chat.completions.create(
			model=self._summary_model_name,
			messages=messages,
			temperature=self._summary_temperature,
			response_model=SummaryResponseModel,
		)
		return self._format_structured_summary(response)

	def _render_history_for_prompt(self) -> str:
		relevant_turns = self._turns[-self.max_messages :] if self.max_messages else list(self._turns)
		lines: List[str] = []
		for index, (role, text, edited) in enumerate(relevant_turns, start=1):
			marker = " (edited)" if edited else ""
			lines.append(f"{index}. {role}{marker}: {text}")
		return "\n".join(lines)

	@staticmethod
	def _format_structured_summary(response: SummaryResponseModel) -> str:
		sections: List[str] = []
		overall = response.summary.strip()
		if overall:
			sections.append(f"Overall: {overall}")

		if response.user_intent:
			sections.append(f"User Intent: {response.user_intent.strip()}")

		if response.assistant_progress:
			progress_lines = [f"- {item.strip()}" for item in response.assistant_progress if item.strip()]
			if progress_lines:
				sections.append("Assistant Progress:\n" + "\n".join(progress_lines))

		if response.outstanding_items:
			outstanding_lines = [f"- {item.strip()}" for item in response.outstanding_items if item.strip()]
			if outstanding_lines:
				sections.append("Outstanding Items:\n" + "\n".join(outstanding_lines))

		if response.risks:
			risk_lines = [f"- {item.strip()}" for item in response.risks if item.strip()]
			if risk_lines:
				sections.append("Risks / Blockers:\n" + "\n".join(risk_lines))

		return "\n\n".join(sections).strip()


class ConversationPerturber:
	"""Coordinates planning, editing, and verification of multi-step perturbations."""

	def __init__(
		self,
		config: PerturbationConfig,
		editor_model: str,
		editor_prompt_path: Union[str, Path],
		*,
		editor_temperature: float = 0.0,
		editor_system_prompt: Optional[str] = None,
		verifier: Optional[InstructionVerifier] = None,
		planner: Optional[PlannerCallable] = None,
		progress_callback: Optional[Callable[[str], None]] = None,
	):
		self.config = config
		self._editor_model = editor_model
		self._editor_prompt_path = Path(editor_prompt_path)
		self._editor_temperature = editor_temperature
		self._editor_system_prompt = editor_system_prompt
		self._verifier = verifier
		self._planner = planner
		self._progress = progress_callback

	def perturb(
		self,
		run: NormalizedAgentRun,
		rubric: RubricInstruction,
		*,
		target_score: Optional[int] = None,
		score_descriptor: str = "",
		score_levels_table: str = "",
	) -> PerturbationOutcome:
		"""
		Generate coherent multi-edit perturbations for a single run.

		When optional ordinal parameters are supplied the planner, editor, and verifier
		are guided toward producing a transcript that targets the requested score.
		"""
		messages = [dict(message) for message in run.get("messages", [])]

		if not messages:
			return PerturbationOutcome(
				perturbed_messages=messages,
				steps=[],
				final_summary="",
				failure_confirmed=False,
			)

		summarizer = self._make_summarizer()
		summarizer.load_transcript(messages)

		# Invokes callable planner to produce a plan
		plan = self._draft_plan(
			run,
			rubric,
			messages,
			target_score=target_score,
			score_descriptor=score_descriptor,
			score_levels_table=score_levels_table,
		)
		if not plan:
			self._notify("planner: no plan available; skipping edits")
			try:
				final_summary = summarizer.snapshot([])
			except Exception as exc:
				logger.warning("Summary generation failed while skipping planner run: %s", exc)
				final_summary = ""

			# We choose to skip the run if the plan fails to generate; maybe consider option for retries.
			return PerturbationOutcome(
				perturbed_messages=messages,
				steps=[],
				final_summary=final_summary,
				failure_confirmed=False,
			)

		# Limit the number of steps to the max edit rounds. This is a hard cap on the number of edits that can be applied. Not sure if this additional validation step is necessary.
		plan_steps = plan.steps[: self.config.max_edit_rounds]

		steps: List[PerturbationStep] = []
		failure_confirmed = False
		failure_rationale = ""

		# Beginning of perturbation loop, iterate through planned steps.
		for round_index, planned_step in enumerate(plan_steps):
			self._notify(f"round {round_index + 1}: executing planned edit on message {planned_step.index}")

			selection, selection_reason = self._build_selection_from_plan(
				messages=messages,
				edits=steps,
				planned_step=planned_step,
				round_index=round_index,
				summarizer=summarizer,
				target_score=target_score,
				score_descriptor=score_descriptor,
				score_levels_table=score_levels_table,
			)

			if not selection:
				self._notify(f"round {round_index + 1}: planner step skipped ({selection_reason})")
				continue

			new_step, new_messages = self._generate_and_apply_edit(
				run=run,
				selection=selection,
				messages=messages,
				rubric=rubric,
				steps=steps,
				plan_thesis=plan.thesis,
				planned_step=planned_step,
				target_score=target_score,
				score_descriptor=score_descriptor,
				score_levels_table=score_levels_table,
			)

			if not new_step or not new_messages:
				self._notify(f"round {round_index + 1}: edit generation failed; stopping plan execution")
				break

			messages = new_messages
			steps.append(new_step)
			updated_message = messages[selection.index]
			summarizer.update_turn(
				index=selection.index,
				role=str(updated_message.get("role", "")),
				content=stringify_content(updated_message.get("content", "")),
				edited=True,
			)
			self._notify(f"round {round_index + 1}: applied planned edit to message {new_step.index}")

			if self.config.trace_messages:
				logger.info(
					"[multi-edit] round=%s message_index=%s reason=%s",
					round_index,
					selection.index,
					selection.reason,
				)

		if self._verifier and steps:
			failure_confirmed, failure_rationale = self._verify_perturbation(
				messages,
				steps,
				rubric,
				target_score=target_score,
				score_descriptor=score_descriptor,
				score_levels_table=score_levels_table,
			)
			self._notify(f"verifier result: {'FAIL' if failure_confirmed else 'PASS'} for {rubric.id}")

		final_summary = self._build_summary(messages, steps)

		return PerturbationOutcome(
			perturbed_messages=messages,
			steps=steps,
			final_summary=final_summary,
			failure_confirmed=failure_confirmed,
			failure_rationale=failure_rationale,
			plan_thesis=plan.thesis,
			planned_steps=plan.steps,
		)

	def _draft_plan(
		self,
		run: NormalizedAgentRun,
		rubric: RubricInstruction,
		messages: Sequence[NormalizedMessage],
		*,
		target_score: Optional[int] = None,
		score_descriptor: str = "",
		score_levels_table: str = "",
	) -> Optional[PlannerPlan]:
		if not self._planner:
			return None

		self._notify("planner: drafting candidate plan")
		plan = self._planner(
			run,
			rubric,
			self.config.max_edit_rounds,
			target_score,
			score_descriptor,
			score_levels_table,
		)
		if not plan:
			self._notify("planner: no plan produced")
			return None

		thesis = plan.thesis.strip()
		if not thesis:
			self._notify("planner: plan thesis empty")
			return None

		sanitized_steps = self._sanitize_plan_steps(plan.steps, messages)
		if not sanitized_steps:
			self._notify("planner: no usable steps in plan")
			return None

		self._notify(f"planner: prepared {len(sanitized_steps)} step(s) with thesis '{thesis}'")
		return PlannerPlan(thesis=thesis, steps=sanitized_steps)

	def _sanitize_plan_steps(
		self,
		steps: Sequence[PlannerStepPlan],
		messages: Sequence[NormalizedMessage],
	) -> List[PlannerStepPlan]:
		usable: List[PlannerStepPlan] = []
		seen: set[int] = set()
		for step in steps:
			index = step.index
			if index < 0 or index >= len(messages):
				continue
			if str(messages[index].get("role", "")).lower() != "assistant":
				continue
			if index in seen:
				continue

			message_id = str(messages[index].get("id", index))
			usable.append(
				PlannerStepPlan(
					index=index,
					message_id=message_id,
					goal=step.goal,
					rationale=step.rationale,
				)
			)
			seen.add(index)
		return usable

	def _build_selection_from_plan(
		self,
		messages: Sequence[NormalizedMessage],
		edits: Sequence[PerturbationStep],
		planned_step: PlannerStepPlan,
		round_index: int,
		summarizer: ConversationSummarizer,
		*,
		target_score: Optional[int] = None,
		score_descriptor: str = "",
		score_levels_table: str = "",
	) -> Tuple[Optional[PlannerSelection], str]:
		"""Prepares and validates a planned edit before execution (check
		for out of range indices, editing of wrong role, etc.). Prepares summary snapshot
		context for use in editor LLM stage.

		Takes a single step from the plan and translates it into a
		selection ready for the Editor LLM stage.

		Returns the PlannerSelection object to guide generation and edit.
		"""
		edited_indices = {step.index for step in edits}

		if planned_step.index < 0 or planned_step.index >= len(messages):
			return None, "planner_target_out_of_range"

		message = messages[planned_step.index]
		role = str(message.get("role", "")).lower()
		base_reason = planned_step.goal or "planner_selected"
		if planned_step.rationale:
			base_reason = f"{base_reason} ({planned_step.rationale})"

		if role != "assistant":
			return None, "planner_target_not_assistant"

		if planned_step.index in edited_indices:
			return None, "planner_target_already_edited"

		summary_snapshot = summarizer.snapshot(edits)

		selection = PlannerSelection(
			index=planned_step.index,
			message_id=planned_step.message_id,
			reason=base_reason,
			summary_snapshot=summary_snapshot,
			message=message,
			round_index=round_index,
			target_score=target_score,
			score_descriptor=score_descriptor,
			score_levels_table=score_levels_table,
		)
		return selection, base_reason

	def _generate_and_apply_edit(
		self,
		run: NormalizedAgentRun,
		selection: PlannerSelection,
		messages: List[NormalizedMessage],
		rubric: RubricInstruction,
		steps: List[PerturbationStep],
		plan_thesis: str,
		planned_step: Optional[PlannerStepPlan],
		*,
		target_score: Optional[int] = None,
		score_descriptor: str = "",
		score_levels_table: str = "",
	) -> Tuple[Optional[PerturbationStep], Optional[List[NormalizedMessage]]]:
		"""
		Generate and apply a single edit to the selected message.

		Args:
		    run (NormalizedAgentRun): _description_
		    selection (PlannerSelection): _description_
		    messages (List[NormalizedMessage]): _description_
		    rubric (RubricInstruction): _description_
		    steps (List[PerturbationStep]): _description_
		    plan_thesis (str): _description_
		    planned_step (Optional[PlannerStepPlan]): _description_

		Returns:
		    Tuple[Optional[PerturbationStep], Optional[List[NormalizedMessage]]]: _description_
		"""
		self._notify(f"round {selection.round_index + 1}: requesting edit draft for message {selection.index}")
		try:
			proposal = single_edit.generate_single_edit(
				run=run,
				selection=selection,
				rubric_text=self._format_rubric_prompt(
					rubric,
					steps,
					plan_thesis=plan_thesis,
					planned_goal=planned_step.goal if planned_step else "",
					planned_rationale=planned_step.rationale if planned_step else "",
					target_score=target_score,
					score_descriptor=score_descriptor,
					score_levels_table=score_levels_table,
				),
				model=self._editor_model,
				prompt_path=self._editor_prompt_path,
				temperature=self._editor_temperature,
				system_prompt=self._editor_system_prompt,
				rubric_id=rubric.id,
				ordinal_guidance=self._ordinal_guidance_text(
					target_score=target_score,
					score_descriptor=score_descriptor,
					score_levels_table=score_levels_table,
				),
			)
		except single_edit.SingleEditError as exc:
			logger.warning(
				"Edit generation failed for run %s message %s: %s",
				run.get("run_id"),
				selection.message_id,
				exc,
			)
			return None, None

		original_content = stringify_content(messages[selection.index].get("content", ""))

		perturbed_messages = single_edit.apply_single_edit(messages, selection, proposal)

		step = PerturbationStep(
			index=selection.index,
			message_id=selection.message_id,
			original_content=original_content,
			edited_content=proposal.new_content,
			selector_reason=selection.reason,
			editor_reason=proposal.reasoning,
			editor_raw_response=proposal.raw_response,
			summary_snapshot=selection.summary_snapshot,
			round_index=selection.round_index,
		)
		return step, perturbed_messages

	def _verify_perturbation(
		self,
		messages: List[NormalizedMessage],
		steps: List[PerturbationStep],
		rubric: RubricInstruction,
		*,
		target_score: Optional[int] = None,
		score_descriptor: str = "",
		score_levels_table: str = "",
	) -> Tuple[bool, str]:
		if not self._verifier:
			return False, ""

		self._notify("verifier: requesting confirmation")
		summary_text = self._build_summary(messages, steps)
		transcript_text = render_transcript(messages)

		ordinal_hint = self._ordinal_guidance_text(
			target_score=target_score,
			score_descriptor=score_descriptor,
			score_levels_table=score_levels_table,
		)

		verifier_instruction = rubric.instructions
		if ordinal_hint:
			verifier_instruction = f"{verifier_instruction}\n\nOrdinal target:\n{ordinal_hint}"

		return self._verifier(
			summary_text,
			transcript_text,
			verifier_instruction,
		)

	def _make_summarizer(self) -> ConversationSummarizer:
		return ConversationSummarizer(
			max_messages=self.config.max_summary_messages,
			model=self.config.summary_model,
			prompt_path=self.config.summary_prompt_path,
			temperature=self.config.summary_temperature,
			system_prompt=self.config.summary_system_prompt,
		)

	def _build_summary(
		self,
		messages: Iterable[NormalizedMessage],
		edits: Sequence[PerturbationStep],
	) -> str:
		summarizer = self._make_summarizer()
		edited_indices = {step.index for step in edits}
		for index, message in enumerate(messages):
			summarizer.push(
				role=str(message.get("role", "")),
				content=stringify_content(message.get("content", "")),
				edited=index in edited_indices,
			)
		return summarizer.snapshot(edits)

	def _notify(self, message: str) -> None:
		if not self._progress:
			return
		try:
			self._progress(message)
		except Exception:
			logger.debug("Progress callback failed: %s", message)

	@staticmethod
	def _format_rubric_prompt(
		rubric: RubricInstruction,
		edits: Sequence[PerturbationStep],
		*,
		plan_thesis: str = "",
		planned_goal: str = "",
		planned_rationale: str = "",
		target_score: Optional[int] = None,
		score_descriptor: str = "",
		score_levels_table: str = "",
	) -> str:
		lines = [
			f"Target rubric instruction ({rubric.id}):",
			rubric.instructions.strip(),
		]
		if plan_thesis:
			lines.append("")
			lines.append("Overall failure thesis:")
			lines.append(plan_thesis.strip())
		if planned_goal:
			lines.append("")
			lines.append("Current edit objective:")
			lines.append(planned_goal.strip())
		if planned_rationale:
			lines.append("")
			lines.append("Planner rationale:")
			lines.append(planned_rationale.strip())
		if target_score is not None:
			lines.append("")
			lines.append("Ordinal guidance for this edit:")
			lines.append(f"- Desired score: {target_score}")
			if score_descriptor:
				lines.append(f"- Descriptor: {score_descriptor.strip()}")
			if score_levels_table:
				lines.append("")
				lines.append("Score scale reference:")
				lines.append(score_levels_table.strip())
		if edits:
			lines.append("")
			lines.append("Edits applied so far:")
			for step in edits:
				lines.append(
					f"- Message {step.index} ({step.message_id}): {step.editor_reason or 'Edited to induce failure.'}"
				)
		return "\n".join(lines).strip()

	@staticmethod
	def _ordinal_guidance_text(
		*,
		target_score: Optional[int],
		score_descriptor: str,
		score_levels_table: str,
	) -> str:
		"""Render the ordinal target context used by planner, editor, and verifier prompts."""
		if target_score is None:
			return ""

		lines: List[str] = ["Ordinal target details:"]
		lines.append(f"- Desired score: {target_score}")
		if score_descriptor:
			lines.append(f"- Descriptor: {score_descriptor.strip()}")
		if score_levels_table:
			lines.append("")
			lines.append("Score scale reference:")
			lines.append(score_levels_table.strip())
		return "\n".join(lines).strip()
