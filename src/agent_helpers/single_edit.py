"""LLM-backed assistant message editing primitives."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import List, Optional, Union

import instructor

from .ingestion import NormalizedAgentRun, NormalizedMessage
from .schemas import EditProposal, EditProposalResponseModel, PlannerSelection
from .text_utils import load_prompt_text, stringify_content

logger = logging.getLogger(__name__)


class SingleEditError(RuntimeError):
	"""Raised when generation or application of a single edit fails."""


def generate_single_edit(
	run: NormalizedAgentRun,
	selection: PlannerSelection,
	rubric_text: str,
	model: str,
	prompt_path: Union[str, Path],
	temperature: float = 0.0,
	system_prompt: Optional[str] = None,
	*,
	rubric_id: Optional[str] = None,
	ordinal_guidance: str = "",
) -> EditProposal:
	"""Request an LLM to propose an edit for the selected assistant message."""
	message = selection.message
	content = message.get("content", "")
	if not isinstance(content, str):
		content = stringify_content(content)

	template = load_prompt_text(prompt_path)
	filled = template.format(
		run_id=run.get("run_id", ""),
		message_id=selection.message_id,
		original_message=content,
		summary=selection.summary_snapshot or "No summary available.",
		rubric=rubric_text or "No rubric provided.",
		ordinal_guidance=ordinal_guidance or "",
	)

	rubric_token = rubric_id or "rubric"
	round_token = selection.round_index + 1

	debug_dump = [
		f"[single-edit-{rubric_token}-{round_token}]",
		f"run_id={run.get('run_id', '')}",
		f"rubric_id={rubric_id or ''}",
		f"model={model}",
		f"prompt_path={prompt_path}",
		f"selection_index={selection.index}",
		f"round_index={selection.round_index}",
		f"message_id={selection.message_id}",
		f"target_score={selection.target_score}",
		filled,
		"",
	]
	debug_path = Path("agent_llm_debug.txt")
	try:
		with debug_path.open("a", encoding="utf-8") as handle:
			handle.write("\n".join(debug_dump) + "\n")
	except OSError:
		logger.debug("Failed to write single edit debug payload to %s", debug_path)

	messages = []
	if system_prompt:
		messages.append({"role": "system", "content": system_prompt})
	messages.append({"role": "user", "content": filled})

	if "/" in model:
		provider, model_name = model.split("/", 1)
	else:
		provider, model_name = "openai", model
	provider_model = f"{provider}/{model_name}"
	client = instructor.from_provider(provider_model)

	try:
		response = client.chat.completions.create(
			model=model_name,
			messages=messages,
			temperature=temperature,
			response_model=EditProposalResponseModel,
		)
	except Exception as exc:
		raise SingleEditError(f"LLM edit generation failed: {exc}") from exc

	raw = response.model_dump_json()
	new_content = response.new_content.strip()
	reasoning = response.explanation

	if not new_content:
		raise SingleEditError("Generated edit content is empty.")

	return EditProposal(
		new_content=new_content,
		reasoning=reasoning,
		raw_response=raw,
	)


def apply_single_edit(
	messages: List[NormalizedMessage],
	selection: PlannerSelection,
	proposal: EditProposal,
) -> List[NormalizedMessage]:
	"""Apply an edit to a copy of the message list, returning the perturbed transcript."""
	if selection.index < 0 or selection.index >= len(messages):
		raise SingleEditError(f"Selection index {selection.index} out of range.")

	perturbed = copy.deepcopy(messages)
	target = perturbed[selection.index]

	if str(target.get("role", "")).lower() != "assistant":
		raise SingleEditError("Selected message is not an assistant turn.")

	target["content"] = proposal.new_content
	return perturbed
