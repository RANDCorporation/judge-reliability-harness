"""Utilities for constructing LLM-based instruction verifiers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import instructor

from .schemas import InstructionVerdictResponseModel
from .text_utils import load_prompt_text

logger = logging.getLogger(__name__)

InstructionVerdict = Tuple[bool, str]
InstructionVerifier = Callable[[str, str, str], InstructionVerdict]


def make_llm_instruction_verifier(
	model: str,
	prompt_path: Union[str, Path],
	temperature: float = 0.0,
	system_prompt: Optional[str] = None,
) -> InstructionVerifier:
	"""Construct an instruction verifier that determines whether failure was induced."""

	template = load_prompt_text(prompt_path)
	if "/" in model:
		provider, model_name = model.split("/", 1)
	else:
		provider, model_name = "openai", model
	provider_model = f"{provider}/{model_name}"
	client = instructor.from_provider(provider_model)

	def _verifier(
		summary: str,
		transcript: str,
		rubric_instruction: str,
	) -> InstructionVerdict:
		filled = template.format(
			summary=summary or "Summary unavailable.",
			transcript=transcript,
			rubric_instruction=rubric_instruction,
		)

		messages = []
		if system_prompt:
			messages.append({"role": "system", "content": system_prompt})
		messages.append({"role": "user", "content": filled})

		try:
			response = client.chat.completions.create(
				model=model_name,
				messages=messages,
				temperature=temperature,
				response_model=InstructionVerdictResponseModel,
			)
		except Exception as exc:
			logger.warning("Instruction verifier call failed: %s", exc)
			return False, f"verifier_error: {exc}"

		verdict = response.verdict.upper()
		rationale = response.rationale.strip() or "no rationale provided"
		return verdict == "FAIL", rationale

	return _verifier
