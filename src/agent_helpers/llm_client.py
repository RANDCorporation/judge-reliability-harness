"""Shared utilities for structured LLM calls across providers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Type

import instructor
from pydantic import BaseModel


def _split_provider_model(model: str) -> Tuple[str, str]:
	if "/" in model:
		provider, model_name = model.split("/", 1)
	else:
		provider, model_name = "openai", model
	return provider, model_name


@lru_cache(maxsize=None)
def _get_client(provider_model: str):
	return instructor.from_provider(provider_model)


def invoke_structured_chat(
	model: str,
	messages: List[Dict[str, Any]],
	response_model: Type[BaseModel],
	*,
	temperature: float = 0.0,
	extra_args: Optional[Dict[str, Any]] = None,
):
	provider, model_name = _split_provider_model(model)
	client = _get_client(f"{provider}/{model_name}")

	payload: Dict[str, Any] = {
		"model": model_name,
		"messages": messages,
		"response_model": response_model,
	}

	if extra_args:
		payload.update(extra_args)

	if provider == "google":
		payload["generation_config"] = {"temperature": temperature}
	else:
		payload["temperature"] = temperature

	return client.chat.completions.create(**payload)
