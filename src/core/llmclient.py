# src/core/llmclient.py

import inspect
from functools import wraps
from textwrap import dedent
from types import MethodType
from typing import Any, Dict, Optional, Type

import instructor
from pydantic import BaseModel
from ratelimit import limits

from schemas import (
	BasicLLMResponseBool,
	LLMClientConfig,
)

from .constants import VALID_TEMPLATES, console
from .resolve_templates import get_prompt


class LLMClient:
	def __init__(self, config: LLMClientConfig):
		"""
		Initialize the LLMClient.

		Args:
		    config (LLMClientConfig): Configuration for the LLMClient model.
		"""
		self.config = config

		if self.config.model == "anthropic/claude-3-haiku-20240307":
			self.client = instructor.from_provider(
				"anthropic/claude-3-haiku-20240307", mode=instructor.Mode.ANTHROPIC_JSON
			)
		else:
			self.client = instructor.from_provider(self.config.model)

		# Apply rate limiting to call() if needed
		original_call_fn = self.__class__.call
		rate_limited_call_fn = self._build_rate_limited_judge_call(self.config.rate_limit)(original_call_fn)
		self.call = MethodType(rate_limited_call_fn, self)

	def _build_rate_limited_judge_call(self, rate_limit: Optional[Dict]):
		"""
		Builds a rate-limited version of a function based on given parameters.

		Args:
		    rate_limit (dict): Rate limiting config with 'calls' and 'period' in seconds.

		Returns:
		    Callable: Decorated function with rate limiting.
		"""

		if not rate_limit:
			return lambda fn: fn  # No-op if no rate limiting

		calls = rate_limit.get("calls", 60)
		period = rate_limit.get("period", 60)  # In seconds

		def decorator(fn):
			rate_limited_fn = limits(calls=calls, period=period)(fn)

			@wraps(fn)
			def wrapper(*args, **kwargs):
				return rate_limited_fn(*args, **kwargs)

			return wrapper

		return decorator

	def _build_prompt(self, template_vars: Dict[str, Any]) -> Optional[str]:
		"""
		Builds a prompt string by interpolating row data into a prompt template.

		Args:
		    row (dict): A dictionary containing row-specific fields.

		Returns:
		    Optional[str]: A formatted prompt ready for model input.
		"""
		template = self.config.template

		if template not in VALID_TEMPLATES:
			console.print(
				f"[WARNING] Template '{template}' for LLM call not found. Defaulting to 'judge/single_autograder'."
			)
			template = "judge/single_autograder"

		if "/" not in template:
			template = f"judge/{template}"

		# Merge defaults with row-specific values, letting template_vars values win
		full_vars = self.config.default_params.copy()
		full_vars.update(template_vars)
		prompt = get_prompt(template, **full_vars)
		return dedent(prompt)

	def _fill_in_args(
		self,
		user_prompt: str,
		response_schema: Type[BaseModel] = BasicLLMResponseBool,
		temperature: Optional[int] = None,
		fixed_seed: Optional[int] = None,
	) -> Dict[str, Any]:
		"""
		Fills in arguments for LLM call, based on given and config parameters.

		Returns
		-------
		args (Dict[str, Any]): arguments for LLM call
		"""

		# Build messages
		messages = []
		system_prompt = self.config.default_params.get("system_prompt")
		if system_prompt:
			messages.append({"role": "system", "content": system_prompt})
		messages.append({"role": "user", "content": user_prompt})

		# Fill in args for LLM call
		args = {
			"messages": messages,
			"response_model": response_schema,
			"max_retries": self.config.retries,
		}

		# Handle Bedrock models - extract modelId from model name
		# For Bedrock, litellm/instructor requires modelId parameter (not "model")
		if self.config.model and self.config.model.startswith("bedrock/"):
			# For Bedrock, model name format is "bedrock/model-id"
			# Extract the model-id part and pass it as modelId parameter
			model_id = self.config.model.split("/", 1)[1]
			args["modelId"] = model_id
			# Note: Do NOT pass "model" parameter for Bedrock - only modelId is valid

		sig = inspect.signature(self.client.chat.completions.create)
		if "temperature" in sig.parameters:
			temperature = temperature if temperature is not None else self.config.temperature
			args["temperature"] = temperature
		if "max_tokens" in sig.parameters:
			args["max_tokens"] = self.config.max_tokens
		if fixed_seed is not None and "seed" in sig.parameters:
			args["seed"] = fixed_seed

		return args

	def call(
		self,
		template_vars: Optional[Dict[str, Any]] = None,
		response_schema: Type[BaseModel] = BasicLLMResponseBool,
		temperature: Optional[int] = None,
		fixed_seed: Optional[int] = None,
	) -> Type[BaseModel]:
		"""
		Runs a single evaluation using a prompt generated from row data.

		Args:
		    row (dict): Input data used to build the prompt.
		    response_schema (Type[BaseModel]): The response schema to output.
		    temperature (int): Override for model temperature parameter.

		Returns:
		    Type[BaseModel]: Object containing score and reasoning score.
		"""
		if self.config.test_debug_mode:
			return BasicLLMResponseBool(score=0, reasoning="ERROR: Test debug mode is active.")

		# Fill in template
		template_vars = template_vars or {}
		user_prompt = self._build_prompt(template_vars)
		if not user_prompt:
			return BasicLLMResponseBool(score=0, reasoning="ERROR: Failed to get user prompt.")

		# Make and return LLM call
		args = self._fill_in_args(user_prompt, response_schema, temperature, fixed_seed)
		try:
			response = self.client.chat.completions.create(**args)
		except Exception as e:
			console.print(f"Error during LLM call: {e}")
			return BasicLLMResponseBool(score=0, reasoning=f"Error during LLM call: {e}")

		return response_schema.model_validate(response)
