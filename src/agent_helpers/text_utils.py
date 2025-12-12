"""Shared text utilities for agent perturbation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Union

try:
	from .ingestion import NormalizedMessage
except ImportError:
	NormalizedMessage = Mapping[str, object]

_MAX_TOOL_ARG_PREVIEW = 200


def load_prompt_text(path: Union[str, Path]) -> str:
	"""Read a prompt template from disk, ensuring the file exists."""
	resolved = Path(path)
	if not resolved.is_file():
		raise FileNotFoundError(f"Prompt not found at {resolved}")
	return resolved.read_text(encoding="utf-8").strip()


def stringify_content(content: object) -> str:
	"""
	Normalize Inspect message `content` fields into plain text.

	Inspect exports may encode assistant responses as:
	- raw strings,
	- lists of rich message segments (dictionaries with `text` or `content`),
	- arbitrary objects such as numbers or nested lists.

	This helper collapses those heterogeneous structures to a newline-delimited
	string so downstream prompts can consume consistent text. Unknown objects
	are coerced via `str()`, and `None` yields the empty string.
	"""
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		parts: List[str] = []
		for item in content:
			if isinstance(item, dict):
				value = item.get("text") or item.get("content")
				parts.append(str(value) if value is not None else str(item))
			else:
				parts.append(str(item))
		return "\n".join(parts)
	return "" if content is None else str(content)


def render_transcript(messages: Iterable[NormalizedMessage]) -> str:
	"""Format a sequence of Inspect messages as numbered transcript lines."""
	lines: List[str] = []
	for index, message in enumerate(messages, start=1):
		if isinstance(message, Mapping):
			role = str(message.get("role", "")).upper()
			header = _format_message_header(index, role, message)
			content_parts = _extract_message_segments(message)
		else:
			role = ""
			header = f"{index:03d}. {role}"
			content_parts = []

		content = "\n".join(part for part in content_parts if part)
		lines.append(f"{header}: {content}" if content else header)
	return "\n".join(lines)


def _extract_message_segments(message: Mapping[str, Any]) -> List[str]:
	segments: List[str] = []
	content = stringify_content(message.get("content"))
	if content:
		segments.append(content)

	tool_calls = message.get("tool_calls")
	if isinstance(tool_calls, list):
		for tool_call in tool_calls:
			if isinstance(tool_call, Mapping):
				formatted = _format_tool_call(tool_call)
				if formatted:
					segments.append(formatted)

	function_call = message.get("function_call")
	if isinstance(function_call, Mapping):
		formatted = _format_function_call(function_call)
		if formatted:
			segments.append(formatted)

	return segments


def _format_message_header(index: int, role: str, message: Mapping[str, Any]) -> str:
	header = f"{index:03d}. {role}"
	lowered = role.lower()

	if lowered in {"tool", "function"}:
		annotations: List[str] = []
		call_id = message.get("tool_call_id") or message.get("id")
		if call_id:
			annotations.append(f"id={call_id}")
		tool_name = message.get("name") or message.get("tool_name")
		function_info = message.get("function")
		if isinstance(function_info, Mapping) and not tool_name:
			tool_name = function_info.get("name")
		if tool_name:
			annotations.append(f"name={tool_name}")
		if annotations:
			header = f"{header} ({', '.join(annotations)})"

	return header


def _format_tool_call(tool_call: Mapping[str, Any]) -> str:
	call_id = tool_call.get("id")
	call_type = tool_call.get("type") or ""

	function_payload = tool_call.get("function")
	if isinstance(function_payload, str):
		tool_name = function_payload
		arguments = tool_call.get("arguments")
	elif isinstance(function_payload, Mapping):
		tool_name = function_payload.get("name") or tool_call.get("name")
		arguments = function_payload.get("arguments")
	else:
		tool_name = tool_call.get("name")
		arguments = tool_call.get("arguments")

	parts: List[str] = []
	if call_id:
		parts.append(f"id={call_id}")
	if tool_name:
		parts.append(f"name={tool_name}")
	if call_type and call_type != "function":
		parts.append(f"type={call_type}")

	arguments_preview = _preview_arguments(arguments)
	if arguments_preview:
		parts.append(f"args={arguments_preview}")

	return "[Tool call] " + ", ".join(parts) if parts else ""


def _format_function_call(function_call: Mapping[str, Any]) -> str:
	tool_name = function_call.get("name")
	arguments = function_call.get("arguments")

	parts: List[str] = []
	if tool_name:
		parts.append(f"name={tool_name}")

	arguments_preview = _preview_arguments(arguments)
	if arguments_preview:
		parts.append(f"args={arguments_preview}")

	return "[Function call] " + ", ".join(parts) if parts else ""


def _preview_arguments(arguments: Any) -> str:
	if arguments is None:
		return ""

	if isinstance(arguments, str):
		text = " ".join(arguments.split())
	else:
		try:
			text = json.dumps(arguments, separators=(",", ":"))
		except TypeError:
			text = str(arguments)

	if len(text) > _MAX_TOOL_ARG_PREVIEW:
		return text[:_MAX_TOOL_ARG_PREVIEW].rstrip() + "..."
	return text
