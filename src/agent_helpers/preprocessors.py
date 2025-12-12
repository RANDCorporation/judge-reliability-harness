"""Example transcript preprocessors for agent perturbation runs."""

from __future__ import annotations

from typing import Any, Dict, List


def drop_tool_messages(run: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Remove tool/function messages from the transcript before perturbation.

	Returns a dict update so the pipeline keeps the original run object but swaps
	in the filtered transcript and annotates how many turns were removed.
	"""
	messages: List[Dict[str, Any]] = list(run.get("messages", []))
	filtered: List[Dict[str, Any]] = [
		message for message in messages if str(message.get("role", "")).lower() not in {"tool", "function"}
	]

	metadata = dict(run.get("metadata") or {})
	metadata["tool_messages_removed"] = len(messages) - len(filtered)

	return {"messages": filtered, "metadata": metadata}


def keep_assistant_comment_lines(run: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Strip executable code from assistant tool-call payloads, leaving only comment lines.
	Assistant message content is preserved; only `tool_calls[*].arguments.code` is modified.
	"""

	def _extract_comments(text: str) -> List[str]:
		return [line for line in text.splitlines() if line.lstrip().startswith("#")]

	filtered_messages: List[Dict[str, Any]] = []
	comment_counts: List[int] = []

	for message in run.get("messages", []):
		if str(message.get("role", "")).lower() != "assistant":
			filtered_messages.append(message)
			continue

		tool_calls = message.get("tool_calls")
		if not isinstance(tool_calls, list):
			filtered_messages.append(message)
			continue

		new_message = dict(message)
		new_tool_calls: List[Dict[str, Any]] = []

		for tool_call in tool_calls:
			if not isinstance(tool_call, dict):
				new_tool_calls.append(tool_call)
				continue

			new_call = dict(tool_call)
			args = tool_call.get("arguments")
			if isinstance(args, dict):
				code_block = args.get("code")
				if isinstance(code_block, str):
					comment_lines = _extract_comments(code_block)
					comment_counts.append(len(comment_lines))
					new_args = dict(args)
					new_args["code"] = "\n".join(comment_lines)
					new_call["arguments"] = new_args

			new_tool_calls.append(new_call)

		new_message["tool_calls"] = new_tool_calls
		filtered_messages.append(new_message)

	metadata = dict(run.get("metadata") or {})
	if comment_counts:
		metadata["assistant_comment_lines"] = sum(comment_counts)

	return {"messages": filtered_messages, "metadata": metadata}


def strip_tool_call_outputs(run: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Remove tool output messages and strip inline tool call outputs from assistant messages.
	"""
	messages: List[Dict[str, Any]] = []
	output_messages_removed = 0
	output_chunks_removed = 0

	for message in run.get("messages", []):
		role = str(message.get("role", "")).lower()
		if role in {"tool", "function"}:
			output_messages_removed += 1
			continue

		new_message = dict(message)
		tool_calls = message.get("tool_calls")
		if isinstance(tool_calls, list):
			new_tool_calls: List[Dict[str, Any]] = []
			for tool_call in tool_calls:
				if not isinstance(tool_call, dict):
					new_tool_calls.append(tool_call)
					continue

				new_call = dict(tool_call)
				if "outputs" in new_call and new_call["outputs"]:
					output_payload = new_call.pop("outputs")
					if isinstance(output_payload, list):
						output_chunks_removed += len(output_payload)
					else:
						output_chunks_removed += 1
				new_tool_calls.append(new_call)
			new_message["tool_calls"] = new_tool_calls

		messages.append(new_message)

	metadata = dict(run.get("metadata") or {})
	if output_messages_removed:
		metadata["tool_output_messages_removed"] = output_messages_removed
	if output_chunks_removed:
		metadata["tool_call_outputs_stripped"] = output_chunks_removed

	return {"messages": messages, "metadata": metadata}


def drop_reasoning_traces(run: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Remove reasoning content blocks from assistant messages.
	Preserves text content and other content types.
	"""
	messages: List[Dict[str, Any]] = []
	reasoning_removed_count = 0

	for message in run.get("messages", []):
		# We only need to check assistant messages for reasoning blocks
		if str(message.get("role", "")).lower() != "assistant":
			messages.append(message)
			continue

		content = message.get("content")

		# If content is a string or None, there are no structured reasoning blocks to remove
		if not isinstance(content, list):
			messages.append(message)
			continue

		# Filter out blocks where type is 'reasoning'
		filtered_content: List[Dict[str, Any]] = []
		for block in content:
			if isinstance(block, dict) and block.get("type") == "reasoning":
				reasoning_removed_count += 1
			else:
				filtered_content.append(block)

		# Create the updated message object
		new_message = dict(message)
		new_message["content"] = filtered_content
		messages.append(new_message)

	metadata = dict(run.get("metadata") or {})
	if reasoning_removed_count > 0:
		metadata["reasoning_traces_removed"] = reasoning_removed_count

	return {"messages": messages, "metadata": metadata}
