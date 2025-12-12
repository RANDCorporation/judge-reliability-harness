from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent_helpers.preprocessors import strip_tool_call_outputs
from schemas import AgentLLMStageConfig, TestAgentPerturbationConfig
from synthetic_data_pipeline.agent_perturbation import _apply_transcript_preprocessor


@pytest.fixture
def agent_editor_prompt_path() -> Path:
	root = Path(__file__).resolve().parents[2]
	return root / "prompts" / "templates" / "synthetic" / "agent_single_edit.md"


def _build_config(tmp_path: Path, prompt_path: Path, preprocessors: List[Any]) -> TestAgentPerturbationConfig:
	log_path = tmp_path / "runs.eval"
	log_path.write_text("[]", encoding="utf-8")
	rubric_path = tmp_path / "rubric.json"
	rubric_path.write_text("[]", encoding="utf-8")

	editor_stage = AgentLLMStageConfig(
		model="openai/gpt-4o-mini",
		prompt_path=prompt_path,
		temperature=0.0,
		system_prompt=None,
	)

	return TestAgentPerturbationConfig(
		input_log_path=log_path,
		rubric_path=rubric_path,
		editor=editor_stage,
		transcript_preprocessors=preprocessors,
	)


def test_transcript_preprocessors_accept_list(tmp_path: Path, agent_editor_prompt_path: Path) -> None:
	config = _build_config(
		tmp_path,
		agent_editor_prompt_path,
		[
			"src.agent_helpers.preprocessors.drop_tool_messages",
			"src.agent_helpers.preprocessors.keep_assistant_comment_lines",
		],
	)

	names = [callable_.__name__ for callable_ in config.transcript_preprocessors]
	assert names == ["drop_tool_messages", "keep_assistant_comment_lines"]


def test_transcript_preprocessors_apply_in_sequence(tmp_path: Path, agent_editor_prompt_path: Path) -> None:
	order: List[str] = []

	def first_preprocessor(run: Dict[str, Any], *, raw_sample: Any = None) -> Dict[str, Any]:
		order.append("first")
		metadata = dict(run.get("metadata") or {})
		metadata["sequence"] = ["first"]
		return {"messages": run.get("messages", []), "metadata": metadata}

	def second_preprocessor(run: Dict[str, Any], *, raw_sample: Any = None) -> Dict[str, Any]:
		order.append("second")
		metadata = dict(run.get("metadata") or {})
		metadata.setdefault("sequence", []).append("second")
		return {"metadata": metadata}

	config = _build_config(
		tmp_path,
		agent_editor_prompt_path,
		[first_preprocessor, second_preprocessor],
	)

	run: Dict[str, Any] = {"messages": [{"role": "assistant", "content": "hello"}]}
	processed = run
	for preprocessor in config.transcript_preprocessors:
		processed = _apply_transcript_preprocessor(processed, preprocessor)

	assert order == ["first", "second"]
	assert processed["metadata"]["sequence"] == ["first", "second"]


def test_strip_tool_call_outputs_removes_tool_messages_and_outputs() -> None:
	run: Dict[str, Any] = {
		"messages": [
			{
				"role": "assistant",
				"content": "Working...",
				"tool_calls": [
					{
						"id": "call-1",
						"function": "test_tool",
						"arguments": {},
						"outputs": [{"content": "secret"}],
					}
				],
			},
			{"role": "tool", "content": "secret", "tool_call_id": "call-1"},
			{"role": "user", "content": "Thanks!"},
		],
		"metadata": {"source": "unit"},
	}

	original_tool_call = run["messages"][0]["tool_calls"][0]
	assert "outputs" in original_tool_call

	result = strip_tool_call_outputs(run)

	assert len(result["messages"]) == 2
	assert all(str(message.get("role", "")).lower() != "tool" for message in result["messages"])
	tool_calls = result["messages"][0]["tool_calls"]
	assert isinstance(tool_calls, list)
	assert "outputs" not in tool_calls[0]
	assert result["metadata"]["source"] == "unit"
	assert result["metadata"]["tool_output_messages_removed"] == 1
	assert result["metadata"]["tool_call_outputs_stripped"] == 1
