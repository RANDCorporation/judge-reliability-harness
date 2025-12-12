import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agent_helpers.schemas import RubricInstruction
from schemas import AgentLLMStageConfig, TestAgentPerturbationConfig

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "synthetic_data_pipeline" / "agent_perturbation.py"
_SPEC = importlib.util.spec_from_file_location("agent_perturbation_test_module", _MODULE_PATH)
assert _SPEC and _SPEC.loader
ap = importlib.util.module_from_spec(_SPEC)
sys.modules["agent_perturbation_test_module"] = ap
_SPEC.loader.exec_module(ap)


@pytest.fixture
def agent_editor_prompt_path() -> Path:
	root = Path(__file__).resolve().parents[2]
	return root / "prompts" / "templates" / "synthetic" / "agent_single_edit.md"


def _build_agent_config(tmp_path: Path, prompt_path: Path, sample_limit: int) -> TestAgentPerturbationConfig:
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
		sample_num_from_orig=sample_limit,
		sampling_seed=7,
		output={"dir": tmp_path / "agent_out"},
	)


def test_subsample_runs_is_deterministic() -> None:
	runs: List[Dict[str, Any]] = [{"run_id": f"run-{idx}"} for idx in range(6)]

	first = ap._subsample_runs(runs, 2, seed=123)
	second = ap._subsample_runs(runs, 2, seed=123)
	none_requested = ap._subsample_runs(runs, None, seed=123)
	full_requested = ap._subsample_runs(runs, 20, seed=123)
	invalid_requested = ap._subsample_runs(runs, 0, seed=123)

	assert len(first) == 2
	assert first == second  # same seed ⇒ same subset
	assert none_requested == runs
	assert full_requested == runs
	assert invalid_requested == runs


def test_generate_agent_perturbations_respects_sample_limit(
	tmp_path: Path, agent_editor_prompt_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _build_agent_config(tmp_path, agent_editor_prompt_path, sample_limit=2)

	mock_runs = [{"run_id": f"run-{idx}", "messages": []} for idx in range(5)]
	mock_rubric = [RubricInstruction(id="agentharm", instructions="Do the thing")]

	monkeypatch.setattr(ap, "console", MagicMock())
	monkeypatch.setattr(ap, "load_inspect_eval_runs", MagicMock(return_value=mock_runs))
	monkeypatch.setattr(ap, "_load_rubric_instructions", MagicMock(return_value=mock_rubric))
	monkeypatch.setattr(ap, "_build_perturber", MagicMock(return_value=object()))

	call_order: List[str] = []

	class DummyPipeline:
		def __init__(self, *_args, test_name: str = "agent_perturbation", **_kwargs):
			self.test_name = test_name

		def generate_for_rubric(self, _run: Dict[str, Any], run_id: str, _rubric_item: RubricInstruction):
			call_order.append(run_id)
			return [({}, {"test_name": self.test_name, "perturbed_idx": run_id, "validation_score": 1})]

	monkeypatch.setattr(ap, "AgentJudgePipeline", DummyPipeline)

	def _ensure_dir(path: Any) -> Path:
		resolved = Path(path).resolve()
		resolved.mkdir(parents=True, exist_ok=True)
		return resolved

	monkeypatch.setattr(ap, "ensure_output_dir", _ensure_dir)
	monkeypatch.setattr(ap, "write_jsonl", MagicMock())

	result = ap.generate_agent_perturbations(config)

	assert len(call_order) == config.sample_num_from_orig
	assert len(result) == config.sample_num_from_orig
	assert sorted(result["perturbed_idx"]) == sorted(call_order)
