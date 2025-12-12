"""Output utilities for agent perturbation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .ingestion import NormalizedAgentRun
from .multi_edit_perturber import PerturbationOutcome, PerturbationStep, RubricInstruction
from .schemas import PlannerStepPlan


def ensure_output_dir(base_dir: Path) -> Path:
	base_dir.mkdir(parents=True, exist_ok=True)
	return base_dir


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as handle:
		for record in records:
			handle.write(json.dumps(record, ensure_ascii=False))
			handle.write("\n")


def build_record(
	run: NormalizedAgentRun,
	outcome: PerturbationOutcome,
	target_rubric: RubricInstruction,
	*,
	extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""Shape a JSON-serializable record summarizing the perturbation outcome."""

	run_identifier_raw = run.get("run_id") or run.get("id")
	run_identifier = str(run_identifier_raw) if run_identifier_raw is not None else None

	perturbation_payload: Dict[str, Any] = {
		"failure_confirmed": outcome.failure_confirmed,
		"failure_rationale": outcome.failure_rationale,
		"final_summary": outcome.final_summary,
		"plan": _serialize_plan(outcome.plan_thesis, outcome.planned_steps),
		"steps": [_serialize_step(step) for step in outcome.steps],
	}

	record: Dict[str, Any] = {
		"run_id": run_identifier,
		"epoch": run.get("epoch"),
		"input": run.get("input"),
		"target": run.get("target"),
		"metadata": run.get("metadata", {}),
		"gold_messages": run.get("messages", []),
		"perturbed_messages": outcome.perturbed_messages,
		"target_rubric": {
			"id": target_rubric.id,
			"instructions": target_rubric.instructions,
		},
		"perturbation": perturbation_payload,
	}

	if extra_metadata:
		record["metadata"] = {**record["metadata"], **extra_metadata}

	return record


def _serialize_step(step: PerturbationStep) -> Dict[str, Any]:
	return {
		"message_index": step.index,
		"message_id": step.message_id,
		"selector_reason": step.selector_reason,
		"editor_reason": step.editor_reason,
		"editor_raw_response": step.editor_raw_response,
		"summary_snapshot": step.summary_snapshot,
		"round_index": step.round_index,
		"original_content": step.original_content,
		"edited_content": step.edited_content,
	}


def _serialize_plan(thesis: str, steps: List[PlannerStepPlan]) -> Dict[str, Any]:
	return {
		"thesis": thesis,
		"steps": [
			{
				"message_index": step.index,
				"message_id": step.message_id,
				"goal": step.goal,
				"rationale": step.rationale,
			}
			for step in steps
		],
	}
