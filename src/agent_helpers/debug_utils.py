"""Debug utilities for inspecting gold vs perturbed agent messages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .multi_edit_perturber import PerturbationOutcome, RubricInstruction
from .text_utils import render_transcript


def write_debug_bundle(
	output_dir: Path,
	run_id: str,
	outcome: PerturbationOutcome,
	gold_messages: List[dict],
	perturbed_messages: List[dict],
	target_rubric: RubricInstruction,
) -> None:
	"""Persist a detailed debug bundle summarizing applied edits."""
	if not outcome.steps:
		return

	output_dir.mkdir(parents=True, exist_ok=True)
	audit_entries: List[dict] = []

	for edit_number, step in enumerate(outcome.steps, start=1):
		original_content = str(gold_messages[step.index].get("content", ""))
		edited_content = str(perturbed_messages[step.index].get("content", ""))
		audit_entries.append(
			{
				"edit_number": edit_number,
				"message_index": step.index,
				"message_id": step.message_id,
				"selector_reason": step.selector_reason,
				"editor_reason": step.editor_reason,
				"original_content": original_content,
				"edited_content": edited_content,
				"summary_snapshot": step.summary_snapshot,
				"editor_raw_response": step.editor_raw_response,
			}
		)

	if audit_entries:
		plan_payload = {
			"thesis": outcome.plan_thesis,
			"steps": [
				{
					"message_index": planned_step.index,
					"message_id": planned_step.message_id,
					"goal": planned_step.goal,
					"rationale": planned_step.rationale,
				}
				for planned_step in outcome.planned_steps
			],
		}
		audit_path = output_dir / f"{run_id}_edits_audit.json"
		payload = {
			"run_id": run_id,
			"target_rubric": {
				"id": target_rubric.id,
				"instructions": target_rubric.instructions,
			},
			"failure_confirmed": outcome.failure_confirmed,
			"failure_rationale": outcome.failure_rationale,
			"final_summary": outcome.final_summary,
			"plan": plan_payload,
			"edits": audit_entries,
		}
		audit_path.write_text(
			json.dumps(payload, indent=2),
			encoding="utf-8",
		)

	_transcripts_dir = output_dir / "transcripts"
	_transcripts_dir.mkdir(parents=True, exist_ok=True)
	original_path = _transcripts_dir / f"{run_id}_original.txt"
	perturbed_path = _transcripts_dir / f"{run_id}_perturbed.txt"
	original_path.write_text(render_transcript(gold_messages), encoding="utf-8")
	perturbed_path.write_text(render_transcript(perturbed_messages), encoding="utf-8")
