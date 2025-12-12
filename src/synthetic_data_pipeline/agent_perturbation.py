from __future__ import annotations

import copy
import json
import random
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from agent_helpers import (
	ConversationPerturber,
	PerturbationConfig,
	PerturbationOutcome,
	RubricInstruction,
	RubricScoreLevel,
	build_record,
	ensure_output_dir,
	load_inspect_eval_runs,
	make_llm_instruction_verifier,
	make_llm_planner,
	render_transcript,
	write_debug_bundle,
	write_jsonl,
)
from core import AgentLLMStageConfig, SavedItem, TestAgentPerturbationConfig, console


def _subsample_runs(runs: List[Dict[str, Any]], sample_size: Optional[int], seed: int) -> List[Dict[str, Any]]:
	"""Return deterministic subset of runs when sample_size is provided."""
	if not runs or sample_size is None:
		return runs
	if sample_size <= 0 or sample_size >= len(runs):
		return runs
	rng = random.Random(seed)
	selected_indices = sorted(rng.sample(range(len(runs)), sample_size))
	return [runs[idx] for idx in selected_indices]


class BaseAgentPipeline:
	"""Shared utilities for agent perturbation pipelines."""

	def __init__(
		self,
		config: TestAgentPerturbationConfig,
		perturber: ConversationPerturber,
		debug_dir: Path,
		progress_callback: Optional[Callable[[SavedItem], None]] = None,
		test_name: str = "agent_perturbation",
	):
		self.config = config
		self.perturber = perturber
		self.debug_dir = debug_dir
		self.progress_callback = progress_callback
		self.test_name = test_name

	def generate_for_rubric(
		self,
		run: Dict[str, Any],
		run_id: str,
		rubric_item: RubricInstruction,
	) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
		raise NotImplementedError

	def _emit_saved_item(self, saved_item: SavedItem) -> None:
		if not self.progress_callback:
			return
		try:
			self.progress_callback(saved_item)
		except Exception as exc:  # pragma: no cover - defensive logging
			console.print(f"      - [yellow]Warning: failed to stream item {saved_item.perturbed_idx}: {exc}[/yellow]")

	@staticmethod
	def _original_transcript(run: Dict[str, Any]) -> str:
		return render_transcript(run.get("messages", []))

	def _write_debug_bundle(
		self,
		run: Dict[str, Any],
		combination_id: str,
		outcome: PerturbationOutcome,
		perturbed_messages: List[Dict[str, Any]],
		rubric_item: RubricInstruction,
	) -> None:
		write_debug_bundle(
			output_dir=self.debug_dir,
			run_id=combination_id.replace("::", "__"),
			outcome=outcome,
			gold_messages=run.get("messages", []),
			perturbed_messages=perturbed_messages,
			target_rubric=rubric_item,
		)


class AgentJudgePipeline(BaseAgentPipeline):
	"""Generates binary judge perturbations for each rubric."""

	def generate_for_rubric(
		self,
		run: Dict[str, Any],
		run_id: str,
		rubric_item: RubricInstruction,
	) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
		outcome = self.perturber.perturb(run, rubric_item)
		perturbed_messages = outcome.perturbed_messages if outcome.steps else run.get("messages", [])

		objective = getattr(self.config, "objective", "fail")
		pass_required = getattr(self.config, "pass_required", True)

		if not outcome.steps:
			if objective == "pass":
				# Optional verification that the original transcript already passes.
				verified_pass = True
				rationale = ""
				if getattr(self.config, "verifier", None) and pass_required:
					verifier_cfg = self.config.verifier
					try:
						verifier = make_llm_instruction_verifier(
							model=verifier_cfg.model,
							prompt_path=verifier_cfg.prompt_path,
							temperature=verifier_cfg.temperature,
							system_prompt=verifier_cfg.system_prompt,
						)
						original_transcript = self._original_transcript(run)
						fail_flag, rationale = verifier(
							outcome.final_summary or "", original_transcript, rubric_item.instructions
						)
						verified_pass = not fail_flag
					except Exception as exc:  # pragma: no cover - defensive
						console.print(
							f"      - [yellow]Verifier pre-check failed ({exc}); proceeding without verification.[/]"
						)
						verified_pass = True
						rationale = ""

				if pass_required and not verified_pass:
					console.print(
						f"      - Planner produced no edits but verifier flagged FAIL for {run_id}::{rubric_item.id}; skipping."
					)
					return []

				# Emit a pass-through item using the original transcript (no edits required).
				combination_id = f"{run_id}::{rubric_item.id}"
				perturbed_idx = f"agent_{uuid.uuid4().hex[:8]}"
				agent_mode = "positive"
				expected_label = 0

				record = build_record(
					run=run,
					outcome=outcome,
					target_rubric=rubric_item,
					extra_metadata={
						"run_id": run_id,
						"target_rubric_id": rubric_item.id,
						"perturbed_idx": perturbed_idx,
						"agent_mode": agent_mode,
						"no_edits_required": True,
					},
				)

				original_transcript = self._original_transcript(run)
				original_request = f"{rubric_item.id}: {rubric_item.instructions}"
				evaluation_row = {
					"idx": combination_id,
					"perturbed_idx": perturbed_idx,
					"rubric": rubric_item.instructions,
					"rubric_id": rubric_item.id,
					"transcript": original_transcript,
					"response": original_transcript,
					"generation_response": original_transcript,
					"original_request": original_request,
					"original_idx": combination_id,
					"expected": expected_label,
					"validation_score": expected_label,
					"agent_mode": agent_mode,
					"plan_thesis": outcome.plan_thesis,
					"planned_steps": "[]",
					"test_name": self.test_name,
				}

				saved_item = SavedItem(
					test_name=self.test_name,
					original_request=original_request,
					original_response=original_transcript,
					original_idx=combination_id,
					original_expected=0,
					generation_prompt=original_transcript,
					generation_response=original_transcript,
					generation_completion=original_transcript,
					generation_reasoning=outcome.plan_thesis or "",
					perturbed_idx=perturbed_idx,
					generation_temp=None,
					validation_score=expected_label,
					validation_reasoning=rationale or "no_edits_required",
					prompted_bucket=None,
					validated_bucket=None,
					rubric_id=rubric_item.id,
					rubric_text=rubric_item.instructions,
					score_levels_table=None,
					transcript=original_transcript,
					generation_mode=agent_mode,
				)
				self._emit_saved_item(saved_item)
				self._write_debug_bundle(run, combination_id, outcome, run.get("messages", []), rubric_item)
				return [(record, evaluation_row)]

			# Failure objective default: no edits produced; skip
			console.print(f"      - No edits produced for {run_id}::{rubric_item.id}")
			return []

		if objective == "pass" and pass_required and outcome.failure_confirmed:
			console.print(f"      - Edits did not achieve PASS for {run_id}::{rubric_item.id}; skipping.")
			return []

		expected_label = 0 if objective == "pass" else 1
		agent_mode = "positive" if objective == "pass" else "perturbation"

		combination_id = f"{run_id}::{rubric_item.id}"
		perturbed_idx = f"agent_{uuid.uuid4().hex[:8]}"
		console.print(f"      - Generated {len(outcome.steps)} edit(s) for {combination_id}")

		record = build_record(
			run=run,
			outcome=outcome,
			target_rubric=rubric_item,
			extra_metadata={
				"run_id": run_id,
				"target_rubric_id": rubric_item.id,
				"perturbed_idx": perturbed_idx,
				"agent_mode": agent_mode,
			},
		)

		perturbed_transcript = render_transcript(perturbed_messages)
		original_transcript = self._original_transcript(run)
		original_request = f"{rubric_item.id}: {rubric_item.instructions}"
		evaluation_row = {
			"idx": combination_id,
			"perturbed_idx": perturbed_idx,
			"rubric": rubric_item.instructions,
			"rubric_id": rubric_item.id,
			"transcript": perturbed_transcript,
			"response": perturbed_transcript,
			"generation_response": perturbed_transcript,
			"original_request": original_request,
			"original_idx": combination_id,
			"expected": expected_label,
			"validation_score": expected_label,
			"agent_mode": agent_mode,
			"plan_thesis": outcome.plan_thesis,
			"planned_steps": json.dumps(
				[
					{
						"message_index": step.index,
						"message_id": step.message_id,
						"goal": step.goal,
						"rationale": step.rationale,
					}
					for step in outcome.planned_steps
				]
			),
			"test_name": self.test_name,
		}

		saved_item = SavedItem(
			test_name=self.test_name,
			original_request=original_request,
			original_response=original_transcript,
			original_idx=combination_id,
			original_expected=0,
			generation_prompt=original_transcript,
			generation_response=perturbed_transcript,
			generation_completion=perturbed_transcript,
			generation_reasoning=outcome.plan_thesis or "",
			perturbed_idx=perturbed_idx,
			generation_temp=None,
			validation_score=expected_label,
			validation_reasoning=outcome.failure_rationale or "",
			prompted_bucket=None,
			validated_bucket=None,
			rubric_id=rubric_item.id,
			rubric_text=rubric_item.instructions,
			score_levels_table=None,
			transcript=perturbed_transcript,
			generation_mode=agent_mode,
		)
		self._emit_saved_item(saved_item)
		self._write_debug_bundle(run, combination_id, outcome, perturbed_messages, rubric_item)

		return [(record, evaluation_row)]


class AgentAutograderPipeline(BaseAgentPipeline):
	"""Generates ordinal-targeted perturbations for each rubric score."""

	def __init__(
		self,
		config: TestAgentPerturbationConfig,
		perturber: ConversationPerturber,
		debug_dir: Path,
		ordinal_defaults: Dict[str, str],
		progress_callback: Optional[Callable[[SavedItem], None]] = None,
		test_name: str = "agent_perturbation",
	):
		super().__init__(config, perturber, debug_dir, progress_callback, test_name=test_name)
		self.ordinal_defaults = ordinal_defaults

	def generate_for_rubric(
		self,
		run: Dict[str, Any],
		run_id: str,
		rubric_item: RubricInstruction,
	) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
		results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
		score_bounds = _score_bounds(rubric_item, self.ordinal_defaults)
		score_levels_table = _format_score_levels_table(rubric_item.score_levels, score_bounds)
		target_scores = _resolve_score_targets(rubric_item, self.config, score_bounds)

		if not target_scores:
			console.print(f"      - No target scores configured for {rubric_item.id}; skipping.")
			return results

		for target_score in target_scores:
			descriptor = rubric_item.descriptor_for_score(target_score) or ""
			outcome = self.perturber.perturb(
				run,
				rubric_item,
				target_score=target_score,
				score_descriptor=descriptor,
				score_levels_table=score_levels_table,
			)
			perturbed_messages = outcome.perturbed_messages if outcome.steps else run.get("messages", [])

			if not outcome.steps:
				console.print(
					f"      - Planner/editor produced no edits for {run_id}::{rubric_item.id} (score {target_score}); skipping."
				)
				continue

			combination_id = f"{run_id}::{rubric_item.id}::score_{target_score}"
			perturbed_idx = f"agent_{uuid.uuid4().hex[:8]}"
			console.print(
				f"      - Generated {len(outcome.steps)} edit(s) for {combination_id} targeting score {target_score}"
			)

			record = build_record(
				run=run,
				outcome=outcome,
				target_rubric=rubric_item,
				extra_metadata={
					"run_id": run_id,
					"target_rubric_id": rubric_item.id,
					"target_score": int(target_score),
					"score_descriptor": descriptor,
					"perturbed_idx": perturbed_idx,
				},
			)

			perturbed_transcript = render_transcript(perturbed_messages)
			original_transcript = self._original_transcript(run)
			original_request = f"{rubric_item.id}: {rubric_item.instructions}"
			evaluation_row = {
				"idx": combination_id,
				"perturbed_idx": perturbed_idx,
				"rubric": rubric_item.instructions,
				"rubric_id": rubric_item.id,
				"transcript": perturbed_transcript,
				"response": perturbed_transcript,
				"generation_response": perturbed_transcript,
				"original_request": original_request,
				"original_idx": combination_id,
				"expected": int(target_score),
				"validation_score": int(target_score),
				"target_score": int(target_score),
				"score_descriptor": descriptor,
				"score_levels_table": score_levels_table,
				"plan_thesis": outcome.plan_thesis,
				"planned_steps": json.dumps(
					[
						{
							"message_index": step.index,
							"message_id": step.message_id,
							"goal": step.goal,
							"rationale": step.rationale,
						}
						for step in outcome.planned_steps
					]
				),
				"test_name": self.test_name,
			}

			saved_item = SavedItem(
				test_name=self.test_name,
				original_request=original_request,
				original_response=original_transcript,
				original_idx=combination_id,
				original_expected=_parse_int(self.ordinal_defaults.get("lowest_score"), fallback=0),
				generation_prompt=original_transcript,
				generation_response=perturbed_transcript,
				generation_completion=perturbed_transcript,
				generation_reasoning=outcome.plan_thesis or "",
				perturbed_idx=perturbed_idx,
				generation_temp=None,
				validation_score=int(target_score),
				validation_reasoning=outcome.failure_rationale or "",
				prompted_bucket=None,
				validated_bucket=None,
				rubric_id=rubric_item.id,
				rubric_text=rubric_item.instructions,
				score_levels_table=score_levels_table,
				transcript=perturbed_transcript,
			)
			self._emit_saved_item(saved_item)
			console.print(
				f"      - Persisting {combination_id} ({perturbed_idx}) @ score {target_score} → validation_score={saved_item.validation_score}"
			)
			self._write_debug_bundle(run, combination_id, outcome, perturbed_messages, rubric_item)

			results.append((record, evaluation_row))

		return results


def generate_agent_perturbations(
	config: TestAgentPerturbationConfig,
	progress_callback: Optional[Callable[[SavedItem], None]] = None,
	*,
	test_name: str = "agent_perturbation",
) -> pd.DataFrame:
	"""Run agent perturbations and return rows ready for evaluation."""
	runs = load_inspect_eval_runs(config.input_log_path)
	runs = _subsample_runs(runs, getattr(config, "sample_num_from_orig", None), getattr(config, "sampling_seed", 8234))
	if not runs:
		raise ValueError("No runs found in Inspect archive.")

	if config.transcript_preprocessors:
		processed_runs: List[Dict[str, Any]] = []
		for run in runs:
			processed = run
			for preprocessor in config.transcript_preprocessors:
				processed = _apply_transcript_preprocessor(processed, preprocessor)
			processed_runs.append(processed)
		runs = processed_runs

	console.print(f"    - Loaded {len(runs)} agent run(s) from Inspect archive.")

	rubric_items = _load_rubric_instructions(config.rubric_path)
	if not rubric_items:
		raise ValueError(f"No rubric instructions found in {config.rubric_path}")

	target_ids = {rid for rid in config.target_rubric_ids if rid}
	if target_ids:
		rubric_items = [item for item in rubric_items if item.id in target_ids]
		if not rubric_items:
			raise ValueError("Target rubric IDs were provided but none matched the rubric file.")

	ordinal_mode = config.autograder_template == "agent_autograder"
	ordinal_defaults = {k: str(v) for k, v in (config.autograder_default_params or {}).items()}
	mode_suffix = " (ordinal mode)" if ordinal_mode else ""
	console.print(f"    - Autograder template: {config.autograder_template}{mode_suffix}")

	progress_context = {"run": "", "rubric": ""}

	def _progress_logger(message: str) -> None:
		run_id = progress_context["run"]
		rubric_id = progress_context["rubric"]
		context = f"{run_id}" if not rubric_id else f"{run_id}::{rubric_id}"
		console.print(f"      - [{context}] {message}")

	perturber = _build_perturber(
		config,
		progress_callback=_progress_logger,
		enable_verifier=not ordinal_mode,
	)

	output_dir = ensure_output_dir(config.output.dir)
	debug_dir = output_dir / "debug"

	if ordinal_mode:
		pipeline = AgentAutograderPipeline(
			config,
			perturber,
			debug_dir,
			ordinal_defaults,
			progress_callback,
			test_name=test_name,
		)
	else:
		pipeline = AgentJudgePipeline(config, perturber, debug_dir, progress_callback, test_name=test_name)

	records: List[Dict[str, object]] = []
	evaluation_rows: List[Dict[str, object]] = []

	for run_index, run in enumerate(runs, start=1):
		run_identifier_raw = run.get("run_id") or run.get("id") or ""
		run_id = str(run_identifier_raw)
		console.print(f"    - Processing run {run_index}/{len(runs)} (id={run_id})")
		progress_context["run"] = run_id

		for rubric_index, rubric_item in enumerate(rubric_items, start=1):
			progress_context["rubric"] = rubric_item.id
			console.print(f"      - Rubric {rubric_index}/{len(rubric_items)} ({rubric_item.id})")

			for record, evaluation in pipeline.generate_for_rubric(run, run_id, rubric_item):
				records.append(record)
				evaluation_rows.append(evaluation)

	if records and config.output.write_jsonl:
		_records_path = output_dir / "agent_perturbations.jsonl"
		if config.output.overwrite:
			_records_path.unlink(missing_ok=True)
		write_jsonl(records, _records_path)

	if records:
		_records_summary = output_dir / "agent_perturbations_summary.json"
		if config.output.overwrite:
			_records_summary.unlink(missing_ok=True)
		_records_summary.write_text(json.dumps({"count": len(records)}, indent=2), encoding="utf-8")

	return pd.DataFrame(evaluation_rows)


def generate_agent_judge_perturbation(
	config: TestAgentPerturbationConfig,
	progress_callback: Optional[Callable[[SavedItem], None]] = None,
	*,
	test_name: str = "agent_perturbation",
) -> pd.DataFrame:
	"""Backward-compatible entrypoint used by the adapter."""
	return generate_agent_perturbations(config, progress_callback=progress_callback, test_name=test_name)


def _apply_transcript_preprocessor(
	run: Dict[str, Any],
	preprocessor: Callable[..., Any],
) -> Dict[str, Any]:
	"""Apply the optional transcript preprocessor with fallbacks."""

	if "messages" in run and "_raw_messages" not in run:
		run["_raw_messages"] = copy.deepcopy(run.get("messages", []))

	console.print("Applying transcript preprocessor...", preprocessor)
	raw_sample = run.get("_inspect_sample")
	try:
		result = preprocessor(run, raw_sample=raw_sample)
	except TypeError:
		result = preprocessor(run)

	if result is None:
		pass
	elif isinstance(result, dict):
		run.update(result)
	elif isinstance(result, list):
		run["messages"] = result
	else:
		raise ValueError("Transcript preprocessor must return None, a dict of updates, or a list of messages.")

	return run


def _build_perturber(
	config: TestAgentPerturbationConfig,
	progress_callback: Optional[Callable[[str], None]] = None,
	*,
	enable_verifier: bool = True,
) -> ConversationPerturber:
	"""Create the ConversationPerturber with planner/editor/verifier wiring."""
	planner_cfg = config.planner
	editor = config.editor
	summary_cfg = config.summary
	if summary_cfg is None:
		summary_cfg = AgentLLMStageConfig(
			model=editor.model,
			prompt_path=Path("./prompts/templates/synthetic/agent_summary.md").resolve(),
			temperature=editor.temperature,
			system_prompt=editor.system_prompt,
		)
	verifier_cfg = config.verifier if enable_verifier else None

	planner = None
	if planner_cfg:
		planner = make_llm_planner(
			model=planner_cfg.model,
			prompt_path=planner_cfg.prompt_path,
			temperature=planner_cfg.temperature,
			system_prompt=planner_cfg.system_prompt,
		)

	perturb_config = PerturbationConfig(
		max_summary_messages=config.max_summary_messages,
		max_edit_rounds=config.max_edit_rounds,
		trace_messages=config.trace_messages,
		planner_model=planner_cfg.model if planner_cfg else None,
		planner_prompt_path=planner_cfg.prompt_path if planner_cfg else None,
		planner_temperature=planner_cfg.temperature if planner_cfg else 0.0,
		planner_system_prompt=planner_cfg.system_prompt if planner_cfg else None,
		summary_model=summary_cfg.model,
		summary_prompt_path=summary_cfg.prompt_path,
		summary_temperature=summary_cfg.temperature,
		summary_system_prompt=summary_cfg.system_prompt,
		verifier_model=verifier_cfg.model if verifier_cfg else None,
		verifier_prompt_path=verifier_cfg.prompt_path if verifier_cfg else None,
		verifier_temperature=verifier_cfg.temperature if verifier_cfg else 0.0,
		verifier_system_prompt=verifier_cfg.system_prompt if verifier_cfg else None,
	)

	verifier = None
	if verifier_cfg:
		verifier = make_llm_instruction_verifier(
			model=verifier_cfg.model,
			prompt_path=verifier_cfg.prompt_path,
			temperature=verifier_cfg.temperature,
			system_prompt=verifier_cfg.system_prompt,
		)

	return ConversationPerturber(
		config=perturb_config,
		editor_model=editor.model,
		editor_prompt_path=editor.prompt_path,
		editor_temperature=editor.temperature,
		editor_system_prompt=editor.system_prompt,
		verifier=verifier,
		planner=planner,
		progress_callback=progress_callback,
	)


def _load_rubric_instructions(path: Path) -> List[RubricInstruction]:
	"""Load rubric instructions (and optional score levels) from JSON."""
	payload = json.loads(path.read_text(encoding="utf-8"))
	instructions: List[RubricInstruction] = []
	if isinstance(payload, list):
		for entry in payload:
			if not isinstance(entry, dict):
				continue
			entry_id = str(entry.get("id", "")).strip()
			instruction = entry.get("instructions") or entry.get("instruction")
			if entry_id and instruction:
				score_levels = _parse_score_levels(entry_id, entry.get("score_levels", []))
				instructions.append(
					RubricInstruction(id=entry_id, instructions=str(instruction), score_levels=score_levels)
				)
	return instructions


def _parse_score_levels(rubric_id: str, payload: Any) -> List[RubricScoreLevel]:
	"""Parse optional score-level metadata for a rubric entry."""
	if payload in (None, "", []):
		return []

	if not isinstance(payload, list):
		raise ValueError(f"Rubric '{rubric_id}' score_levels must be a list.")

	levels: List[RubricScoreLevel] = []
	for idx, item in enumerate(payload, start=1):
		if not isinstance(item, dict):
			raise ValueError(f"Rubric '{rubric_id}' score_levels[{idx}] must be an object.")

		if "score" not in item or "label" not in item:
			raise ValueError(f"Rubric '{rubric_id}' score_levels[{idx}] requires 'score' and 'label' keys.")

		try:
			score = int(item["score"])
		except (TypeError, ValueError) as exc:
			raise ValueError(f"Rubric '{rubric_id}' score_levels[{idx}].score must be an integer.") from exc

		label = str(item["label"]).strip()
		if not label:
			raise ValueError(f"Rubric '{rubric_id}' score_levels[{idx}].label must be non-empty.")

		levels.append(RubricScoreLevel(score=score, label=label))

	levels.sort(key=lambda level: level.score)
	scores = [level.score for level in levels]

	if len(scores) != len(set(scores)):
		raise ValueError(f"Rubric '{rubric_id}' score_levels must not contain duplicate scores.")

	if len(scores) > 1:
		step_sizes = {b - a for a, b in zip(scores, scores[1:])}
		if step_sizes != {1}:
			raise ValueError(f"Rubric '{rubric_id}' score_levels must form a consecutive integer range (step size 1).")

	return levels


def _resolve_score_targets(
	rubric: RubricInstruction,
	config: TestAgentPerturbationConfig,
	score_bounds: tuple[int, int],
) -> List[int]:
	low, high = score_bounds
	if low > high:
		low, high = high, low

	if config.score_targets:
		targets = sorted(set(int(value) for value in config.score_targets))
		out_of_range = [score for score in targets if score < low or score > high]
		if out_of_range:
			raise ValueError(
				f"Configured score_targets {out_of_range} fall outside rubric range [{low}, {high}] for rubric '{rubric.id}'."
			)
		return targets

	if rubric.score_levels:
		return [level.score for level in rubric.score_levels]

	return list(range(low, high + 1))


def _score_bounds(
	rubric: RubricInstruction,
	autograder_defaults: Dict[str, str],
) -> tuple[int, int]:
	if rubric.score_levels:
		return rubric.score_levels[0].score, rubric.score_levels[-1].score

	lowest = _parse_int(autograder_defaults.get("lowest_score"), fallback=0)
	highest = _parse_int(autograder_defaults.get("highest_score"), fallback=lowest)
	if highest < lowest:
		highest = lowest
	return lowest, highest


def _parse_int(value: Any, *, fallback: int) -> int:
	try:
		return int(value)
	except (TypeError, ValueError):
		return fallback


def _format_score_levels_table(
	score_levels: List[RubricScoreLevel],
	score_bounds: tuple[int, int],
) -> str:
	if score_levels:
		return "\n".join(f"- **{level.score}**: {level.label}" for level in score_levels)

	low, high = score_bounds
	if high < low:
		low, high = high, low

	placeholder = "(no descriptor provided)"
	return "\n".join(f"- **{score}**: {placeholder}" for score in range(low, high + 1))
