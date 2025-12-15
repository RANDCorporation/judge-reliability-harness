"""
TEMPORARY script to evaluate LLM juries on an XLSX with rubric/transcript and true_label.
The script will be worked into the main harness soon, as time permits.
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import instructor
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError

try:
	import anthropic
except ImportError:
	anthropic = None

try:
	from google import genai as google_genai
	from google.genai import types as google_genai_types
except ImportError:
	google_genai = None
	google_genai_types = None

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

# Common judge-capable models across major providers.
# You can override via --models on the CLI.
COMMON_JUDGE_MODELS: List[str] = [
	"anthropic/claude-sonnet-4-5",
	"anthropic/claude-haiku-4-5",
	"anthropic/claude-opus-4-1",
	"google/gemini-2.5-flash",
	"google/gemini-2.5-flash-lite",
	"google/gemini-2.5-pro",
	"openai/gpt-5.1",
	"openai/gpt-5",
	"openai/gpt-5-mini",
	"openai/gpt-5-nano",
	"openai/gpt-4.1",
	"openai/gpt-4.1-mini",
	"openai/gpt-4o",
	"openai/gpt-4o-mini",
]

_GOOGLE_SCHEMA = None
_GOOGLE_SAFETY_SETTINGS = None
_ANTHROPIC_RESPONSE_INSTRUCTIONS = 'Return ONLY a JSON object with keys \'label\' (0 or 1) and \'rationale\' (string). Example: {"label": 0, "rationale": "Concise explanation"}.'


class JudgeResponseModel(BaseModel):
	"""
	The verdict of the judge on whether the transcript violates the rubric.
	"""

	label: int = Field(..., ge=0, le=1, description="0 (no violation) or 1 (violation)")
	rationale: str = ""


@dataclass
class JudgeCallError(Exception):
	message: str


def make_client(model: str):
	"""
	Return an instructor client and provider-local model name.
	Accepts strings like:
	- 'openai/gpt-4o-mini'
	- 'anthropic/claude-3-haiku-20240307'
	- 'gpt-4o-mini' (assumes openai)
	"""
	if "/" in model:
		provider, model_name = model.split("/", 1)
	else:
		provider, model_name = "openai", model
	if provider == "google":
		if google_genai is None or google_genai_types is None:
			raise RuntimeError("google-genai>=1.50.1 is required for Google Gemini models.")
		client = google_genai.Client()
	elif provider == "anthropic":
		if anthropic is None:
			raise RuntimeError("anthropic>=0.71.0 is required for Anthropic models.")
		raw_client = anthropic.Anthropic()
		client = instructor.from_anthropic(raw_client)
	else:
		provider_model = f"{provider}/{model_name}"
		client = instructor.from_provider(provider_model)
	return client, provider, model_name


def call_judge_model(
	client,
	provider: str,
	model_name: str,
	prompt_text: str,
	temperature: float = 0.0,
) -> Optional[int]:
	"""
	Invoke an LLM judge to return a binary label via structured extraction.
	Returns 0 or 1. On failure, returns None.
	"""
	if provider == "google":
		return _call_google_judge_model(client, model_name, prompt_text, temperature)

	messages: List[Dict[str, str]] = [{"role": "user", "content": prompt_text}]

	payload = {
		"model": model_name,
		"messages": messages,
		"response_model": JudgeResponseModel,
		"temperature": temperature,
		"max_retries": 2,  # Instructor will auto-retry validation errors twice
	}

	if provider == "anthropic":
		payload["max_tokens"] = 1024

	logger.debug(
		"Calling judge model provider=%s model=%s temperature=%s prompt_chars=%s",
		provider,
		model_name,
		temperature,
		len(prompt_text),
	)

	try:
		response: JudgeResponseModel = client.chat.completions.create(**payload)

	except Exception as exc:
		if _should_retry_without_temperature(provider, exc, payload):
			logger.info(
				"Retrying provider=%s model=%s without explicit temperature due to API constraints.",
				provider,
				model_name,
			)
			payload.pop("temperature", None)
			try:
				response = client.chat.completions.create(**payload)
			except Exception as exc_retry:
				logger.exception(
					"Judge model retry failed provider=%s model=%s: %s",
					provider,
					model_name,
					exc_retry,
				)
				return None
		else:
			logger.exception(
				"Judge model call failed provider=%s model=%s: %s",
				provider,
				model_name,
				exc,
			)
			return None

	label = int(response.label)

	logger.debug(
		"Judge model provider=%s model=%s returned label=%s rationale_len=%s",
		provider,
		model_name,
		label,
		len(response.rationale or ""),
	)

	return 1 if label >= 1 else 0


def _should_retry_without_temperature(provider: str, exc: Exception, payload: Dict[str, object]) -> bool:
	if "temperature" not in payload:
		return False
	if provider not in {"openai"}:
		return False
	text = str(exc).lower()
	return "temperature" in text and ("unsupported" in text or "does not support" in text)


def _google_response_schema():
	global _GOOGLE_SCHEMA
	if google_genai_types is None:
		raise RuntimeError("google-genai types are unavailable.")
	if _GOOGLE_SCHEMA is None:
		_GOOGLE_SCHEMA = google_genai_types.Schema(
			type=google_genai_types.Type.OBJECT,
			required=["label", "rationale"],
			properties={
				"label": google_genai_types.Schema(
					type=google_genai_types.Type.INTEGER,
					description="0 if rubric not violated, else 1.",
				),
				"rationale": google_genai_types.Schema(
					type=google_genai_types.Type.STRING,
					description="Short explanation for the judgment.",
				),
			},
		)
	return _GOOGLE_SCHEMA


def _google_safety_settings():
	global _GOOGLE_SAFETY_SETTINGS
	if google_genai_types is None:
		raise RuntimeError("google-genai types are unavailable.")
	if _GOOGLE_SAFETY_SETTINGS is None:
		threshold = google_genai_types.HarmBlockThreshold.BLOCK_NONE
		_GOOGLE_SAFETY_SETTINGS = [
			google_genai_types.SafetySetting(
				category=google_genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
				threshold=threshold,
			),
			google_genai_types.SafetySetting(
				category=google_genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
				threshold=threshold,
			),
			google_genai_types.SafetySetting(
				category=google_genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
				threshold=threshold,
			),
			google_genai_types.SafetySetting(
				category=google_genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
				threshold=threshold,
			),
		]
	return _GOOGLE_SAFETY_SETTINGS


def _call_google_judge_model(
	client,
	model_name: str,
	prompt_text: str,
	temperature: float,
) -> Optional[int]:
	if google_genai is None or google_genai_types is None:
		logger.error("google-genai is not installed but a Google model was requested.")
		return None
	logger.debug(
		"Calling judge model provider=google model=%s temperature=%s prompt_chars=%s",
		model_name,
		temperature,
		len(prompt_text),
	)
	config = google_genai_types.GenerateContentConfig(
		temperature=temperature,
		response_mime_type="application/json",
		response_schema=_google_response_schema(),
		safety_settings=_google_safety_settings(),
	)
	contents = [
		google_genai_types.Content(
			role="user",
			parts=[google_genai_types.Part(text=prompt_text)],
		)
	]
	try:
		response = client.models.generate_content(
			model=model_name,
			contents=contents,
			config=config,
		)
	except Exception as exc:
		logger.exception("Judge model call failed provider=google model=%s: %s", model_name, exc)
		return None

	parsed_payload = getattr(response, "parsed", None)
	if parsed_payload is None:
		response_text = getattr(response, "text", None) or ""
		try:
			parsed_payload = json.loads(response_text)
		except Exception:
			logger.warning("Gemini response was not structured JSON: %s", response_text[:200])
			return None

	if isinstance(parsed_payload, dict):
		payload = parsed_payload
	elif hasattr(parsed_payload, "model_dump"):
		payload = parsed_payload.model_dump()
	else:
		payload = {
			"label": getattr(parsed_payload, "label", 0),
			"rationale": getattr(parsed_payload, "rationale", ""),
		}

	try:
		judge_response = JudgeResponseModel.model_validate(payload)
	except ValidationError as exc:
		logger.warning("Gemini response validation failed: %s", exc)
		return None

	logger.debug(
		"Judge model provider=google model=%s returned label=%s rationale_len=%s",
		model_name,
		int(judge_response.label),
		len(judge_response.rationale or ""),
	)
	return 1 if int(judge_response.label) >= 1 else 0


def majority_vote(labels: Sequence[int]) -> int:
	"""
	Compute majority vote for a sequence of binary labels.
	Ties resolve to 0 by default (conservative per rubric guidance).
	"""
	if not labels:
		return 0
	total = int(np.sum(labels))
	half = len(labels) / 2.0
	return 1 if total > half else 0


def compute_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int]) -> np.ndarray:
	"""
	Compute a binary 2x2 confusion matrix in order [[TN, FP],[FN, TP]]
	assuming labels are 0/1.
	"""
	matrix = np.zeros((2, 2), dtype=int)
	for t, p in zip(y_true, y_pred):
		t_i = 1 if int(t) == 1 else 0
		p_i = 1 if int(p) == 1 else 0
		matrix[t_i, p_i] += 1
	# Reorder to [[TN, FP],[FN, TP]]
	tn = matrix[0, 0]
	fp = matrix[0, 1]
	fn = matrix[1, 0]
	tp = matrix[1, 1]
	return np.array([[tn, fp], [fn, tp]], dtype=int)


def _parse_true_label(value: object) -> int:
	"""
	Map common encodings to binary labels:
	- 'C'/'correct' -> 0
	- 'I'/'incorrect' -> 1
	- 'pass' -> 0, 'fail' -> 1
	- numeric/text '0'/'1' supported
	Default fallback: 0
	"""
	try:
		# Handle pandas NA/None quickly
		if value is None or (isinstance(value, float) and np.isnan(value)):
			return 0
	except Exception:
		pass

	# Numeric-like
	try:
		as_int = int(value)  # type: ignore[arg-type]
		return 1 if as_int == 1 else 0
	except Exception:
		pass

	text = str(value).strip().lower()
	if text in {"1", "true", "t", "yes", "y", "violation", "fail", "failed", "incorrect", "i"}:
		return 1
	if text in {"0", "false", "f", "no", "n", "pass", "passed", "correct", "c", "ok"}:
		return 0
	return 0


def plot_confusion_heatmap(cm: np.ndarray, out_path: Path) -> None:
	ax = sns.heatmap(
		cm,
		annot=True,
		fmt="d",
		cmap="Blues",
		xticklabels=["Pred 0", "Pred 1"],
		yticklabels=["True 0", "True 1"],
	)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_title("Confusion Matrix")
	plt.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=200)
	plt.close()


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
	accuracy = float(np.mean((y_true == y_pred).astype(float))) if len(y_true) > 0 else 0.0
	cm = compute_confusion_matrix(y_true, y_pred)
	tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
	neg_total = tn + fp
	pos_total = tp + fn
	fpr = float(fp / neg_total) if neg_total > 0 else 0.0
	fnr = float(fn / pos_total) if pos_total > 0 else 0.0
	return {
		"accuracy": accuracy,
		"error_rate": 1.0 - accuracy,
		"false_positive_rate": fpr,
		"false_negative_rate": fnr,
		"confusion_matrix": {
			"labels": ["True0_Pred0", "True0_Pred1", "True1_Pred0", "True1_Pred1"],
			"values": [tn, fp, fn, tp],
		},
		"samples": int(len(y_true)),
	}


def evaluate_xlsx(
	xlsx_path: Path,
	models: List[str],
	prompt_path: Path,
	rubric_col: str,
	transcript_col: str,
	label_col: str,
	rubric_id_col: Optional[str],
	temperature: float,
	output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
	df = pd.read_excel(xlsx_path)

	for required in (rubric_col, transcript_col, label_col):
		if required not in df.columns:
			raise ValueError(f"Missing required column '{required}' in {xlsx_path}")

	# Local import after sys.path adjustment to avoid E402 and allow direct script execution
	from src.agent_helpers.text_utils import load_prompt_text

	prompt_template = load_prompt_text(prompt_path)

	clients: Dict[str, Tuple[object, str, str]] = {}
	for model in models:
		clients[model] = make_client(model)

	pred_cols: List[str] = []
	for model in models:
		col_name = f"pred_{model.replace('/', '_')}"
		pred_cols.append(col_name)
		df[col_name] = pd.NA

	majority_col = "pred_majority"
	df[majority_col] = pd.NA

	true_vals = []
	majority_vals = []

	for idx, row in df.iterrows():
		rubric_text = str(row[rubric_col])
		transcript_text = str(row[transcript_col])
		rubric_id = str(row[rubric_id_col]) if rubric_id_col and rubric_id_col in df.columns else "unknown"
		true_label = _parse_true_label(row[label_col])

		user_prompt = prompt_template.format(
			rubric_id=rubric_id,
			rubric=rubric_text,
			transcript=transcript_text,
		)

		sample_votes: List[int] = []
		for model in models:
			client, provider, model_name = clients[model]
			logger.debug(
				"Submitting sample idx=%s rubric_id=%s to model=%s",
				idx,
				rubric_id,
				model,
			)
			label = call_judge_model(
				client,
				provider,
				model_name,
				user_prompt,
				temperature=temperature,
			)
			col_name = f"pred_{model.replace('/', '_')}"
			if label is None:
				df.at[idx, col_name] = pd.NA
			else:
				df.at[idx, col_name] = int(label)
				sample_votes.append(int(label))

		mv = majority_vote(sample_votes)
		df.at[idx, majority_col] = int(mv)

		true_vals.append(true_label)
		majority_vals.append(mv)

	df["__true_label"] = [int(v) for v in true_vals]
	y_true = df["__true_label"].to_numpy(dtype=int)
	y_pred = np.array([int(v) for v in majority_vals], dtype=int)

	# Metrics for majority across all models
	all_models_metrics = _compute_metrics(y_true, y_pred)

	# Per-model metrics
	per_model_metrics: Dict[str, Dict[str, object]] = {}
	for model in models:
		col = f"pred_{model.replace('/', '_')}"
		mask = df[col].notna()
		if mask.any():
			model_preds = df.loc[mask, col].astype(int).to_numpy(dtype=int)
			model_truth = df.loc[mask, "__true_label"].astype(int).to_numpy(dtype=int)
		else:
			model_preds = np.array([], dtype=int)
			model_truth = np.array([], dtype=int)
		per_model_metrics[model] = _compute_metrics(model_truth, model_preds)

	# Best trio selection (post-hoc)
	from itertools import combinations

	best_trio_models: List[str] = []
	best_trio_metrics: Dict[str, object] = {}
	best_trio_preds: Optional[np.ndarray] = None
	best_acc = -1.0
	if len(models) >= 3:
		for trio in combinations(models, 3):
			cols = [f"pred_{m.replace('/', '_')}" for m in trio]
			mat = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=int)
			# Majority vote across the 3 models (sum >= 2 => 1 else 0)
			trio_pred = (mat.sum(axis=1) >= 2).astype(int)
			metrics = _compute_metrics(y_true, trio_pred)
			acc = float(metrics["accuracy"])
			if acc > best_acc:
				best_acc = acc
				best_trio_models = list(trio)
				best_trio_metrics = metrics
				best_trio_preds = trio_pred

	report = {
		"input_xlsx": str(xlsx_path),
		"models": models,
		"columns": {
			"rubric": rubric_col,
			"transcript": transcript_col,
			"true_label": label_col,
			"rubric_id": rubric_id_col or "",
		},
		"metrics": all_models_metrics,
		"per_model_metrics": per_model_metrics,
		"best_trio": {
			"models": best_trio_models,
			"metrics": best_trio_metrics,
		},
	}

	output_dir.mkdir(parents=True, exist_ok=True)
	df.drop(columns=["__true_label"], inplace=True)

	df.to_csv(output_dir / "predictions.csv", index=False)

	with (output_dir / "report.json").open("w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)

	if best_trio_preds is not None:
		cm_to_plot = compute_confusion_matrix(y_true, best_trio_preds)
	else:
		cm_to_plot = compute_confusion_matrix(y_true, y_pred)
	plot_confusion_heatmap(cm_to_plot, output_dir / "confusion_matrix.png")

	return df, report


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate LLM judge models on an XLSX with rubric/transcript and true_label."
	)
	parser.add_argument(
		"--xlsx",
		type=Path,
		required=True,
		help="Path to input XLSX file. Must contain rubric, transcript, and true_label columns.",
	)
	parser.add_argument(
		"--models",
		type=str,
		nargs="+",
		required=False,
		default=None,
		help="List of models, e.g. openai/gpt-4o-mini anthropic/claude-3-haiku-20240307. If omitted, uses COMMON_JUDGE_MODELS.",
	)
	parser.add_argument(
		"--prompt-path",
		type=Path,
		default=Path("prompts/templates/judge/agent_judge_inspect.md"),
		help="Path to prompt template. Should accept {rubric_id}, {rubric}, {transcript}.",
	)
	parser.add_argument(
		"--rubric-col",
		type=str,
		default="rubric_text",
		help="Column name for rubric text.",
	)
	parser.add_argument(
		"--rubric-id-col",
		type=str,
		default="rubric_id",
		help="Optional column name for rubric id. If missing, 'unknown' is used.",
	)
	parser.add_argument(
		"--transcript-col",
		type=str,
		default="transcript",
		help="Column name for transcript text.",
	)
	parser.add_argument(
		"--label-col",
		type=str,
		default="validation_score",
		help="Column name for true labels (0/1).",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="Sampling temperature for judge models.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Output directory. Defaults to outputs/judge_eval_YYYYmmdd_HHMMSS/",
	)
	return parser.parse_args()


def main() -> None:
	if not logging.getLogger().handlers:
		logging.basicConfig(
			level=logging.INFO,
			format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
		)
	args = parse_args()
	xlsx_path: Path = args.xlsx
	models: Optional[List[str]] = args.models
	prompt_path: Path = args.prompt_path
	rubric_col: str = args.rubric_col
	rubric_id_col: Optional[str] = args.rubric_id_col
	transcript_col: str = args.transcript_col
	label_col: str = args.label_col
	temperature: float = args.temperature
	output_dir: Optional[Path] = args.output_dir

	if not models:
		models = list(COMMON_JUDGE_MODELS)

	if output_dir is None:
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_dir = Path("outputs") / f"judge_eval_{ts}"

	_ = evaluate_xlsx(
		xlsx_path=xlsx_path,
		models=models,
		prompt_path=prompt_path,
		rubric_col=rubric_col,
		transcript_col=transcript_col,
		label_col=label_col,
		rubric_id_col=rubric_id_col,
		temperature=temperature,
		output_dir=output_dir,
	)
	print(f"Wrote predictions and report to: {output_dir}")


if __name__ == "__main__":
	main()
