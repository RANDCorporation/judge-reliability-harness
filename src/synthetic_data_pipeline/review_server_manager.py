# src/review_server/review_server_manager.py

import numbers
import time
from typing import Any, Callable, Dict, Optional

import pandas as pd

from core import console
from review_server import (
	build_item_from_perturbation_record,
	build_items_from_perturbations,
	run_review_session,
)
from schemas import (
	PerturbationConfig,
	SavedItem,
)

from .base_pipeline import BasePipeline
from .data_registry import DataRegistry


def _coerce_bool(value: Any) -> Optional[bool]:
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)) and value in (0, 1):
		return bool(value)
	if isinstance(value, str):
		lowered = value.strip().lower()
		if lowered in {"true", "false"}:
			return lowered == "true"
		if lowered in {"1", "0"}:
			return lowered == "1"
	return None


def _coerce_int(value: Any) -> Optional[int]:
	if isinstance(value, bool):
		return int(value)
	if isinstance(value, int):
		return value
	if isinstance(value, str):
		stripped = value.strip()
		if stripped.startswith("-"):
			digits = stripped[1:]
		else:
			digits = stripped
		if digits.isdigit():
			try:
				return int(stripped)
			except ValueError:
				return None
	return None


def _coerce_float(value: Any) -> Optional[float]:
	if isinstance(value, bool):
		return float(int(value))
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str):
		stripped = value.strip()
		try:
			return float(stripped)
		except ValueError:
			return None
	return None


def _normalize_expected_label(value: Any, reference: Any = None) -> Any:
	"""
	Convert UI strings into the closest native type based on the existing column
	dtype so pandas/sklearn downstream stay consistent.
	"""
	if value is None:
		return None

	# Try to match the existing dtype first
	if reference is not None:
		if isinstance(reference, bool):
			coerced = _coerce_bool(value)
			if coerced is not None:
				return coerced
		elif isinstance(reference, numbers.Integral):
			coerced = _coerce_int(value)
			if coerced is not None:
				return coerced
		elif isinstance(reference, numbers.Real):
			coerced = _coerce_float(value)
			if coerced is not None:
				return coerced

	# Fall back to best-effort parsing
	for coercer in (_coerce_bool, _coerce_int, _coerce_float):
		coerced = coercer(value)
		if coerced is not None:
			return coerced

	if isinstance(value, str):
		return value.strip()
	return value


def _get_existing_label(df: pd.DataFrame, mask: pd.Series) -> Any:
	if mask.any():
		current = df.loc[mask, "validation_score"]
		if not current.empty:
			return current.iloc[0]
	return None


class ReviewServerManager:
	def __init__(self, perturbation_config: PerturbationConfig, data_registry: DataRegistry):
		self.perturbation_config = perturbation_config
		self.data_registry = data_registry

	def run_HITL_server(self, test_name: str, pipeline: BasePipeline) -> pd.DataFrame:
		"""Run HITL review for registry-backed pipelines."""
		return self._run_review_session(
			test_name=test_name,
			initial_items=pipeline.get_data_to_review(),
			runner=pipeline.run,
			append_records=False,
		)

	def run_agentic_session(
		self,
		test_name: str,
		generator: Callable[[Callable[[SavedItem], None]], pd.DataFrame],
	) -> pd.DataFrame:
		"""Run HITL review for agentic pipelines that emit SavedItems via callbacks."""
		return self._run_review_session(
			test_name=test_name,
			initial_items=self._get_unreviewed_registry_items(),
			runner=generator,
			append_records=True,
		)

	def _get_unreviewed_registry_items(self) -> pd.DataFrame:
		df = self.data_registry.get_data()
		if df.empty or "human_reviewed" not in df.columns:
			return df
		mask = ~(df["human_reviewed"].fillna(False))
		return df[mask].copy()

	def _get_reviewed_registry_items(self) -> pd.DataFrame:
		"""Return all rows that have been accepted/reviewed so far."""
		df = self.data_registry.get_data()
		if df.empty or "human_reviewed" not in df.columns:
			return df
		mask = df["human_reviewed"].fillna(False)
		return df[mask].copy()

	def _run_review_session(
		self,
		test_name: str,
		*,
		initial_items: pd.DataFrame,
		runner: Callable[[Callable[[SavedItem], None]], pd.DataFrame],
		append_records: bool,
	) -> pd.DataFrame:
		batch_id = f"{test_name}_{time.strftime('%Y%m%d%H%M%S')}"
		items = build_items_from_perturbations(initial_items) if not initial_items.empty else []

		with run_review_session(
			batch_id=batch_id,
			items=[],
			output_dir=self.perturbation_config.review_log_dir,
		) as session:
			if items:
				session.add_items(items)

			def _on_record(record: SavedItem) -> None:
				if append_records:
					self.data_registry.append(record)
				try:
					item_payload = build_item_from_perturbation_record(record)
					session.add_item(item_payload)
				except Exception as exc:
					console.print(
						f"  [yellow]Warning: Failed to stream item {record.perturbed_idx} to review UI ({exc}).[/yellow]"
					)

			perturbed_df = runner(_on_record)

			if perturbed_df is None or perturbed_df.empty:
				console.print("[WARNING] Synthetic data pipeline produced no data.")
				return pd.DataFrame()

			console.print(
				"\n[blue]Synthetic data pipeline complete.\nClick 'Finalize Review' on the UI or 'Enter' on your keyboard to proceed.[/]"
			)

			session.wait_for_finalize()
			review_payload = session.collect_results()
			self._apply_review_results(perturbed_df, review_payload)
			return self.data_registry.get_data()

	def _apply_review_results(self, dataset: pd.DataFrame, review_payload: Dict[str, Any]):
		"""
		Apply reviewer decisions to the dataset, removing rejected items and
		overwriting edited responses or expected labels. Returns a curated copy.
		"""
		if not review_payload:
			return dataset

		# delete rejected items first
		rejected_ids = [item["sample_id"] for item in review_payload.get("rejected", [])]

		for sample_id in rejected_ids:
			self.data_registry.delete(sample_id)

		if rejected_ids:
			df = dataset[~dataset["perturbed_idx"].isin(rejected_ids)].copy()
		else:
			df = dataset.copy()

		# update accepted items
		accepted_by_id = {item["sample_id"]: item for item in review_payload.get("accepted", [])}
		for sample_id, payload in accepted_by_id.items():
			updates = {"human_reviewed": True}
			mask = df["perturbed_idx"].astype(str) == sample_id
			reference_value = _get_existing_label(df, mask)
			normalized_label = None
			if "generated_response" in payload:
				updates["generation_response"] = payload["generated_response"]
			if "expected_label" in payload:
				normalized_label = _normalize_expected_label(payload["expected_label"], reference_value)
				updates["validation_score"] = normalized_label
			self.data_registry.update(sample_id, **updates)

			# update local df copy
			if "generated_response" in payload:
				df.loc[mask, "generation_response"] = payload["generated_response"]
			if "expected_label" in payload:
				df.loc[mask, "validation_score"] = normalized_label

		# Apply edits
		edited_by_id = {item["sample_id"]: item for item in review_payload.get("edited", [])}
		for sample_id, payload in edited_by_id.items():
			updates = {}
			updates["human_reviewed"] = True
			mask = df["perturbed_idx"].astype(str) == sample_id
			reference_value = _get_existing_label(df, mask)
			normalized_label = None
			if "generated_response" in payload:
				updates["generation_response"] = payload["generated_response"]
			if "expected_label" in payload:
				normalized_label = _normalize_expected_label(payload["expected_label"], reference_value)
				updates["validation_score"] = normalized_label
			if updates:
				self.data_registry.update(sample_id, **updates)

			# update local df copy
			if "generated_response" in payload:
				df.loc[mask, "generation_response"] = payload["generated_response"]
			if "expected_label" in payload:
				df.loc[mask, "validation_score"] = normalized_label

		return df
