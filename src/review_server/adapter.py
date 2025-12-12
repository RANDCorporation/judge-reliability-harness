# src/review_server/adapter.py

from __future__ import annotations

from typing import List

import pandas as pd

from schemas import ReviewItem, SavedItem


def build_item_from_perturbation_record(saved_item: SavedItem) -> ReviewItem:
	"""Converts SavedItem to ReviewItem."""
	agent_mode = getattr(saved_item, "generation_mode", None) or ""
	is_agentic = saved_item.test_name == "agent_perturbation"
	source_type = (
		"agentic_positive"
		if (is_agentic and agent_mode == "positive")
		else ("agentic_transcript" if is_agentic else "synthetic_generation")
	)
	args = {
		"sample_id": saved_item.perturbed_idx,
		"test_name": saved_item.test_name,
		"original_request": saved_item.original_request,
		"original_response": saved_item.original_response,
		"generated_response": saved_item.generation_response,
		"expected_label": saved_item.validation_score,
		"source_type": source_type,
		"metadata": {
			"original_expected": saved_item.original_expected,
			"prompted_bucket": saved_item.prompted_bucket,
			"validated_bucket": saved_item.validated_bucket,
			"generation_temp": saved_item.generation_temp,
			"validation_reasoning": saved_item.validation_reasoning,
			"source_type": source_type,
			"generation_reasoning": saved_item.generation_reasoning,
			"agent_mode": agent_mode or None,
		},
	}

	if saved_item.test_name in {"agent_perturbation", "agent_positives"}:
		# Preserve the transcript for diffing in the UI.
		transcript = saved_item.original_response or saved_item.generation_prompt or ""
		args["metadata"]["original_transcript"] = transcript

	return ReviewItem(**args)


def build_items_from_perturbations(df: pd.DataFrame) -> List[ReviewItem]:
	"""Converts raw synthetic ordinal pipeline output to review item dicts."""
	items = []
	for record in df.to_dict("records"):
		# Replace pandas NaN placeholders with real None values so SavedItem validation passes.
		for key, value in list(record.items()):
			if pd.isna(value):
				record[key] = None

		# Normalize SavedItem data types
		if "prompted_bucket" in record and record["prompted_bucket"] is not None:
			record["prompted_bucket"] = str(record["prompted_bucket"])
		items.append(build_item_from_perturbation_record(SavedItem(**record)))
	return items
