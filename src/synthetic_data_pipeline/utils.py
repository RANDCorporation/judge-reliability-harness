# src/synthetic_data_pipeline/utils.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Set
from sentence_transformers import SentenceTransformer

from core import console
from schemas import OriginalDataPointConfig


class SimilarityFilter:
	"""Encapsulates similarity checking logic."""

	def __init__(self, threshold: float, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
		self.threshold = threshold
		self.encoder = SentenceTransformer(model_name)
		self.encoder.max_seq_length = 512
		self.seed_vectors = None
		self.kept_vectors = []

	def load_seed_data(self, texts: List[str]):
		if texts:
			console.print(f"Encoding {len(texts)} seed documents for similarity filtering...")
			self.seed_vectors = self._encode(texts)

	def check(self, text: str) -> bool:
		"""Returns True if the item is unique enough, False otherwise."""
		vec = self._encode([text])[0]

		if self.seed_vectors is not None:
			if np.max(self.seed_vectors @ vec) >= self.threshold:
				console.print("Too similar to seed data.")
				return False

		if self.kept_vectors:
			if np.max(np.vstack(self.kept_vectors) @ vec) >= self.threshold:
				console.print("Too similar to existing synthetic data.")
				return False

		self.kept_vectors.append(vec)
		return True

	def _encode(self, texts: List[str]) -> np.ndarray:
		return self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def build_synthetic_ordinal_template(
	seed_df: pd.DataFrame,
	source_item: OriginalDataPointConfig,
	target_bucket: int,
	num_examples: int,
	items_by_bucket: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
	"""Builds the prompt variables, including optional few-shot examples."""
	examples = []
	if num_examples > 0:
		source_df = None
		bucket_df = items_by_bucket.get(target_bucket)

		if bucket_df is not None and len(bucket_df) >= num_examples:
			source_df = bucket_df
		elif len(seed_df) >= num_examples:
			source_df = seed_df

		if source_df is not None:
			examples = source_df.sample(num_examples).to_dict("records")

	if not examples and num_examples > 0:
		console.print(f"Not enough seed examples for bucket {target_bucket}. Skipping.")
		return None  # Cannot generate without examples

	examples_block = "\n\n".join(f"#### Example scored {r['expected']}\n{r['response']}" for r in examples)
	return {
		"source_prompt": source_item.request,
		"target_bucket": target_bucket,
		"num_examples": len(examples),
		"examples_block": examples_block,
	}


class BucketManager:
	def __init__(self, df: pd.DataFrame, seed: int):
		self.df = df
		self.seed = seed
		self.buckets = self.define_buckets()
		self.items_by_bucket = self.prepare_item_buckets()

	def define_buckets(self) -> List[int]:
		"""Returns sorted list of unique buckets from seed data."""
		return sorted(self.df["expected"].dropna().unique())

	def prepare_item_buckets(self) -> Dict[int, pd.DataFrame]:
		"""Pre-sorts items into their respective buckets for efficient access."""
		return {bucket: group_df for bucket, group_df in self.df.groupby("expected") if pd.notna(bucket)}

	def visualize_status(self, source_id: str, validated_buckets: Set):
		"""Display a visual representation of validator-confirmed buckets for a source item."""
		filled_buckets = []
		empty_buckets = []

		for bucket in self.buckets:
			if (source_id, bucket) in validated_buckets:
				filled_buckets.append(str(bucket))
			else:
				empty_buckets.append(str(bucket))

		filled_str = " ".join(f"[{b}]" for b in filled_buckets) if filled_buckets else "none"
		empty_str = " ".join(f"({b})" for b in empty_buckets) if empty_buckets else "none"
		console.print(f"\nSTATUS ({source_id}): Filled {filled_str} | Empty {empty_str}\n")


def get_sample(
	df: pd.DataFrame, sample_size: Optional[int] = None, seed: int = 8234, with_replacement=False
) -> pd.DataFrame:
	"""Gets sample of input data"""
	if sample_size is None or sample_size <= 0:
		return df
	if with_replacement:
		sample_df = df.sample(n=sample_size, random_state=seed, replace=True)
	else:
		sample_size = min(sample_size, len(df))
		sample_df = df.sample(n=sample_size, random_state=seed)
	return sample_df
