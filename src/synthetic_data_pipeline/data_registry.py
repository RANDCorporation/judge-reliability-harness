import os
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from core import console
from schemas import SavedItem


class DataRegistry:
	"""
	A registry that tracks all generated synthetic data and their metadata.
	Uses a single CSV file for all SavedItem records.
	Thread-safe for parallel generation.
	"""

	def __init__(self, path: Path, autosave_every: int = 1):
		self.autosave_every = autosave_every
		self.path = path
		self._unsaved_changes = 0
		self._lock = threading.Lock()
		self._load()

	def _load(self):
		"""Thread-safe load of DataFrame"""
		if self.path.exists():
			suffix = self.path.suffix.lower()
			if suffix == ".csv":
				self.state = pd.read_csv(self.path)
			elif suffix == ".xlsx":
				self.state = pd.read_excel(self.path)
			else:
				raise ValueError(f"Unsupported registry format: {suffix}")
			self._ensure_columns()
		else:
			# Initialize empty DataFrame with SavedItem columns
			self.state = pd.DataFrame(columns=SavedItem.model_fields)

	def _ensure_columns(self):
		expected_columns = list(SavedItem.model_fields.keys())
		missing = [col for col in expected_columns if col not in self.state.columns]
		for col in missing:
			self.state[col] = None

		extras = [col for col in self.state.columns if col not in expected_columns]
		self.state = self.state[expected_columns + extras]

	def _save(self, verbose=True):
		"""Thread-safe save of full DataFrame"""
		with self._lock:
			tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
			suffix = self.path.suffix.lower()
			if suffix == ".csv":
				self.state.to_csv(tmp_path, index=False)
			elif suffix == ".xlsx":
				self.state.to_excel(tmp_path, index=False)
			else:
				raise ValueError(f"Unsupported registry format: {suffix}")
			os.replace(tmp_path, self.path)
			self._unsaved_changes = 0
		if verbose:
			console.print(f"[DataRegistry] Saved {len(self.state)} rows → {self.path}")

	def get_data(self) -> pd.DataFrame:
		"""Thread-safe return the full DataFrame."""
		with self._lock:
			return self.state.copy()

	def append(self, saved_item: SavedItem):
		"""Thread-safe append a SavedItem to the registry."""
		do_save = False
		with self._lock:
			df_new = pd.DataFrame([saved_item.model_dump()])
			if self.state.empty:
				self.state = df_new
			else:
				self.state = pd.concat([self.state, df_new], ignore_index=True)

			self._unsaved_changes += 1
			do_save = self._unsaved_changes >= self.autosave_every

		if do_save:
			self._save(verbose=False)

	def update(self, perturbed_idx: str, **updates):
		"""Thread-safe update one row by perturbed_idx."""
		do_save = False
		with self._lock:
			mask = self.state["perturbed_idx"] == perturbed_idx
			if not mask.any():
				return

			for k, v in updates.items():
				if k in self.state.columns:
					self.state.loc[mask, k] = v

			self._unsaved_changes += 1
			do_save = self._unsaved_changes >= self.autosave_every

		if do_save:
			self._save(verbose=False)

	def delete(self, perturbed_idx: str):
		"""Thread-safe delete a row by perturbed_idx."""
		do_save = False
		with self._lock:
			self.state = self.state[self.state["perturbed_idx"] != perturbed_idx].copy()
			self._unsaved_changes += 1
			do_save = self._unsaved_changes >= self.autosave_every

		if do_save:
			self._save(verbose=False)

	def get(self, perturbed_idx: str) -> Optional[SavedItem]:
		"""Thread-safe return a SavedItem object by perturbed_idx."""
		with self._lock:
			result = self.state.loc[self.state["perturbed_idx"] == perturbed_idx]
			if result.empty:
				return None
			record = result.iloc[0].to_dict()
			# Normalize data types for SavedItem (CSV may have different types)
			if "prompted_bucket" in record and record["prompted_bucket"] is not None:
				record["prompted_bucket"] = str(record["prompted_bucket"])
			return SavedItem(**record)

	def close(self):
		"""Ensure unsaved changes are written."""
		if self._unsaved_changes > 0:
			self._save()
