"""Canonical ingestion routines for Inspect-AI `.eval` agent logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from inspect_ai.log import read_eval_log

NormalizedAgentRun = Dict[str, Any]
NormalizedMessage = Dict[str, Any]


class AgentIngestionError(RuntimeError):
	"""Raised when an Inspect-AI log cannot be loaded into a run."""


def load_inspect_eval_runs(path: Union[str, Path]) -> List[NormalizedAgentRun]:
	"""
	Load an Inspect-AI `.eval` archive and return each sample as a dict.

	Args:
	    path (Path): Path to an Inspect-AI `.eval` archive.

	Returns:
	    List[NormalizedAgentRun]: List of runs suitable for downstream perturbation logic.
	"""
	resolved = Path(path)
	if not resolved.exists():
		raise FileNotFoundError(f"Log path not found: {resolved}")

	try:
		log = read_eval_log(str(resolved))
	except Exception as exc:
		raise AgentIngestionError(f"Failed to load Inspect log {resolved}: {exc}") from exc

	samples = getattr(log, "samples", None) or []

	runs = [sample.model_dump(mode="json") for sample in samples]

	return runs
