# src/reliability_tests/eval_metrics.py

import pandas as pd
from pathlib import Path
from sklearn import metrics
import numpy as np
import scipy.stats
import json
import matplotlib.pyplot as plt
import seaborn as sns

from schemas import AdminConfig
from core import console


def _normalize_label(value):
	"""Best-effort conversion so gold/pred labels share a dtype for sklearn metrics."""
	if value is None or (isinstance(value, float) and np.isnan(value)):
		return value

	if isinstance(value, bool):
		return int(value)

	if isinstance(value, (int, float)):
		return value

	if isinstance(value, str):
		text = value.strip()
		if not text:
			return text
		lowered = text.lower()
		if lowered in {"true", "false"}:
			return 1 if lowered == "true" else 0
		if lowered in {"yes", "no"}:
			return 1 if lowered == "yes" else 0
		if lowered in {"pass", "fail"}:
			return lowered  # keep categorical labels aligned
		try:
			if "." in text:
				return float(text)
			return int(text)
		except ValueError:
			return text

	return str(value)


class EvalMetrics:
	def __init__(self, admin_config: AdminConfig):
		self.admin_config = admin_config
		self.evaluation_config = admin_config.evaluation_config

	def _get_error_bars(self, data: pd.DataFrame, confidence: float = 0.95):
		data = np.array(data)
		n = len(data)
		m = np.mean(data)
		se = scipy.stats.sem(data)
		h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
		return m, m - h, m + h

	def _get_bootstrapped_metrics(
		self,
		results: pd.DataFrame,
	) -> dict:
		"""
		Get bootstrapped score
		"""
		null_result = {"mean": "N/A", "lower_ci": "N/A", "upper_ci": "N/A"}
		if results.empty:
			return null_result

		if self.evaluation_config.bootstrap_size is None:
			bootstrap_size = len(results)
			bootstrap_repetitions = 1
		else:
			bootstrap_size = int(self.evaluation_config.bootstrap_size * len(results))
			bootstrap_repetitions = self.evaluation_config.bootstrap_repetitions

		if bootstrap_size == 0:
			return null_result

		metric_name = self.evaluation_config.metric
		metric_fn = getattr(metrics, metric_name, None)

		bootstrap_score_list = []
		for _ in range(bootstrap_repetitions):
			sample = results.sample(n=bootstrap_size, replace=True)
			if metric_fn is not None:
				metric_score = metric_fn(sample["validation_score"], sample["autograder_score"])
			else:
				metric_score = (sample["autograder_score"] == sample["validation_score"]).sum() / len(sample)
			bootstrap_score_list.append(metric_score)

		mean, lower_ci, upper_ci = self._get_error_bars(bootstrap_score_list)
		return {"mean": mean, "lower_ci": lower_ci, "upper_ci": upper_ci}

	def calculate_metrics(self, results: pd.DataFrame) -> dict:
		"""
		Calculate metrics from test results using the metric specified in evaluation_config.metric.

		Args:
		    results: DataFrame with 'autograder_score' and 'validation_score' columns.

		Returns:
		    Dict with metric score, pass rate, and counts.
		"""
		null_return = {
			"metrics": {
				"pass_rate": np.nan,
				"passed": np.nan,
				"total": np.nan,
				"metric_name": None,
				"metric_score": None,
			},
			"bootstrap": {"mean": np.nan, "lower_ci": np.nan, "upper_ci": np.nan},
		}
		if results.empty:
			print("[WARNING] Results empty!")
			return null_return

		required_cols = {"autograder_score", "validation_score"}
		if not required_cols.issubset(results.columns):
			console.print("[WARNING] Missing required columns: 'autograder_score' and 'validation_score'.")
			return null_return

		results = results.copy()
		results["validation_score"] = results["validation_score"].map(_normalize_label)
		results["autograder_score"] = results["autograder_score"].map(_normalize_label)

		total_count = len(results)
		if total_count == 0:
			return null_return

		passed = (results["autograder_score"] == results["validation_score"]).sum()
		pass_rate = passed / total_count

		# Try to get the sklearn metric function
		metric_name = self.evaluation_config.metric
		metric_score = None
		metric_fn = getattr(metrics, metric_name, None)
		if metric_fn is not None:
			metric_score = metric_fn(results["validation_score"], results["autograder_score"])

		return {
			"metrics": {
				"pass_rate": f"{pass_rate:.2%}",
				"passed": passed,
				"total": total_count,
				"metric_type": metric_name,
				"metric_score": f"{metric_score:.2%}",
			},
			"bootstrap": self._get_bootstrapped_metrics(results),
		}

	def _make_cost_curves(self, output_dir: Path):
		"""
		Generates plot of cost curve for each autograder model.
		Produces a heatmap with tests on the y-axis and models on the x-axis.
		"""
		report_files = list(Path("./outputs/").glob("*/*_report.json"))

		if not report_files:
			print("No report files found in ./outputs")
			return

		# Collect metrics from each model's report
		model_scores = {}
		test_ids = set()

		for report_file in report_files:
			with open(report_file, "r", encoding="utf-8") as f:
				report = json.load(f)

			model_name = report["meta_data"]["autograder_model"]
			metrics = report["all_metrics"]

			# assume metrics is a dict like: { test_id: { "cost": value, ... }, ... }
			model_scores[model_name] = {}
			for test_id, test_metrics in metrics.items():
				test_ids.add(test_id)
				# choose the metric you want to visualize
				cost_value = test_metrics["metrics"].get("metric_score", None)
				if isinstance(cost_value, str) and cost_value.endswith("%"):
					cost_value = float(cost_value.rstrip("%")) / 100
				model_scores[model_name][test_id] = cost_value

		# Build a dataframe with test_ids as rows and model_names as columns
		df = pd.DataFrame.from_dict(model_scores, orient="columns")
		df = df.reindex(sorted(test_ids))  # order tests

		# Plot heatmap
		plt.figure(figsize=(10, max(6, len(test_ids) * 0.4)))
		sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "Cost"})
		plt.title("Cost Curve Heatmap by Test and Model")
		plt.xlabel("Model")
		plt.ylabel("Test ID")
		plt.tight_layout()

		plot_path = output_dir / "cost_curve_heatmap.png"
		plt.savefig(plot_path, dpi=300)
		plt.close()

		print(f"✅ Cost curve heatmap saved to {plot_path}")

	def generate_report(self, all_metrics: dict):
		"""Generates a JSON report of the results."""
		output_dir = Path(self.admin_config.output_dir)

		report_content = {
			"meta_data": {
				"module_name": self.admin_config.module_name,
				"dataset": output_dir / self.admin_config.dataset_config.dataset_name,
				"autograder_model": self.evaluation_config.autograder_model_name,
			},
			"all_metrics": all_metrics,
		}

		if all_metrics:
			model_name = self.evaluation_config.autograder_model_name.replace("/", "_")
			report_path = output_dir / f"{model_name}_report.json"
			with open(report_path, "w", encoding="utf-8") as f:
				json.dump(report_content, f, indent=2, default=str)

		if self.evaluation_config.get_cost_curves:
			self._make_cost_curves(output_dir)
