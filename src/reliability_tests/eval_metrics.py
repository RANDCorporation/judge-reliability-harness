# src/reliability_tests/eval_metrics.py

import pandas as pd
from pathlib import Path
from sklearn import metrics
import numpy as np
import scipy.stats
import json
import matplotlib.pyplot as plt
from litellm import token_counter, cost_per_token

from schemas import AdminConfig, LLMClientConfig
from core import LLMClient, console, threaded_executor


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
		if lowered in {"yes", "no", "y", "n"}:
			return 1 if lowered in {"yes", "y"} else 0
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

		# Ensure both columns have the same dtype after normalization
		# Check if we have mixed types and convert to a common type
		validation_types = set(type(v).__name__ for v in results["validation_score"].dropna().unique()[:100])
		autograder_types = set(type(v).__name__ for v in results["autograder_score"].dropna().unique()[:100])

		# If we have mixed types, try to convert both to numeric (int/float)
		if len(validation_types) > 1 or len(autograder_types) > 1 or validation_types != autograder_types:
			# Try to convert both to numeric, coercing errors to NaN
			results["validation_score"] = pd.to_numeric(results["validation_score"], errors="coerce")
			results["autograder_score"] = pd.to_numeric(results["autograder_score"], errors="coerce")
			# Fill NaN with a default value (0) if needed
			results["validation_score"] = results["validation_score"].fillna(0).astype(int)
			results["autograder_score"] = results["autograder_score"].fillna(0).astype(int)

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

	def _calculate_test_cost(
		self, graded_df: pd.DataFrame, judge: LLMClient, model: str, test_name: str = None
	) -> float:
		"""
		Calculate the total cost for a test by counting tokens for all rows.

		Args:
		    graded_df: DataFrame with evaluation results
		    judge: LLMClient instance for accessing config and building prompts
		    model: Model name for token counting
		    test_name: Optional test name for special handling (e.g., agent_perturbation)

		Returns:
		    float: Total cost in USD for this test
		"""
		total_cost = 0.0

		# Extract model name for litellm token_counter (it expects format like "gpt-3.5-turbo" or "gemini/gemini-2.5-pro")
		# If model is "google/gemini-2.5-pro", litellm expects "gemini/gemini-2.5-pro"
		if model and "/" in model:
			provider, model_name = model.split("/", 1)
			if provider == "google":
				# litellm uses "gemini/" prefix for Google models
				litellm_model = f"gemini/{model_name}"
			else:
				litellm_model = model
		else:
			litellm_model = model

		# Rename columns to match what the template expects (same as evaluator does)
		graded_df_for_prompt = graded_df.copy()

		# Special handling for agent_perturbation test (same as in evaluator)
		if test_name == "agent_perturbation":
			if "rubric" not in graded_df_for_prompt.columns and "rubric_text" in graded_df_for_prompt.columns:
				graded_df_for_prompt["rubric"] = graded_df_for_prompt["rubric_text"]
			if "transcript" not in graded_df_for_prompt.columns:
				graded_df_for_prompt["transcript"] = graded_df_for_prompt.get("generation_response", "")
			if "score_levels_table" not in graded_df_for_prompt.columns:
				graded_df_for_prompt["score_levels_table"] = ""

		if "original_request" in graded_df_for_prompt.columns:
			graded_df_for_prompt = graded_df_for_prompt.rename(columns={"original_request": "request"})
		if "generation_response" in graded_df_for_prompt.columns:
			graded_df_for_prompt = graded_df_for_prompt.rename(columns={"generation_response": "response"})

		# Get system prompt from config
		system_prompt = judge.config.default_params.get("system_prompt", "")

		# Build prompts and count tokens for each row using threaded_executor
		for row, user_prompt in threaded_executor(
			graded_df_for_prompt, lambda r: judge._build_prompt(r), self.evaluation_config.max_workers
		):
			if not user_prompt:
				continue

			# Build messages for token counting
			messages = []
			if system_prompt:
				messages.append({"role": "system", "content": system_prompt})
			messages.append({"role": "user", "content": user_prompt})

			# Count tokens for input prompt using litellm model format
			token_count_input = token_counter(model=litellm_model, messages=messages)

			# Count tokens for LLM output - reasoning and score
			autograder_score = row.get("autograder_score", "")
			autograder_reasoning = row.get("autograder_reasoning", "")

			# Create a dict representation of the response (like model_dump() would return)
			response_dict = {"score": autograder_score, "reasoning": autograder_reasoning or ""}
			# Convert to JSON string and count tokens
			response_json = json.dumps(response_dict)
			output_messages_json = [{"role": "assistant", "content": response_json}]
			token_count_output = token_counter(model=litellm_model, messages=output_messages_json)

			prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(
				model=litellm_model, prompt_tokens=token_count_input, completion_tokens=token_count_output
			)
			cost_usd_dollar = prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar
			total_cost += cost_usd_dollar

		return total_cost

	def _load_accuracies(self, output_dir: Path, model_name: str) -> dict:
		"""
		Loads accuracy metrics from the report JSON file.

		Returns:
		    dict: Dictionary mapping test_name to accuracy (0-1)
		"""
		report_path = output_dir / f"{model_name}_report.json"
		if report_path.exists():
			with open(report_path, "r", encoding="utf-8") as f:
				report = json.load(f)

			# Extract accuracies from all_metrics
			all_metrics = report.get("all_metrics", {})
			test_accuracies = {}
			for test_name, test_metrics in all_metrics.items():
				metric_score_str = test_metrics.get("metrics", {}).get("metric_score", "0%")
				if isinstance(metric_score_str, str) and metric_score_str.endswith("%"):
					accuracy = float(metric_score_str.rstrip("%")) / 100
				else:
					accuracy = float(metric_score_str) if metric_score_str else 0.0
				test_accuracies[test_name] = accuracy
			return test_accuracies

		# no metrics found, return empty dict
		console.print(f"[yellow]Report file not found: {report_path}. Cannot load accuracies.[/yellow]")
		return {}

	def _make_cost_curves(self, output_dir: Path):
		"""
		Generates plot of cost curve and data table for each autograder model.
		Supports both single-judge and multi-judge modes.
		Produces a heatmap with tests on the y-axis and models on the x-axis.
		Also creates a cost vs accuracy scatter plot.
		"""
		if self.evaluation_config.is_multi_judge_mode():
			self._make_cost_curves_multi_judge(output_dir)
		else:
			self._make_cost_curves_single_judge(output_dir)

	def _make_cost_curves_single_judge(self, output_dir: Path):
		"""
		Generates cost curves for single-judge mode.
		"""
		console.print(f"Making cost curves for {self.evaluation_config.autograder_model_name}...")

		judge = LLMClient(self.evaluation_config.autograder_model_config)
		model = judge.config.model
		model_name = self.evaluation_config.autograder_model_name.replace("/", "_").replace(":", "_")
		extension = self.evaluation_config.output_file_format

		test_costs = {}  # {test_name: total_cost}
		test_accuracies = self._load_accuracies(output_dir, model_name)

		# Get test names from the accuracies (which come from the report JSON)
		test_names = list(test_accuracies.keys())

		# Load saved evaluation result files to calculate costs
		for test_name in test_names:
			print("test_name: ", test_name)
			filename = f"{test_name}_results_{model_name}.{extension}"
			path = output_dir / filename

			if not path.exists():
				console.print(f"[yellow]Evaluation results file not found: {filename}. Skipping.[/yellow]")
				continue

			# Load the graded results - check file extension
			if extension == "csv":
				graded_df = pd.read_csv(path, index_col=0)
			elif extension == "xlsx":
				graded_df = pd.read_excel(path)
			else:
				console.print(f"[yellow]Unsupported file extension: {extension}. Skipping.[/yellow]")
				continue

			if graded_df.empty:
				continue

			# Calculate total cost for this test
			test_costs[test_name] = self._calculate_test_cost(graded_df, judge, model, test_name)

		# Create cost vs accuracy scatter plot and save data to CSV
		if test_costs and test_accuracies:
			print("test_costs: ", test_costs)
			print("test_accuracies: ", test_accuracies)
			self._save_cost_data_to_csv(test_costs, test_accuracies, output_dir, model_name)
			self._plot_cost_vs_accuracy(test_costs, test_accuracies, output_dir, model_name)

	def _make_cost_curves_multi_judge(self, output_dir: Path):
		"""
		Generates cost curves for multi-judge mode.
		Reads from majority_vote_report.json and generates cost curves for each judge.
		"""
		console.print("Making cost curves for multi-judge mode...")

		# Load majority_vote_report.json
		report_path = output_dir / "majority_vote_report.json"
		if not report_path.exists():
			console.print(f"[yellow]majority_vote_report.json not found: {report_path}. Skipping cost curves.[/yellow]")
			return

		with open(report_path, "r", encoding="utf-8") as f:
			report = json.load(f)

		all_metrics = report.get("all_metrics", {})
		if not all_metrics:
			console.print("[yellow]No metrics found in majority_vote_report.json. Skipping cost curves.[/yellow]")
			return

		# Parse metric keys to extract {test_name}_{judge_name} pairs
		# Group by judge_name to get all tests per judge
		judge_metrics = {}  # {judge_name: {test_name: accuracy}}
		judge_configs = {}  # {judge_name: JudgeConfig}

		# Build a mapping of judge_name -> JudgeConfig
		if self.evaluation_config.judges:
			for judge_config in self.evaluation_config.judges:
				judge_configs[judge_config.name] = judge_config

		# Parse metrics and group by judge
		for metric_key, metric_data in all_metrics.items():
			# Skip majority_vote metrics (they don't have individual judge results)
			if metric_key.endswith("_majority_vote"):
				continue

			# Parse {test_name}_{judge_name} format
			# Find the last underscore to split test_name and judge_name
			parts = metric_key.rsplit("_", 1)
			if len(parts) != 2:
				console.print(f"[yellow]Unexpected metric key format: {metric_key}. Skipping.[/yellow]")
				continue

			test_name, judge_name = parts

			# Extract accuracy from metric
			metric_score_str = metric_data.get("metrics", {}).get("metric_score", "0%")
			if isinstance(metric_score_str, str) and metric_score_str.endswith("%"):
				accuracy = float(metric_score_str.rstrip("%")) / 100
			else:
				accuracy = float(metric_score_str) if metric_score_str else 0.0

			if judge_name not in judge_metrics:
				judge_metrics[judge_name] = {}
			judge_metrics[judge_name][test_name] = accuracy

		# Process each judge
		all_judge_data = {}  # {judge_name: {test_costs: dict, test_accuracies: dict}}
		extension = self.evaluation_config.output_file_format

		for judge_name, test_accuracies in judge_metrics.items():
			if judge_name not in judge_configs:
				console.print(f"[yellow]Judge config not found for '{judge_name}'. Skipping cost calculation.[/yellow]")
				continue

			judge_config = judge_configs[judge_name]
			# Create LLMClientConfig for this judge
			base_config = self.evaluation_config.autograder_model_config
			llm_client_config = LLMClientConfig(
				model=judge_config.model,
				template=judge_config.template or self.evaluation_config.template,
				default_params=base_config.default_params.copy() if base_config else {},
				temperature=base_config.temperature if base_config else 0.0,
				test_debug_mode=base_config.test_debug_mode if base_config else False,
				rate_limit=base_config.rate_limit.copy() if base_config else {},
				retries=base_config.retries if base_config else 3,
				max_tokens=base_config.max_tokens if base_config else 1200,
			)

			judge = LLMClient(llm_client_config)
			model = judge.config.model
			model_name = model.replace("/", "_").replace(":", "_")

			console.print(f"  Processing judge: {judge_name} (model: {model})")

			test_costs = {}
			test_names = list(test_accuracies.keys())

			# Load saved evaluation result files to calculate costs
			# In multi-judge mode, files are named: {test_name}_results_{judge_name}_{model_name}.{extension}
			for test_name in test_names:
				filename = f"{test_name}_results_{judge_name}_{model_name}.{extension}"
				path = output_dir / filename

				if not path.exists():
					console.print(f"    [yellow]Evaluation results file not found: {filename}. Skipping.[/yellow]")
					continue

				# Load the graded results
				if extension == "csv":
					graded_df = pd.read_csv(path, index_col=0)
				else:
					graded_df = pd.read_excel(path)

				if graded_df.empty:
					continue

				# Calculate total cost for this test
				test_costs[test_name] = self._calculate_test_cost(graded_df, judge, model)

			# Store data for this judge
			if test_costs:
				all_judge_data[judge_name] = {
					"test_costs": test_costs,
					"test_accuracies": test_accuracies,
					"model_name": model_name,
				}

				# Generate individual cost curves for this judge
				self._save_cost_data_to_csv(test_costs, test_accuracies, output_dir, model_name)
				self._plot_cost_vs_accuracy(test_costs, test_accuracies, output_dir, model_name)

		# Generate combined comparison plot
		if len(all_judge_data) > 1:
			self._plot_cost_vs_accuracy_combined(all_judge_data, output_dir)

	def _save_cost_data_to_csv(self, test_costs: dict, test_accuracies: dict, output_dir: Path, model_name: str):
		"""
		Saves cost and accuracy data to a CSV file.

		Args:
		    test_costs: Dictionary mapping test_name to total cost in USD
		    test_accuracies: Dictionary mapping test_name to accuracy (0-1)
		    output_dir: Directory to save the CSV
		    model_name: Name of the model for the filename
		"""
		data = []
		for test_name in test_costs.keys():
			cost = test_costs[test_name]
			accuracy = test_accuracies.get(test_name, 0.0)

			# Calculate efficiency metrics
			# Cost per accuracy point: cost per 1% accuracy point
			# accuracy is stored as decimal (0-1), so multiply by 100 to get percentage
			cost_per_accuracy_point = cost / (accuracy * 100) if accuracy > 0 else float("inf")

			data.append(
				{
					"test_name": test_name,
					"total_cost_usd": cost,
					"accuracy": accuracy,
					"cost_per_accuracy_point": cost_per_accuracy_point,
				}
			)

		df = pd.DataFrame(data)
		csv_path = output_dir / f"{model_name}_token_costs.csv"
		df.to_csv(csv_path, index=False)
		console.print(f"Token costs data saved to {csv_path}")

	def _plot_cost_vs_accuracy(self, test_costs: dict, test_accuracies: dict, output_dir: Path, model_name: str):
		"""
		Creates a scatter plot of cost vs accuracy.

		Args:
		    test_costs: Dictionary mapping test_name to total cost in USD
		    test_accuracies: Dictionary mapping test_name to accuracy (0-1)
		    output_dir: Directory to save the plot
		    model_name: Name of the model for the plot title
		"""
		# Prepare data for plotting
		test_names = list(test_costs.keys())
		costs = [test_costs[name] for name in test_names]
		accuracies = [test_accuracies.get(name, 0.0) for name in test_names]

		# Create scatter plot
		plt.figure(figsize=(10, 6))
		plt.scatter(costs, accuracies, alpha=0.6, s=100)

		# Add test name labels
		for i, test_name in enumerate(test_names):
			plt.annotate(test_name, (costs[i], accuracies[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)

		plt.xlabel("Total Cost (USD)", fontsize=12)
		plt.ylabel("Model Accuracy", fontsize=12)
		plt.title(f"Cost vs Accuracy: {model_name}", fontsize=14)
		plt.grid(True, alpha=0.3)
		plt.tight_layout()

		# Save plot
		plot_path = output_dir / f"cost_vs_accuracy_{model_name}.png"
		plt.savefig(plot_path, dpi=300)
		plt.close()

		console.print(f"✅ Cost vs accuracy plot saved to {plot_path}")

	def _plot_cost_vs_accuracy_combined(self, all_judge_data: dict, output_dir: Path):
		"""
		Creates a combined scatter plot comparing cost vs accuracy for all judges.

		Args:
		    all_judge_data: Dictionary mapping judge_name to {test_costs, test_accuracies, model_name}
		    output_dir: Directory to save the plot
		"""
		plt.figure(figsize=(12, 8))

		# Plot each judge with different colors/markers
		colors = plt.cm.tab10(range(len(all_judge_data)))
		markers = ["o", "s", "D", "v", "^", "<", ">", "p", "*", "h"]

		for idx, (judge_name, judge_data) in enumerate(all_judge_data.items()):
			test_costs = judge_data["test_costs"]
			test_accuracies = judge_data["test_accuracies"]

			test_names = list(test_costs.keys())
			costs = [test_costs[name] for name in test_names]
			accuracies = [test_accuracies.get(name, 0.0) for name in test_names]

			plt.scatter(
				costs,
				accuracies,
				alpha=0.6,
				s=100,
				label=judge_name,
				color=colors[idx],
				marker=markers[idx % len(markers)],
			)

			# Add test name labels (only for first judge to avoid clutter)
			if idx == 0:
				for i, test_name in enumerate(test_names):
					plt.annotate(
						test_name,
						(costs[i], accuracies[i]),
						xytext=(5, 5),
						textcoords="offset points",
						fontsize=7,
						alpha=0.7,
					)

		plt.xlabel("Total Cost (USD)", fontsize=12)
		plt.ylabel("Model Accuracy", fontsize=12)
		plt.title("Cost vs Accuracy: All Judges Comparison", fontsize=14)
		plt.legend(loc="best", fontsize=10)
		plt.grid(True, alpha=0.3)
		plt.tight_layout()

		# Save plot
		plot_path = output_dir / "cost_vs_accuracy_combined.png"
		plt.savefig(plot_path, dpi=300)
		plt.close()

		console.print(f"✅ Combined cost vs accuracy plot saved to {plot_path}")

	def generate_report(self, all_metrics: dict):
		"""Generates a JSON report of the results."""
		output_dir = Path(self.admin_config.output_dir)

		# Determine model name and report identifier
		# In multi-judge mode with aggregation, use aggregation method name
		# Otherwise, use autograder_model_name
		if self.evaluation_config.is_multi_judge_mode() and self.evaluation_config.aggregation_method:
			model_name = self.evaluation_config.aggregation_method
			autograder_model = f"{self.evaluation_config.aggregation_method} (aggregated)"
		else:
			model_name = self.evaluation_config.autograder_model_name.replace("/", "_")
			autograder_model = self.evaluation_config.autograder_model_name

		report_content = {
			"meta_data": {
				"module_name": self.admin_config.module_name,
				"dataset": output_dir / self.admin_config.dataset_config.dataset_name,
				"autograder_model": autograder_model,
			},
			"all_metrics": all_metrics,
		}

		if all_metrics:
			model_name = self.evaluation_config.autograder_model_name.replace("/", "_").replace(":", "_")
			report_path = output_dir / f"{model_name}_report.json"
			with open(report_path, "w", encoding="utf-8") as f:
				json.dump(report_content, f, indent=2, default=str)

		if self.evaluation_config.get_cost_curves:
			self._make_cost_curves(output_dir)
