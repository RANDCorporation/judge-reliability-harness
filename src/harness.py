# src/harness.py

import uuid

import litellm
import pandas as pd
import truststore
from typing import Any, Dict, Optional

from core import console
from reliability_tests import EvalMetrics, Evaluator
from schemas import AdminConfig
from synthetic_data_pipeline import SyntheticDataAdapter


def _get_majority_vote(row: pd.Series, judge_scores: Dict[str, Dict[str, Any]]) -> Any:
	"""
	Calculate majority vote from judge scores for a single row.

	Args:
	    row: DataFrame row containing judge score columns
	    judge_scores: Dictionary mapping judge_name -> {perturbed_idx -> score}

	Returns:
	    Majority vote result (most common score value)
	"""
	scores = []
	for judge_name in judge_scores.keys():
		score_col = f"{judge_name}_score"
		if score_col in row and pd.notna(row[score_col]):
			scores.append(row[score_col])

	if not scores:
		return None

	# Count occurrences of each score
	from collections import Counter

	counts = Counter(scores)
	# Return the most common score; if tie, return the first one encountered
	return counts.most_common(1)[0][0]


def get_aggregation_function(method: str):
	"""
	Get aggregation function for the specified method.

	Args:
	    method: Aggregation method name (e.g., 'majority_vote')

	Returns:
	    Aggregation function or None if method not supported
	"""
	if method == "majority_vote":
		return _get_majority_vote
	return None


class Harness:
	"""
	Executes the full Judge Reliability Harness (JRH) pipeline:
	- Loads the dataset
	- Runs each configured reliability test (perturb + grade)
	- Saves outputs and reports performance
	"""

	def __init__(self, admin_config: AdminConfig, synthetic_data_adapter: SyntheticDataAdapter):
		"""
		Initialize the Harness with configuration and prepare key variables.

		Args:
		    config: A AdminConfig object containing configuration settings.
		"""
		self.admin_config = admin_config
		self.synthetic_data_adapter = synthetic_data_adapter
		self.evaluator = Evaluator(self.admin_config.evaluation_config)
		self.eval_metrics = EvalMetrics(self.admin_config)

		# lite llm ignore ssl warnings
		truststore.inject_into_ssl()
		litellm.ssl_verify = False
		litellm.verbose = False
		litellm.enable_json_schema_validation = True

	def _load_and_preprocess_data(self) -> pd.DataFrame:
		"""
		Pre-processes the original dataset by renaming external column names to internal names
		based on configuration. Allows 'expected' column to be optional.
		"""
		dataset_path = self.admin_config.dataset_config.dataset_path
		suffix = dataset_path.suffix.lower()
		if suffix in {".xlsx", ".xls"}:
			reader = pd.read_excel
		elif suffix == ".csv":
			reader = pd.read_csv
		else:
			configured_format = (self.admin_config.output_file_format or "").lower()
			reader = pd.read_excel if configured_format == "xlsx" else pd.read_csv

		try:
			original_dataset = reader(dataset_path)
		except FileNotFoundError:
			console.print("[yellow]Dataset file not found![/yellow]")
			return pd.DataFrame()
		except Exception as exc:
			console.print(f"[yellow]Dataset could not be read ({exc}). Proceeding without dataset.[/yellow]")
			return pd.DataFrame()

		if "original_idx" not in original_dataset.columns:
			original_dataset["original_idx"] = [f"seed_{uuid.uuid4().hex[:8]}" for _ in range(len(original_dataset))]

		column_map = self.admin_config.perturbation_config.preprocess_columns_map
		required_keys = {k for k in column_map if k != "expected"}
		missing_columns = {column_map[k] for k in required_keys if column_map[k] not in original_dataset.columns}
		if missing_columns:
			console.print(f"[yellow]Dataset missing required preprocess values: {', '.join(missing_columns)}[/yellow]")
			return pd.DataFrame()

		# Only relabel columns that exist in the dataset
		relabel_mapping = {v: k for k, v in column_map.items() if v in original_dataset.columns}
		return original_dataset.rename(columns=relabel_mapping)

	def _run_eval_against_original_data(self, original_dataset: pd.DataFrame) -> pd.DataFrame:
		"""
		Evaluates the original dataset using aiautograder if an 'expected' column is not present,
		and if configuration allows using model-generated scores as expected values.
		"""
		if original_dataset.empty:
			return pd.DataFrame()

		if "expected" in original_dataset.columns or not self.admin_config.dataset_config.use_original_data_as_expected:
			return original_dataset

		evaluated_dataset = self.evaluator.grade_original_preprocess(original_dataset)

		if evaluated_dataset.empty:
			console.print(
				"[WARNING]: Evaluation of original dataset failed. No 'expected' column in original dataset found."
			)
			return original_dataset

		evaluated_dataset = evaluated_dataset.drop(columns=["autograder_reasoning"], errors="ignore").rename(
			columns={"autograder_score": "expected"}
		)

		# Save preprocessed dataset
		input_path = self.admin_config.dataset_config.dataset_path
		file_format = self.admin_config.output_file_format
		suffix = ".xlsx" if file_format == "xlsx" else ".csv"
		output_path = input_path.with_name(input_path.stem + "_preprocessed" + suffix)
		if file_format == "xlsx":
			evaluated_dataset.to_excel(output_path, index=False)
		else:
			evaluated_dataset.to_csv(output_path, index=False)
		console.print(
			f"[yellow]Saved preprocessed dataset to: {output_path}.\n[bold]NOTE:[/bold] Make sure to update dataset name when running JRH again.[/yellow]"
		)

		return evaluated_dataset

	def _validate_aggregation_prerequisites(
		self, test_name: str, judge_results: Dict[str, pd.DataFrame]
	) -> tuple[Optional[str], bool]:
		"""
		Validates that current setup allows for aggregation.

		Returns:
		    Tuple of (aggregation_method, is_valid). Returns (None, False) if validation fails.
		"""
		if not self.admin_config.evaluation_config.is_multi_judge_mode():
			return None, False

		aggregation_method = self.admin_config.evaluation_config.aggregation_method
		if not aggregation_method:
			return None, False

		if len(judge_results) < 2:
			console.print(
				f"[yellow]Warning: Need at least 2 judges for aggregation. Skipping for {test_name}.[/yellow]"
			)
			return None, False

		return aggregation_method, True

	def _get_base_dataframe(self, test_name: str, judge_results: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
		"""
		Finds and validates the base DataFrame from judge results.

		Returns:
		    Base DataFrame or None if no valid base found.
		"""
		# Use the first judge with non-empty results as the base (they all have the same perturbed_idx)
		base_df = None
		for judge_name, df in judge_results.items():
			# Ensure df is a DataFrame, not a Series
			if not isinstance(df, pd.DataFrame):
				if isinstance(df, pd.Series):
					df = df.to_frame().T  # Convert Series to DataFrame
				else:
					console.print(
						f"[yellow]Warning: Judge '{judge_name}' result is not a DataFrame or Series. Skipping.[/yellow]"
					)
					continue

			if not df.empty and "perturbed_idx" in df.columns:
				base_df = df.copy()
				break

		if base_df is None or base_df.empty:
			console.print(
				f"[yellow]Warning: No valid judge results found for aggregation. Skipping for {test_name}.[/yellow]"
			)
			return None

		if "perturbed_idx" not in base_df.columns:
			console.print(
				f"[yellow]Warning: Missing 'perturbed_idx' column for majority vote. Skipping for {test_name}.[/yellow]"
			)
			return None

		return base_df

	def _clean_judge_vote_columns(self, base_df: pd.DataFrame) -> pd.DataFrame:
		"""
		Removes judge vote columns from base_df if in single-judge mode.

		In multi-judge mode, these columns are kept for reference.
		"""
		essential_columns = {
			"perturbed_idx",
			"original_expected",
			"expected",
			"autograder_score",
			"sample_id",
			"original_idx",
		}

		# Remove judge vote columns in single-judge mode but keep them in multi-judge mode for reference.
		if not self.admin_config.evaluation_config.is_multi_judge_mode():
			# Remove judge vote columns from original dataset based on each judge's expected column configuration
			# These are the CSV column names specified in each judge's preprocess_columns_map.expected
			judge_vote_columns = set()
			for judge in self.admin_config.evaluation_config.judges:
				expected_csv_col = judge.preprocess_columns_map.get("expected")
				if (
					expected_csv_col
					and expected_csv_col in base_df.columns
					and expected_csv_col not in essential_columns
				):
					judge_vote_columns.add(expected_csv_col)

			if judge_vote_columns:
				base_df = base_df.drop(columns=judge_vote_columns)
				console.print(
					f"[yellow]Removed original dataset vote columns from base_df: {judge_vote_columns}[/yellow]"
				)

		return base_df

	def _collect_judge_scores(
		self, test_name: str, judge_results: Dict[str, pd.DataFrame]
	) -> Optional[Dict[str, Dict[str, Any]]]:
		"""
		Collects scores from all judges into a dictionary mapping judge_name -> {perturbed_idx -> score}.

		Returns:
		    Dictionary of judge scores or None if insufficient valid judges.
		"""
		judge_scores = {}
		for judge_name, df in judge_results.items():
			# Ensure df is a DataFrame, not a Series
			if not isinstance(df, pd.DataFrame):
				if isinstance(df, pd.Series):
					df = df.to_frame().T  # Convert Series to DataFrame
				else:
					console.print(
						f"[yellow]Warning: Judge '{judge_name}' result is not a DataFrame or Series. Skipping.[/yellow]"
					)
					continue

			if df.empty or "autograder_score" not in df.columns or "perturbed_idx" not in df.columns:
				console.print(
					f"[yellow]Warning: Judge '{judge_name}' missing required columns. Skipping for majority vote.[/yellow]"
				)
				continue

			# Create a mapping of perturbed_idx -> autograder_score for this judge
			judge_scores[judge_name] = df.set_index("perturbed_idx")["autograder_score"].to_dict()

		if len(judge_scores) < 2:
			console.print(
				f"[yellow]Warning: Not enough valid judges for aggregation. Skipping for {test_name}.[/yellow]"
			)
			return None

		return judge_scores

	def _apply_aggregation_to_dataframe(
		self, base_df: pd.DataFrame, aggregation_method: str, judge_scores: Dict[str, Dict[str, Any]]
	) -> Optional[pd.DataFrame]:
		"""
		Applies aggregation function to create aggregated scores in the DataFrame.

		Returns:
		    DataFrame with aggregation applied, or None if aggregation function not found.
		"""
		# Get aggregation function for the specified method
		aggregation_fn = get_aggregation_function(aggregation_method)
		if not aggregation_fn:
			console.print(
				f"[yellow]Warning: Unknown aggregation method '{aggregation_method}'. Skipping aggregation.[/yellow]"
			)
			return None

		# Add individual judge scores as columns
		for judge_name in judge_scores.keys():
			base_df[f"{judge_name}_score"] = base_df["perturbed_idx"].map(judge_scores[judge_name].get)

		# Calculate aggregation based on method
		aggregation_column_name = f"{aggregation_method}_score"  # e.g., "majority_vote_score"

		# Currently, we only have get_majority_vote function. See reliability_tests/aggregation.py for more details.
		base_df[aggregation_column_name] = base_df.apply(lambda row: aggregation_fn(row, judge_scores), axis=1)

		return base_df

	def _merge_and_compare_reference_column(
		self, base_df: pd.DataFrame, original_dataset: pd.DataFrame, aggregation_column_name: str
	) -> pd.DataFrame:
		"""
		Merges reference column from original dataset and compares with aggregated result.

		Returns:
		    DataFrame with aggregation_match column added if reference column exists.
		"""
		reference_column = self.admin_config.evaluation_config.aggregation_reference_column

		if not reference_column:
			return base_df

		# Use the preprocessed original_dataset that was passed in (which contains original_idx)
		# instead of re-reading from disk. This addresses the issue where re-reading the raw CSV
		# would lose the synthetic original_idx column created during preprocessing.
		if original_dataset.empty or reference_column not in original_dataset.columns:
			console.print(f"[yellow]Warning: Column '{reference_column}' not found in original dataset.[/yellow]")
			return base_df

		# Find a common column to merge on between base_df and original_dataset
		merge_key = None
		if "sample_id" in base_df.columns and "sample_id" in original_dataset.columns:
			merge_key = "sample_id"
		elif "original_idx" in base_df.columns and "original_idx" in original_dataset.columns:
			merge_key = "original_idx"
		else:
			# Not able to merge on a common column.
			console.print(
				"[yellow]Warning: Neither 'sample_id' nor 'original_idx' found in both datasets for aggregation comparison.[/yellow]"
			)
			console.print(f"[yellow]  base_df columns: {list(base_df.columns)}[/yellow]")
			console.print(f"[yellow]  original_dataset columns: {list(original_dataset.columns)}[/yellow]")
			return base_df

		# Merge to get reference column from original dataset
		original_ref_col = f"{reference_column}_original"
		columns_to_merge = original_dataset[[merge_key, reference_column]].rename(
			columns={reference_column: original_ref_col}
		)

		merged = base_df.merge(columns_to_merge, on=merge_key, how="left")

		# Compare calculated aggregation with original dataset reference column
		if original_ref_col in merged.columns:
			# Normalize both columns for comparison (handle case/whitespace differences)
			agg_normalized = merged[aggregation_column_name].astype(str).str.lower().str.strip()
			ref_normalized = merged[original_ref_col].astype(str).str.lower().str.strip()
			merged["aggregation_match"] = agg_normalized == ref_normalized
			return merged

		return base_df

	def _combine_judges_and_aggregate(
		self, test_name: str, judge_results: Dict[str, pd.DataFrame], original_dataset: pd.DataFrame
	) -> pd.DataFrame:
		"""
		Combines multiple judge results into a single DataFrame with aggregation.

		Args:
		    test_name: Name of the test
		    judge_results: Dictionary mapping judge_name to their results DataFrame
		    original_dataset: The preprocessed original dataset (with original_idx).
		        This MUST be the in-memory preprocessed dataset, not re-read from disk,
		        to ensure original_idx is available for merging.

		Returns:
		    Combined DataFrame with aggregated result and comparison columns
		"""
		# Validate prerequisites
		aggregation_method, is_valid = self._validate_aggregation_prerequisites(test_name, judge_results)
		if not is_valid:
			return pd.DataFrame()

		# Get base DataFrame
		base_df = self._get_base_dataframe(test_name, judge_results)
		if base_df is None:
			return pd.DataFrame()

		# Clean judge vote columns if needed
		base_df = self._clean_judge_vote_columns(base_df)

		# Collect scores from all judges
		judge_scores = self._collect_judge_scores(test_name, judge_results)
		if judge_scores is None:
			return pd.DataFrame()

		# Apply aggregation
		aggregation_column_name = f"{aggregation_method}_score"
		base_df = self._apply_aggregation_to_dataframe(base_df, aggregation_method, judge_scores)
		if base_df is None:
			return pd.DataFrame()

		# Merge and compare with reference column
		base_df = self._merge_and_compare_reference_column(base_df, original_dataset, aggregation_column_name)

		return base_df

	def run(self) -> None:
		"""
		Executes the full Judge Reliability Harness (JRH) pipeline:
		- Loads the dataset
		- Runs evaluator on original data set to create an 'expected' column, if required by the config.
		- Runs each synthetic data pipeline and saves results
		- Evaluates aiautograder performance
		- Aggregates and reports test results
		"""
		# Step 1: Load and preprocess data
		console.print("\n[bold]Step 1: Loading and Preprocessing original data.[/bold]")
		test_entries = getattr(self.synthetic_data_adapter, "validated_test_list", None)
		if isinstance(test_entries, dict):
			if test_entries:
				requires_dataset = any(getattr(entry, "requires_dataset", True) for entry in test_entries.values())
			else:
				requires_dataset = False
		else:
			requires_dataset = True

		if requires_dataset:
			original_dataset = self._load_and_preprocess_data()
			original_dataset = self._run_eval_against_original_data(original_dataset)
		else:
			original_dataset = pd.DataFrame()
			console.print("[dim]No dataset required by selected tests; skipping dataset load.[/dim]")

		# Step 2: Perturbations
		synthetic_tests_to_run = self.synthetic_data_adapter.get_validated_test_names()
		console.print(f"\n[bold]Step 2: Running {len(synthetic_tests_to_run)} Reliability Test(s)[/bold]")
		perturbed_datasets = {
			test_name: self.synthetic_data_adapter.run(original_dataset, test_name)
			for test_name in synthetic_tests_to_run
		}

		# Step 3: Evaluations
		tests_to_evaluate = self.admin_config.evaluation_config.tests_to_evaluate
		console.print(f"\n[bold]Step 3: Evaluating {len(tests_to_evaluate)} Test(s)[/bold]")

		# Determine if we're in multi-judge mode
		is_multi_judge = self.admin_config.evaluation_config.is_multi_judge_mode()
		# judges = self.admin_config.evaluation_config.judges if is_multi_judge else None

		# Structure: Dict[test_name, DataFrame] for single-judge, Dict[test_name, Dict[judge_name, DataFrame]] for multi-judge
		evaluated_datasets = {}

		for test_name in tests_to_evaluate:
			perturbed_df = perturbed_datasets.get(test_name)
			if perturbed_df is None or perturbed_df.empty:
				perturbed_df = self.synthetic_data_adapter.load_perturbations(test_name)
				console.print("Loaded perturbed of length: ", len(perturbed_df))

			# grade_synthetic_data now returns Dict[str, pd.DataFrame] for both modes
			graded_results = self.evaluator.grade_synthetic_data(test_name, perturbed_df)

			if is_multi_judge:
				# Multi-judge mode: results are Dict[judge_name, DataFrame]
				evaluated_datasets[test_name] = graded_results
			else:
				# Single-judge mode: results are {"single_judge": DataFrame}
				# Extract the DataFrame for backward compatibility
				evaluated_datasets[test_name] = graded_results.get("single_judge", pd.DataFrame())

			# Save results
			self.evaluator.save_evaluated_output(test_name, graded_results)

		# Step 3.5: Combine judges and aggregate (if multi-judge mode)
		aggregation_method = self.admin_config.evaluation_config.aggregation_method
		aggregated_results = {}  # Dict[test_name, DataFrame] for aggregated results
		if self.admin_config.evaluation_config.is_multi_judge_mode() and aggregation_method:
			console.print(f"\n[bold]Step 3.5: Combining Judges and Aggregating ({aggregation_method})[/bold]")
			for test_name, judge_results in evaluated_datasets.items():
				combined_df = self._combine_judges_and_aggregate(test_name, judge_results, original_dataset)
				if not combined_df.empty:
					# Save combined results
					output_path = self.admin_config.output_dir / f"{test_name}_{aggregation_method}.csv"
					combined_df.to_csv(output_path, index=False)
					console.print(f"[green]Saved {aggregation_method} results to: {output_path}[/green]")
					aggregated_results[test_name] = combined_df

		# Step 4: Metrics and Report
		console.print("\n[bold]Step 4: Collecting Metrics and Generating Report[/bold]")
		all_metrics = {}
		for test_name, result in evaluated_datasets.items():
			# Handle both single-judge (DataFrame) and multi-judge (Dict[judge_name, DataFrame]) structures
			if isinstance(result, dict):
				# Multi-judge mode: calculate metrics for each judge
				for judge_name, df in result.items():
					if not df.empty:
						metrics = self.eval_metrics.calculate_metrics(df)
						all_metrics[f"{test_name}_{judge_name}"] = metrics | self.synthetic_data_adapter.get_meta_data(
							test_name
						)
			else:
				# Single-judge mode
				metrics = self.eval_metrics.calculate_metrics(result)
				all_metrics[test_name] = metrics | self.synthetic_data_adapter.get_meta_data(test_name)

		# Calculate metrics for aggregated results (if any)
		if aggregated_results:
			reference_column = self.admin_config.evaluation_config.aggregation_reference_column
			aggregation_column_name = f"{aggregation_method}_score"  # e.g., "majority_vote_score"
			for test_name, agg_df in aggregated_results.items():
				# Prepare DataFrame for metrics calculation
				# Use aggregated_score as 'score' and reference_column_original as 'expected'
				if aggregation_column_name in agg_df.columns:
					# Find the reference column (should be {reference_column}_original after merge)
					ref_col_original = f"{reference_column}_original" if reference_column else None
					if ref_col_original and ref_col_original in agg_df.columns:
						# Select only the columns we need and rename them
						# calculate_metrics expects 'autograder_score' and 'validation_score'
						metrics_df = pd.DataFrame(
							{
								"autograder_score": agg_df[aggregation_column_name],
								"validation_score": agg_df[ref_col_original],
							}
						)
						# Calculate metrics
						agg_metrics = self.eval_metrics.calculate_metrics(metrics_df)
						all_metrics[f"{test_name}_{aggregation_method}"] = agg_metrics

		self.eval_metrics.generate_report(all_metrics)
