# src/harness.py

import uuid

import litellm
import pandas as pd
import truststore

from core import console
from reliability_tests import EvalMetrics, Evaluator
from schemas import AdminConfig
from synthetic_data_pipeline import SyntheticDataAdapter


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

		evaluated_dataset = evaluated_dataset.drop(columns=["reasoning"], errors="ignore").rename(
			columns={"score": "expected"}
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
		tests_to_evaluate = self.admin_config.evaluation_config.tests_to_evaluate or synthetic_tests_to_run
		console.print(f"\n[bold]Step 3: Evaluating {len(tests_to_evaluate)} Test(s)[/bold]")
		evaluated_datasets = {}
		for test_name in tests_to_evaluate:
			perturbed_df = perturbed_datasets.get(test_name)
			if perturbed_df is None or perturbed_df.empty:
				perturbed_df = self.synthetic_data_adapter.load_perturbations(test_name)
				console.print("Loaded perturbed of length: ", len(perturbed_df))
			graded_df = self.evaluator.grade_synthetic_data(test_name, perturbed_df)
			evaluated_datasets[test_name] = graded_df
			self.evaluator.save_evaluated_output(test_name, graded_df)

		# Step 4: Metrics and Report
		console.print("\n[bold]Step 4: Collecting Metrics and Generating Report[/bold]")
		all_metrics = {}
		for test_name, df in evaluated_datasets.items():
			metrics = self.eval_metrics.calculate_metrics(df)
			all_metrics[test_name] = metrics | self.synthetic_data_adapter.get_meta_data(test_name)

		self.eval_metrics.generate_report(all_metrics)
