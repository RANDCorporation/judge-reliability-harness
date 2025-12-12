# src/reliability_tests/evaluator.py

import pandas as pd

from core import LLMClient, console, get_response_schema, threaded_executor
from schemas import (
	EvaluationConfig,
)


class Evaluator:
	"""
	Evaluator for the Judge Reliability Harness (JRH).
	"""

	def __init__(self, evaluation_config: EvaluationConfig):
		"""
		Initialize a Evaluator instance.

		Args:
		    evaluation_config (EvaluationConfig): Config for this EvaluationService.
		"""
		self.evaluation_config = evaluation_config

	def _grade_single(self, dataset: pd.DataFrame) -> pd.DataFrame:
		"""
		Grade self.test_df using the AIAutograder.

		Args:
		    dataset (pd.DataFrame): Input data.

		Returns:
		    pd.DataFrame: DataFrame augmented with reasoning and score columns.
		"""
		if dataset.empty:
			return pd.DataFrame()

		judge = LLMClient(self.evaluation_config.autograder_model_config)
		template = self.evaluation_config.autograder_model_config.template
		response_schema = get_response_schema(template)

		results = []
		for row, result in threaded_executor(
			dataset, lambda r: judge.call(r, response_schema), self.evaluation_config.max_workers
		):
			augmented_row = {**row, **result.model_dump()}
			results.append(augmented_row)

		return pd.DataFrame(results)

	def grade_synthetic_data(self, test_name: str, perturbed_df: pd.DataFrame) -> pd.DataFrame:
		"""
		Generates evaluation results for a specific test.

		Args:
		    test_name (str): The name of the test to run.
		    perturbed_df (pd.DataFrame): The dataset to generate results on (perturbed dataset).

		Returns:
		    pd.DataFrame: The results dataset. Empty DataFrame if no results generated.
		"""

		if test_name == "stochastic_stability":
			graded = self.grade_original_preprocess(perturbed_df)
			graded["validation_score"] = graded["expected"]
			return graded

		model_name = self.evaluation_config.autograder_model_config.model.replace("/", "_")
		extension = self.evaluation_config.output_file_format
		filename = f"{test_name}_results_{model_name}.{extension}"

		path = self.evaluation_config.output_dir / filename
		if path.exists() and not self.evaluation_config.overwrite_results:
			if extension == "csv":
				cached = pd.read_csv(path, index_col=0)
			else:
				cached = pd.read_excel(path)
		else:
			cached = pd.DataFrame()

		if cached.empty and perturbed_df.empty:
			return pd.DataFrame()

		processed_ids = set(cached["perturbed_idx"]) if "perturbed_idx" in cached else set()
		to_process = perturbed_df[~perturbed_df["perturbed_idx"].isin(processed_ids)]

		# Run evaluation
		if not to_process.empty:
			if test_name in {"agent_perturbation", "agent_positives"}:
				to_process = to_process.copy()
				if "rubric" not in to_process.columns and "rubric_text" in to_process.columns:
					to_process["rubric"] = to_process["rubric_text"]
				to_process["transcript"] = to_process.get("generation_response", "")
				if "score_levels_table" not in to_process.columns:
					to_process["score_levels_table"] = ""
			to_process = to_process.rename(columns={"original_request": "request", "generation_response": "response"})
			new_output = self._grade_single(to_process)
			new_output = new_output.rename(
				columns={
					"request": "original_request",
					"response": "generation_response",
					"score": "autograder_score",
					"reasoning": "autograder_reasoning",
				}
			)
			combined = pd.concat([cached, new_output], ignore_index=True)
		else:
			combined = cached

		if combined.empty:
			console.print(f"[yellow]Results for {test_name} returned no data.[/yellow]")

		return combined

	def grade_original_preprocess(self, original_df: pd.DataFrame) -> pd.DataFrame:
		"""
		Generates evaluation results as a preprocessing step of the original dataframe.

		Args:
		    original_df (pd.DataFrame): The dataset to generate results on (original dataset).

		Returns:
		    pd.DataFrame: The results dataset. Empty DataFrame if no results generated.
		"""
		if original_df.empty:
			console.print("[yellow]Cannot evaluate dataset because it is empty. Skipping. [/]")
			return pd.DataFrame()

		new_output = self._grade_single(original_df)
		new_output = new_output.rename(
			columns={
				"score": "autograder_score",
				"reasoning": "autograder_reasoning",
			}
		)
		return new_output

	def save_evaluated_output(self, test_name: str, graded_df: pd.DataFrame):
		"""
		Saves results of graded synthetic data in new file.
		"""
		if graded_df.empty:
			console.print(f"[yellow]No graded rows to save for {test_name}. Skipping save.[/]")
			return

		model_name = self.evaluation_config.autograder_model_config.model.replace("/", "_")
		extension = self.evaluation_config.output_file_format
		filename = f"{test_name}_results_{model_name}.{extension}"
		path = self.evaluation_config.output_dir / filename

		if path.exists() and not self.evaluation_config.overwrite_results:
			console.print(f"[cyan]Updating {filename} with newly graded rows while preserving existing data.[/cyan]")

		if extension == "csv":
			graded_df.to_csv(path)
		else:
			graded_df.to_excel(path, index=False)
