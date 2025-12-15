# src/reliability_tests/evaluator.py

import pandas as pd
from typing import Dict, Optional

from core import LLMClient, console, get_response_schema, threaded_executor
from schemas import (
	EvaluationConfig,
	LLMClientConfig,
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

	def _grade_single(self, dataset: pd.DataFrame, llm_client_config: Optional[LLMClientConfig] = None) -> pd.DataFrame:
		"""
		Grade dataset using the AIAutograder.

		Args:
		    dataset (pd.DataFrame): Input data.
		    llm_client_config (Optional[LLMClientConfig]): LLM client configuration.
		        If None (default for single judge mode), uses self.evaluation_config.autograder_model_config.

		Returns:
		    pd.DataFrame: DataFrame augmented with reasoning and score columns.
		"""
		if dataset.empty:
			return pd.DataFrame()

		## We need to do this for multi-judge mode, where each judge has its own model config.
		if llm_client_config is None:
			llm_client_config = self.evaluation_config.autograder_model_config

		judge = LLMClient(llm_client_config)
		template = llm_client_config.template
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
		    Dict[str, pd.DataFrame]: Dictionary mapping judge name to results dataset.
		                             For single judge mode, key is "single_judge".
		                             For multi-judge mode, keys are judge names.
		"""
		if self.evaluation_config.is_multi_judge_mode():
			return self._grade_multi_judge(test_name, perturbed_df)
		else:
			return self._grade_single_judge(test_name, perturbed_df)

	def _grade_single_judge(self, test_name: str, perturbed_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
		"""Grade using single judge mode (backward compatible)."""
		if test_name == "stochastic_stability":
			graded = self.grade_original_preprocess(perturbed_df)
			graded["validation_score"] = graded["expected"]
			return {"single_judge": graded}

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
			return {"single_judge": pd.DataFrame()}

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

		return {"single_judge": combined}

	def _grade_multi_judge(self, test_name: str, perturbed_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
		"""Grade using multiple judges mode with per-judge column mappings."""
		results = {}

		for judge in self.evaluation_config.judges:
			judge_name = judge.name
			model_name = judge.model.replace("/", "_")
			extension = self.evaluation_config.output_file_format
			filename = f"{test_name}_results_{judge_name}_{model_name}.{extension}"

			path = self.evaluation_config.output_dir / filename
			if path.exists() and not self.evaluation_config.overwrite_results:
				if extension == "csv":
					cached = pd.read_csv(path, index_col=0)
				else:
					cached = pd.read_excel(path)
			else:
				cached = pd.DataFrame()

			if cached.empty and perturbed_df.empty:
				results[judge_name] = pd.DataFrame()
				continue

			# Map columns to internal format (request, response, expected)
			to_process = perturbed_df.copy()
			column_map = judge.preprocess_columns_map or {}
			fallback_map = {
				"request": "original_request",
				"response": "generation_response",
				"expected": "original_expected",
			}

			rename_dict = {}
			for target in ["request", "response", "expected"]:
				# If target already exists, keep it (unless CSV mapping overrides)
				if target in to_process.columns:
					source = column_map.get(target)
					if source and source in to_process.columns and source != target:
						# Save existing target and use CSV column
						rename_dict[target] = f"{target}_backup_{judge_name}"
						rename_dict[source] = target
					continue

				# Try CSV column mapping first
				source = column_map.get(target)
				if source and source in to_process.columns:
					rename_dict[source] = target
				# Fallback to internal column names for perturbed data
				elif target in fallback_map and fallback_map[target] in to_process.columns:
					rename_dict[fallback_map[target]] = target

			if rename_dict:
				to_process = to_process.rename(columns=rename_dict)

			# Ensure required columns exist
			if "request" not in to_process.columns or "response" not in to_process.columns:
				console.print(
					f"[yellow]Warning: Judge '{judge_name}' missing required columns (request/response). Skipping.[/yellow]"
				)
				results[judge_name] = pd.DataFrame()
				continue

			# Handle expected column
			if "expected" not in to_process.columns:
				if "original_expected" in to_process.columns:
					to_process["expected"] = to_process["original_expected"]
				else:
					console.print(
						f"[yellow]Warning: Judge '{judge_name}' missing 'expected' column. Skipping.[/yellow]"
					)
					results[judge_name] = pd.DataFrame()
					continue

			processed_ids = set(cached["perturbed_idx"]) if "perturbed_idx" in cached.columns else set()
			to_grade = to_process[~to_process["perturbed_idx"].isin(processed_ids)].copy()

			# Run evaluation
			if not to_grade.empty:
				if test_name == "agent_perturbation":
					if "rubric" not in to_grade.columns and "rubric_text" in to_grade.columns:
						to_grade["rubric"] = to_grade["rubric_text"]
					if "transcript" not in to_grade.columns:
						to_grade["transcript"] = to_grade.get("generation_response", "")
					if "score_levels_table" not in to_grade.columns:
						to_grade["score_levels_table"] = ""

				# Create LLMClientConfig for this judge
				# Copy base config from autograder_model_config and override with judge-specific values
				base_config = self.evaluation_config.autograder_model_config
				llm_client_config = LLMClientConfig(
					model=judge.model,  # Judge-specific model
					template=judge.template or self.evaluation_config.template,  # Judge-specific template
					default_params=base_config.default_params.copy() if base_config else {},
					temperature=base_config.temperature if base_config else 0.0,
					test_debug_mode=base_config.test_debug_mode
					if base_config
					else False,  # IMPORTANT: Copy from base config
					rate_limit=base_config.rate_limit.copy() if base_config else {},
					retries=base_config.retries if base_config else 3,
					max_tokens=base_config.max_tokens if base_config else 1200,
				)

				new_output = self._grade_single(to_grade, llm_client_config)
				new_output = new_output.rename(
					columns={
						"score": "autograder_score",
						"reasoning": "autograder_reasoning",
					}
				)
				combined = pd.concat([cached, new_output], ignore_index=True)
			else:
				combined = cached

			if combined.empty:
				console.print(f"[yellow]Results for {test_name} (judge: {judge_name}) returned no data.[/yellow]")

			results[judge_name] = combined

		return results

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

	def save_evaluated_output(
		self, test_name: str, graded_results: Dict[str, pd.DataFrame], judge_name: Optional[str] = None
	):
		"""
		Saves results of graded synthetic data in new file.

		Args:
		    test_name (str): The name of the test.
		    graded_results (Dict[str, pd.DataFrame]): Dictionary mapping judge name to results.
		        For single-judge mode, this should be {"single_judge": DataFrame}.
		        For multi-judge mode, keys are judge names.
		    judge_name (Optional[str]): Optional judge name for backward compatibility.
		        If provided and graded_results is a single DataFrame, uses this name.
		"""
		# Handle backward compatibility: if graded_results is a DataFrame, convert to dict
		if isinstance(graded_results, pd.DataFrame):
			if judge_name:
				graded_results = {judge_name: graded_results}
			else:
				graded_results = {"single_judge": graded_results}

		for judge_key, graded_df in graded_results.items():
			if graded_df.empty:
				console.print(f"[yellow]No graded rows to save for {test_name} (judge: {judge_key}). Skipping save.[/]")
				continue

			# Determine model name based on judge
			if judge_key == "single_judge":
				model_name = self.evaluation_config.autograder_model_config.model.replace("/", "_")
				filename = f"{test_name}_results_{model_name}.{self.evaluation_config.output_file_format}"
			else:
				# Find judge config to get model name
				judge = next((j for j in self.evaluation_config.judges if j.name == judge_key), None)
				if judge:
					model_name = judge.model.replace("/", "_")
					filename = (
						f"{test_name}_results_{judge_key}_{model_name}.{self.evaluation_config.output_file_format}"
					)
				else:
					# Fallback: use judge_key as model name
					filename = f"{test_name}_results_{judge_key}.{self.evaluation_config.output_file_format}"

			path = self.evaluation_config.output_dir / filename

			if path.exists() and not self.evaluation_config.overwrite_results:
				console.print(
					f"[cyan]Updating {filename} with newly graded rows while preserving existing data.[/cyan]"
				)

			if self.evaluation_config.output_file_format == "csv":
				graded_df.to_csv(path)
			else:
				graded_df.to_excel(path, index=False)
