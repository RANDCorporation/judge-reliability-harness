# src/synthetic_data_pipeline/registry.py

import json
from pathlib import Path
from typing import Any, Dict

from core import (
	console,
	resolve_agent_perturbation_config,
	resolve_basic_perturbations_config,
	resolve_stochastic_stability_config,
	resolve_synthetic_ordinal_config,
)
from schemas import (
	AdminConfig,
	TestRegistryEntry,
)


def build_registry(admin_config: AdminConfig, base_config: Dict[str, Any]) -> Dict[str, TestRegistryEntry]:
	"""
	Builds registry of perturbation and resolve_config functions for each valid test.
	"""
	prompt_path = Path("./prompts/synthetic_generation_prompts/basic_perturbation_instructions.json")
	if not prompt_path.exists():
		raise FileNotFoundError(f"Missing test instruction file: {prompt_path}")
	with open(prompt_path, "r") as file:
		test_instruction_list = json.load(file)

	registry = {}
	tests_to_run = admin_config.perturbation_config.tests_to_run

	if "stochastic_stability" in tests_to_run and "test_stochastic_stability_config" in base_config.keys():
		registry["stochastic_stability"] = TestRegistryEntry(
			test_name="stochastic_stability",
			config=resolve_stochastic_stability_config(base_config.get("test_stochastic_stability_config", {})),
			description="Score should remain the same under repeated samples and varying keys.",
		)
	if "synthetic_ordinal" in tests_to_run and "synthetic_data_params" in base_config.keys():
		registry["synthetic_ordinal"] = TestRegistryEntry(
			test_name="synthetic_ordinal",
			config=resolve_synthetic_ordinal_config(admin_config, base_config.get("synthetic_data_params", {})),
			description="Judge should match the ordinal target score on synthetic essays, reliability measure",
		)
	if "agent_perturbation" in tests_to_run and "test_agent_perturbation_config" in base_config.keys():
		registry["agent_perturbation"] = TestRegistryEntry(
			test_name="agent_perturbation",
			config=resolve_agent_perturbation_config(
				admin_config,
				base_config.get("test_agent_perturbation_config", {}),
				base_config.get("synthetic_data_params", {}),
			),
			description="Agent transcript edits should trigger rubric violations in autograder evaluation",
			requires_dataset=False,
		)
	if "agent_positives" in tests_to_run:
		# Derive positives config from dedicated block if present, otherwise from perturbation config.
		positives_source = {}
		if "test_agent_positives_config" in base_config.keys():
			positives_source = dict(base_config.get("test_agent_positives_config", {}))
		elif "test_agent_perturbation_config" in base_config.keys():
			positives_source = dict(base_config.get("test_agent_perturbation_config", {}))
		else:
			console.print("[yellow]  - [Warning] 'agent_positives' requested but no config block found; skipping.[/]")
			positives_source = None
		if positives_source is not None:
			# Force positive objective; if no explicit output, set a default agent_positives dir.
			positives_source["objective"] = "pass"
			positives_source.setdefault("pass_required", True)
			output_conf = positives_source.get("output", {})
			if not output_conf or not output_conf.get("dir"):
				output_conf = dict(output_conf) if output_conf else {}
				output_conf.setdefault("dir", "./outputs/agent_positives")
				positives_source["output"] = output_conf
			registry["agent_positives"] = TestRegistryEntry(
				test_name="agent_positives",
				config=resolve_agent_perturbation_config(
					admin_config,
					positives_source,
					base_config.get("synthetic_data_params", {}),
				),
				description="Agent transcript edits should satisfy rubric criteria (positive generation)",
				requires_dataset=False,
			)

	for test_name in tests_to_run:
		if test_name in test_instruction_list["discriminative_tests"]:
			registry[test_name] = TestRegistryEntry(
				test_name=test_name,
				config=resolve_basic_perturbations_config(
					admin_config,
					test_instruction_list["discriminative_tests"][test_name],
					base_config.get("synthetic_data_params", {}),
				),
				description="Score should flip for discriminative tests",
			)
		elif test_name in test_instruction_list["consistency_tests"]:
			registry[test_name] = TestRegistryEntry(
				test_name=test_name,
				config=resolve_basic_perturbations_config(
					admin_config,
					test_instruction_list["consistency_tests"][test_name],
					base_config.get("synthetic_data_params", {}),
				),
				description="Score should remain consistent for consistency tests",
			)
		elif test_name not in ["stochastic_stability", "synthetic_ordinal", "agent_perturbation", "agent_positives"]:
			console.print(f"[yellow]  - [Warning] Unknown test '{test_name}' — skipping.[/]")

	return registry
