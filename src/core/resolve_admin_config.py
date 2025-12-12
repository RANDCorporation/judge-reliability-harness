# src/core/load_configs.py

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from schemas import (
	AdminConfig,
	DatasetConfig,
	EvaluationConfig,
	LLMClientConfig,
	PerturbationConfig,
)


def _fill_defaults(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
	"""Recursively fills in default values into config."""
	config = deepcopy(config)
	for key, default_value in default_config.items():
		if key not in config:
			config[key] = default_value
		elif isinstance(config[key], dict) and isinstance(default_value, dict):
			config[key] = _fill_defaults(config[key], default_value)
	return config


def _load_dataset_config(admin_section: Dict[str, Any], base_dir: Path) -> DatasetConfig:
	"""Loads the dataset configuration."""
	ds_vars = admin_section["dataset_config"].copy()
	ds_vars["dataset_path"] = (base_dir / ds_vars["dataset_name"]).resolve()
	default_params = ds_vars.get("default_params", {}).copy()

	for var_name, file_name in ds_vars.get("default_params_path", {}).items():
		var_path = base_dir / file_name
		if var_path.exists():
			default_params[var_name] = var_path.read_text(encoding="utf-8").strip()
		else:
			print(f"[WARN] Missing parameter file: {var_path}")

	ds_vars["default_params"] = default_params
	return DatasetConfig(**ds_vars)


def _load_perturbation_config(
	admin_section: Dict[str, Any], output_dir: Path, output_file_format: str
) -> PerturbationConfig:
	"""Loads the perturbation configuration."""
	pert_vars = admin_section["perturbation_config"].copy()
	pert_vars["output_dir"] = output_dir
	pert_vars["output_file_format"] = output_file_format

	if pert_vars.get("use_HITL_process"):
		review_log_dir = output_dir / "review"
		review_log_dir.mkdir(parents=True, exist_ok=True)
		pert_vars["review_log_dir"] = review_log_dir

	return PerturbationConfig(**pert_vars)


def _load_evaluation_config(
	admin_section: Dict[str, Any], default_params: Dict[str, Any], output_dir: Path, output_file_format: str
) -> EvaluationConfig:
	"""Loads the evaluation configuration."""
	eval_vars = admin_section["evaluation_config"].copy()
	llm_client_config = LLMClientConfig(
		model=eval_vars["autograder_model_name"],
		template=eval_vars["template"],
		default_params=default_params,
		test_debug_mode=admin_section["test_debug_mode"],
	)
	eval_vars["output_dir"] = output_dir
	eval_vars["autograder_model_config"] = llm_client_config
	eval_vars["output_file_format"] = output_file_format
	return EvaluationConfig(**eval_vars)


def _load_admin_config(admin_section: Dict[str, Any]) -> AdminConfig:
	"""
	Gets admin_config from base_config.
	"""
	base_module_name = admin_section["module_name"]
	time_stamp = admin_section.get("time_stamp")
	if time_stamp is None:
		time_stamp = time.strftime("%Y%m%d_%H%M")
	else:
		# Ensure time_stamp is always a string (YAML may parse numbers as int)
		time_stamp = str(time_stamp)
	full_module_name = f"{base_module_name}_{time_stamp}"
	output_dir = Path(f"./outputs/{full_module_name}").resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	base_dir = Path(f"./inputs/data/{base_module_name}")

	output_file_format = admin_section.get("output_file_format")
	if not output_file_format:
		if base_module_name.lower().startswith("stratus"):
			output_file_format = "xlsx"
		else:
			output_file_format = "csv"

	dataset_config = _load_dataset_config(admin_section, base_dir)
	perturbation_config = _load_perturbation_config(admin_section, output_dir, output_file_format)
	evaluation_config = _load_evaluation_config(
		admin_section, dataset_config.default_params, output_dir, output_file_format
	)

	return AdminConfig(
		base_module_name=base_module_name,
		module_name=full_module_name,
		time_stamp=time_stamp,
		test_debug_mode=admin_section.get("test_debug_mode", True),
		output_dir=output_dir,
		output_file_format=output_file_format,
		dataset_config=dataset_config,
		perturbation_config=perturbation_config,
		evaluation_config=evaluation_config,
	)


def _load_yaml_config(path: Path) -> dict:
	if not path.exists():
		raise FileNotFoundError(f"Config not found at: {path}")
	with path.open("r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def get_validated_admin_config(
	base_config: Dict[str, Any], default_config_path: Path = Path("./src/configs/default_config.yml")
) -> Tuple[AdminConfig, Dict[str, Any]]:
	"""
	Loads and validates a configuration, filling in defaults and ensuring test configs are properly set up.
	"""
	default_config = _load_yaml_config(default_config_path)
	merged_config = _fill_defaults(base_config, default_config)
	admin_config = _load_admin_config(merged_config["admin"])

	return (admin_config, merged_config)
