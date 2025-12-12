# src/utils.py

from pathlib import Path
import yaml


def save_config(output_dir: Path, file_stem: str, config: dict):
	"""
	Save a generic config to output dir.
	"""
	config_path = output_dir / file_stem
	try:
		with open(config_path, "w", encoding="utf-8") as f:
			yaml.dump(config, f, default_flow_style=False, sort_keys=False)
	except Exception as e:
		print(f"Failed to save config: {e}")
