import re
from pathlib import Path
from typing import Optional

from .constants import console

_BASE_PROMPT_DIR = Path(__file__).parent


def _normalize_template_path(prompt_name: str) -> Path:
	"""Return the template path fragment, ensuring a ``.md`` suffix."""
	candidate = Path(prompt_name)
	if candidate.suffix:
		return candidate
	return candidate.with_suffix(".md")


def _ordered_candidates(prompt_name: str, custom_prompts_dir: str | None) -> list[Path]:
	"""Build the ordered list of template locations to inspect."""
	template_fragment = _normalize_template_path(prompt_name)
	search_order: list[Path] = []

	if Path(prompt_name).suffix == ".md":
		search_order.append(Path(prompt_name))
	if custom_prompts_dir:
		search_order.append(Path(custom_prompts_dir) / template_fragment)
	if "/" not in prompt_name and template_fragment.parent == Path("."):
		raise FileNotFoundError("Prompt identifier must include a namespace (e.g. 'judge/single_judge').")
	search_order.append(_BASE_PROMPT_DIR / template_fragment)
	search_order.append(_BASE_PROMPT_DIR / template_fragment.name)

	return search_order


def _resolve_template_path(prompt_name: str, custom_prompts_dir: str | None) -> Path:
	"""Resolve a prompt identifier to a concrete template path."""
	for candidate in _ordered_candidates(prompt_name, custom_prompts_dir):
		if candidate.exists():
			return candidate

	normalized = _normalize_template_path(prompt_name)
	raise FileNotFoundError(f"Template '{prompt_name}' not found (looked for {normalized.name})")


def load_prompt_template(prompt_name: str, custom_prompts_dir: str | None = "./prompts/templates") -> str:
	"""Return the raw template text without performing variable substitution."""
	template_path = _resolve_template_path(prompt_name, custom_prompts_dir)
	return template_path.read_text(encoding="utf-8")


def get_prompt(prompt_name: str, custom_prompts_dir: str | None = "./prompts/templates", **variables) -> Optional[str]:
	"""Get a prompt template and substitute variables."""
	template = load_prompt_template(prompt_name, custom_prompts_dir).strip()

	if template.startswith("---\n"):
		parts = template.split("---\n", 2)
		if len(parts) >= 3:
			template = parts[2].strip()

	template_vars = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_.]*)\}", template))

	missing_vars = template_vars - set(variables.keys())

	if missing_vars:
		console.print(f"[red] [WARNING] Missing required variables for '{prompt_name}': {missing_vars}[/]")

	result = template
	for var_name, value in variables.items():
		if var_name in template_vars:
			result = result.replace(f"{{{var_name}}}", str(value))

	return result
