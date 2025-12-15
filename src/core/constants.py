# src/core/constants.py

from rich.console import Console
from typing import Type
from pydantic import BaseModel

from schemas import BasicLLMResponseInt, BasicLLMResponseBool, BasicLLMResponseStr


console = Console()

VALID_TEMPLATES = [
	"single_autograder",
	"single_judge",
	"single_judge_str",
	"multiclass_judge",
	"synthetic/basic_perturbation",
	"synthetic/standard_generation",
	"synthetic/standard_validation",
	"agent_judge",
	"agent_autograder",
]


def get_response_schema(template: str) -> Type[BaseModel]:
	if template not in VALID_TEMPLATES:
		return BasicLLMResponseStr
	response_schema_map = {
		"single_judge": BasicLLMResponseBool,
		"single_autograder": BasicLLMResponseInt,
		"agent_autograder": BasicLLMResponseInt,
		"agent_judge": BasicLLMResponseInt,
		"multiclass_judge": BasicLLMResponseStr,
		"single_judge_str": BasicLLMResponseStr,
	}
	if template in response_schema_map:
		return response_schema_map[template]
	return BasicLLMResponseStr
