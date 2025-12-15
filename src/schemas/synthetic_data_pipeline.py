# src/core/schemas/synthetic_data_pipeline.py

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class OriginalDataPointConfig(BaseModel):
	# Allows and ignores extra fields
	model_config = ConfigDict(extra="ignore")

	original_idx: str = Field(..., description="index of data point from original dataset")
	request: str = Field(..., description="request")
	response: str = Field(..., description="response")
	expected: Union[int, str] = Field(..., description="expected")


class SavedItem(BaseModel):
	"""Config object that defines how synthetic data will be saved in CSV files."""

	test_name: str = Field(..., description="test name")
	# original data
	original_request: str = Field(..., description="source question")
	original_response: Optional[str] = Field(None, description="source response or transcript")
	original_idx: str = Field(..., description="source item id")
	original_expected: Union[int, str] = Field(..., description="source gold answer")
	# generation data
	generation_prompt: Optional[str] = Field(None, description="Prompt/transcript shown to the generator.")
	generation_response: Union[str, int, bool] = Field(..., description="generation response")
	generation_completion: Optional[Union[str, int, bool]] = Field(None, description="Raw completion when available.")
	generation_reasoning: Optional[str] = Field(None, description="generation reasoning")
	perturbed_idx: str = Field(..., description="perturbation id")
	generation_temp: Optional[float] = Field(None, description="temperature")
	generation_mode: Optional[str] = Field(
		None, description="Agentic generation mode, e.g., 'perturbation' or 'positive'."
	)
	# validation data
	validation_score: Union[str, int, bool] = Field(..., description="validation score")
	validation_reasoning: Optional[str] = Field(None, description="validation reasoning")
	validated_bucket: Optional[str] = Field(None, description="Bucket/rubric id associated with validation.")
	rubric_id: Optional[str] = Field(None, description="Identifier for agent judge/autograder rubric rows.")
	rubric_text: Optional[str] = Field(None, description="Full rubric text associated with the item.")
	score_levels_table: Optional[str] = Field(None, description="Formatted ordinal score descriptors.")
	transcript: Optional[str] = Field(None, description="Full transcript evaluated by the agent judge/autograder.")
	# synthetic_ordinal / metadata
	prompted_bucket: Optional[str] = Field(None, description="prompted bucket")
	# HTIL part
	human_reviewed: Optional[bool] = Field(False, description="Whether item has been human reviewed yet or not.")


class BasicLLMResponseInt(BaseModel):
	"""The Pydantic schema for the LLM's response."""

	score: int = Field(..., description="The LLM score or response.")
	reasoning: str = Field(..., description="Reasoning behind the LLM response.")


class BasicLLMResponseBool(BaseModel):
	"""The Pydantic schema for the LLM's response."""

	score: bool = Field(..., description="The LLM score or response.")
	reasoning: str = Field(..., description="Reasoning behind the LLM response.")


class BasicLLMResponseStr(BaseModel):
	"""The Pydantic schema for the LLM's response."""

	score: str = Field(..., description="The LLM score or response.")
	reasoning: str = Field(..., description="Reasoning behind the LLM response.")
