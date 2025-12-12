# src/core/schemas/review_server.py

from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class ReviewItem(BaseModel):
	"""Serializable item presented to the reviewer with original context and output."""

	sample_id: str
	test_name: str
	source_type: str = "synthetic_generation"
	original_request: Optional[str] = None
	original_response: Optional[str] = None
	generated_response: Union[str, int, bool]
	expected_label: Optional[Union[int, str, bool]] = None
	metadata: Dict[str, Any] = Field(default_factory=dict)


class ReviewDecision(BaseModel):
	"""Reviewer-submitted decision payload with optional edited fields."""

	sample_id: str
	status: str  # accepted | rejected | edited
	edited_response: Optional[str] = None
	edited_expected: Optional[Union[int, str, bool]] = None
	reviewer_id: Optional[str] = None
	notes: Optional[str] = None
