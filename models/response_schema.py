"""
Pydantic models for structured JSON output from LLMs.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ModelResponse(BaseModel):
    """Schema for model responses in JSON format."""

    open_ended_answer: str = Field(
        ...,
        description="Full detailed explanation or answer to the medical question"
    )

    choice: Optional[str] = Field(
        None,
        description="Letter choice (A, B, C, D, or E) for multiple choice questions"
    )

    brief: str = Field(
        ...,
        description="Brief summary of the answer in 1-2 sentences"
    )

    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence score from 0 to 100"
    )

    @field_validator('choice')
    def validate_choice(cls, v):
        """Ensure choice is a valid letter A-E if provided."""
        if v is not None and v not in ['A', 'B', 'C', 'D']:
            # Try to extract a valid letter if present
            for letter in ['A', 'B', 'C', 'D']:
                if letter in v.upper():
                    return letter
            # If no valid letter found, return None
            return None
        return v

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "open_ended_answer": "Celecoxib is metabolized by CYP2C9, so it should be avoided in poor metabolizers.",
                "choice": "D",
                "brief": "Poor CYP2C9 metabolizers should avoid celecoxib.",
                "confidence": 90
            }
        }