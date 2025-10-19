"""
Structured output helpers for model predictions.

These utilities provide a single place to parse model responses into the
competition's canonical schema using Pydantic validation.  They are used by
the evaluation framework to normalise predictions and to record clean
reasoning traces without depending on brittle ad-hoc JSON parsing.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


def _strip_or_empty(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip()


class PredictionPayload(BaseModel):
    """Canonical schema for model responses."""

    open_ended_answer: str = Field(
        default="",
        description="Detailed explanation or final free-text answer.",
    )
    choice: Optional[str] = Field(
        default=None,
        description="Multiple choice letter (A-D) when applicable.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Concise clinical reasoning supporting the answer.",
    )
    confidence: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Self-reported confidence score from 0 to 100.",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @field_validator("open_ended_answer", mode="before")
    @classmethod
    def _coerce_answer(cls, value: Any) -> str:
        return _strip_or_empty(value)

    @field_validator("choice", mode="before")
    @classmethod
    def _normalise_choice(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        candidate = _strip_or_empty(value).upper()
        if candidate in {"A", "B", "C", "D"}:
            return candidate
        # Treat explicit null markers as missing data.
        if candidate in {"", "NONE", "NULL", "NOTAVALUE", "NOT_APPLICABLE"}:
            return ""
        return ""

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = _strip_or_empty(value)
        return text or None

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            num = int(float(value))
        except (TypeError, ValueError):
            return None
        return max(0, min(100, num))

    def ensure_defaults(self, *, fallback_reasoning: str) -> "PredictionPayload":
        """Return a copy with guaranteed non-empty reasoning/value defaults."""
        copy = self.model_copy()
        if not copy.reasoning:
            copy.reasoning = fallback_reasoning
        if copy.confidence is None:
            copy.confidence = 60
        return copy

    def to_prediction_dict(self, question_type: str) -> dict[str, Any]:
        """Convert to the dictionary shape used by the submission code."""
        data = self.model_dump()
        choice = data.get("choice") or ""
        if question_type == "open_ended":
            data["choice"] = "NOTAVALUE"
        else:
            data["choice"] = choice or ""
        return {
            "choice": data["choice"],
            "open_ended_answer": data.get("open_ended_answer", ""),
        }


class StructuredTrace(BaseModel):
    """Reasoning trace persisted alongside each prediction."""

    open_ended_answer: str = Field(default="")
    choice: str = Field(default="")
    reasoning: str = Field(default="")
    confidence: int = Field(default=60, ge=0, le=100)
    parsing_strategy: str = Field(default="json")
    source_label: str = Field(default="response")
    raw_response_excerpt: str = Field(default="")
    errors: Optional[List[str]] = Field(default=None)
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


def _iter_json_candidates(text: str) -> Iterable[str]:
    """Yield JSON substrings that might decode successfully."""
    if not text:
        return

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        yield stripped

    fence_tag = "```"
    if fence_tag in text:
        parts = text.split(fence_tag)
        for idx in range(1, len(parts), 2):
            candidate = parts[idx].strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate:
                yield candidate

    # Balanced brace extraction
    depth = 0
    start = None
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : idx + 1]
                    start = None


def parse_prediction_response(
    raw_text: str,
    question_type: str,
    *,
    fallback_reasoning: str = "Clinical decision based on multi-agent analysis",
    source_label: str = "response",
) -> Tuple[PredictionPayload, StructuredTrace]:
    """
    Parse a raw model response into PredictionPayload + StructuredTrace.

    The parser first attempts to validate any JSON-looking sections using
    Pydantic.  If all attempts fail we fall back to heuristic parsing so that
    evaluation can continue without dropping predictions entirely.
    """
    errors: List[str] = []
    candidates = list(_iter_json_candidates(raw_text))

    for candidate in candidates:
        try:
            payload = PredictionPayload.model_validate_json(candidate)
        except ValidationError as exc:
            errors.append(str(exc))
            continue
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        else:
            payload = payload.ensure_defaults(fallback_reasoning=fallback_reasoning)
            choice = payload.choice or ""
            trace = StructuredTrace(
                open_ended_answer=payload.open_ended_answer,
                choice=choice if question_type != "open_ended" else "NOTAVALUE",
                reasoning=payload.reasoning or fallback_reasoning,
                confidence=payload.confidence or 60,
                parsing_strategy="json",
                source_label=source_label,
                raw_response_excerpt=raw_text.strip()[:500],
            )
            return payload, trace

    # JSON parsing failed - use heuristic fallback
    choice = ""
    if question_type in {"multi_choice", "open_ended_multi_choice"}:
        choice = _heuristic_extract_choice(raw_text)

    payload = PredictionPayload(
        open_ended_answer=_strip_or_empty(raw_text),
        choice=choice or "",
        reasoning=fallback_reasoning,
        confidence=55,
    ).ensure_defaults(fallback_reasoning=fallback_reasoning)

    trace = StructuredTrace(
        open_ended_answer=payload.open_ended_answer,
        choice=payload.choice or "",
        reasoning=payload.reasoning or fallback_reasoning,
        confidence=payload.confidence or 55,
        parsing_strategy="fallback",
        source_label=source_label,
        raw_response_excerpt=_strip_or_empty(raw_text)[:500],
        errors=errors or None,
    )
    return payload, trace


def _heuristic_extract_choice(text: str) -> str:
    if not text:
        return ""
    upper = text.upper()
    for marker in ("CHOICE:", "ANSWER:", "THE ANSWER IS"):
        if marker in upper:
            idx = upper.index(marker) + len(marker)
            remainder = upper[idx : idx + 10]
            for char in remainder:
                if char in "ABCD":
                    return char
    for char in upper[:2]:
        if char in "ABCD":
            return char
    return ""
