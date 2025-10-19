"""Core tool interface definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolRequest:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    name: str
    ok: bool
    content: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolProvider:
    """Abstract base for providers; subclasses should override execute."""

    provider_name: str = "base"

    def execute(self, request: ToolRequest) -> ToolResult:  # pragma: no cover - override required
        raise NotImplementedError

