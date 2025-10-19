"""Tool provider stubs for ToolUniverse/TxAgent integration."""

from __future__ import annotations

import logging
import os
from typing import Dict

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None

from .interface import ToolProvider, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


class TxAgentProvider(ToolProvider):
    provider_name = "txagent"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: int = 20):
        self.base_url = base_url or os.getenv("TXAGENT_ENDPOINT")
        self.api_key = api_key or os.getenv("TXAGENT_API_KEY")
        self.timeout = timeout

    def execute(self, request: ToolRequest) -> ToolResult:
        if not self.base_url:
            return ToolResult(request.name, False, None, error="TXAgent endpoint not configured")
        if requests is None:
            return ToolResult(request.name, False, None, error="requests library not available")

        url = f"{self.base_url.rstrip('/')}/tool"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "tool_name": request.name,
            "parameters": request.params,
            "metadata": request.metadata,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return ToolResult(request.name, True, data, metadata={"status_code": resp.status_code})
        except Exception as e:  # pragma: no cover - network operations
            logger.warning("TxAgent call failed: %s", e)
            return ToolResult(request.name, False, None, error=str(e))


class ToolUniverseProvider(ToolProvider):
    provider_name = "tooluniverse"

    def __init__(self, base_url: str | None = None, timeout: int = 20):
        self.base_url = base_url or os.getenv("TOOLUNIVERSE_ENDPOINT")
        self.timeout = timeout
        self.tu = None
        self._initialize_sdk()

    def _initialize_sdk(self):
        """Initialize the ToolUniverse SDK if available."""
        try:
            from tooluniverse import ToolUniverse
            self.tu = ToolUniverse()
            self.tu.load_tools()
            logger.info("ToolUniverse SDK initialized successfully with %d tools", len(self.tu.tools) if hasattr(self.tu, 'tools') else 600)
        except ImportError:
            logger.warning("ToolUniverse SDK not installed. Falling back to HTTP mode.")
            self.tu = None
        except Exception as e:
            logger.warning("Failed to initialize ToolUniverse SDK: %s", e)
            self.tu = None

    def execute(self, request: ToolRequest) -> ToolResult:
        # First try to use the SDK if available
        if self.tu is not None:
            return self._execute_with_sdk(request)

        # Fallback to HTTP mode
        return self._execute_with_http(request)

    def _execute_with_sdk(self, request: ToolRequest) -> ToolResult:
        """Execute tool using the ToolUniverse Python SDK."""
        try:
            # Format request according to ToolUniverse SDK specification
            sdk_request = {
                "name": request.name,
                "arguments": request.params
            }

            # Execute using the SDK
            result = self.tu.run(sdk_request)

            # Handle the response
            if result is not None:
                return ToolResult(
                    request.name,
                    True,
                    result,
                    metadata={"provider": "tooluniverse_sdk"}
                )
            else:
                return ToolResult(
                    request.name,
                    False,
                    None,
                    error="Tool returned no result"
                )
        except Exception as e:
            logger.warning("ToolUniverse SDK execution failed: %s", e)
            # Fallback to HTTP if SDK fails
            return self._execute_with_http(request)

    def _execute_with_http(self, request: ToolRequest) -> ToolResult:
        """Execute tool using HTTP API (fallback mode)."""
        if not self.base_url:
            return ToolResult(request.name, False, None, error="ToolUniverse endpoint not configured")
        if requests is None:
            return ToolResult(request.name, False, None, error="requests library not available")

        url = f"{self.base_url.rstrip('/')}/execute"
        payload = {
            "tool": request.name,
            "arguments": request.params,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return ToolResult(request.name, True, resp.json(), metadata={"status_code": resp.status_code, "provider": "tooluniverse_http"})
        except Exception as e:  # pragma: no cover - network operations
            logger.warning("ToolUniverse HTTP call failed: %s", e)
            return ToolResult(request.name, False, None, error=str(e))

    def find_tools(self, description: str, limit: int = 10) -> ToolResult:
        """Find relevant tools based on description."""
        if self.tu is not None:
            try:
                # Use Tool_Finder_Keyword operation
                result = self.tu.run({
                    "name": "Tool_Finder_Keyword",
                    "arguments": {"description": description, "limit": limit}
                })
                return ToolResult("Tool_Finder_Keyword", True, result, metadata={"provider": "tooluniverse_sdk"})
            except Exception as e:
                logger.warning("Tool discovery failed: %s", e)
                return ToolResult("Tool_Finder_Keyword", False, None, error=str(e))
        else:
            return ToolResult("Tool_Finder_Keyword", False, None, error="ToolUniverse SDK not available")


class NoOpProvider(ToolProvider):
    provider_name = "noop"

    def execute(self, request: ToolRequest) -> ToolResult:
        return ToolResult(request.name, True, {"message": "Tool execution skipped (offline mode)"})
