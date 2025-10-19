import json
import logging
import sys
import os
from typing import Any, Dict, List, Tuple

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.retry import retry_with_exponential_backoff

from tools.providers import (
    NoOpProvider,
    ToolRequest,
    ToolResult,
    ToolUniverseProvider,
    TxAgentProvider,
)

logger = logging.getLogger(__name__)


class MultiAgentModel:
    """Multi-agent therapeutic reasoning orchestrator with tool support."""

    ROLE_ORDER = ["planner", "router", "workers", "verifier", "decider"]

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.roles: Dict[str, Any] = {}
        self.runtime: Dict[str, Any] = {}
        self.prompts: Dict[str, str] = {}
        self.step_budget: int = 6
        self.tool_budget: int = 0
        self.self_consistency_k: int = 1
        self.default_temperature: float = 0.3
        self.default_top_p: float = 0.9
        self.role_temperatures: Dict[str, float] = {}
        self.tool_providers: List[Any] = []
        self.trace: List[Dict[str, Any]] = []
        self.telemetry: Dict[str, Any] = {}
        self.last_telemetry: Dict[str, Any] = {}

    def load(self, **kwargs):
        config = kwargs.get("config", {}) or {}
        self.runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
        roles_cfg = self.runtime.get("agent_roles", {})
        self.default_temperature = float(self.runtime.get("temperature", 0.3))
        self.default_top_p = float(self.runtime.get("top_p", 0.9))
        self.self_consistency_k = int(self.runtime.get("self_consistency_k", 1))
        self.step_budget = int(self.runtime.get("step_budget", 6))
        self.tool_budget = int(self.runtime.get("tool_budget", 0))
        self.role_temperatures = self.runtime.get("role_temperatures", {}) or {}

        default_model_name = roles_cfg.get("default", self.model_name or "gpt-5")

        def create_model(name: str):
            model_id = name or default_model_name
            lower = model_id.lower()
            try:
                if "gemini" in lower:
                    from models.gemini_model import GeminiModel

                    mdl = GeminiModel(model_id)
                    mdl.load()
                    return mdl
                from eval_framework import ChatGPTModel  # Lazy import avoids circular deps

                mdl = ChatGPTModel(model_id)
                mdl.load()
                return mdl
            except Exception as exc:  # pragma: no cover - runtime configuration failure
                raise RuntimeError(f"Failed to initialise model '{model_id}': {exc}") from exc

        for role in self.ROLE_ORDER:
            role_model_name = roles_cfg.get(role, default_model_name)
            self.roles[role] = create_model(role_model_name)

        self.prompts = {
            "planner": self._read_prompt_file(
                "prompts/system_planner.txt",
                "Decompose the user’s therapeutic question into clinical sub-goals and a short plan.",
            ),
            "router": self._read_prompt_file(
                "prompts/system_router.txt",
                "Respond with JSON listing tool tasks prioritising FDA labels, then guidelines, then literature.",
            ),
            "worker": self._read_prompt_file(
                "prompts/system_worker.txt",
                "Summarise evidence from tools focusing on dosing, safety, interactions, and population adjustments.",
            ),
            "verifier": self._read_prompt_file(
                "prompts/system_verifier.txt",
                "Check contradictions across the proposed evidence. Enforce hierarchy: label > guideline > RCT/SR > primary.",
            ),
            "decider": self._read_prompt_file(
                "prompts/system_decider.txt",
                "Issue a final recommendation with safety emphasis and cite strongest evidence.",
            ),
        }

        self.tool_providers = self._initialise_tool_providers(self.runtime.get("tools", {}))

    def _initialise_tool_providers(self, tools_cfg: Dict[str, Any]) -> List[Any]:
        enabled = tools_cfg.get("enabled", False)
        providers = tools_cfg.get("providers", []) or []
        if not enabled or not providers:
            return [NoOpProvider()]

        instantiated = []
        for name in providers:
            lower = name.lower()
            if lower == "txagent":
                instantiated.append(TxAgentProvider(
                    base_url=tools_cfg.get("txagent_endpoint"),
                    api_key=tools_cfg.get("txagent_api_key"),
                    timeout=int(tools_cfg.get("timeout", 20)),
                ))
            elif lower == "tooluniverse":
                instantiated.append(ToolUniverseProvider(
                    base_url=tools_cfg.get("tooluniverse_endpoint"),
                    timeout=int(tools_cfg.get("timeout", 20)),
                ))
            else:
                instantiated.append(NoOpProvider())
        return instantiated or [NoOpProvider()]

    def _read_prompt_file(self, path: str, default: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except Exception:  # pragma: no cover - prompt missing
            return default

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def inference(self, prompt: str, max_tokens: int = 1024, **kwargs) -> Tuple[str, List[Dict]]:
        # Accept kwargs for compatibility
        _ = kwargs
        self.trace = []
        self.telemetry = {
            "tool_calls": [],
            "steps": [],
            "self_consistency": self.self_consistency_k,
            "step_budget": self.step_budget,
        }

        planner_text = self._planner_step(prompt)
        tasks, _ = self._router_step(prompt, planner_text)
        tool_results = self._execute_tools(tasks)
        evidence_text = self._worker_step(prompt, planner_text, tool_results)
        verifier_text = self._verifier_step(prompt, planner_text, evidence_text, tool_results)
        final_text = self._decider_step(
            prompt,
            evidence_text,
            verifier_text,
            tool_results,
            max_tokens=max_tokens,
        )
        self.last_telemetry = self.telemetry

        return final_text.strip(), self.trace

    # ------------------------------------------------------------------
    # Individual stages
    # ------------------------------------------------------------------

    def _planner_step(self, question: str) -> str:
        planner_prompt = f"{self.prompts['planner']}\n\nQuestion: {question}\nPlan:"
        response = self._invoke_role("planner", planner_prompt, max_tokens=512)
        return response

    def _router_step(self, question: str, plan: str) -> Tuple[List[Dict[str, Any]], str]:
        router_prompt = (
            f"{self.prompts['router']}\n\nClinical question: {question}\nPlan: {plan}\n"
            "Respond with JSON as specified."
        )
        router_text = self._invoke_role("router", router_prompt, max_tokens=400)

        # First discover relevant tools if ToolUniverse is available
        tasks = self._parse_router_output(router_text)

        # If we have tool discovery capability, enhance tasks with actual tool names
        if self.tool_budget > 0 and tasks:
            enhanced_tasks = self._enhance_tasks_with_tool_discovery(tasks)
            if enhanced_tasks:
                tasks = enhanced_tasks

        self.telemetry["steps"].append({"stage": "router", "tasks": tasks})
        return tasks, router_text

    def _worker_step(self, question: str, plan: str, tool_results: List[Dict[str, Any]]) -> str:
        evidence_block = self._format_tool_results(tool_results)
        worker_prompt = (
            f"{self.prompts['worker']}\n\nClinical question: {question}\nPlan: {plan}\n"
            f"Tool evidence:\n{evidence_block}\nSummary:" 
        )
        return self._invoke_role("workers", worker_prompt, max_tokens=700)

    def _verifier_step(
        self,
        question: str,
        plan: str,
        evidence: str,
        tool_results: List[Dict[str, Any]],
    ) -> str:
        evidence_block = self._format_tool_results(tool_results)
        verifier_prompt = (
            f"{self.prompts['verifier']}\n\nQuestion: {question}\nPlan: {plan}\n"
            f"Tool evidence:\n{evidence_block}\nWorker summary: {evidence}\nVerdict:"
        )
        return self._invoke_role("verifier", verifier_prompt, max_tokens=400)

    def _decider_step(
        self,
        question: str,
        evidence: str,
        verifier: str,
        tool_results: List[Dict[str, Any]],
        max_tokens: int,
    ) -> str:
        evidence_block = self._format_tool_results(tool_results)
        answers = []
        raw_runs = []

        for run in range(max(1, self.self_consistency_k)):
            temperature = self._role_temperature("decider", jitter=True if run else False)
            decider_prompt = (
                f"{self.prompts['decider']}\n\nQuestion: {question}\n"
                f"Evidence summary: {evidence}\nVerifier notes: {verifier}\n"
                f"Tool evidence:\n{evidence_block}\nFinal answer:"
            )
            reply = self._invoke_role(
                "decider",
                decider_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            raw_runs.append(reply)
            answers.append(self._extract_choice(reply))

        final_answer = self._aggregate_answers(raw_runs, answers)
        self.telemetry["steps"].append({
            "stage": "decider",
            "runs": raw_runs,
            "choices": answers,
            "final": final_answer,
        })
        return final_answer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @retry_with_exponential_backoff(
        max_retries=5,
        initial_sleep=2.0,
        max_sleep=60.0,
        exponential_base=2.0,
    )
    def _invoke_model_with_retry(self, model, prompt: str, generation_args: Dict[str, Any]) -> Tuple[str, Any]:
        """Invoke model with retry logic for transient errors."""
        return model.inference(prompt, **generation_args)

    def _invoke_role(
        self,
        role: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        model = self.roles[role]
        model_name = getattr(model, "model_name", "")
        generation_args: Dict[str, Any] = {}

        # Detect if this is a Gemini model
        is_gemini = self._is_gemini_model(model)
        is_reasoning = self._is_reasoning_model_name(model_name)

        if is_reasoning and not is_gemini:
            # GPT-5 reasoning model parameters
            generation_args["reasoning_effort"] = self.runtime.get("reasoning_effort", "high")
            if self.runtime.get("verbosity"):
                generation_args["verbosity"] = self.runtime["verbosity"]
            generation_args["max_completion_tokens"] = int(self.runtime.get("max_completion_tokens", max_tokens))
            # Don't pass max_tokens for reasoning models
        elif is_gemini:
            # Gemini model parameters
            generation_args["temperature"] = (
                temperature if temperature is not None else self._role_temperature(role)
            )
            generation_args["top_p"] = top_p if top_p is not None else self.default_top_p
            generation_args["max_tokens"] = max_tokens
            # Gemini doesn't support reasoning_effort or verbosity
        else:
            # Standard GPT model parameters
            generation_args["temperature"] = (
                temperature if temperature is not None else self._role_temperature(role)
            )
            generation_args["top_p"] = top_p if top_p is not None else self.default_top_p
            generation_args["max_tokens"] = max_tokens

        # Call model with retry logic
        try:
            response, _ = self._invoke_model_with_retry(model, prompt, generation_args)
        except Exception as e:
            logger.error(f"Failed to invoke {role} after all retries: {e}")
            # Return error response instead of raising to allow pipeline to continue
            response = f"Error in {role}: {str(e)}"

        self.trace.append({
            "role": role,
            "prompt": prompt,
            "response": response,
            "metadata": generation_args,
        })
        return response

    def _role_temperature(self, role: str, jitter: bool = False) -> float:
        base = float(self.role_temperatures.get(role, self.default_temperature))
        if jitter:
            return min(1.0, base + 0.1)
        return base

    @staticmethod
    def _is_reasoning_model_name(name: str) -> bool:
        if not name:
            return False
        lower = name.lower()
        markers = ["gpt-5", "o1", "o3"]
        if any(marker in lower for marker in markers):
            if "gpt-5" in lower and "chat" in lower and "reasoning" not in lower:
                return False
            return True
        return False

    @staticmethod
    def _is_gemini_model(model) -> bool:
        """Check if a model is a Gemini model."""
        if hasattr(model, "__class__"):
            class_name = model.__class__.__name__
            if "GeminiModel" in class_name:
                return True
        if hasattr(model, "model_name"):
            model_name = str(getattr(model, "model_name", "")).lower()
            if "gemini" in model_name:
                return True
        return False

    def _parse_router_output(self, text: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(self._extract_json(text))
            if isinstance(data, dict) and isinstance(data.get("tasks"), list):
                tasks = [task for task in data["tasks"] if isinstance(task, dict)]
                return tasks[: max(0, self.tool_budget)]
        except Exception:
            pass

        # Fallback: simple heuristic parsing
        tasks = []
        for line in text.splitlines():
            if ":" in line:
                parts = line.split(":", 1)
                tool = parts[0].strip().lower()
                query = parts[1].strip()
                if tool:
                    tasks.append({"tool": tool, "query": query, "category": "unknown"})
        return tasks[: max(0, self.tool_budget)]

    def _enhance_tasks_with_tool_discovery(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance tasks with actual tool discovery using ToolUniverse."""
        if not self.tool_providers:
            return tasks

        enhanced_tasks = []
        for task in tasks:
            query = task.get("query", "")
            category = task.get("category", "unknown")

            # Try to find relevant tools based on the query
            provider = self._select_provider_for_discovery()
            if provider and hasattr(provider, "find_tools"):
                try:
                    # Use the query to find relevant tools
                    tool_result = provider.find_tools(query, limit=3)
                    if tool_result.ok and tool_result.content:
                        # Extract tool names from the discovery result
                        discovered_tools = self._extract_tool_names_from_discovery(tool_result.content)
                        if discovered_tools:
                            # Create multiple tasks for the discovered tools
                            for tool_name in discovered_tools[:2]:  # Limit to 2 tools per query
                                enhanced_tasks.append({
                                    "tool": tool_name,
                                    "query": query,
                                    "category": category,
                                    "original_intent": task.get("tool", "")
                                })
                        else:
                            # Keep original task if no tools discovered
                            enhanced_tasks.append(task)
                    else:
                        # Keep original task if discovery failed
                        enhanced_tasks.append(task)
                except Exception as e:
                    logger.debug("Tool discovery failed for query '%s': %s", query, e)
                    enhanced_tasks.append(task)
            else:
                enhanced_tasks.append(task)

        return enhanced_tasks[: max(0, self.tool_budget)]

    def _select_provider_for_discovery(self):
        """Select a provider that supports tool discovery (ToolUniverse)."""
        for provider in self.tool_providers:
            if hasattr(provider, "find_tools"):
                return provider
        return None

    def _extract_tool_names_from_discovery(self, discovery_result) -> List[str]:
        """Extract tool names from ToolUniverse discovery result."""
        tool_names = []
        if isinstance(discovery_result, dict):
            # Handle different possible response formats
            if "tools" in discovery_result:
                tools = discovery_result["tools"]
                if isinstance(tools, list):
                    for tool in tools:
                        if isinstance(tool, dict) and "name" in tool:
                            tool_names.append(tool["name"])
                        elif isinstance(tool, str):
                            tool_names.append(tool)
            elif "results" in discovery_result:
                # Alternative format
                results = discovery_result["results"]
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict) and "name" in result:
                            tool_names.append(result["name"])
        elif isinstance(discovery_result, list):
            # Direct list of tools
            for item in discovery_result:
                if isinstance(item, dict) and "name" in item:
                    tool_names.append(item["name"])
                elif isinstance(item, str):
                    tool_names.append(item)
        return tool_names

    def _extract_json(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        raise ValueError("No JSON object detected")

    def _execute_tools(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not tasks:
            logger.debug("No tasks provided for tool execution")
            return results

        if not self.tool_providers:
            logger.warning("No tool providers configured")
            return results

        if self.tool_budget <= 0:
            logger.warning("Tool budget is %d, cannot execute tools", self.tool_budget)
            return results

        logger.info("Executing %d tools (budget: %d)", min(len(tasks), self.tool_budget), self.tool_budget)

        for task in tasks[: self.tool_budget]:
            provider = self._select_provider(task)
            request = ToolRequest(
                name=task.get("tool", "unknown"),
                params={"query": task.get("query")},
                metadata={"category": task.get("category")},
            )

            logger.debug("Executing tool '%s' with query: %s", request.name, request.params.get("query"))

            try:
                outcome: ToolResult = provider.execute(request)
                result_entry = {
                    "tool": request.name,
                    "query": request.params.get("query"),
                    "provider": getattr(provider, "provider_name", "unknown"),
                    "ok": outcome.ok,
                    "content": outcome.content,
                    "error": outcome.error,
                }

                if outcome.ok:
                    logger.debug("Tool '%s' executed successfully", request.name)
                else:
                    logger.warning("Tool '%s' failed: %s", request.name, outcome.error)

            except Exception as e:
                logger.error("Exception while executing tool '%s': %s", request.name, e)
                result_entry = {
                    "tool": request.name,
                    "query": request.params.get("query"),
                    "provider": getattr(provider, "provider_name", "unknown"),
                    "ok": False,
                    "content": None,
                    "error": str(e),
                }

            self.telemetry["tool_calls"].append(result_entry)
            results.append(result_entry)

        logger.info("Completed tool execution: %d successful, %d failed",
                   sum(1 for r in results if r["ok"]),
                   sum(1 for r in results if not r["ok"]))

        return results

    def _select_provider(self, task: Dict[str, Any]):
        # Currently returns first provider, but task parameter is kept for future routing logic
        _ = task  # Mark as intentionally unused
        if not self.tool_providers:
            return NoOpProvider()
        return self.tool_providers[0]

    def get_telemetry(self) -> Dict[str, Any]:
        return self.last_telemetry

    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "- No external tools executed."
        formatted = []
        for item in results:
            status = "OK" if item.get("ok") else f"ERROR: {item.get('error')}"
            content = item.get("content")
            if isinstance(content, dict):
                content_string = json.dumps(content)
            else:
                content_string = str(content)
            formatted.append(
                f"- Tool={item.get('tool')} Provider={item.get('provider')} Status={status} Query={item.get('query')}\n  {content_string}"
            )
        return "\n".join(formatted)

    def _extract_choice(self, response: str) -> str:
        if not response:
            return ""
        text = response.strip().upper()
        if text and text[0] in "ABCDE":
            return text[0]
        for marker in ["ANSWER IS", "ANSWER:", "CHOICE", "OPTION"]:
            if marker in text:
                idx = text.index(marker) + len(marker)
                remainder = text[idx:].strip()
                if remainder and remainder[0] in "ABCDE":
                    return remainder[0]
        return ""

    def _aggregate_answers(self, raw_runs: List[str], choices: List[str]) -> str:
        votes: Dict[str, int] = {}
        for choice in choices:
            if choice:
                votes[choice] = votes.get(choice, 0) + 1
        if votes:
            winner = max(votes.items(), key=lambda kv: kv[1])[0]
            # Return the run corresponding to winner if available
            for run, choice in zip(raw_runs, choices):
                if choice == winner:
                    return run
            return winner
        # Fallback to last run
        return raw_runs[-1] if raw_runs else ""


class DummyModel:
    """A deterministic, offline-safe model for dry runs."""

    def __init__(self, *args, **kwargs):
        # Accept any arguments for compatibility
        _ = (args, kwargs)
        pass

    def load(self, **kwargs):
        # Accept any arguments for compatibility
        _ = kwargs
        pass

    def inference(self, prompt: str, **kwargs) -> Tuple[str, List[Dict]]:
        # Accept any arguments for compatibility
        _ = kwargs
        text = ""
        up = (prompt or "").lower()
        if "answer with only the letter" in up or "multi-choice answer" in up or "final choice" in up:
            text = "A"
        elif "open-ended question" in up:
            text = "This is a placeholder open-ended answer focusing on safety, dosing, and contraindications."
        else:
            text = "A"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
        ]
        return text, messages
