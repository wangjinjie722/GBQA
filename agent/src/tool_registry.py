"""Planner-visible tool registry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, List, Optional

from .codebase_types import UniversalCodebaseAdapter
from .environment_clients import CodeToolAdapter, RuntimeLogAdapter
from .log_analyzer import LogAnalyzer
from .types import CapabilityDescriptor, Observation


ToolPayload = Dict[str, Any]
ToolRuntimeContext = Dict[str, Any]
ToolHandler = Callable[[ToolPayload, ToolRuntimeContext], "ToolInvocationResult"]
ToolActionParser = Callable[[str], ToolPayload]


@dataclass
class ToolInvocationResult:
    """Normalized result returned by a registry tool invocation."""

    observation: Observation
    refreshed_capability: Optional[CapabilityDescriptor] = None


@dataclass
class Tool:
    """Describes a planner-visible callable tool."""

    name: str
    description: str
    action_format: str
    handler: ToolHandler
    action_parser: ToolActionParser

    def invoke(
        self,
        payload: ToolPayload,
        runtime_context: ToolRuntimeContext,
    ) -> ToolInvocationResult:
        return self.handler(payload, runtime_context)

    def parse_action(self, action_text: str) -> ToolPayload:
        return self.action_parser(action_text)


class ToolRegistry:
    """Registers planner-visible tools and dispatches invocations."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def parse_action(self, name: str, action_text: str) -> ToolPayload:
        return self._get(name).parse_action(action_text)

    def invoke(
        self,
        name: str,
        payload: ToolPayload,
        runtime_context: ToolRuntimeContext,
    ) -> ToolInvocationResult:
        return self._get(name).invoke(payload, runtime_context)

    def render_prompt_section(self) -> str:
        lines = ["## Available Tools:"]
        for tool in self.list_tools():
            lines.append(
                f"- {tool.name}: {tool.description} Format: `{tool.action_format}`."
            )
        return "\n".join(lines)

    def _get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]


def register_environment_action_tool(
    registry: ToolRegistry,
    handler: ToolHandler,
) -> None:
    """Register the primary environment-action tool."""
    registry.register(
        Tool(
            name="environment_action",
            description=(
                "Execute one semantic environment action through the operator and active execution backend"
            ),
            action_format="semantic action string",
            handler=handler,
            action_parser=lambda action_text: {"action": _require_action(action_text)},
        )
    )


def register_code_tools(
    registry: ToolRegistry,
    adapter: CodeToolAdapter,
    codebase_adapter: Optional[UniversalCodebaseAdapter] = None,
) -> None:
    """Register white-box source-code tools."""
    # Use provided codebase_adapter or fall back to HTTP adapter wrapper
    ca = codebase_adapter or UniversalCodebaseAdapter(api_client=adapter)

    registry.register(
        Tool(
            name="code_list_files",
            description="List available source code files for the current environment",
            action_format="any non-empty text (ignored)",
            handler=lambda payload, runtime: _invoke_code_list(payload, runtime, ca),
            action_parser=lambda _action_text: {},
        )
    )
    registry.register(
        Tool(
            name="code_read_file",
            description="Read a source file, optionally with a line range",
            action_format="path or path:start-end",
            handler=lambda payload, runtime: _invoke_code_read(payload, runtime, ca),
            action_parser=_parse_code_read_action,
        )
    )
    registry.register(
        Tool(
            name="code_search",
            description="Search source code using a regex pattern",
            action_format="pattern",
            handler=lambda payload, runtime: _invoke_code_search(payload, runtime, ca),
            action_parser=lambda action_text: {"pattern": _require_action(action_text)},
        )
    )
    registry.register(
        Tool(
            name="code_write_file",
            description="Modify a source file using JSON payload or path:old->new patch shorthand",
            action_format="JSON string or path:old_text->new_text",
            handler=lambda payload, runtime: _invoke_code_write(payload, runtime, ca),
            action_parser=_parse_code_write_action,
        )
    )
    registry.register(
        Tool(
            name="code_restore_file",
            description="Restore a file previously modified by code_write_file",
            action_format="path",
            handler=lambda payload, runtime: _invoke_code_restore(payload, runtime, ca),
            action_parser=lambda action_text: {"path": _require_action(action_text)},
        )
    )


def register_runtime_log_tool(
    registry: ToolRegistry,
    adapter: RuntimeLogAdapter,
) -> None:
    """Register runtime debug-log access."""
    registry.register(
        Tool(
            name="code_read_debug_logs",
            description=(
                "Read or clear runtime debug logs for the current active environment session; "
                "the session id is inferred automatically"
            ),
            action_format="read or clear",
            handler=lambda payload, runtime: _invoke_runtime_log_tool(
                payload,
                runtime,
                adapter,
            ),
            action_parser=_parse_debug_log_action,
        )
    )


def register_log_analysis_tool(
    registry: ToolRegistry,
    adapter: RuntimeLogAdapter,
    analyzer: LogAnalyzer,
) -> None:
    """Register session-log analysis using the active API-backed session."""
    registry.register(
        Tool(
            name="log_analyze",
            description=(
                "Analyze the current environment session log for anomalies and optionally "
                "show filtered commands"
            ),
            action_format=(
                "analyze, failures, or JSON object with start_turn/end_turn/"
                "failures_only/limit/include_debug_output"
            ),
            handler=lambda payload, runtime: _invoke_log_analysis_tool(
                payload,
                runtime,
                adapter,
                analyzer,
            ),
            action_parser=_parse_log_analysis_action,
        )
    )


def _require_action(action_text: str) -> str:
    text = str(action_text).strip()
    if not text:
        raise ValueError("Planner action must not be empty")
    return text


def _parse_code_read_action(action_text: str) -> ToolPayload:
    text = _require_action(action_text)
    path, separator, line_spec = text.rpartition(":")
    if not separator or "-" not in line_spec:
        return {"path": text}
    start_text, dash, end_text = line_spec.partition("-")
    if not dash or not start_text.isdigit() or not end_text.isdigit():
        return {"path": text}
    return {
        "path": path.strip(),
        "start_line": int(start_text),
        "end_line": int(end_text),
    }


def _parse_code_write_action(action_text: str) -> ToolPayload:
    text = _require_action(action_text)
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict) or not str(payload.get("path", "")).strip():
            raise ValueError("code_write_file JSON must include a non-empty 'path'")
        return payload
    path, separator, patch_spec = text.partition(":")
    if not separator or "->" not in patch_spec:
        raise ValueError(
            "code_write_file action must be JSON or use path:old_text->new_text"
        )
    search_text, arrow, replace_text = patch_spec.partition("->")
    if not arrow:
        raise ValueError(
            "code_write_file patch shorthand must use path:old_text->new_text"
        )
    return {
        "path": path.strip(),
        "patch": {
            "search": search_text,
            "replace": replace_text,
        },
    }


def _parse_debug_log_action(action_text: str) -> ToolPayload:
    text = _require_action(action_text).lower()
    if text not in {"read", "clear"}:
        raise ValueError("code_read_debug_logs action must be 'read' or 'clear'")
    return {"clear": text == "clear"}


def _parse_log_analysis_action(action_text: str) -> ToolPayload:
    text = _require_action(action_text)
    lowered = text.lower()
    if lowered in {"analyze", "summary"}:
        return {"include_debug_output": True}
    if lowered == "failures":
        return {"include_debug_output": True, "failures_only": True}
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("log_analyze JSON action must decode to an object")
        payload.setdefault("include_debug_output", True)
        return payload
    raise ValueError(
        "log_analyze action must be 'analyze', 'failures', or a JSON object"
    )


def _invoke_code_list(payload: ToolPayload, runtime: ToolRuntimeContext, adapter: UniversalCodebaseAdapter) -> ToolInvocationResult:
    files = adapter.list_files()
    result = {"success": True, "files": [{"path": f.path, "is_dir": f.is_dir} for f in files]}
    return ToolInvocationResult(observation=_tool_observation("code_list_files", payload, result))

def _invoke_code_read(payload: ToolPayload, runtime: ToolRuntimeContext, adapter: UniversalCodebaseAdapter) -> ToolInvocationResult:
    path = payload.get("path", "")
    content = adapter.read_file(path)
    result = {"success": content is not None, "path": path, "content": content}
    return ToolInvocationResult(observation=_tool_observation("code_read_file", payload, result))

def _invoke_code_search(payload: ToolPayload, runtime: ToolRuntimeContext, adapter: UniversalCodebaseAdapter) -> ToolInvocationResult:
    matches = adapter.search_code(payload.get("pattern", ""))
    result = {"success": True, "matches": [{"path": m["path"], "line": m["line"], "text": m["content"]} for m in matches]}
    return ToolInvocationResult(observation=_tool_observation("code_search", payload, result))

def _invoke_code_write(payload: ToolPayload, runtime: ToolRuntimeContext, adapter: UniversalCodebaseAdapter) -> ToolInvocationResult:
    success = adapter.write_file(payload.get("path", ""), payload.get("content", ""))
    result = {"success": success, "path": payload.get("path")}
    return ToolInvocationResult(observation=_tool_observation("code_write_file", payload, result))

def _invoke_code_restore(payload: ToolPayload, runtime: ToolRuntimeContext, adapter: UniversalCodebaseAdapter) -> ToolInvocationResult:
    success = adapter.restore_file(payload.get("path", ""))
    result = {"success": success, "path": payload.get("path")}
    return ToolInvocationResult(observation=_tool_observation("code_restore_file", payload, result))


def _invoke_runtime_log_tool(
    payload: ToolPayload,
    runtime_context: ToolRuntimeContext,
    adapter: RuntimeLogAdapter,
) -> ToolInvocationResult:
    session = runtime_context.get("session")
    if session is None:
        raise RuntimeError("code_read_debug_logs requires an active session")
    
    backend_type = getattr(session, "backend_type", "api")
    session_id = getattr(session, "session_id", "")
    
    if backend_type == "api":
        result = adapter.read_debug_logs(session_id, clear=bool(payload.get("clear", False)))
    else:
        # Fallback to CUA client logs if available
        client = session.raw.get("client") if isinstance(session.raw, dict) else None
        if client and hasattr(client, "read_browser_logs"):
            result = {"success": True, "logs": client.read_browser_logs()}
        else:
            result = {"success": False, "message": f"Debug logs not supported for {backend_type}"}

    return ToolInvocationResult(
        observation=_tool_observation("code_read_debug_logs", payload, result),
    )


def _invoke_log_analysis_tool(
    payload: ToolPayload,
    runtime_context: ToolRuntimeContext,
    adapter: RuntimeLogAdapter,
    analyzer: LogAnalyzer,
) -> ToolInvocationResult:
    session = runtime_context.get("session")
    if session is None:
        raise RuntimeError("log_analyze requires an active session")

    backend_type = getattr(session, "backend_type", "api")
    session_id = getattr(session, "session_id", "")
    
    # Logic similar to what we did for PR 6
    if backend_type == "api":
        session_result = adapter.read_session_log(session_id)
        if not bool(session_result.get("success", False)):
            return ToolInvocationResult(observation=_tool_observation("log_analyze", payload, session_result))
        session_data = session_result.get("data", {})
        debug_output = ""
        if bool(payload.get("include_debug_output", True)):
            debug_result = adapter.read_debug_logs(session_id, clear=False)
            debug_output = str(debug_result.get("logs", "")) if debug_result.get("success") else ""
    else:
        session_data = {"commands": runtime_context.get("history", [])}
        debug_output = ""
        client = session.raw.get("client") if isinstance(session.raw, dict) else None
        if client and hasattr(client, "read_browser_logs"):
            try: debug_output = client.read_browser_logs()
            except Exception: pass

    result: Dict[str, Any] = {
        "success": True,
        "session_id": session_id,
        "analysis": analyzer.analyze_session(session_data, debug_output),
    }
    return ToolInvocationResult(observation=_tool_observation("log_analyze", payload, result))


def _tool_observation(
    tool_name: str,
    payload: ToolPayload,
    result: Dict[str, Any],
) -> Observation:
    success = bool(result.get("success", False))
    summary = _tool_summary(tool_name, result)
    message = str(result.get("message", "")).strip() or summary
    execution: Dict[str, Any] = {
        "attempts": [],
        "diagnostics": {
            "tool": tool_name,
            "tool_payload": payload,
        },
    }
    if not success:
        execution["diagnostics"]["error"] = message
        execution["diagnostics"]["error_kind"] = "tool_execution_error"
        execution["suspected_origin"] = "execution"
    return Observation(
        success=success,
        message=message,
        state={},
        raw=result,
        summary=summary,
        env_state={},
        artifacts={},
        execution=execution,
    )


def _tool_summary(tool_name: str, result: Dict[str, Any]) -> str:
    if tool_name == "code_list_files":
        files = result.get("files", [])
        return "Code tool result (file list):\n" + "\n".join([f.get("path") for f in files if f.get("path")])
    if tool_name == "code_read_file":
        path = str(result.get("path", "")).strip()
        content = str(result.get("content", "")).strip()
        return f"Code tool result (read file: {path}):\n{content}"
    if tool_name == "code_search":
        matches = result.get("matches", [])
        lines = [f"{m.get('path')}:{m.get('line')} {m.get('text')}" for m in matches]
        return "Code tool result (search matches):\n" + "\n".join(lines) if lines else "No matches found."
    if tool_name == "code_read_debug_logs":
        logs = str(result.get("logs", "")).strip()
        return f"Runtime log result:\n{logs}" if logs else "No debug logs found."
    if tool_name == "log_analyze":
        return f"Log analysis result: {result.get('analysis', {}).get('summary', 'Done')}"
    
    path = str(result.get("path", "")).strip()
    message = str(result.get("message", "")).strip()
    if path and message:
        return f"Code tool result ({path}): {message}"
    return f"Code tool result: {message or 'Success'}"
