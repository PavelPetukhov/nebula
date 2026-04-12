"""
ADK → MLflow run persistence.

Run model
---------
ONE run per ADK session — a new run is created each time the user starts a
new session (e.g. clicks "New session" in the ADK web UI).  All agents in
the same session (orchestrator + sub-agents) share that run, keyed by
session_id via the module-level _session_runs dict.

Each agent invocation increments a per-session step counter so metrics are
distinguishable in the MLflow UI, and artifacts are stored under
turns/<step>-<agent>/.

  [Experiment: adk/nebula]          ← fixed via MLFLOW_EXPERIMENT_NAME
  ├── adk-20260412-145300 (RUNNING) ← session 1
  │     metrics (step=0): latency_ms, …  ← orchestrator turn 1
  │     metrics (step=1): latency_ms, …  ← sub-agent turn 1
  │     artifacts/turns/0000-nebula_orchestrator/conversation.json
  └── adk-20260412-160000 (RUNNING) ← session 2 (user clicked New session)
        …

Usage
-----
    from shared.mlflow_tracking import MLflowAdkTracker

    tracker = MLflowAdkTracker(is_root=True)   # orchestrator — owns run lifecycle
    tracker = MLflowAdkTracker()               # sub-agents — attach to active run
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state — one run per orchestrator session
#
# The orchestrator (is_root=True) owns the run lifecycle: it creates a new
# run when its session_id changes (i.e. the user clicked "New session").
# Sub-agents always attach to _current_run_id, whatever it is, because
# AgentTool gives them a brand-new InMemorySession with a different session_id.
# ---------------------------------------------------------------------------
_current_run_id: str | None = None      # run_id of the active orchestrator session
_current_session_id: str | None = None  # orchestrator session_id that owns the run
_step_counter: int = 0
_global_lock = threading.Lock()


def get_experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT_NAME", "adk/nebula")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize(obj: Any, _depth: int = 0) -> Any:
    if _depth > 10:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize(v, _depth + 1) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v, _depth + 1) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        try:
            return _serialize(obj.model_dump(), _depth + 1)
        except Exception:
            pass
    if hasattr(obj, "_pb"):
        try:
            from google.protobuf.json_format import MessageToDict
            return _serialize(MessageToDict(obj._pb), _depth + 1)
        except Exception:
            pass
    if hasattr(obj, "parts") and hasattr(obj, "role"):
        try:
            return {"role": obj.role, "parts": _serialize(list(obj.parts), _depth + 1)}
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _serialize(
                {k: v for k, v in obj.__dict__.items() if not k.startswith("_")},
                _depth + 1,
            )
        except Exception:
            pass
    return str(obj)


def _extract_text(content: Any) -> str:
    try:
        if isinstance(content, str):
            return content
        if hasattr(content, "parts"):
            return " ".join(p.text for p in content.parts if hasattr(p, "text") and p.text)
        return str(content)[:500]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class MLflowAdkTracker:
    """
    ADK callback set that logs all agent activity into a single MLflow run.

    Parameters
    ----------
    tracking_uri / tracking_token:
        Fall back to MLFLOW_TRACKING_URI / MLFLOW_TRACKING_TOKEN env vars.
    """

    def __init__(
        self,
        is_root: bool = False,
        tracking_uri: str | None = None,
        tracking_token: str | None = None,
    ) -> None:
        self._is_root = is_root
        self._tracking_uri = tracking_uri
        self._tracking_token = tracking_token

        self._client: MlflowClient | None = None
        self._experiment_id: str | None = None

        # invocation_id -> per-invocation in-memory state
        self._state: dict[str, dict[str, Any]] = {}
        self._inv_lock = threading.Lock()

        # (session_id, agent_name) pairs already logged to avoid duplicates
        self._logged_agents: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> bool:
        if self._experiment_id is not None:
            return True
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)

            uri = self._tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
            if not uri:
                raise ValueError("MLFLOW_TRACKING_URI is not set.")
            mlflow.set_tracking_uri(uri)

            if self._tracking_token:
                os.environ["MLFLOW_TRACKING_TOKEN"] = self._tracking_token

            client = MlflowClient()
            name = get_experiment_name()
            exp = client.get_experiment_by_name(name)
            experiment_id = exp.experiment_id if exp else client.create_experiment(name)

            self._client = client
            self._experiment_id = experiment_id
            logger.warning("MLflow connected: uri=%s  experiment=%s", uri, name)
            return True
        except Exception as exc:
            logger.warning("MLflow connection failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Process run
    # ------------------------------------------------------------------

    def _ensure_run(self, session_id: str) -> str | None:
        """
        For the root tracker: create a new run when the orchestrator session_id
        changes (user clicked "New session"), or return the existing run_id.
        For non-root trackers: return whatever run the root already created.
        """
        global _current_run_id, _current_session_id, _step_counter
        with _global_lock:
            if self._is_root:
                if session_id != _current_session_id:
                    # New session — create a new MLflow run
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    run_name = f"adk-{ts}"
                    run = self._client.create_run(
                        experiment_id=self._experiment_id,
                        run_name=run_name,
                        tags={
                            "mlflow.runName": run_name,
                            "adk.type": "session",
                            "adk.session_id": session_id,
                            "status": "running",
                            "started_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    _current_run_id = run.info.run_id
                    _current_session_id = session_id
                    _step_counter = 0
                    logger.warning(
                        "MLflow run created: session_id=%s run_id=%s",
                        session_id, _current_run_id,
                    )
                return _current_run_id
            else:
                # Sub-agent: attach to whatever run the root created
                return _current_run_id

    def _next_step(self) -> int:
        global _step_counter
        with _global_lock:
            step = _step_counter
            _step_counter += 1
        return step

    # ------------------------------------------------------------------
    # Prompt logging
    # ------------------------------------------------------------------

    def _log_prompt_once(self, run_id: str, session_id: str, agent_name: str, callback_context: Any) -> None:
        """
        Log the agent's system instruction once per MLflow run.
        Stored as prompts/<agent_name>.txt so each run contains a full
        snapshot of every prompt active during that session.
        """
        key = (run_id, agent_name)
        if key in self._logged_agents:
            return
        self._logged_agents.add(key)

        try:
            agent_obj = callback_context._invocation_context.agent
            instruction = getattr(agent_obj, "instruction", None)
            if not instruction:
                return
            model = str(getattr(agent_obj, "model", "unknown"))

            content = f"# {agent_name}\n# model: {model}\n\n{instruction}"

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, f"{agent_name}.txt")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(content)
                self._safe_log(
                    self._client.log_artifact, run_id, path,
                    artifact_path="prompts",
                )
            logger.warning("MLflow prompt logged: prompts/%s.txt", agent_name)
        except Exception as exc:
            logger.debug("_log_prompt_once failed: %s", exc)

    # ------------------------------------------------------------------
    # Invocation state
    # ------------------------------------------------------------------

    def _init_state(self, invocation_id: str, run_id: str, step: int) -> dict[str, Any]:
        state: dict[str, Any] = {
            "run_id": run_id,
            "step": step,
            "monotonic_start": time.monotonic(),
            "wall_start": datetime.now(timezone.utc).isoformat(),
            "tool_calls": [],
            "model_turns": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_call_count": 0,
        }
        with self._inv_lock:
            self._state[invocation_id] = state
        return state

    def _pop_state(self, invocation_id: str) -> dict[str, Any] | None:
        with self._inv_lock:
            return self._state.pop(invocation_id, None)

    def _peek_state(self, invocation_id: str) -> dict[str, Any] | None:
        with self._inv_lock:
            return self._state.get(invocation_id)

    def _safe_log(self, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            logger.debug("MLflow log failed (%s): %s", fn.__name__, exc)

    # ------------------------------------------------------------------
    # ADK callbacks
    # ------------------------------------------------------------------

    def before_agent_callback(self, callback_context: Any) -> None:
        try:
            if not self._ensure_connected():
                return None

            session_id = callback_context.session.id
            run_id = self._ensure_run(session_id)
            if run_id is None:
                return None  # root hasn't fired yet

            step = self._next_step()
            invocation_id: str = callback_context.invocation_id
            agent_name: str = callback_context.agent_name

            logger.warning(
                "MLflow logging invocation: agent=%s step=%d run_id=%s",
                agent_name, step, run_id,
            )

            self._safe_log(self._client.set_tag, run_id, f"step.{step}.agent", agent_name)
            self._safe_log(self._client.set_tag, run_id, f"step.{step}.invocation_id", invocation_id)

            # Tag the current OTel span with session.id so MLflow groups all
            # traces from this ADK session under the same MLflow Session.
            try:
                from opentelemetry import trace as otel_trace
                span = otel_trace.get_current_span()
                if span and span.is_recording():
                    span.set_attribute("session.id", session_id)
            except Exception:
                pass

            # Log the agent's system prompt once per session run
            self._log_prompt_once(run_id, session_id, agent_name, callback_context)

            self._init_state(invocation_id, run_id, step)
        except Exception as exc:
            logger.warning("before_agent_callback error: %s", exc)
        return None

    def after_agent_callback(self, callback_context: Any) -> None:
        try:
            invocation_id: str = callback_context.invocation_id
            state = self._pop_state(invocation_id)
            if state is None:
                return None

            run_id: str = state["run_id"]
            step: int = state["step"]
            elapsed_ms = (time.monotonic() - state["monotonic_start"]) * 1000
            agent_name: str = callback_context.agent_name

            # Metrics at this step so each invocation appears as a data point
            for name, value in (
                ("latency_ms", elapsed_ms),
                ("input_tokens", state["input_tokens"]),
                ("output_tokens", state["output_tokens"]),
                ("total_tokens", state["input_tokens"] + state["output_tokens"]),
                ("tool_call_count", state["tool_call_count"]),
                ("model_turn_count", len(state["model_turns"])),
            ):
                self._safe_log(self._client.log_metric, run_id, name, value, step=step)

            final_response = None
            if state["model_turns"]:
                final_response = state["model_turns"][-1].get("response_content")

            conversation = {
                "schema_version": "1",
                "step": step,
                "agent_name": agent_name,
                "invocation_id": invocation_id,
                "started_at": state["wall_start"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "latency_ms": round(elapsed_ms, 2),
                "token_usage": {
                    "input": state["input_tokens"],
                    "output": state["output_tokens"],
                    "total": state["input_tokens"] + state["output_tokens"],
                },
                "model_turns": state["model_turns"],
                "tool_calls": state["tool_calls"],
                "final_response": final_response,
            }

            # Store under turns/<step>-<agent>/conversation.json
            artifact_subdir = f"turns/{step:04d}-{agent_name}"
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "conversation.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(conversation, fh, indent=2, default=str)
                self._safe_log(
                    self._client.log_artifact, run_id, path,
                    artifact_path=artifact_subdir,
                )
        except Exception as exc:
            logger.warning("after_agent_callback error: %s", exc)
        return None

    def before_tool_callback(self, tool: Any, args: dict[str, Any], tool_context: Any) -> None:
        try:
            state = self._peek_state(tool_context.invocation_id)
            if state is None:
                return None
            state["tool_calls"].append({
                "tool": tool.name,
                "args": _serialize(args),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "result": None,
                "finished_at": None,
            })
        except Exception as exc:
            logger.debug("before_tool_callback: %s", exc)
        return None

    def after_tool_callback(
        self, tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any
    ) -> None:
        try:
            state = self._peek_state(tool_context.invocation_id)
            if state is None:
                return None
            state["tool_call_count"] += 1
            end_time = datetime.now(timezone.utc).isoformat()
            for entry in reversed(state["tool_calls"]):
                if entry["tool"] == tool.name and entry["result"] is None:
                    entry["result"] = _serialize(tool_response)
                    entry["finished_at"] = end_time
                    break
        except Exception as exc:
            logger.debug("after_tool_callback: %s", exc)
        return None

    def before_model_callback(self, callback_context: Any, llm_request: Any) -> None:
        try:
            state = self._peek_state(callback_context.invocation_id)
            if state is None:
                return None
            contents = None
            try:
                contents = _serialize(list(llm_request.contents))
            except Exception:
                pass
            state["model_turns"].append({
                "turn": len(state["model_turns"]) + 1,
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "request_contents": contents,
                "response_content": None,
                "responded_at": None,
                "usage": None,
            })
        except Exception as exc:
            logger.debug("before_model_callback: %s", exc)
        return None

    def after_model_callback(self, callback_context: Any, llm_response: Any) -> None:
        try:
            state = self._peek_state(callback_context.invocation_id)
            if state is None:
                return None
            usage_dict: dict[str, int] | None = None
            usage_meta = getattr(llm_response, "usage_metadata", None)
            if usage_meta is not None:
                prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
                cand_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
                state["input_tokens"] += prompt_tokens
                state["output_tokens"] += cand_tokens
                usage_dict = {
                    "prompt_token_count": prompt_tokens,
                    "candidates_token_count": cand_tokens,
                }
            if state["model_turns"]:
                turn = state["model_turns"][-1]
                content = getattr(llm_response, "content", None)
                turn["response_content"] = _serialize(content) if content is not None else None
                turn["responded_at"] = datetime.now(timezone.utc).isoformat()
                turn["usage"] = usage_dict
        except Exception as exc:
            logger.debug("after_model_callback: %s", exc)
        return None
