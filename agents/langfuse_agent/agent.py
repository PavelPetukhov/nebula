"""LangFuse Anomaly Detection Agent — ADK agent definition."""

import logging
from datetime import datetime

from google.adk.agents import Agent
from google.adk.models.anthropic_llm import AnthropicLlm
from langfuse import Langfuse

from agents.langfuse_agent.anomaly import compute_anomaly_score
from agents.langfuse_agent.config import config
from agents.langfuse_agent.tools import fetch_baseline_stats, fetch_recent_traces
from shared.alerts import dispatch_alert
from shared.mlflow_tracking import MLflowAdkTracker
from shared.models import AlertPayload, AnomalyFinding, Severity

logger = logging.getLogger(__name__)


def _overall_severity(findings: list[AnomalyFinding]) -> Severity:
    order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    if not findings:
        return Severity.LOW
    return max(findings, key=lambda f: order.index(f.severity)).severity


def run_anomaly_check(project: str, hours: int | None = None) -> dict:
    """
    Entry-point tool: fetch traces, score anomalies, dispatch alert if needed.
    Returns a dict summary suitable for MCP responses.
    """
    hours = hours or config.default_trace_hours
    lf = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host,
    )

    with lf.start_as_current_span(
        name="langfuse_anomaly_check",
        input={"project": project, "hours": hours},
    ):
        current = fetch_recent_traces(project, hours)
        baseline = fetch_baseline_stats(project, config.baseline_lookback_days)
        findings = compute_anomaly_score(
            current,
            baseline,
            judge_z_threshold=config.judge_score_z_threshold,
            latency_z_threshold=config.latency_z_threshold,
            token_z_threshold=config.token_usage_z_threshold,
            error_rate_z_threshold=config.error_rate_z_threshold,
            volume_z_threshold=config.trace_volume_z_threshold,
        )

        result = {
            "project": project,
            "hours_checked": hours,
            "trace_count": current.trace_count,
            "findings_count": len(findings),
            "findings": [f.model_dump() for f in findings],
            "timestamp": datetime.utcnow().isoformat(),
        }

        if findings:
            severity = _overall_severity(findings)
            alert = AlertPayload(
                title=f"Anomaly detected in {project}",
                severity=severity,
                agent="langfuse_agent",
                project=project,
                findings=findings,
                summary=f"{len(findings)} anomaly/anomalies detected in {project} over the last {hours}h.",
            )
            dispatch_alert(alert, channel=config.alert_channel)
            result["alert_dispatched"] = True
        else:
            result["alert_dispatched"] = False

        lf.update_current_trace(output=result)

    lf.flush()
    return result


_tracker = MLflowAdkTracker()

# ADK Agent definition
langfuse_agent = Agent(
    name="langfuse_anomaly_agent",
    model=AnthropicLlm(model=config.model),
    description=(
        "Monitors production AI traces in LangFuse. "
        "Detects quality, latency, token usage, error rate, and volume anomalies. "
        "Dispatches structured alerts when anomalies are found."
    ),
    instruction=(
        "You are the LangFuse Anomaly Detection Agent. "
        "Your job is to interpret the results of deterministic anomaly scoring and produce "
        "clear, actionable alert narratives. "
        "Do NOT perform statistical calculations yourself — always use the provided tool functions. "
        "When anomalies are found, explain what they mean in plain language for the on-call team."
    ),
    tools=[run_anomaly_check],
    before_agent_callback=_tracker.before_agent_callback,
    after_agent_callback=_tracker.after_agent_callback,
    before_tool_callback=_tracker.before_tool_callback,
    after_tool_callback=_tracker.after_tool_callback,
    before_model_callback=_tracker.before_model_callback,
    after_model_callback=_tracker.after_model_callback,
)

root_agent = langfuse_agent
