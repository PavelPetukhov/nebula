"""LangFuse API tool functions — called by the ADK agent."""

import logging
from datetime import datetime, timedelta

from langfuse import Langfuse

from agents.langfuse_agent.anomaly import BaselineMetrics, TraceMetrics, compute_baseline
from agents.langfuse_agent.config import config

logger = logging.getLogger(__name__)


def _get_client() -> Langfuse:
    return Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host,
    )


def fetch_recent_traces(project: str, hours: int = 6) -> TraceMetrics:
    """
    Pull traces from LangFuse for a given project over the last N hours.
    Returns a TraceMetrics dataclass for anomaly scoring.
    """
    client = _get_client()
    since = datetime.utcnow() - timedelta(hours=hours)

    traces = client.fetch_traces(
        name=project,
        from_timestamp=since,
        limit=500,
    ).data

    judge_scores: list[float] = []
    latencies: list[float] = []
    tokens: list[float] = []
    errors = 0
    tool_names: set[str] = set()

    for trace in traces:
        # Collect judge scores from scores attached to trace
        for score in (trace.scores or []):
            if score.name and "judge" in score.name.lower():
                if score.value is not None:
                    judge_scores.append(float(score.value))

        # Latency in ms
        if trace.latency is not None:
            latencies.append(trace.latency * 1000)  # seconds → ms

        # Token usage
        if trace.usage and trace.usage.total_tokens:
            tokens.append(float(trace.usage.total_tokens))

        # Errors
        if trace.level == "ERROR":
            errors += 1

        # Tool names from observations
        for obs in (trace.observations or []):
            if obs.type == "TOOL" and obs.name:
                tool_names.add(obs.name)

    error_rate = errors / len(traces) if traces else 0.0

    logger.info(
        "Fetched %d traces for project=%s over last %dh", len(traces), project, hours
    )
    return TraceMetrics(
        judge_scores=judge_scores,
        latencies_p95=latencies,
        token_counts=tokens,
        error_rate=error_rate,
        trace_count=len(traces),
        tool_names=tool_names,
    )


def fetch_baseline_stats(project: str, lookback_days: int = 7) -> BaselineMetrics:
    """
    Compute rolling baseline metrics from historical daily windows.
    """
    client = _get_client()
    windows: list[TraceMetrics] = []

    for day_offset in range(1, lookback_days + 1):
        end = datetime.utcnow() - timedelta(days=day_offset - 1)
        start = end - timedelta(days=1)
        traces = client.fetch_traces(
            name=project,
            from_timestamp=start,
            to_timestamp=end,
            limit=500,
        ).data

        judge_scores: list[float] = []
        latencies: list[float] = []
        tokens: list[float] = []
        errors = 0
        tool_names: set[str] = set()

        for trace in traces:
            for score in (trace.scores or []):
                if score.name and "judge" in score.name.lower() and score.value is not None:
                    judge_scores.append(float(score.value))
            if trace.latency is not None:
                latencies.append(trace.latency * 1000)
            if trace.usage and trace.usage.total_tokens:
                tokens.append(float(trace.usage.total_tokens))
            if trace.level == "ERROR":
                errors += 1
            for obs in (trace.observations or []):
                if obs.type == "TOOL" and obs.name:
                    tool_names.add(obs.name)

        error_rate = errors / len(traces) if traces else 0.0
        windows.append(TraceMetrics(
            judge_scores=judge_scores,
            latencies_p95=latencies,
            token_counts=tokens,
            error_rate=error_rate,
            trace_count=len(traces),
            tool_names=tool_names,
        ))

    return compute_baseline(windows)
