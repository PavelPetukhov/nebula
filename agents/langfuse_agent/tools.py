"""LangFuse API tool functions — called by the ADK agent."""

import logging
from datetime import datetime, timedelta

from langfuse.api.client import FernLangfuse

from agents.langfuse_agent.anomaly import BaselineMetrics, TraceMetrics, compute_baseline
from agents.langfuse_agent.config import config

_TRACE_PAGE_LIMIT = 100  # max allowed by Langfuse API

logger = logging.getLogger(__name__)


def _get_client() -> FernLangfuse:
    return FernLangfuse(
        base_url=config.langfuse_host,
        username=config.langfuse_public_key,
        password=config.langfuse_secret_key,
    )


def _window_metrics(
    client: FernLangfuse,
    project: str,
    since: datetime,
    until: datetime | None = None,
) -> TraceMetrics:
    """Fetch and aggregate metrics for one time window (3 bulk API calls)."""
    trace_kwargs = {"to_timestamp": until} if until else {}
    obs_kwargs = {"to_start_time": until} if until else {}
    score_kwargs = {"to_timestamp": until} if until else {}

    # 1. Traces — paginate through all pages (max 100 per page)
    traces = []
    page = 1
    while True:
        resp = client.trace.list(
            from_timestamp=since,
            limit=_TRACE_PAGE_LIMIT,
            page=page,
            **trace_kwargs,
        )
        traces.extend(resp.data)
        if page >= resp.meta.total_pages:
            break
        page += 1

    latencies = [t.latency * 1000 for t in traces if t.latency is not None]
    trace_ids = {t.id for t in traces}

    if not trace_ids:
        return TraceMetrics(
            judge_scores=[],
            latencies_p95=[],
            token_counts=[],
            error_rate=0.0,
            trace_count=0,
            tool_names=set(),
        )

    # 2. Scores — paginate (max 100/page), filter to our trace IDs and judge score names
    judge_scores: list[float] = []
    score_page = 1
    while True:
        scores_resp = client.score_v_2.get(
            from_timestamp=since,
            limit=100,
            page=score_page,
            **score_kwargs,
        )
        for score in scores_resp.data:
            if (
                score.trace_id in trace_ids
                and score.name
                and "judge" in score.name.lower()
                and isinstance(getattr(score, "value", None), (int, float))
            ):
                judge_scores.append(float(score.value))
        if score_page >= scores_resp.meta.total_pages:
            break
        score_page += 1

    # 3. Observations — bulk fetch for tokens (GENERATION), errors, and tool names
    obs_resp = client.observations.get_many(
        from_start_time=since,
        limit=100,
        **obs_kwargs,
    )
    tokens: list[float] = []
    error_trace_ids: set[str] = set()
    tool_names: set[str] = set()

    for obs in obs_resp.data:
        if obs.trace_id not in trace_ids:
            continue
        if obs.type == "GENERATION" and obs.usage_details:
            total = obs.usage_details.get("total")
            if total:
                tokens.append(float(total))
        if obs.level and obs.level.value == "ERROR":
            error_trace_ids.add(obs.trace_id)
        if obs.type == "TOOL" and obs.name:
            tool_names.add(obs.name)

    error_rate = len(error_trace_ids) / len(traces) if traces else 0.0

    logger.info(
        "Window %s→%s: project=%s traces=%d judge_scores=%d tokens=%d errors=%d",
        since.isoformat(),
        until.isoformat() if until else "now",
        project,
        len(traces),
        len(judge_scores),
        len(tokens),
        len(error_trace_ids),
    )
    return TraceMetrics(
        judge_scores=judge_scores,
        latencies_p95=latencies,
        token_counts=tokens,
        error_rate=error_rate,
        trace_count=len(traces),
        tool_names=tool_names,
    )


def fetch_recent_traces(project: str, hours: int = 6) -> TraceMetrics:
    """
    Pull traces from LangFuse for a given project over the last N hours.
    Returns a TraceMetrics dataclass for anomaly scoring.
    """
    client = _get_client()
    since = datetime.utcnow() - timedelta(hours=hours)
    return _window_metrics(client, project, since)


def fetch_baseline_stats(project: str, lookback_days: int = 7) -> BaselineMetrics:
    """
    Compute rolling baseline metrics from historical daily windows.
    """
    client = _get_client()
    windows: list[TraceMetrics] = []

    for day_offset in range(1, lookback_days + 1):
        end = datetime.utcnow() - timedelta(days=day_offset - 1)
        start = end - timedelta(days=1)
        windows.append(_window_metrics(client, project, start, end))

    return compute_baseline(windows)
