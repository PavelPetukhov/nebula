"""Prefect flow wrapping the LangFuse Anomaly Detection Agent."""

import logging

from prefect import flow, task

from agents.langfuse_agent.agent import run_anomaly_check
from agents.langfuse_agent.config import config

logger = logging.getLogger(__name__)


@task(name="check_project_anomalies", retries=2, retry_delay_seconds=60)
def check_project(project: str, hours: int) -> dict:
    return run_anomaly_check(project=project, hours=hours)


@flow(
    name="langfuse-anomaly-detection",
    description="Periodic anomaly detection across all monitored LangFuse projects.",
    log_prints=True,
)
def langfuse_anomaly_flow(
    projects: list[str] | None = None,
    hours: int | None = None,
) -> list[dict]:
    """
    Run anomaly checks across all configured projects.

    Args:
        projects: Override the list of projects to check (defaults to config).
        hours: Override the trace window in hours (defaults to config).
    """
    target_projects = projects or config.monitored_projects
    trace_hours = hours or config.default_trace_hours

    if not target_projects:
        logger.warning("No projects configured — set MONITORED_PROJECTS or pass projects list")
        return []

    results = []
    for project in target_projects:
        result = check_project(project, trace_hours)
        results.append(result)
        logger.info(
            "Project=%s | traces=%d | findings=%d | alert=%s",
            project,
            result.get("trace_count", 0),
            result.get("findings_count", 0),
            result.get("alert_dispatched", False),
        )

    return results


if __name__ == "__main__":
    langfuse_anomaly_flow()
