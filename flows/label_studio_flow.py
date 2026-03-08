"""Prefect flow wrapping the Label Studio Annotation Quality Agent."""

import logging

from prefect import flow, task

from agents.label_studio_agent.tools import detect_queue_backlog, flag_low_agreement_tasks

logger = logging.getLogger(__name__)


@task(name="check_project_quality", retries=2, retry_delay_seconds=60)
def check_project(project_id: int, backlog_threshold: int) -> dict:
    backlog = detect_queue_backlog(project_id, backlog_threshold)
    low_agreement = flag_low_agreement_tasks(project_id)
    return {
        "project_id": project_id,
        "backlog": backlog,
        "low_agreement_count": len(low_agreement),
        "low_agreement_task_ids": low_agreement[:20],  # cap for log readability
    }


@flow(
    name="label-studio-annotation-quality",
    description="Periodic annotation quality and queue health check.",
    log_prints=True,
)
def label_studio_quality_flow(
    project_ids: list[int],
    backlog_threshold: int = 100,
) -> list[dict]:
    results = []
    for pid in project_ids:
        result = check_project(pid, backlog_threshold)
        results.append(result)
        logger.info(
            "Project=%d | backlog=%s | low_agreement_tasks=%d",
            pid,
            result["backlog"]["backlog_flagged"],
            result["low_agreement_count"],
        )
    return results


if __name__ == "__main__":
    label_studio_quality_flow(project_ids=[1, 2, 3])
