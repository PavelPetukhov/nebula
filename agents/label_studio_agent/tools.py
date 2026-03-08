"""Label Studio API tools."""

import logging

from label_studio_sdk import Client

from agents.label_studio_agent.config import config
from shared.models import AnnotationQueueStatus

logger = logging.getLogger(__name__)


def _get_client() -> Client:
    return Client(url=config.label_studio_url, api_key=config.label_studio_api_key)


def fetch_annotation_stats(project_id: int) -> AnnotationQueueStatus:
    """Pull task completion rates and agreement scores for a project."""
    client = _get_client()
    project = client.get_project(project_id)
    stats = project.get_stats()

    return AnnotationQueueStatus(
        project=str(project_id),
        total_tasks=stats.get("total_tasks", 0),
        completed_tasks=stats.get("total_annotations", 0),
        pending_tasks=stats.get("skipped_annotations", 0),
        in_review_tasks=stats.get("in_review_annotations", 0),
    )


def detect_queue_backlog(project_id: int, threshold: int | None = None) -> dict:
    """Flag a stalled annotation queue if pending tasks exceed threshold."""
    threshold = threshold or config.backlog_threshold
    status = fetch_annotation_stats(project_id)
    flagged = status.pending_tasks > threshold
    return {
        "project_id": project_id,
        "pending_tasks": status.pending_tasks,
        "threshold": threshold,
        "backlog_flagged": flagged,
    }


def flag_low_agreement_tasks(project_id: int) -> list[int]:
    """Return task IDs where inter-annotator agreement is below threshold."""
    client = _get_client()
    project = client.get_project(project_id)
    tasks = project.get_tasks()

    low_agreement: list[int] = []
    for task in tasks:
        annotations = task.get("annotations", [])
        if len(annotations) < 2:
            continue
        # Simple agreement: fraction of annotations matching the first
        first = annotations[0].get("result")
        matches = sum(1 for a in annotations[1:] if a.get("result") == first)
        agreement = matches / (len(annotations) - 1)
        if agreement < config.agreement_threshold:
            low_agreement.append(task["id"])

    logger.info(
        "Found %d low-agreement tasks in project %d", len(low_agreement), project_id
    )
    return low_agreement
