"""Deterministic annotation quality / agreement scoring logic."""

from shared.models import AnnotationQueueStatus


def compute_mean_agreement(tasks: list[dict]) -> float:
    """Compute mean inter-annotator agreement across all multi-annotated tasks."""
    scores: list[float] = []
    for task in tasks:
        annotations = task.get("annotations", [])
        if len(annotations) < 2:
            continue
        first = annotations[0].get("result")
        matches = sum(1 for a in annotations[1:] if a.get("result") == first)
        scores.append(matches / (len(annotations) - 1))
    return sum(scores) / len(scores) if scores else 1.0


def enrich_queue_status(status: AnnotationQueueStatus, tasks: list[dict]) -> AnnotationQueueStatus:
    """Add agreement stats to an existing AnnotationQueueStatus."""
    mean_agreement = compute_mean_agreement(tasks)
    low_ids = [
        t["id"]
        for t in tasks
        if len(t.get("annotations", [])) >= 2
        and (
            sum(
                1
                for a in t["annotations"][1:]
                if a.get("result") == t["annotations"][0].get("result")
            )
            / (len(t["annotations"]) - 1)
        )
        < 0.7
    ]
    return status.model_copy(
        update={"mean_agreement_score": mean_agreement, "low_agreement_task_ids": low_ids}
    )
