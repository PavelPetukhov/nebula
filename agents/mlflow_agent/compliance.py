"""Deterministic compliance report generation for MLflow runs."""

from datetime import datetime

from shared.models import ComplianceReport


def generate_compliance_report(
    team: str,
    runs: list[dict],
    period_start: datetime,
    period_end: datetime,
) -> ComplianceReport:
    """Produce a structured compliance audit summary for a team's runs."""
    violations = []
    compliant = 0

    for run in runs:
        missing_tags = [
            t for t in ["team", "dataset_version", "model_version", "owner"]
            if t not in run.get("tags", {})
        ]
        if missing_tags:
            violations.append({
                "run_id": run["run_id"],
                "experiment_name": run.get("experiment_name", "unknown"),
                "missing_tags": missing_tags,
            })
        else:
            compliant += 1

    return ComplianceReport(
        team=team,
        period_start=period_start,
        period_end=period_end,
        total_runs=len(runs),
        compliant_runs=compliant,
        violations=violations,
    )
