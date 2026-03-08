"""Prefect flow wrapping the MLflow Governance & Compliance Agent."""

import logging
from datetime import datetime, timedelta

from prefect import flow, task

from agents.mlflow_agent.compliance import generate_compliance_report
from agents.mlflow_agent.tools import detect_experiment_anomalies, fetch_experiments

logger = logging.getLogger(__name__)


@task(name="fetch_and_audit_team", retries=2, retry_delay_seconds=60)
def audit_team(team: str, start: datetime, end: datetime) -> dict:
    runs = fetch_experiments(team, start, end)
    report = generate_compliance_report(team, runs, start, end)
    anomalies = detect_experiment_anomalies(runs)
    return {
        "team": team,
        "report": report.model_dump(),
        "anomalies": anomalies,
    }


@flow(
    name="mlflow-governance-compliance",
    description="Nightly governance compliance sweep across all teams.",
    log_prints=True,
)
def mlflow_compliance_flow(
    teams: list[str],
    lookback_days: int = 1,
) -> list[dict]:
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    results = []
    for team in teams:
        result = audit_team(team, start, end)
        results.append(result)
        report = result["report"]
        logger.info(
            "Team=%s | runs=%d | compliant=%d | violations=%d",
            team,
            report["total_runs"],
            report["compliant_runs"],
            len(report["violations"]),
        )
    return results


if __name__ == "__main__":
    mlflow_compliance_flow(teams=["team-a", "team-b"])
