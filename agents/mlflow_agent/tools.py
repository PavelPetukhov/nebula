"""MLflow tracking API tools."""

import logging
import os
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from agents.mlflow_agent.config import config

logger = logging.getLogger(__name__)


def _get_client() -> MlflowClient:
    os.environ["MLFLOW_TRACKING_URI"] = config.mlflow_tracking_uri
    if config.mlflow_tracking_token:
        os.environ["MLFLOW_TRACKING_TOKEN"] = config.mlflow_tracking_token
    return MlflowClient()


def fetch_experiments(team: str, start_date: datetime, end_date: datetime) -> list[dict]:
    """Pull experiment runs for a team within a date range."""
    client = _get_client()
    experiments = client.search_experiments(filter_string=f"tags.team = '{team}'")
    runs = []
    for exp in experiments:
        exp_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"attributes.start_time >= {int(start_date.timestamp() * 1000)} "
                          f"AND attributes.start_time <= {int(end_date.timestamp() * 1000)}",
            max_results=200,
        )
        for run in exp_runs:
            runs.append({
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "experiment_name": exp.name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "tags": dict(run.data.tags),
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
            })
    logger.info("Fetched %d runs for team=%s", len(runs), team)
    return runs


def check_governance_completeness(run: dict) -> dict:
    """Validate required tags and params are present on a run."""
    missing_tags = [t for t in config.required_tags if t not in run.get("tags", {})]
    missing_params = [p for p in config.required_params if p not in run.get("params", {})]
    return {
        "run_id": run["run_id"],
        "compliant": not missing_tags and not missing_params,
        "missing_tags": missing_tags,
        "missing_params": missing_params,
    }


def detect_experiment_anomalies(runs: list[dict]) -> list[dict]:
    """Flag runs with missing lineage or unusual parameter patterns."""
    anomalies = []
    for run in runs:
        check = check_governance_completeness(run)
        if not check["compliant"]:
            anomalies.append({
                "run_id": run["run_id"],
                "issue": "missing_governance_metadata",
                "details": check,
            })
    return anomalies
