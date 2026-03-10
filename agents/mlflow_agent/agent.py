"""MLflow Governance & Compliance Agent — ADK agent definition."""

from google.adk.agents import Agent

from agents.mlflow_agent.config import config

# Tools imported here for ADK registration
from agents.mlflow_agent.tools import (
    check_governance_completeness,
    detect_experiment_anomalies,
    fetch_experiments,
)

mlflow_agent = Agent(
    name="mlflow_governance_agent",
    model=config.model,
    description=(
        "Automates MLflow experiment governance reporting. "
        "Validates required metadata tags, detects missing lineage, "
        "and produces compliance summaries for audit teams."
    ),
    instruction=(
        "You are the MLflow Governance & Compliance Agent. "
        "Use the provided tools to fetch runs, check governance completeness, "
        "and generate compliance reports. "
        "Summarize findings in clear language suitable for team leads and compliance officers. "
        "Do not modify experiment data — report only."
    ),
    tools=[fetch_experiments, check_governance_completeness, detect_experiment_anomalies],
)

root_agent = mlflow_agent
