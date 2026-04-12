"""MLflow Governance & Compliance Agent — ADK agent definition."""

from google.adk.agents import Agent
from google.adk.models.anthropic_llm import AnthropicLlm

from agents.mlflow_agent.config import config

# Tools imported here for ADK registration
from agents.mlflow_agent.tools import (
    check_governance_completeness,
    detect_experiment_anomalies,
    fetch_experiments,
)
from shared.mlflow_tracking import MLflowAdkTracker

_tracker = MLflowAdkTracker()

mlflow_agent = Agent(
    name="mlflow_governance_agent",
    model=AnthropicLlm(model=config.model),
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
    before_agent_callback=_tracker.before_agent_callback,
    after_agent_callback=_tracker.after_agent_callback,
    before_tool_callback=_tracker.before_tool_callback,
    after_tool_callback=_tracker.after_tool_callback,
    before_model_callback=_tracker.before_model_callback,
    after_model_callback=_tracker.after_model_callback,
)

root_agent = mlflow_agent
