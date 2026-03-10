"""Nebula Orchestrator — routes user questions to the correct specialist agent."""

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from agents.langfuse_agent.agent import langfuse_agent
from agents.label_studio_agent.agent import label_studio_agent
from agents.mlflow_agent.agent import mlflow_agent

orchestrator = Agent(
    name="nebula_orchestrator",
    model="anthropic/claude-sonnet-4-6",
    description="Central router for the Nebula AI governance system.",
    instruction=(
        "You are the Nebula orchestrator. Your job is to understand the user's question "
        "and delegate it to the right specialist agent:\n\n"
        "- langfuse_anomaly_agent: production trace health — latency spikes, error rates, "
        "judge score drops, token usage anomalies\n"
        "- mlflow_governance_agent: experiment tracking, model metadata compliance, "
        "lineage validation, audit reports\n"
        "- label_studio_quality_agent: annotation queue health, inter-annotator agreement, "
        "backlog detection, labeler performance\n\n"
        "If the question spans multiple domains, call each relevant agent and synthesize "
        "the results into a single coherent response. "
        "Always attribute which agent produced which findings."
    ),
    tools=[
        AgentTool(agent=langfuse_agent),
        AgentTool(agent=mlflow_agent),
        AgentTool(agent=label_studio_agent),
    ],
)

root_agent = orchestrator
