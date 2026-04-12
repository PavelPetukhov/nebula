"""Label Studio Annotation Quality Agent — ADK agent definition."""

from google.adk.agents import Agent
from google.adk.models.anthropic_llm import AnthropicLlm

from agents.label_studio_agent.config import config
from agents.label_studio_agent.tools import (
    detect_queue_backlog,
    fetch_annotation_stats,
    flag_low_agreement_tasks,
)
from shared.mlflow_tracking import MLflowAdkTracker

_tracker = MLflowAdkTracker()

label_studio_agent = Agent(
    name="label_studio_quality_agent",
    model=AnthropicLlm(model=config.model),
    description=(
        "Monitors Label Studio annotation workflow health. "
        "Detects queue backlogs, low inter-annotator agreement, and stalled projects."
    ),
    instruction=(
        "You are the Label Studio Annotation Quality Agent. "
        "Use the provided tools to check annotation queue health and agreement scores. "
        "Summarize issues for project managers and ML engineers in clear, concise language. "
        "Do not modify annotations or task assignments."
    ),
    tools=[fetch_annotation_stats, detect_queue_backlog, flag_low_agreement_tasks],
    before_agent_callback=_tracker.before_agent_callback,
    after_agent_callback=_tracker.after_agent_callback,
    before_tool_callback=_tracker.before_tool_callback,
    after_tool_callback=_tracker.after_tool_callback,
    before_model_callback=_tracker.before_model_callback,
    after_model_callback=_tracker.after_model_callback,
)

root_agent = label_studio_agent
