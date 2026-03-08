"""Label Studio Annotation Quality Agent — ADK agent definition."""

from google.adk.agents import Agent

from agents.label_studio_agent.config import config
from agents.label_studio_agent.tools import (
    detect_queue_backlog,
    fetch_annotation_stats,
    flag_low_agreement_tasks,
)

label_studio_agent = Agent(
    name="label_studio_quality_agent",
    model=config.model,
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
)
