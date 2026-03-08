"""Shared Pydantic data models used across all agents."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyFinding(BaseModel):
    metric: str
    current_value: float
    baseline_value: float
    z_score: float
    severity: Severity
    description: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentRunSummary(BaseModel):
    agent_name: str
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str  # "running" | "success" | "failed"
    findings_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlertPayload(BaseModel):
    title: str
    severity: Severity
    agent: str
    team: str | None = None
    project: str | None = None
    findings: list[AnomalyFinding] = Field(default_factory=list)
    summary: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ComplianceReport(BaseModel):
    team: str
    period_start: datetime
    period_end: datetime
    total_runs: int
    compliant_runs: int
    violations: list[dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class AnnotationQueueStatus(BaseModel):
    project: str
    total_tasks: int
    completed_tasks: int
    pending_tasks: int
    in_review_tasks: int
    mean_agreement_score: float | None = None
    backlog_flag: bool = False
    low_agreement_task_ids: list[int] = Field(default_factory=list)
