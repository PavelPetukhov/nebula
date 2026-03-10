"""LangFuse agent configuration — thresholds, project list, schedules."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class LangFuseAgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Anomaly detection thresholds
    judge_score_z_threshold: float = 2.0   # z-score to flag judge score drop
    latency_z_threshold: float = 2.5       # z-score for P95 latency spike
    token_usage_z_threshold: float = 3.0   # z-score for token usage increase
    error_rate_z_threshold: float = 2.0    # z-score for error rate spike
    trace_volume_z_threshold: float = 2.5  # z-score for volume drop (negative)

    # Baseline window
    baseline_lookback_days: int = 7

    # Default fetch window
    default_trace_hours: int = 6

    # Projects to monitor (comma-separated env var, or set directly)
    monitored_projects: list[str] = []

    # Alert channel
    alert_channel: str = "slack"

    # ADK model
    model: str = "anthropic/claude-sonnet-4-6"


config = LangFuseAgentConfig()
