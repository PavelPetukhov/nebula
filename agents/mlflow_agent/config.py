"""MLflow agent configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class MLflowAgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_tracking_token: str = ""

    # Governance required tags — runs missing these are flagged
    required_tags: list[str] = ["team", "dataset_version", "model_version", "owner"]
    required_params: list[str] = []

    model: str = "claude-sonnet-4-6"
    alert_channel: str = "slack"


config = MLflowAgentConfig()
