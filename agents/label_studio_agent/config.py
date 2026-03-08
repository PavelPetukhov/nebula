"""Label Studio agent configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class LabelStudioAgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    label_studio_url: str = "http://localhost:8080"
    label_studio_api_key: str = ""

    # Backlog threshold: flag queue if pending tasks exceed this count
    backlog_threshold: int = 100

    # Agreement threshold: flag tasks below this inter-annotator agreement score
    agreement_threshold: float = 0.7

    model: str = "claude-sonnet-4-6"
    alert_channel: str = "slack"


config = LabelStudioAgentConfig()
