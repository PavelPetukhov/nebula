# Nebula

Multi-agent AI governance system. Each agent is a specialized, autonomous operator for one platform component — monitoring, compliance, and annotation quality across internal AI teams.

## Agents

| Agent | Responsibility | Phase |
|---|---|---|
| LangFuse Agent | Production trace anomaly detection | 1 |
| MLflow Agent | Experiment governance & compliance reporting | 2 |
| Label Studio Agent | Annotation quality & queue health | 3 |
| Orchestrator | Cross-agent workflow coordination | 4 |
| MCP Server | External-facing entry point | 5 |

## Project Structure

```
agents/
  langfuse_agent/   # Trace anomaly detection
  mlflow_agent/     # Governance & compliance
  label_studio_agent/ # Annotation quality
orchestrator/       # Master agent (Phase 4)
mcp_server/         # External interface (Phase 5)
flows/              # Prefect flow definitions
shared/             # Models, auth, alerts
tests/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in endpoints and keys
```

## Running Tests

```bash
pytest tests/
```

## Configuration

All configuration is via environment variables. See `.env.example` for the full list. Key variables:

- `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_TOKEN`
- `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY`
- `SLACK_WEBHOOK_URL`

## Design Principles

- **Deterministic anomaly logic** — all statistical scoring is pure Python; the LLM layer generates narratives only
- **Independently deployable** — each agent runs as its own Prefect flow with no shared runtime state
- **No external network calls** — all services are internal
