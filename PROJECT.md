# AgentFlow Multi-Agent System — Project Overview

## Context

AgentFlow is an internal AI governance platform that wraps several infrastructure systems
into a unified governance layer for multiple teams building AI/LLM applications across
the organization.

**Platform components:**
- **MLflow 3** — Experiment tracking, compliance reporting, model governance
- **Label Studio** — Human-in-the-loop annotation workflows
- **LangFuse** — Production AI observability and tracing
- **Prefect** — Pipeline orchestration and scheduling
- **DVC** — Dataset versioning

**Infrastructure:**
- Red Hat Enterprise Linux VMs
- PostgreSQL (schema-per-team isolation)
- S3 Sonic object storage
- Internal network only — no external internet access
- SSO authentication (GDT tokens, OIDC)

**Teams served:** Multiple internal teams across RAG/Q&A and chatbot use cases, with
additional teams onboarding.

---

## Initiative: AgentFlow Multi-Agent System

Build a multi-agent system where each agent is a specialized, autonomous operator for one
AgentFlow platform component. Agents run independently but can be orchestrated for
cross-cutting workflows. External clients (Claude.ai, Claude Code, internal tooling) communicate
with the system through a single MCP server that acts as the front door.

**Framework:** Google ADK (Agent Development Kit)  
**External interface:** MCP Server (single entry point for all external consumers)  
**Scheduling:** Prefect (already deployed)  
**Tracing the agents themselves:** LangFuse (recursive — agents are also governed)

---

## System Architecture

```
External Consumers
├── Claude.ai / Claude Code
├── Internal developer tooling
└── Future Slack bot / portal
        │
        │  (MCP protocol)
        ▼
AgentFlow MCP Server              ← single auth/audit boundary, front door
        │
        │  (ADK internally)
        ▼
AgentFlow Orchestrator Agent      ← master agent, routes to specialists
        │
        ├── LangFuse Agent        → production anomaly detection
        ├── MLflow Agent          → experiment governance & compliance
        └── Label Studio Agent    → annotation quality & HITL workflows
```

### Why MCP as the Front Door

- **Single auth boundary** — all external traffic enters through one controlled gate,
  clean story for risk and compliance teams
- **Reusable interface** — Claude Code, Claude.ai, and future consumers all use the
  same MCP tools without duplicating integration logic
- **ADK stays internal** — agent internals remain an implementation detail;
  consumers only see the high-level MCP tool surface
- **Conversational access** — with AgentFlow MCP connected to Claude.ai, you can
  query your own platform conversationally in real time

### MCP Server Tool Surface

The AgentFlow MCP server exposes high-level tools that map to orchestrator capabilities:

- `run_anomaly_check(team, hours)` — Trigger LangFuse anomaly scan
- `generate_compliance_report(team, period)` — Trigger MLflow compliance report
- `get_annotation_queue_status(project)` — Query Label Studio health
- `get_platform_health_summary()` — Cross-agent status overview
- `trigger_cross_agent_workflow(scenario)` — Initiate coordinated remediation

---

## Agent Roster

### 1. LangFuse Anomaly Detection Agent
**Status:** Build first  
**Responsibility:** Monitor production AI traces and detect quality/performance anomalies
across teams.

**Core tools:**
- `fetch_recent_traces(project, hours)` — Pull traces from LangFuse API for a given team/project
- `fetch_baseline_stats(project, lookback_days)` — Compute rolling baseline metrics
- `compute_anomaly_score(current_metrics, baseline)` — Deterministic statistical comparison
- `dispatch_alert(findings, channel)` — Send structured alert to Slack or internal channel

**Metrics monitored:**

| Metric | Anomaly Signal |
|---|---|
| LLM-as-judge scores | Rolling mean drop > configurable threshold |
| Trace latency (P95) | Spike vs. 7-day baseline |
| Token usage per trace | Sudden increase (prompt injection risk) |
| Error rate | Any spike above threshold |
| Tool call patterns | New or unexpected tools appearing |
| Trace volume | Drop indicating silent failures |

**Key architectural constraint:**  
Anomaly logic must be **deterministic Python** (statistical functions). The LLM layer handles
only interpretation and alert narrative generation. This is non-negotiable for audit
reproducibility.

**Trigger:** Prefect schedule, every N hours (configurable per team SLA)

---

### 2. MLflow Governance & Compliance Agent
**Status:** Build second  
**Responsibility:** Automate experiment governance reporting and surface compliance artifacts
from MLflow across teams.

**Core tools:**
- `fetch_experiments(team, date_range)` — Pull experiment runs for a team
- `check_governance_completeness(run)` — Validate required tags, params, dataset refs are logged
- `generate_compliance_report(team, period)` — Produce structured audit summary
- `detect_experiment_anomalies(runs)` — Flag unusual parameter changes or missing lineage

**Value delivered:**
- Nightly compliance summaries per team (reduces manual audit effort)
- Alert when teams log experiments without required governance metadata
- Cross-team experiment comparison reports for leadership

**Trigger:** Prefect nightly schedule + on-demand via API

---

### 3. Label Studio Annotation Quality Agent
**Status:** Build third  
**Responsibility:** Monitor annotation workflow health, inter-annotator agreement, and
HITL queue backlogs.

**Core tools:**
- `fetch_annotation_stats(project)` — Pull completion rates, agreement scores
- `detect_queue_backlog(project, threshold)` — Flag stalled annotation queues
- `flag_low_agreement_tasks(project)` — Surface tasks with annotator disagreement
- `generate_annotation_report(project, period)` — Weekly annotation quality summary

**Trigger:** Prefect schedule, configurable per project

---

## Cross-Agent Orchestration (Phase 2)

Once individual agents are stable, introduce an optional orchestrator for cross-cutting
workflows. Example trigger chain:

```
LangFuse Agent detects quality drop in RAG pipeline
    → MLflow Agent checks if a recent experiment deploy correlates
    → Label Studio Agent triggers re-annotation of affected evaluation examples
    → MLflow Agent creates new experiment run tracking the remediation
```

This orchestration is where the compounded value of the multi-agent system emerges.

**Do not build this until all three agents operate reliably in isolation.**

---

## Implementation Principles

### 1. Deterministic logic, LLM narration
Statistical computations, thresholds, and anomaly scoring must be pure Python functions.
The LLM layer is used only to generate human-readable summaries and alert messages.
This ensures reproducibility for compliance and audit teams.

### 2. Each agent is independently deployable
No shared state between agents at runtime. Each agent is a separate Prefect flow with its
own schedule, failure handling, and alerting. One agent crashing does not affect others.

### 3. Agents are themselves governed
All agent runs are traced in LangFuse (recursive governance). Agent runs should also log
a summary artifact in MLflow for audit purposes. Agents eat their own dog food.

### 4. Auth follows existing patterns
Use established AgentFlow authentication patterns — GDT token extraction via sidecar,
header-based auth for internal services. Do not introduce new auth mechanisms.

### 5. No external network calls
All API calls are to internal services only. LangFuse, MLflow, and Label Studio are all
deployed internally. This is a hard infrastructure constraint.

---

## Suggested Project Structure

```
agentflow-agents/
│
├── mcp_server/                   # Phase 5 — external-facing front door
│   ├── server.py                 # MCP server definition and tool registration
│   ├── tools.py                  # High-level tool handlers (delegate to orchestrator)
│   └── auth.py                   # GDT token validation at entry point
│
├── orchestrator/
│   ├── coordinator.py            # Master ADK agent — routes to specialist agents
│   └── workflows.py              # Cross-agent workflow definitions (Phase 4)
│
├── agents/
│   ├── langfuse_agent/
│   │   ├── agent.py              # ADK agent definition
│   │   ├── tools.py              # LangFuse API tool functions
│   │   ├── anomaly.py            # Deterministic anomaly scoring logic
│   │   └── config.py             # Thresholds, project list, schedules
│   │
│   ├── mlflow_agent/
│   │   ├── agent.py
│   │   ├── tools.py              # MLflow tracking API tools
│   │   ├── compliance.py         # Report generation logic
│   │   └── config.py
│   │
│   └── label_studio_agent/
│       ├── agent.py
│       ├── tools.py              # Label Studio API tools
│       ├── quality.py            # Agreement scoring logic
│       └── config.py
│
├── flows/
│   ├── langfuse_flow.py          # Prefect flow wrapping LangFuse agent
│   ├── mlflow_flow.py
│   └── label_studio_flow.py
│
├── shared/
│   ├── auth.py                   # GDT token / header auth helpers
│   ├── alerts.py                 # Unified alert dispatching
│   └── models.py                 # Shared Pydantic data models
│
└── tests/
    ├── test_anomaly.py
    ├── test_compliance.py
    ├── test_quality.py
    └── test_mcp_server.py
```

---

## Build Sequence

| Phase | Deliverable | Notes |
|---|---|---|
| 1 | LangFuse Anomaly Detection Agent | ADK + Prefect, standalone |
| 2 | MLflow Governance & Compliance Agent | ADK + Prefect, standalone |
| 3 | Label Studio Annotation Quality Agent | ADK + Prefect, standalone |
| 4 | AgentFlow Orchestrator Agent | ADK master agent, routes to 1-3 |
| 5 | AgentFlow MCP Server | Front door — only after orchestrator is stable |

**Do not build Phase 5 until Phase 4 is stable.** The MCP server is only valuable once
the orchestrator has something meaningful to expose.

---

## Out of Scope (for now)

- UI for agent management (use Prefect UI for run visibility)
- Agent self-healing or auto-remediation (alert and report only)
- External system integration beyond LangFuse, MLflow, Label Studio
- Real-time streaming (scheduled batch runs are sufficient)
- MCP servers per individual agent — one MCP server at the orchestrator level is sufficient

---

*Last updated: March 2026*
