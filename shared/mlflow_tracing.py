"""
Wire ADK's built-in OpenTelemetry spans to MLflow's trace ingestion endpoint.

ADK instruments every model call, tool call, and agent hop as OTel spans
automatically.  All we need to do is point the OTel exporter at MLflow.

This is driven entirely by env vars (set in .env):

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5001
    OTEL_EXPORTER_OTLP_HEADERS=x-mlflow-experiment-id=<id>
    LANGFUSE_OTEL_EXPORT_ENABLED=false   ← prevents Langfuse from overriding

Call setup_mlflow_tracing() once at the top of the orchestrator module,
before any agents are imported, so the OTel provider is in place when ADK
registers its instrumentation.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_mlflow_tracing() -> None:
    """
    Install the OTel → MLflow trace exporter.

    Reads OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS from
    the environment (loaded from .env if not already set).  No-ops if the
    endpoint is not configured, so missing env vars never crash the agent.
    """
    from dotenv import load_dotenv
    load_dotenv(override=False)

    # Prefer the traces-specific endpoint to avoid ADK auto-creating logs/metrics
    # exporters (MLflow has no /v1/logs or /v1/metrics, causing 404 spam when
    # the generic OTEL_EXPORTER_OTLP_ENDPOINT is used instead).
    endpoint = (
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    if not endpoint:
        logger.warning(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT not set — MLflow trace ingestion disabled."
        )
        return

    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        # Parse headers from the env var  (key=value,key=value format)
        raw_headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
        headers: dict[str, str] = {}
        for pair in raw_headers.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                headers[k.strip()] = v.strip()

        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)

        logger.warning(
            "MLflow tracing active → %s (headers: %s)", endpoint, headers
        )
    except Exception as exc:
        logger.warning("MLflow tracing setup failed: %s", exc)
