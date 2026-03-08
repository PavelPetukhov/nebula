"""Unit tests for deterministic anomaly scoring logic."""

import pytest

from agents.langfuse_agent.anomaly import (
    BaselineMetrics,
    TraceMetrics,
    compute_anomaly_score,
    compute_baseline,
)
from shared.models import Severity


def _make_baseline(**overrides) -> BaselineMetrics:
    defaults = dict(
        judge_score_mean=0.85,
        judge_score_std=0.05,
        latency_p95_mean=500.0,
        latency_p95_std=50.0,
        token_mean=1000.0,
        token_std=100.0,
        error_rate_mean=0.02,
        error_rate_std=0.01,
        trace_volume_mean=200.0,
        trace_volume_std=20.0,
        known_tools={"search", "calculator"},
    )
    defaults.update(overrides)
    return BaselineMetrics(**defaults)


def _make_current(**overrides) -> TraceMetrics:
    defaults = dict(
        judge_scores=[0.85] * 50,
        latencies_p95=[500.0] * 50,
        token_counts=[1000.0] * 50,
        error_rate=0.02,
        trace_count=200,
        tool_names={"search", "calculator"},
    )
    defaults.update(overrides)
    return TraceMetrics(**defaults)


def test_no_anomaly_when_normal():
    baseline = _make_baseline()
    current = _make_current()
    findings = compute_anomaly_score(current, baseline)
    assert findings == []


def test_judge_score_drop_flagged():
    baseline = _make_baseline(judge_score_mean=0.85, judge_score_std=0.05)
    current = _make_current(judge_scores=[0.50] * 50)  # large drop
    findings = compute_anomaly_score(current, baseline)
    metrics = {f.metric for f in findings}
    assert "llm_judge_score" in metrics


def test_latency_spike_flagged():
    baseline = _make_baseline(latency_p95_mean=500.0, latency_p95_std=50.0)
    current = _make_current(latencies_p95=[900.0] * 50)  # spike
    findings = compute_anomaly_score(current, baseline)
    metrics = {f.metric for f in findings}
    assert "latency_p95_ms" in metrics


def test_token_spike_flagged():
    baseline = _make_baseline(token_mean=1000.0, token_std=100.0)
    current = _make_current(token_counts=[1500.0] * 50)  # 5 sigma spike
    findings = compute_anomaly_score(current, baseline)
    metrics = {f.metric for f in findings}
    assert "token_usage_per_trace" in metrics


def test_error_rate_spike_flagged():
    baseline = _make_baseline(error_rate_mean=0.02, error_rate_std=0.01)
    current = _make_current(error_rate=0.15)  # big spike
    findings = compute_anomaly_score(current, baseline)
    metrics = {f.metric for f in findings}
    assert "error_rate" in metrics


def test_volume_drop_flagged():
    baseline = _make_baseline(trace_volume_mean=200.0, trace_volume_std=20.0)
    current = _make_current(trace_count=50)  # silent failure
    findings = compute_anomaly_score(current, baseline)
    metrics = {f.metric for f in findings}
    assert "trace_volume" in metrics


def test_new_tool_flagged():
    baseline = _make_baseline(known_tools={"search"})
    current = _make_current(tool_names={"search", "exfil_tool"})
    findings = compute_anomaly_score(current, baseline)
    metrics = {f.metric for f in findings}
    assert "unexpected_tools" in metrics
    tool_finding = next(f for f in findings if f.metric == "unexpected_tools")
    assert tool_finding.severity == Severity.HIGH


def test_compute_baseline_aggregates_windows():
    windows = [
        TraceMetrics([0.8, 0.9], [400.0, 500.0], [900.0, 1100.0], 0.01, 180, {"search"}),
        TraceMetrics([0.85, 0.88], [480.0, 520.0], [950.0, 1050.0], 0.02, 210, {"search", "calc"}),
    ]
    baseline = compute_baseline(windows)
    assert 0.8 < baseline.judge_score_mean < 0.9
    assert baseline.known_tools == {"search", "calc"}
