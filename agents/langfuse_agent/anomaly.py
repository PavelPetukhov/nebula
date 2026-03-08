"""
Deterministic anomaly scoring logic for LangFuse traces.

All statistical computations live here as pure Python functions.
The LLM layer must NOT perform anomaly detection — only narration.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from shared.models import AnomalyFinding, Severity


@dataclass
class TraceMetrics:
    judge_scores: list[float]
    latencies_p95: list[float]   # ms
    token_counts: list[float]
    error_rate: float             # 0.0–1.0
    trace_count: int
    tool_names: set[str]


@dataclass
class BaselineMetrics:
    judge_score_mean: float
    judge_score_std: float
    latency_p95_mean: float
    latency_p95_std: float
    token_mean: float
    token_std: float
    error_rate_mean: float
    error_rate_std: float
    trace_volume_mean: float
    trace_volume_std: float
    known_tools: set[str]


def compute_baseline(windows: list[TraceMetrics]) -> BaselineMetrics:
    """Compute rolling baseline statistics from historical windows."""
    judge = [s for w in windows for s in w.judge_scores]
    latencies = [l for w in windows for l in w.latencies_p95]
    tokens = [t for w in windows for t in w.token_counts]
    error_rates = [w.error_rate for w in windows]
    volumes = [w.trace_count for w in windows]
    all_tools: set[str] = set()
    for w in windows:
        all_tools |= w.tool_names

    def safe_std(values: list[float]) -> float:
        return float(np.std(values)) if len(values) > 1 else 1.0

    return BaselineMetrics(
        judge_score_mean=float(np.mean(judge)) if judge else 0.0,
        judge_score_std=safe_std(judge),
        latency_p95_mean=float(np.mean(latencies)) if latencies else 0.0,
        latency_p95_std=safe_std(latencies),
        token_mean=float(np.mean(tokens)) if tokens else 0.0,
        token_std=safe_std(tokens),
        error_rate_mean=float(np.mean(error_rates)) if error_rates else 0.0,
        error_rate_std=safe_std(error_rates),
        trace_volume_mean=float(np.mean(volumes)) if volumes else 0.0,
        trace_volume_std=safe_std(volumes),
        known_tools=all_tools,
    )


def _z_score(value: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (value - mean) / std


def _severity_from_z(z: float, thresholds: tuple[float, float, float]) -> Severity:
    """Map absolute z-score to severity using (medium, high, critical) thresholds."""
    az = abs(z)
    if az >= thresholds[2]:
        return Severity.CRITICAL
    if az >= thresholds[1]:
        return Severity.HIGH
    if az >= thresholds[0]:
        return Severity.MEDIUM
    return Severity.LOW


def compute_anomaly_score(
    current: TraceMetrics,
    baseline: BaselineMetrics,
    judge_z_threshold: float = 2.0,
    latency_z_threshold: float = 2.5,
    token_z_threshold: float = 3.0,
    error_rate_z_threshold: float = 2.0,
    volume_z_threshold: float = 2.5,
) -> list[AnomalyFinding]:
    """
    Compare current window metrics against baseline and return anomaly findings.
    Returns an empty list when everything is within normal bounds.
    """
    findings: list[AnomalyFinding] = []

    # Judge score — flag drops (negative z)
    if current.judge_scores:
        current_judge_mean = float(np.mean(current.judge_scores))
        z = _z_score(current_judge_mean, baseline.judge_score_mean, baseline.judge_score_std)
        if z <= -judge_z_threshold:
            findings.append(AnomalyFinding(
                metric="llm_judge_score",
                current_value=current_judge_mean,
                baseline_value=baseline.judge_score_mean,
                z_score=z,
                severity=_severity_from_z(z, (judge_z_threshold, judge_z_threshold + 1, judge_z_threshold + 2)),
                description=f"LLM-as-judge mean score dropped to {current_judge_mean:.3f} "
                            f"(baseline {baseline.judge_score_mean:.3f}, z={z:.2f})",
            ))

    # P95 latency — flag spikes (positive z)
    if current.latencies_p95:
        current_p95 = float(np.percentile(current.latencies_p95, 95))
        z = _z_score(current_p95, baseline.latency_p95_mean, baseline.latency_p95_std)
        if z >= latency_z_threshold:
            findings.append(AnomalyFinding(
                metric="latency_p95_ms",
                current_value=current_p95,
                baseline_value=baseline.latency_p95_mean,
                z_score=z,
                severity=_severity_from_z(z, (latency_z_threshold, latency_z_threshold + 1, latency_z_threshold + 2)),
                description=f"P95 latency spiked to {current_p95:.0f}ms "
                            f"(baseline {baseline.latency_p95_mean:.0f}ms, z={z:.2f})",
            ))

    # Token usage — flag increases (positive z, prompt injection risk)
    if current.token_counts:
        current_token_mean = float(np.mean(current.token_counts))
        z = _z_score(current_token_mean, baseline.token_mean, baseline.token_std)
        if z >= token_z_threshold:
            findings.append(AnomalyFinding(
                metric="token_usage_per_trace",
                current_value=current_token_mean,
                baseline_value=baseline.token_mean,
                z_score=z,
                severity=_severity_from_z(z, (token_z_threshold, token_z_threshold + 1, token_z_threshold + 2)),
                description=f"Token usage per trace increased to {current_token_mean:.0f} "
                            f"(baseline {baseline.token_mean:.0f}, z={z:.2f}) — possible prompt injection",
            ))

    # Error rate — flag spikes (positive z)
    z = _z_score(current.error_rate, baseline.error_rate_mean, baseline.error_rate_std)
    if z >= error_rate_z_threshold:
        findings.append(AnomalyFinding(
            metric="error_rate",
            current_value=current.error_rate,
            baseline_value=baseline.error_rate_mean,
            z_score=z,
            severity=_severity_from_z(z, (error_rate_z_threshold, error_rate_z_threshold + 1, error_rate_z_threshold + 2)),
            description=f"Error rate spiked to {current.error_rate:.1%} "
                        f"(baseline {baseline.error_rate_mean:.1%}, z={z:.2f})",
        ))

    # Trace volume — flag drops (negative z = silent failure)
    z = _z_score(current.trace_count, baseline.trace_volume_mean, baseline.trace_volume_std)
    if z <= -volume_z_threshold:
        findings.append(AnomalyFinding(
            metric="trace_volume",
            current_value=float(current.trace_count),
            baseline_value=baseline.trace_volume_mean,
            z_score=z,
            severity=_severity_from_z(z, (volume_z_threshold, volume_z_threshold + 1, volume_z_threshold + 2)),
            description=f"Trace volume dropped to {current.trace_count} "
                        f"(baseline {baseline.trace_volume_mean:.0f}, z={z:.2f}) — possible silent failure",
        ))

    # Tool call patterns — flag new/unexpected tools
    new_tools = current.tool_names - baseline.known_tools
    if new_tools:
        findings.append(AnomalyFinding(
            metric="unexpected_tools",
            current_value=float(len(new_tools)),
            baseline_value=0.0,
            z_score=float(len(new_tools)),
            severity=Severity.HIGH,
            description=f"New unexpected tool(s) appeared: {', '.join(sorted(new_tools))}",
        ))

    return findings
