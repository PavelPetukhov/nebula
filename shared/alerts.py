"""Unified alert dispatching — Slack webhooks and internal channels."""

import json
import logging
import os

import httpx

from shared.models import AlertPayload, Severity

logger = logging.getLogger(__name__)

SEVERITY_EMOJI = {
    Severity.LOW: ":information_source:",
    Severity.MEDIUM: ":warning:",
    Severity.HIGH: ":rotating_light:",
    Severity.CRITICAL: ":fire:",
}


def _slack_blocks(payload: AlertPayload) -> list[dict]:
    emoji = SEVERITY_EMOJI.get(payload.severity, ":bell:")
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{emoji} {payload.title}"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Agent:*\n{payload.agent}"},
                {"type": "mrkdwn", "text": f"*Severity:*\n{payload.severity.value.upper()}"},
                {"type": "mrkdwn", "text": f"*Team:*\n{payload.team or 'N/A'}"},
                {"type": "mrkdwn", "text": f"*Project:*\n{payload.project or 'N/A'}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Summary:*\n{payload.summary}"},
        },
    ]
    if payload.findings:
        finding_lines = []
        for f in payload.findings[:5]:  # cap at 5 for readability
            finding_lines.append(
                f"• `{f.metric}`: {f.current_value:.3f} vs baseline {f.baseline_value:.3f}"
                f" (z={f.z_score:.2f}, {f.severity.value})"
            )
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Findings:*\n" + "\n".join(finding_lines)},
            }
        )
    return blocks


def dispatch_alert(payload: AlertPayload, channel: str = "slack") -> bool:
    """
    Send a structured alert to the specified channel.

    Args:
        payload: Structured alert payload.
        channel: Currently supports 'slack'. Extend for other channels.

    Returns:
        True on success, False on failure.
    """
    if channel == "slack":
        return _send_slack_alert(payload)
    logger.warning("Unknown alert channel: %s", channel)
    return False


def _send_slack_alert(payload: AlertPayload) -> bool:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.error("SLACK_WEBHOOK_URL not configured — alert not sent")
        return False

    body = {"blocks": _slack_blocks(payload)}
    try:
        resp = httpx.post(webhook_url, content=json.dumps(body), timeout=10.0)
        resp.raise_for_status()
        logger.info("Slack alert sent: %s", payload.title)
        return True
    except httpx.HTTPError as exc:
        logger.error("Failed to send Slack alert: %s", exc)
        return False
