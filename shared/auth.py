"""GDT token / header auth helpers for internal service access."""

import os
from typing import Any

import httpx


def get_gdt_token() -> str | None:
    """Retrieve the GDT token from the environment (injected by sidecar)."""
    return os.getenv("GDT_TOKEN")


def internal_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build standard headers for internal service calls."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = get_gdt_token()
    if token:
        header_name = os.getenv("GDT_TOKEN_HEADER", "X-GDT-Token")
        headers[header_name] = token
    if extra:
        headers.update(extra)
    return headers


def build_httpx_client(base_url: str, timeout: float = 30.0) -> httpx.Client:
    """Return a pre-configured synchronous HTTPX client for internal services."""
    return httpx.Client(
        base_url=base_url,
        headers=internal_headers(),
        timeout=timeout,
    )


def build_async_httpx_client(base_url: str, timeout: float = 30.0) -> httpx.AsyncClient:
    """Return a pre-configured async HTTPX client for internal services."""
    return httpx.AsyncClient(
        base_url=base_url,
        headers=internal_headers(),
        timeout=timeout,
    )


def validate_gdt_token(token: str) -> dict[str, Any]:
    """
    Validate a GDT token at the MCP server entry point.
    Returns decoded claims or raises ValueError on failure.
    Placeholder — wire to your SSO/OIDC introspection endpoint.
    """
    if not token:
        raise ValueError("Missing GDT token")
    # TODO: call internal OIDC introspection endpoint
    return {"sub": "unknown", "token": token}
