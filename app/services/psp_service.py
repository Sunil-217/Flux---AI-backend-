"""Fluxway PSP-orchestration client.

Talks to the standalone Fluxway payments backend over HTTP using a brand
**environment secret token** (`X-SECRET-TOKEN`). That token is bound to exactly
one brand+environment, so anything onboarded through it lands at that brand
level — which is exactly what the Payment Gateways admin needs.

Two operations back the admin UI:
  • list_catalog()  → GET /flow-types/definitions?type=<PSP_FLOW_TYPE>
                      the PSP "templates" (FlowTargets) + their credential JSON
                      Schema + the action/definition ids used to wire flows.
  • onboard_psp()   → POST /psps/external
                      validates credentials against the target schema, encrypts
                      them, and enables the PSP at brand level.

Secrets policy: credentials are forwarded to Fluxway once (it encrypts + stores
them) and are NEVER persisted or logged on the Close AI side.

Everything is env-gated: if FLUXWAY_BASE_URL / FLUXWAY_SECRET_TOKEN are unset,
`is_configured()` is False and callers fall back to local-only behaviour.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import requests

from app.core.config import (
    FLUXWAY_BASE_URL,
    FLUXWAY_PSP_FLOW_TYPE,
    FLUXWAY_SECRET_TOKEN,
    FLUXWAY_TIMEOUT,
)


class FluxwayError(Exception):
    """A Fluxway call failed — message is safe to surface to the admin."""


def is_configured() -> bool:
    return bool(FLUXWAY_BASE_URL and FLUXWAY_SECRET_TOKEN)


def _headers() -> dict[str, str]:
    return {
        "X-SECRET-TOKEN": FLUXWAY_SECRET_TOKEN,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _unwrap(resp: requests.Response) -> Any:
    """Pull `data` out of Fluxway's `{timestamp, code, message, data}` envelope,
    raising a clean FluxwayError on any non-2xx or transport problem."""
    try:
        body = resp.json()
    except ValueError:
        body = None
    if not resp.ok:
        msg = None
        if isinstance(body, dict):
            msg = body.get("message")
        raise FluxwayError(msg or f"Fluxway returned HTTP {resp.status_code}.")
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


def _get(path: str, params: Optional[dict] = None) -> Any:
    try:
        resp = requests.get(
            f"{FLUXWAY_BASE_URL}{path}", headers=_headers(), params=params, timeout=FLUXWAY_TIMEOUT
        )
    except requests.RequestException as exc:
        raise FluxwayError(f"Could not reach Fluxway: {exc}") from exc
    return _unwrap(resp)


def _post(path: str, payload: dict) -> Any:
    try:
        resp = requests.post(
            f"{FLUXWAY_BASE_URL}{path}", headers=_headers(), json=payload, timeout=FLUXWAY_TIMEOUT
        )
    except requests.RequestException as exc:
        raise FluxwayError(f"Could not reach Fluxway: {exc}") from exc
    return _unwrap(resp)


def list_catalog() -> dict:
    """Return the PSP catalog for the configured flow type:
    `{"flow_type": str, "targets": [ {id, name, logo, credential_schema,
      input_schema, operations:[{flow_action_id, flow_definition_id}]} ]}`.

    `operations` is pre-built from each target's actions that have a definition,
    ready to pass straight to onboarding."""
    data = _get("/flow-types/definitions", params={"type": FLUXWAY_PSP_FLOW_TYPE})
    targets_out = []
    for t in (data or {}).get("targets", []) if isinstance(data, dict) else []:
        ops = []
        for a in t.get("actions", []) or []:
            definition = a.get("definition") or {}
            if definition.get("id"):
                ops.append(
                    {"flow_action_id": a.get("id"), "flow_definition_id": definition.get("id")}
                )
        targets_out.append(
            {
                "id": t.get("id"),
                "name": t.get("name"),
                "logo": t.get("logo"),
                "credential_schema": t.get("credentialSchema"),
                "input_schema": t.get("inputSchema"),
                "operations": ops,
            }
        )
    return {"flow_type": (data or {}).get("name") if isinstance(data, dict) else FLUXWAY_PSP_FLOW_TYPE,
            "targets": targets_out}


def get_target(flow_target_id: str) -> Optional[dict]:
    """Find one target (with its pre-built operations) in the catalog."""
    for t in list_catalog()["targets"]:
        if t["id"] == flow_target_id:
            return t
    return None


def onboard_psp(
    name: str,
    flow_target_id: str,
    credential: dict,
    operations: list[dict],
    description: Optional[str] = None,
    logo: Optional[str] = None,
) -> dict:
    """Create + enable a PSP at brand level on Fluxway. `credential` is sent as a
    JSON string (Fluxway parses, schema-validates, then per-value encrypts it).
    `operations` is the list from the catalog. Returns the created PSP details
    (includes the remote `id` / `brandId` / `environmentId`)."""
    payload = {
        "name": name,
        "description": description,
        "logo": logo,
        "credential": json.dumps(credential),  # Fluxway expects a JSON string
        # PspDto marks brandId/environmentId as required, so the body must carry
        # non-empty values to pass Pydantic validation — but /psps/external
        # OVERRIDES both from the X-SECRET-TOKEN context (create_external), so
        # these placeholders are never persisted. The real ids come back in the
        # response and are what we store locally.
        "brandId": "__from_secret_token__",
        "environmentId": "__from_secret_token__",
        "flowTargetId": flow_target_id,
        "status": "ENABLED",
        "operations": [
            {
                "flowActionId": op["flow_action_id"],
                "flowDefinitionId": op["flow_definition_id"],
                "status": "ENABLED",
            }
            for op in operations
        ],
    }
    return _post("/psps/external", payload)
