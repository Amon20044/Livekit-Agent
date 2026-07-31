from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import aiohttp
from livekit.agents import utils

from app.models import RuntimeConfig


@dataclass(slots=True)
class JobMetadata:
    call_id: str = ""
    swarm_id: str = ""
    direction: str = "inbound"


def parse_job_metadata(raw: str | None) -> JobMetadata:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    direction = str(value.get("direction") or "inbound").lower()
    if direction not in {"inbound", "outbound"}:
        direction = "inbound"
    return JobMetadata(
        call_id=str(value.get("callId") or ""),
        swarm_id=str(value.get("swarmId") or ""),
        direction=direction,
    )


class ControlPlaneClient:
    def __init__(self, base_url: str, internal_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "X-Internal-Key": internal_key,
            "Content-Type": "application/json",
        }
        self._timeout = aiohttp.ClientTimeout(total=20)

    async def runtime(self, swarm_id: str, call_id: str = "") -> RuntimeConfig:
        path = f"/runtime/swarms/{swarm_id}"
        if call_id:
            path += f"?callId={call_id}"
        payload = await self._request("GET", path)
        return RuntimeConfig.from_payload(payload)

    async def create_inbound_call(
        self,
        *,
        swarm_id: str,
        room: str,
        caller: str,
        dialed: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/runtime/calls/inbound",
            {
                "swarmId": swarm_id,
                "room": room,
                "from": caller,
                "to": dialed,
                "context": context or {},
                "metadata": {"source": "livekit"},
            },
        )

    async def complete_call(
        self,
        call_id: str,
        transcript: list[dict[str, Any]],
        failure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/runtime/calls/{call_id}/complete",
            {"transcript": transcript, "failure": failure},
        )

    async def record_handoff(
        self,
        call_id: str,
        *,
        from_node: str,
        to_node: str,
        reason: str,
        captured: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/runtime/calls/{call_id}/handoff",
            {
                "fromNode": from_node,
                "toNode": to_node,
                "reason": reason,
                "captured": captured,
            },
        )

    async def search_catalog(self, base_url: str, query: str) -> dict[str, Any]:
        url = base_url.rstrip("/") + "/search/advanced"
        session = utils.http_context.http_session()
        async with session.post(
            url,
            json={"query": query, "page": 1, "pageSize": 5},
            timeout=self._timeout,
        ) as response:
            if response.status >= 400:
                raise RuntimeError(f"catalog returned {response.status}")
            return await response.json()

    async def lookup_order(
        self, base_url: str, order_number: str, phone_last4: str
    ) -> dict[str, Any]:
        url = base_url.rstrip("/") + "/internal/voice/orders/lookup"
        session = utils.http_context.http_session()
        async with session.post(
            url,
            json={"orderNumber": order_number, "phoneLast4": phone_last4},
            headers=self._headers,
            timeout=self._timeout,
        ) as response:
            body = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError("order verification failed")
            if not isinstance(body, dict):
                raise RuntimeError("order service returned an invalid payload")
            return body

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        session = utils.http_context.http_session()
        async with session.request(
            method,
            self._base_url + path,
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
        ) as response:
            body = await response.json(content_type=None)
            if response.status >= 400:
                message = (
                    (body.get("error") or {}).get("message")
                    if isinstance(body, dict)
                    else None
                ) or f"control plane returned {response.status}"
                raise RuntimeError(message)
            if not isinstance(body, dict):
                raise RuntimeError("control plane returned an invalid payload")
            return body
