"""Safety Gate (SGRG) interoperability gateway — synchronous client.

Vendored from `ai-alert-assistant/src/utils/`: `interoperability.py` (auth,
retry), `sgrg_core_methods.py` (envelope, methods) and `prepare_training_data.py`
(attachment discovery).

    DO NOT FIX A GATEWAY BUG HERE. Fix it in ai-alert-assistant and re-vendor,
    or the two copies drift and whichever one nobody is looking at stays broken.
    Extracting a shared package is the tracked follow-up; vendoring is the
    deliberate short-term choice because that repo is a FastAPI application, not
    something installable.

Two differences from the original, both deliberate:

**Synchronous.** The original is `async`. A Haystack component's `run()` is
called synchronously, and `asyncio.run()` inside a component that a deepset
`AsyncPipeline` is already driving raises "cannot be called from a running event
loop". `httpx.Client` sidesteps the question rather than working around it.

**Envelope preserved verbatim.** The request `message.content` is base64-encoded
XML inside JSON. The response is *not* symmetric — see `_send`. Attachment bytes
are base64 one level further down. This asymmetry is the part most easily broken
by tidying, so it is spelled out at each site rather than abstracted.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

import httpx

logger = logging.getLogger(__name__)

GATEWAY_ENDPOINT = "/interoperability/rest/sendMessage"

NS_GET_NOTIFICATION = "https://webgate.ec.europa.eu/Safety-Gate/get_notification"
NS_NOTIFICATION_ATTACHMENT = (
    "https://webgate.ec.europa.eu/Safety-Gate/notification_attachment"
)

T = TypeVar("T")


# --- configuration ------------------------------------------------------------


def security_config() -> Dict[str, Any]:
    """The gateway's two-part security block, verbatim from
    `interoperability.get_security_config`. Every value comes from the
    environment; on deepset these must be workspace secrets, never init
    parameters."""
    return {
        "forSgrGateway": {
            "calledSystem": None,
            "callingSystem": os.environ.get("SGR_GATEWAY_CALLING_SYSTEM"),
            "login": os.environ.get("SGR_GATEWAY_LOGIN"),
            "password": os.environ.get("SGR_GATEWAY_PASSWORD"),
            "organisationCode": int(os.environ.get("SGR_GATEWAY_ORGANISATION_CODE", 0)),
            "accessProfile": os.environ.get("SGR_GATEWAY_ACCESS_PROFILE"),
        },
        "forEndSystem": {
            "calledSystem": "RAS",
            "callingSystem": None,
            "login": os.environ.get("END_SYSTEM_LOGIN"),
            "password": os.environ.get("END_SYSTEM_PASSWORD"),
            "organisationCode": int(os.environ.get("END_SYSTEM_ORGANISATION_CODE", 0)),
            "accessProfile": os.environ.get("END_SYSTEM_ACCESS_PROFILE"),
        },
    }


# Every variable the gateway needs. deepset injects a secret into the runtime as
# an environment variable of the SAME NAME, so each of these is one hand-typed
# entry on the Secrets page — there is no grouped or nested form.
REQUIRED_ENV = (
    "API_GATEWAY_URL",
    "API_GATEWAY_USER",
    "API_GATEWAY_SECRET",
    "SGR_GATEWAY_CALLING_SYSTEM",
    "SGR_GATEWAY_LOGIN",
    "SGR_GATEWAY_PASSWORD",
    "SGR_GATEWAY_ORGANISATION_CODE",
    "SGR_GATEWAY_ACCESS_PROFILE",
    "END_SYSTEM_LOGIN",
    "END_SYSTEM_PASSWORD",
    "END_SYSTEM_ORGANISATION_CODE",
    "END_SYSTEM_ACCESS_PROFILE",
)


def assert_env_configured() -> None:
    """Fail loudly on a missing secret rather than building a broken envelope.

    `security_config` reads with `os.environ.get`, so an absent or misspelled
    secret becomes `None` — or `0` for the two organisationCode fields — and the
    gateway answers with a generic auth failure that names none of them. Twelve
    variables have to be typed by hand into the deepset UI; assume one will be
    wrong and say which.
    """
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise ValueError(
            "SGRG gateway is not configured — missing environment variable(s): "
            + ", ".join(missing)
            + ". On deepset each is a secret of the same name "
            "(Settings > Workspace > Secrets)."
        )


def _retry(
    call: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
) -> T:
    """Retry on timeouts only.

    Anything else — bad credentials, a 4xx, a malformed envelope — will fail
    identically on a second attempt, so it is raised immediately rather than
    spending 60s of an officer's wait discovering that.
    """
    delay = initial_delay
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return call()
        except (httpx.TimeoutException, httpx.ReadTimeout) as exc:
            last = exc
            if attempt < max_retries - 1:
                logger.warning(
                    "SGRG attempt %d timed out (%s); retrying in %.0fs",
                    attempt + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
    if last is not None:
        raise last
    raise RuntimeError("retry loop exited without a result")


# --- XML bodies, verbatim from sgrg_core_methods ------------------------------


def notification_body(alert_id: str, return_type: str = "JSON") -> str:
    return f"""
            <parameters xmlns="{NS_GET_NOTIFICATION}" xsi:schemaLocation="{NS_GET_NOTIFICATION} schema.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            <notificationId>{alert_id}</notificationId>
            <returnType>{return_type}</returnType>
            </parameters>
            """  # noqa: E501


def attachment_body(
    alert_id: str,
    attachment_id: str,
    attachment_type: str = "PHOTO",
    return_type: str = "JSON",
) -> str:
    """GET_ATTACHMENT and GET_PUBLISHED_ATTACHMENT share this body; only the
    method key differs."""
    return f"""
            <parameters xmlns="{NS_NOTIFICATION_ATTACHMENT}" xsi:schemaLocation="{NS_NOTIFICATION_ATTACHMENT} schema.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            <notificationId>{alert_id}</notificationId>
            <attachmentId>{attachment_id}</attachmentId>
            <attachmentType>{attachment_type}</attachmentType>
            <returnType>{return_type}</returnType>
            </parameters>
            """  # noqa: E501


# --- attachment discovery, from prepare_training_data.extract_attachment_refs -


def attachment_refs(notification: Dict[str, Any]) -> List[Dict[str, Any]]:
    """-> [{attachmentId, attachmentType, fileName, mainPicture}].

    Images are at `product.photos[]`. Document containers are NOT confirmed in
    the restricted body, so four plausible keys are scanned and whatever is found
    is logged — the original's words, and still true. This is the weak point of
    the whole fetch: documents gate `risk_type`, `risk_description` and
    `legal_provision`, which the validator forces to "Unknown" when `has_docs` is
    false. A missed document is therefore a silent accuracy loss, not an error.
    """
    refs: List[Dict[str, Any]] = []
    product = notification.get("product") or {}

    for photo in product.get("photos") or []:
        if not isinstance(photo, dict):
            continue
        attachment_id = photo.get("id") or photo.get("attachmentId")
        if attachment_id is None:
            continue
        refs.append(
            {
                "attachmentId": str(attachment_id),
                "attachmentType": "PHOTO",
                "fileName": photo.get("fileName") or "",
                "mainPicture": bool(photo.get("mainPicture")),
            }
        )

    candidates = [
        (product, "documents"),
        (product, "attachments"),
        (notification, "documents"),
        (notification, "attachments"),
    ]
    for container, key in candidates:
        value = container.get(key) if isinstance(container, dict) else None
        if not value:
            continue
        logger.info("attachment_refs: found '%s' = %s", key, value)
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, dict):
                continue
            attachment_id = item.get("id") or item.get("attachmentId")
            if attachment_id is None:
                continue
            refs.append(
                {
                    "attachmentId": str(attachment_id),
                    "attachmentType": item.get("attachmentType") or "DOCUMENT",
                    "fileName": item.get("fileName") or item.get("filename") or "",
                    "mainPicture": False,
                }
            )

    return refs


def decode_attachment_bytes(attachment: Dict[str, Any]) -> Optional[bytes]:
    """The file itself: the `content` field of an attachment response is base64.

    This is the *second* encoding layer, one level below the envelope — see
    `_send` for why the first one is not symmetric.
    """
    raw = attachment.get("content")
    if raw is None:
        return None
    return base64.b64decode(raw)


# --- the gateway --------------------------------------------------------------


class SafetyGateGateway:
    """Synchronous wrapper over the gateway's single `sendMessage` endpoint.

    :param base_url: defaults to `API_GATEWAY_URL`.
    :param timeout: per-request, seconds. The original uses 120s for
        sendMessage and 30s for the token call; both are kept.
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url or os.environ.get("API_GATEWAY_URL", "")
        self.timeout = timeout
        self.max_retries = max_retries

    def _token(self) -> str:
        client_id = os.environ.get("API_GATEWAY_USER")
        client_secret = os.environ.get("API_GATEWAY_SECRET")
        if not client_id or not client_secret:
            raise ValueError(
                "API_GATEWAY_USER and API_GATEWAY_SECRET must be set "
                "(on deepset: as workspace secrets)"
            )
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{self.base_url}/token",
                params={"grant_type": "client_credentials"},
                auth=(client_id, client_secret),
            )
            response.raise_for_status()
        return str(response.json()["access_token"])

    def _send(self, method: str, content: str) -> Dict[str, Any]:
        """POST one base64-XML body and return the decoded response content."""
        url = self.base_url + GATEWAY_ENDPOINT

        def call() -> Dict[str, Any]:
            payload = {
                "security": security_config(),
                "message": {
                    "content": base64.b64encode(content.encode("utf-8")).decode(
                        "utf-8"
                    ),
                    "type": "REQUEST",
                    "method": method,
                    "callBackUrl": "",
                },
            }
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._token()}"},
                )
                response.raise_for_status()

            body = response.json()
            message = body.get("message") or {}
            if "content" not in message:
                raise ValueError(f"{method}: response carries no message.content")

            # json.loads, NOT b64decode — the envelope is asymmetric. The REQUEST
            # content is base64-encoded XML; the RESPONSE `message.content` comes
            # back as a plain JSON string (`sgrg_core_methods.py:88`). That file's
            # own docstring calls the response "doubly base64-encoded", which
            # describes the attachment `content` field one level down, not this.
            # Adding a b64decode here to make it look symmetric breaks every call.
            return cast(Dict[str, Any], json.loads(message["content"]))

        return _retry(call, max_retries=self.max_retries)

    def notification(self, alert_id: str, published: bool) -> Dict[str, Any]:
        """One notification body. The two scopes are different documents, and the
        same photo carries a DIFFERENT attachment id in each — so both must be
        fetched and each scope's attachments discovered with its own ids
        (`prepare_training_data.py:123`)."""
        method = "GET_PUBLISHED_NOTIFICATION" if published else "GET_NOTIFICATION"
        return self._send(method, notification_body(alert_id))

    def attachment(
        self,
        alert_id: str,
        attachment_id: str,
        attachment_type: str,
        published: bool,
    ) -> Dict[str, Any]:
        """One attachment's metadata; use `decode_attachment_bytes` for the file."""
        method = "GET_PUBLISHED_ATTACHMENT" if published else "GET_ATTACHMENT"
        return self._send(
            method, attachment_body(alert_id, attachment_id, attachment_type)
        )
