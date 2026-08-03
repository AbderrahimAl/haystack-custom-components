"""Safety Gate alert fetcher — an alert id in, that alert's files out.

    alert_id
       │
       ├─ SGRG gateway: published + restricted notification bodies
       ├─ SGRG gateway: every attachment of both scopes → bytes
       ├─ upload each file into the deepset workspace   (side effect: starts the index)
       └─ outputs: sources (ByteStream[]), manifest, notification

Why a component and not an orchestrator
---------------------------------------
In production the request goes straight to deepset, so there is no service in the
request path to sequence anything — and deepset cannot chain pipelines, fire on an
index completing, or call itself. Whatever fetches the files therefore has to run
*inside* a pipeline. This is that thing.

Why it uploads AND returns the bytes
------------------------------------
The two consumers need the same files at different times:

  - `SafetyGateBarcodeDecoder`, wired to `sources` in this same pipeline, gets
    every image of the alert in one call — which is the whole condition its
    aggregation needs (`barcode_decoder.py:5-27`). No retrieval, no file
    download, no waiting for an index. The partial-result failure it warns about
    stops being possible rather than being guarded against.
  - OCR and document extraction happen out of band, in the index the upload
    triggers, because they are far too slow to sit in a synchronous request.

So the upload exists solely to make OCR asynchronous. That is its only job.

    STILL UNVERIFIED, and the whole design rests on it: whether a deepset
    component can reach `webgate.ec.europa.eu`, and whether `write_mode=OVERWRITE`
    reuses a file id or mints a new one. If it mints a new one, re-fetching an
    alert produces duplicate OCR Documents and any count-based readiness check
    silently breaks. `_upload` logs the returned id to make that a one-line test.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
from typing import Any, Dict, List, Optional, cast

import httpx
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace

from dc_custom_component.components.safety_gate.naming import (
    build_prefixed_filename,
)
from dc_custom_component.components.safety_gate.sgrg_client import (
    SafetyGateGateway,
    assert_env_configured,
    attachment_refs,
    decode_attachment_bytes,
)

logger = logging.getLogger(__name__)

DEEPSET_API_URL = "https://api.cloud.deepset.ai"

# Both scopes, published first — the order `SafetyGateBarcodeDecoder.sort_key`
# expects, so `sources` already arrives in the reference order.
SCOPES = (("published", True), ("restricted", False))

# deepset accepts any file type, but the index routes on MIME type
# (`index_pipeline.IMAGE_MIME_TYPES`), so an image uploaded as
# application/octet-stream would never reach the OCR branch.
_FALLBACK_MIME = "application/octet-stream"

# >=4 digits: alert ids are 8-digit notification ids, and this is loose enough to
# survive the wrapping below without matching an incidental small number.
_ALERT_ID_RX = re.compile(r"\d{4,}")


def parse_alert_id(raw: str) -> str:
    r"""Pull the alert id out of whatever deepset hands over as the query.

    deepset wraps a pipeline's `query` input in chat templating before it reaches
    a component: passing "10099538" arrived as
    `Chat History: []\n\nCurrent Question: 10099538` (observed 2026-07-30 — the
    reason `barcode_pipeline.py:111` refuses to wire `query` to the decoder's
    `alert_id` at all). Taking the first run of >=4 digits makes the raw id and
    the wrapped form resolve identically, so the component behaves the same run
    locally as deployed.
    """
    match = _ALERT_ID_RX.search(str(raw))
    if not match:
        raise ValueError(f"no alert id found in query {raw!r}")
    return match.group(0)


@component
class SafetyGateAlertFetcher:
    """Downloads one alert's attachments and stages them in deepset.

    :param api_key: deepset workspace key, for the file upload. Workspace *write*
        is enough — the Organization Admin key is only needed to share a custom
        component, so #42 does not gate this.
    :param workspace: deepset workspace to upload into; defaults to
        `WORKSPACE_NAME`.
    :param upload: set False to fetch without writing anything. The barcode path
        works fully in this mode, which makes the component testable and locally
        runnable without touching a workspace.
    :param gateway_timeout: seconds per SGRG call.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("DEEPSET_API_KEY", strict=False),
        workspace: Secret = Secret.from_env_var("WORKSPACE_NAME", strict=False),
        api_url: str = DEEPSET_API_URL,
        upload: bool = True,
        gateway_timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.workspace = workspace
        self.api_url = api_url.rstrip("/")
        self.upload = upload
        self.gateway_timeout = gateway_timeout
        self._gateway: Optional[SafetyGateGateway] = None

    def to_dict(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            default_to_dict(
                self,
                api_key=self.api_key.to_dict(),
                workspace=self.workspace.to_dict(),
                api_url=self.api_url,
                upload=self.upload,
                gateway_timeout=self.gateway_timeout,
            ),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyGateAlertFetcher":
        deserialize_secrets_inplace(
            data["init_parameters"], keys=["api_key", "workspace"]
        )
        return cast("SafetyGateAlertFetcher", default_from_dict(cls, data))

    def warm_up(self) -> None:
        """Resolve every credential BEFORE the pipeline runs, not mid-request.

        deepset calls `warm_up` during deployment validation, so checking here
        turns a missing secret into a deployment error naming the variable instead
        of a 500 on an officer's button press. deepset injects each secret as an
        environment variable of the same name, which is why `Secret.from_env_var`
        and the gateway's plain `os.environ` reads work identically on the
        platform and locally from `.env`.

        An injected gateway (tests, local runs) skips the environment check —
        nothing it would guard is used.
        """
        if self.upload:
            if not self.api_key.resolve_value() or not self.workspace.resolve_value():
                raise ValueError(
                    "DEEPSET_API_KEY and WORKSPACE_NAME must be set to upload "
                    "(or construct with upload=False)"
                )
        if self._gateway is None:
            assert_env_configured()
            self._gateway = SafetyGateGateway(timeout=self.gateway_timeout)

    @component.output_types(
        sources=List[ByteStream],
        manifest=Dict[str, Any],
        notification=Dict[str, Any],
    )
    def run(self, alert_id: str) -> Dict[str, Any]:
        """Fetch both scopes of one alert and stage every attachment.

        `sources` carries images *and* documents; the barcode decoder filters to
        images itself (`barcode_decoder.IMAGE_EXTS`), so the selection stays in
        one place rather than being duplicated here.

        `manifest` reports what was staged. `expected_images` / `expected_docs`
        are what a later readiness check compares the indexed Document count
        against — the fetcher is the only place that knows the true total, so it
        is the only place that can supply it.
        """
        self.warm_up()
        gateway = cast(SafetyGateGateway, self._gateway)
        alert = parse_alert_id(alert_id)

        bodies: Dict[str, Dict[str, Any]] = {}
        sources: List[ByteStream] = []
        staged: List[Dict[str, Any]] = []
        skipped = 0

        for folder, published in SCOPES:
            try:
                body = gateway.notification(alert, published=published)
            except Exception as exc:
                # One scope failing is recoverable — an alert may legitimately have
                # no published body yet. Both failing leaves nothing to do, which
                # the empty manifest below reports.
                logger.warning(
                    "Alert %s: %s notification failed: %s", alert, folder, exc
                )
                continue
            bodies[folder] = body

            for ref in attachment_refs(body):
                stream = self._fetch_one(gateway, alert, folder, ref, published)
                if stream is None:
                    skipped += 1
                    continue
                sources.append(stream)
                staged.append(
                    {
                        "file_name": stream.meta["file_name"],
                        "folder": folder,
                        "kind": stream.meta["kind"],
                        "file_id": stream.meta.get("deepset_file_id"),
                    }
                )

        images = sum(1 for item in staged if item["kind"] == "image")
        manifest = {
            "alert_id": alert,
            "files": staged,
            "expected_images": images,
            "expected_docs": len(staged) - images,
            "skipped": skipped,
            "uploaded": self.upload,
        }
        logger.info(
            "Alert %s staged %d images + %d documents (%d skipped)",
            alert,
            images,
            len(staged) - images,
            skipped,
        )
        return {"sources": sources, "manifest": manifest, "notification": bodies}

    # --- one attachment -------------------------------------------------------

    def _fetch_one(
        self,
        gateway: SafetyGateGateway,
        alert: str,
        folder: str,
        ref: Dict[str, Any],
        published: bool,
    ) -> Optional[ByteStream]:
        """-> a ByteStream named and tagged for the three components, or None.

        A single unreadable attachment must not fail the alert: nine images that
        decode are worth more than an exception, and `skipped` in the manifest
        keeps the shortfall visible rather than silent.
        """
        attachment_id = str(ref["attachmentId"])
        try:
            response = gateway.attachment(
                alert, attachment_id, str(ref["attachmentType"]), published=published
            )
            data = decode_attachment_bytes(response)
        except Exception as exc:
            logger.warning(
                "Alert %s: %s attachment %s failed: %s",
                alert,
                folder,
                attachment_id,
                exc,
            )
            return None

        if not data:
            logger.warning(
                "Alert %s: %s attachment %s carried no content",
                alert,
                folder,
                attachment_id,
            )
            return None

        # The response's own fileName wins — it is what the local reference wrote
        # to disk (`prepare_training_data.py:100`), so `title:` and `source:` stay
        # byte-comparable. The ref's name and the id are fallbacks only.
        raw_name = response.get("fileName") or ref.get("fileName") or attachment_id
        file_name = build_prefixed_filename(alert, folder, str(raw_name))
        mime = mimetypes.guess_type(file_name)[0] or _FALLBACK_MIME

        file_id = self._upload(file_name, data, alert, folder) if self.upload else None

        # Metadata AND the filename prefix, on purpose. Metadata is the
        # components' first resolution source and survives a file download that
        # discards the original path; the prefix is what makes the name unique in
        # a flat namespace and what carries provenance if metadata is ever lost.
        return ByteStream(
            data=data,
            mime_type=mime,
            meta={
                "alert_id": alert,
                "folder": folder,
                "file_name": file_name,
                "kind": "image" if str(mime).startswith("image/") else "document",
                "attachment_id": attachment_id,
                "deepset_file_id": file_id,
            },
        )

    def _upload(
        self, file_name: str, data: bytes, alert: str, folder: str
    ) -> Optional[str]:
        """POST one file into the workspace. -> the deepset file id, if returned.

        `write_mode=OVERWRITE` so a second button-press on the same alert replaces
        the files instead of duplicating them. Whether OVERWRITE reuses the id or
        mints a new one is UNVERIFIED — it is logged here so a double-fetch test
        settles it without instrumenting anything.

        The `meta` form field is what makes provenance the components' *first*
        resolution source rather than relying on the filename.
        """
        key = self.api_key.resolve_value()
        workspace = self.workspace.resolve_value()
        if not key or not workspace:
            raise ValueError(
                "DEEPSET_API_KEY and WORKSPACE_NAME must be set to upload "
                "(or construct with upload=False)"
            )

        url = f"{self.api_url}/api/v1/workspaces/{workspace}/files"
        mime = mimetypes.guess_type(file_name)[0] or _FALLBACK_MIME
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    url,
                    params={"write_mode": "OVERWRITE"},
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": (file_name, data, mime)},
                    data={"meta": json.dumps({"alert_id": alert, "folder": folder})},
                )
                response.raise_for_status()
                body = response.json() if response.content else {}
        except Exception as exc:
            # Raise: a file that is not in the workspace will never be OCR'd, and
            # the barcode result alone would look like a complete alert.
            raise RuntimeError(f"upload of {file_name} failed: {exc}") from exc

        file_id = body.get("file_id") or body.get("id")
        logger.info("Uploaded %s -> file_id=%s", file_name, file_id)
        return str(file_id) if file_id else None
