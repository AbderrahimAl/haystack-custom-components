"""Tests for SafetyGateAlertFetcher.

No network: the gateway is replaced with a fake returning canned notification
bodies and attachments. What is worth testing is not the HTTP — it is the five
things that silently degrade the pipeline if they are wrong:

  1. the alert id survives deepset's chat-templating of `query`
  2. both scopes are fetched, published first, so `sources` arrives in the order
     the barcode decoder's sort expects
  3. every ByteStream carries provenance in metadata AND in its filename
  4. images get an image/* mime type, or the index never routes them to OCR
  5. one bad attachment is skipped, not fatal, and is counted in the manifest
"""

import base64
from typing import Any, Dict, List, Optional, Set

import pytest

from dc_custom_component.components.safety_gate.alert_fetcher import (
    SafetyGateAlertFetcher,
    parse_alert_id,
)

ALERT = "10099538"


def body(
    photos: List[Dict[str, Any]],
    documents: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    product: Dict[str, Any] = {"photos": photos}
    if documents is not None:
        product["documents"] = documents
    return {"product": product, "reference": "A12/0001/26"}


class FakeGateway:
    """Canned gateway. `broken` names attachment ids that raise on fetch."""

    def __init__(
        self,
        bodies: Dict[str, Dict[str, Any]],
        broken: Optional[Set[str]] = None,
    ) -> None:
        self.bodies = bodies
        self.broken = broken or set()
        self.calls: List[str] = []

    def notification(self, alert_id: str, published: bool) -> Dict[str, Any]:
        folder = "published" if published else "restricted"
        self.calls.append(f"notification:{folder}")
        if folder not in self.bodies:
            raise RuntimeError(f"no {folder} body")
        return self.bodies[folder]

    def attachment(
        self,
        alert_id: str,
        attachment_id: str,
        attachment_type: str,
        published: bool,
    ) -> Dict[str, Any]:
        folder = "published" if published else "restricted"
        self.calls.append(f"attachment:{folder}:{attachment_id}")
        if attachment_id in self.broken:
            raise RuntimeError("gateway said no")
        return {
            "fileName": f"file-{attachment_id}.jpg",
            "content": base64.b64encode(b"\xff\xd8bytes").decode(),
        }


def fetcher(gateway: FakeGateway) -> SafetyGateAlertFetcher:
    component = SafetyGateAlertFetcher(upload=False)
    component._gateway = gateway  # type: ignore[assignment]  # skip the real one
    return component


# --- 1. the query wrapping ----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "10099538",
        "  10099538 ",
        "Chat History: []\n\nCurrent Question: 10099538",
        '{"query": "10099538"}',
    ],
)
def test_alert_id_survives_deepset_query_wrapping(raw: str) -> None:
    """deepset wraps `query` in chat templating before a component sees it."""
    assert parse_alert_id(raw) == "10099538"


@pytest.mark.parametrize("raw", ["", "no digits here", "12"])
def test_unparseable_query_raises(raw: str) -> None:
    with pytest.raises(ValueError, match="no alert id"):
        parse_alert_id(raw)


# --- 2. both scopes, published first ------------------------------------------


def test_fetches_both_scopes_published_first() -> None:
    gateway = FakeGateway(
        {
            "published": body([{"id": "p1", "fileName": "front.jpg"}]),
            "restricted": body([{"id": "r1", "fileName": "back.jpg"}]),
        }
    )

    result = fetcher(gateway).run(alert_id=ALERT)

    folders = [stream.meta["folder"] for stream in result["sources"]]
    assert folders == ["published", "restricted"]
    assert gateway.calls[0] == "notification:published"


def test_a_missing_scope_is_not_fatal() -> None:
    """An alert can legitimately have no published body yet."""
    gateway = FakeGateway({"restricted": body([{"id": "r1", "fileName": "x.jpg"}])})

    result = fetcher(gateway).run(alert_id=ALERT)

    assert len(result["sources"]) == 1
    assert "published" not in result["notification"]


# --- 3 & 4. provenance and mime ----------------------------------------------


def test_every_source_carries_provenance_both_ways() -> None:
    gateway = FakeGateway({"published": body([{"id": "p1", "fileName": "front.jpg"}])})

    stream = fetcher(gateway).run(alert_id=ALERT)["sources"][0]

    # metadata — the components' first resolution source
    assert stream.meta["alert_id"] == ALERT
    assert stream.meta["folder"] == "published"
    # and the filename prefix, which survives a download that drops the metadata
    assert stream.meta["file_name"].startswith(f"{ALERT}__published__")


def test_images_get_an_image_mime_type() -> None:
    """An image uploaded as application/octet-stream never reaches the OCR
    branch, because the index routes on mime type."""
    gateway = FakeGateway({"published": body([{"id": "p1", "fileName": "front.jpg"}])})

    stream = fetcher(gateway).run(alert_id=ALERT)["sources"][0]

    assert stream.mime_type == "image/jpeg"
    assert stream.meta["kind"] == "image"


# --- 5. a bad attachment is visible, not fatal -------------------------------


def test_one_broken_attachment_is_skipped_and_counted() -> None:
    gateway = FakeGateway(
        {
            "published": body(
                [
                    {"id": "good", "fileName": "a.jpg"},
                    {"id": "bad", "fileName": "b.jpg"},
                ]
            )
        },
        broken={"bad"},
    )

    result = fetcher(gateway).run(alert_id=ALERT)

    assert len(result["sources"]) == 1
    assert result["manifest"]["skipped"] == 1
    assert result["manifest"]["expected_images"] == 1


def test_manifest_reports_what_a_readiness_check_needs() -> None:
    gateway = FakeGateway({"published": body([{"id": "p1", "fileName": "a.jpg"}])})

    manifest = fetcher(gateway).run(alert_id=ALERT)["manifest"]

    assert manifest["alert_id"] == ALERT
    assert manifest["expected_images"] == 1
    assert manifest["expected_docs"] == 0
    assert manifest["uploaded"] is False


# --- configuration errors name themselves ------------------------------------


def test_missing_gateway_secret_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing secret must not become a `None` in the security block and a
    generic auth failure — the whole point is knowing WHICH one."""
    from dc_custom_component.components.safety_gate.sgrg_client import (
        REQUIRED_ENV,
        assert_env_configured,
    )

    for name in REQUIRED_ENV:
        monkeypatch.setenv(name, "x")
    monkeypatch.delenv("SGR_GATEWAY_LOGIN")

    with pytest.raises(ValueError, match="SGR_GATEWAY_LOGIN"):
        assert_env_configured()


def test_warm_up_checks_credentials_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """deepset calls warm_up at deployment validation, so a secret nobody created
    should fail there and name itself, not 500 on an officer's button press.

    The absent-secret case is an UNSET env var, not an empty token — Haystack
    refuses to build `Secret.from_token("")` at all, so that is not a shape this
    can ever be in.
    """
    monkeypatch.delenv("DEEPSET_API_KEY", raising=False)
    monkeypatch.delenv("WORKSPACE_NAME", raising=False)

    component = SafetyGateAlertFetcher(upload=True)

    with pytest.raises(ValueError, match="must be set to upload"):
        component.warm_up()


def test_upload_without_credentials_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSET_API_KEY", raising=False)
    monkeypatch.delenv("WORKSPACE_NAME", raising=False)

    component = SafetyGateAlertFetcher(upload=True)

    with pytest.raises(ValueError, match="must be set to upload"):
        component._upload("10099538__published__a.jpg", b"x", "10099538", "published")


# --- serialisation ------------------------------------------------------------


def test_round_trips_through_to_dict() -> None:
    """deepset serialises the pipeline to YAML, so the secrets must survive it
    without ever being written into it."""
    original = SafetyGateAlertFetcher(upload=False)

    data = original.to_dict()
    assert "env_vars" in data["init_parameters"]["api_key"]

    restored = SafetyGateAlertFetcher.from_dict(data)
    assert restored.upload is False
    assert restored.api_url == original.api_url
