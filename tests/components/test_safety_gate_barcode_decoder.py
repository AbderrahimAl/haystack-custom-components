"""Tests for SafetyGateBarcodeDecoder.

This is the only component that is alert-level rather than per-file, so the tests
concentrate on the four aggregation properties that byte-parity with the reference
`barcodes.md` depends on:

  1. iteration order — published before restricted, then a CASE-SENSITIVE
     filename sort
  2. dedup on (type, value) across every image of the alert
  3. attribution of each value to the FIRST image that produced it
  4. a Document is emitted even when nothing decodes (`decoded: 0`)

Plus the batch-grouping behaviour that makes it work inside an index pipeline at
all. No cv2 required — decoding is driven with fakes.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from haystack.dataclasses import ByteStream

from dc_custom_component.components.safety_gate import barcode_decoder as mod
from dc_custom_component.components.safety_gate.barcode_decoder import (
    SafetyGateBarcodeDecoder,
    document_id,
    render_barcode_md,
    sort_key,
)

# --- the markdown contract ----------------------------------------------------

EMPTY_MD = """---
title: "Barcode decodes"
alert_id: 10098308
artifact: barcode-decode
engine: opencv
decoded: 0
generated: "2026-07-04T20:46:25"
---

# Barcode decodes (alert 10098308)

_No barcodes decoded from any image of this alert._
"""


def test_empty_render_matches_the_reference() -> None:
    """Golden test against sample_data/10098308/output/barcodes.md."""
    assert (
        render_barcode_md("10098308", "opencv", [], "2026-07-04T20:46:25") == EMPTY_MD
    )


def test_render_with_decodes_embeds_a_json_block() -> None:
    decodes = [
        {
            "file": "20251112_134414.jpg",
            "folder": "published",
            "type": "BARCODE",
            "value": "6920452022174",
        }
    ]

    rendered = render_barcode_md("10099538", "opencv", decodes, "T")

    assert "decoded: 1" in rendered
    assert "```json" in rendered
    assert '"value": "6920452022174"' in rendered
    assert "Programmatically decoded symbols (decoder output, not OCR)." in rendered
    # The JSON block must be indent=2, as json.dumps(..., indent=2) produces.
    assert json.dumps(decodes, ensure_ascii=False, indent=2) in rendered


def test_artifact_line_is_present_unlike_document_markdown() -> None:
    """Barcode and image markdown carry `artifact:`; document markdown does not."""
    assert "artifact: barcode-decode" in render_barcode_md("1", "opencv", [], "T")


def test_document_id_is_per_alert_only() -> None:
    """One barcode Document per alert — no folder or filename in the key, so a
    second batch for the same alert overwrites rather than duplicating."""
    assert document_id("10099538") == document_id("10099538")
    assert document_id("10099538") != document_id("10099539")


# --- ordering -----------------------------------------------------------------


def test_published_sorts_before_restricted() -> None:
    assert sort_key("published", "z.jpg") < sort_key("restricted", "a.jpg")


def test_filename_sort_is_case_sensitive() -> None:
    """`sorted(d.iterdir())` is case-sensitive; ocr_extract.list_images
    lowercases. The difference is preserved because dedup attribution
    depends on this order."""
    assert sort_key("published", "Z.jpg") < sort_key("published", "a.jpg")


def test_unknown_folders_sort_last() -> None:
    assert sort_key("published", "a") < sort_key("elsewhere", "a")
    assert sort_key("restricted", "a") < sort_key("elsewhere", "a")


# --- aggregation --------------------------------------------------------------


class _ScriptedDecoder(SafetyGateBarcodeDecoder):
    """Subclass that scripts decodes per source filename, in read order."""

    def __init__(self, script: Dict[str, List[Dict[str, str]]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._script = script
        self._resolved_engine = "opencv"
        self.read_order: List[str] = []

    def _decode_alert(self, alert_id: str, items: Any) -> Any:
        # Record the order the aggregation visits files in, then delegate.
        self.read_order.extend(name for _, name, _ in items)
        self._queue = [self._script.get(name, []) for _, name, _ in items]
        return super()._decode_alert(alert_id, items)

    def _read(self, path: Path) -> List[Dict[str, str]]:
        return self._queue.pop(0) if self._queue else []


def _stream(
    name: str, alert: str = "10099538", folder: str = "published"
) -> ByteStream:
    return ByteStream(
        data=b"fake", meta={"file_name": name, "alert_id": alert, "folder": folder}
    )


def test_one_document_per_alert_not_per_image() -> None:
    comp = _ScriptedDecoder({})

    documents = comp.run([_stream("a.jpg"), _stream("b.jpg"), _stream("c.jpg")])[
        "documents"
    ]

    assert len(documents) == 1
    assert documents[0].meta["images_scanned"] == 3


def test_document_emitted_even_when_nothing_decodes() -> None:
    """`decoded: 0` keeps 'ran and found nothing' distinct from 'never ran'."""
    comp = _ScriptedDecoder({})

    doc = comp.run([_stream("a.jpg")])["documents"][0]

    assert doc.meta["decoded"] == 0
    assert doc.content is not None
    assert "_No barcodes decoded from any image of this alert._" in doc.content


def test_duplicate_values_are_deduped_across_images() -> None:
    same = [{"type": "BARCODE", "value": "6920452022174"}]
    comp = _ScriptedDecoder({"a.jpg": same, "b.jpg": same})

    doc = comp.run([_stream("a.jpg"), _stream("b.jpg")])["documents"][0]

    assert doc.meta["decoded"] == 1


def test_first_image_wins_attribution() -> None:
    same = [{"type": "BARCODE", "value": "111"}]
    comp = _ScriptedDecoder({"b.jpg": same, "a.jpg": same})

    # b uploaded first, but a sorts first -> a must own the value
    doc = comp.run([_stream("b.jpg"), _stream("a.jpg")])["documents"][0]

    assert comp.read_order == ["a.jpg", "b.jpg"]
    assert doc.content is not None
    payload = json.loads(doc.content.split("```json\n")[1].split("\n```")[0])
    assert payload[0]["file"] == "a.jpg"


def test_published_image_wins_over_restricted_for_the_same_value() -> None:
    same = [{"type": "QRCODE", "value": "https://example.test"}]
    comp = _ScriptedDecoder({"r.jpg": same, "p.jpg": same})

    doc = comp.run(
        [_stream("r.jpg", folder="restricted"), _stream("p.jpg", folder="published")]
    )["documents"][0]

    assert doc.content is not None
    payload = json.loads(doc.content.split("```json\n")[1].split("\n```")[0])
    assert payload[0]["folder"] == "published"


def test_different_values_are_all_kept() -> None:
    comp = _ScriptedDecoder(
        {
            "a.jpg": [{"type": "BARCODE", "value": "111"}],
            "b.jpg": [{"type": "QRCODE", "value": "222"}],
        }
    )

    doc = comp.run([_stream("a.jpg"), _stream("b.jpg")])["documents"][0]

    assert doc.meta["decoded"] == 2


def test_same_value_different_type_is_not_a_duplicate() -> None:
    """The dedup key is (type, value), not value alone."""
    comp = _ScriptedDecoder(
        {
            "a.jpg": [{"type": "BARCODE", "value": "111"}],
            "b.jpg": [{"type": "QRCODE", "value": "111"}],
        }
    )

    assert (
        comp.run([_stream("a.jpg"), _stream("b.jpg")])["documents"][0].meta["decoded"]
        == 2
    )


# --- batch grouping -----------------------------------------------------------


def test_a_batch_with_two_alerts_yields_two_documents() -> None:
    comp = _ScriptedDecoder({})

    documents = comp.run(
        [_stream("a.jpg", alert="111"), _stream("b.jpg", alert="222")]
    )["documents"]

    assert len(documents) == 2
    assert {d.meta["alert_id"] for d in documents} == {"111", "222"}
    assert {d.id for d in documents} == {document_id("111"), document_id("222")}


def test_non_images_are_ignored() -> None:
    comp = _ScriptedDecoder({})

    documents = comp.run(
        [_stream("report.pdf"), _stream("sheet.xlsx"), _stream("photo.jpg")]
    )["documents"]

    assert len(documents) == 1
    assert documents[0].meta["images_scanned"] == 1


def test_a_batch_of_only_non_images_yields_nothing() -> None:
    """No images means no barcode pass ran — emitting `decoded: 0` here would
    claim otherwise."""
    assert _ScriptedDecoder({}).run([_stream("report.pdf")])["documents"] == []


def test_filename_prefix_provenance() -> None:
    comp = _ScriptedDecoder({})
    stream = ByteStream(
        data=b"x", meta={"file_name": "10099538__restricted__photo.jpg"}
    )

    resolved = comp._resolve(stream, {})

    assert resolved == {
        "alert_id": "10099538",
        "folder": "restricted",
        "file_name": "photo.jpg",
    }


# --- engine selection ---------------------------------------------------------


def test_pyzbar_is_never_chosen_by_default(monkeypatch: Any) -> None:
    """libzbar segfaults on arm64 — OpenCV must win unless explicitly asked."""
    monkeypatch.delenv("BARCODE_ENGINE", raising=False)
    pytest.importorskip("cv2")

    assert mod.detect_engine("auto") == "opencv"


def test_barcode_env_var_forces_pyzbar(monkeypatch: Any) -> None:
    monkeypatch.setenv("BARCODE_ENGINE", "pyzbar")
    pytest.importorskip("pyzbar")

    assert mod.detect_engine("auto") == "pyzbar"


def test_missing_contrib_opencv_is_reported(monkeypatch: Any) -> None:
    """Plain opencv-python silently decodes QR only — indistinguishable from an
    alert with no barcodes, so it must be surfaced."""
    import sys
    import types

    fake = types.ModuleType("cv2")  # no `barcode` attribute
    monkeypatch.setitem(sys.modules, "cv2", fake)

    warning = mod.assert_engine_capable("opencv")

    assert warning is not None and "cv2.barcode is missing" in warning


def test_capable_opencv_reports_no_warning() -> None:
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "barcode"):
        pytest.skip("this environment has plain opencv-python")

    assert mod.assert_engine_capable("opencv") is None


def test_serialization_roundtrip() -> None:
    comp = SafetyGateBarcodeDecoder(
        engine="opencv", default_folder="restricted", default_alert_id="10099538"
    )

    restored = SafetyGateBarcodeDecoder.from_dict(comp.to_dict())

    assert restored.engine == "opencv"
    assert restored.default_folder == "restricted"
    assert restored.default_alert_id == "10099538"
