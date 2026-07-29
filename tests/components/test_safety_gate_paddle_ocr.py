"""Tests for SafetyGatePaddleOCR.

The markdown format is a contract, not an implementation detail: the issue-#37
acceptance check byte-diffs it against the reference `<alertId>/output/` files
produced by `preprocessing/ocr_render_md.py`. The golden-string test below is
therefore the most important one here — it is copied from a real reference file
(`sample_data/10098308/output/published__APEX TATOO.jpg.md`) with only the
values this test controls substituted.

Nothing here requires paddle: rendering is pure, and the OCR paths are driven
with fake readers.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest
from haystack.dataclasses import ByteStream

from dc_custom_component.components.safety_gate.paddle_ocr import (
    CYRILLIC_KEY,
    PRIMARY_KEY,
    SafetyGatePaddleOCR,
    document_id,
    entry_reliability,
    parse_prefixed_filename,
    render_image_md,
    table_text,
    yaml_str,
)


class _FakeReader:
    """Stands in for a PaddleOCR instance."""

    def __init__(self, texts: List[str], scores: List[float]) -> None:
        self._payload = [{"rec_texts": texts, "rec_scores": scores}]
        self.calls = 0

    def predict(self, _: Any) -> List[Dict[str, Any]]:
        self.calls += 1
        return self._payload


def _component(**kwargs: Any) -> SafetyGatePaddleOCR:
    """A component with warm_up() short-circuited, so paddle is never imported."""
    comp = SafetyGatePaddleOCR(**kwargs)
    comp._primary = _FakeReader(["hello"], [0.99])
    comp._engine = "paddleocr 3.7.0"
    return comp


# --- markdown rendering: the byte-level contract ------------------------------

EXPECTED_MD = """---
title: "APEX TATOO.jpg"
alert_id: 10098308
artifact: image-ocr
source: "published/APEX TATOO.jpg"
folder: published
file_type: jpg
language_model: PP-OCRv6
mean_confidence: 0.93
reliability: high
lines: 5
engine: "paddleocr 3.7.0"
detection_model: PP-OCRv6_medium_det
generated: "2026-07-03T10:22:10"
ocr_generated: "2026-07-02T18:19:58"
---

# OCR — published/APEX TATOO.jpg (alert 10098308)

Confidence is the recognizer's probability per line (0–1): ≥0.85 reliable, \
0.65–0.85 mixed, <0.65 often misread — treat low-confidence lines as hints.

| Conf | Text |
|------|------|
| 0.99 | Eternal Ink |
| 0.93 | APEX |
| 1.00 | TATTOO INK |
| 0.99 | REACH COMPLIANT |
| 0.77 | WYSTIOUE I Red-Violer |
"""


def test_render_matches_the_reference_markdown_byte_for_byte() -> None:
    """Golden test against a real ocr_render_md.py output file."""
    entry = {
        "text": "Eternal Ink\nAPEX\nTATTOO INK\nREACH COMPLIANT\nWYSTIOUE I Red-Violer",
        "language_model": "PP-OCRv6",
        "mean_confidence": 0.9299,
        "lines": [
            {"text": "Eternal Ink", "confidence": 0.99},
            {"text": "APEX", "confidence": 0.93},
            {"text": "TATTOO INK", "confidence": 1.0},
            {"text": "REACH COMPLIANT", "confidence": 0.99},
            {"text": "WYSTIOUE I Red-Violer", "confidence": 0.77},
        ],
    }

    rendered = render_image_md(
        entry,
        alert_id="10098308",
        folder="published",
        image_name="APEX TATOO.jpg",
        engine="paddleocr 3.7.0",
        ocr_generated="2026-07-02T18:19:58",
        generated="2026-07-03T10:22:10",
    )

    assert rendered == EXPECTED_MD


def test_no_text_detected_body() -> None:
    entry: Dict[str, Any] = {"text": "", "mean_confidence": 0.0, "lines": []}

    rendered = render_image_md(
        entry, "1", "published", "a.png", "paddleocr 3.7.0", "T", generated="T"
    )

    assert "_No text detected._" in rendered
    assert "reliability: none" in rendered
    assert "lines: 0" in rendered


def test_extraction_error_body() -> None:
    rendered = render_image_md(
        entry := {"error": "OSError: broken"},
        "1",
        "restricted",
        "a.png",
        "paddleocr 3.7.0",
        "T",
        generated="T",
    )

    assert entry_reliability(entry) == "error"
    assert "**Extraction error:** OSError: broken" in rendered
    assert "reliability: error" in rendered


def test_alternative_reading_block_excludes_the_winner() -> None:
    entry = {
        "text": "АБВ",
        "language_model": CYRILLIC_KEY,
        "mean_confidence": 0.81,
        "lines": [{"text": "АБВ", "confidence": 0.81}],
        "by_model": {
            PRIMARY_KEY: {"lines": 1, "mean_confidence": 0.42, "text": "ABB"},
            CYRILLIC_KEY: {"lines": 1, "mean_confidence": 0.81, "text": "АБВ"},
        },
    }

    rendered = render_image_md(
        entry, "1", "published", "a.png", "paddleocr 3.7.0", "T", generated="T"
    )

    assert "> Alternative reading by `PP-OCRv6` (mean confidence 0.42):" in rendered
    assert "> ABB" in rendered
    # The winning model must not be echoed back as its own alternative.
    assert "Alternative reading by `cyrillic`" not in rendered


def test_alternative_model_error_is_rendered() -> None:
    entry = {
        "text": "x",
        "language_model": PRIMARY_KEY,
        "mean_confidence": 0.5,
        "lines": [{"text": "x", "confidence": 0.5}],
        "by_model": {
            PRIMARY_KEY: {"lines": 1, "mean_confidence": 0.5, "text": "x"},
            CYRILLIC_KEY: {"error": "RuntimeError: boom"},
        },
    }

    rendered = render_image_md(
        entry, "1", "published", "a.png", "paddleocr 3.7.0", "T", generated="T"
    )

    assert "> Alternative model `cyrillic` failed: RuntimeError: boom" in rendered


def test_empty_alternative_text_renders_placeholder() -> None:
    entry = {
        "text": "x",
        "language_model": PRIMARY_KEY,
        "mean_confidence": 0.5,
        "lines": [{"text": "x", "confidence": 0.5}],
        "by_model": {
            PRIMARY_KEY: {"lines": 1, "mean_confidence": 0.5, "text": "x"},
            CYRILLIC_KEY: {"lines": 0, "mean_confidence": 0.0, "text": ""},
        },
    }

    assert "> _(empty)_" in render_image_md(
        entry, "1", "published", "a.png", "e", "T", generated="T"
    )


def test_table_text_escapes_pipes_backslashes_and_newlines() -> None:
    assert table_text("a|b") == "a\\|b"
    assert table_text("a\\b") == "a\\\\b"
    assert table_text("a\nb") == "a b"


def test_yaml_str_escapes_quotes() -> None:
    assert yaml_str('a"b') == '"a\\"b"'


def test_reliability_bands() -> None:
    def _entry(conf: float) -> Dict[str, Any]:
        return {"mean_confidence": conf, "lines": [{"text": "x", "confidence": conf}]}

    assert entry_reliability(_entry(0.85)) == "high"
    assert entry_reliability(_entry(0.8499)) == "medium"
    assert entry_reliability(_entry(0.65)) == "medium"
    assert entry_reliability(_entry(0.6499)) == "low"


# --- escalation semantics -----------------------------------------------------


def test_low_confidence_triggers_escalation_and_best_score_wins(tmp_path: Path) -> None:
    comp = _component()
    comp._primary = _FakeReader(["bad"], [0.30])
    comp._cyrillic_reader = _FakeReader(["good", "reading"], [0.90, 0.92])
    comp._load_image = lambda path: "sentinel"  # type: ignore[method-assign]

    entry = comp._ocr_image(tmp_path / "x.png", force_cyrillic=False)

    # score = lines x mean_conf -> 2 * 0.91 = 1.82 beats 1 * 0.30
    assert entry["language_model"] == CYRILLIC_KEY
    assert "by_model" in entry
    assert entry["by_model"][PRIMARY_KEY]["mean_confidence"] == 0.3


def test_high_confidence_does_not_escalate(tmp_path: Path) -> None:
    comp = _component()
    comp._primary = _FakeReader(["fine"], [0.95])
    cyrillic = _FakeReader(["never"], [0.99])
    comp._cyrillic_reader = cyrillic
    comp._load_image = lambda path: "sentinel"  # type: ignore[method-assign]

    entry = comp._ocr_image(tmp_path / "x.png", force_cyrillic=False)

    assert entry["language_model"] == PRIMARY_KEY
    assert "by_model" not in entry
    assert cyrillic.calls == 0


def test_zero_line_reading_never_escalates(tmp_path: Path) -> None:
    """Faithful port of a quirk in ocr_extract.py:174 — an empty reading is
    falsy, so the one case escalation might help most is the case it skips."""
    comp = _component()
    comp._primary = _FakeReader([], [])
    cyrillic = _FakeReader(["something"], [0.9])
    comp._cyrillic_reader = cyrillic
    comp._load_image = lambda path: "sentinel"  # type: ignore[method-assign]

    entry = comp._ocr_image(tmp_path / "x.png", force_cyrillic=False)

    assert cyrillic.calls == 0
    assert entry["lines"] == []


def test_country_forces_escalation_even_at_high_confidence(tmp_path: Path) -> None:
    comp = _component()
    comp._primary = _FakeReader(["latin"], [0.99])
    cyrillic = _FakeReader(["кир"], [0.95])
    comp._cyrillic_reader = cyrillic
    comp._load_image = lambda path: "sentinel"  # type: ignore[method-assign]

    entry = comp._ocr_image(tmp_path / "x.png", force_cyrillic=True)

    assert cyrillic.calls == 1
    assert "by_model" in entry


def test_include_cyrillic_false_disables_escalation(tmp_path: Path) -> None:
    comp = _component(include_cyrillic=False)
    comp._primary = _FakeReader(["bad"], [0.10])
    cyrillic = _FakeReader(["good"], [0.99])
    comp._cyrillic_reader = cyrillic
    comp._load_image = lambda path: "sentinel"  # type: ignore[method-assign]

    entry = comp._ocr_image(tmp_path / "x.png", force_cyrillic=True)

    assert cyrillic.calls == 0
    assert entry["language_model"] == PRIMARY_KEY


def test_cyrillic_failure_falls_back_to_primary(tmp_path: Path) -> None:
    class _Exploding:
        def predict(self, _: Any) -> Any:
            raise RuntimeError("boom")

    comp = _component()
    comp._primary = _FakeReader(["ok"], [0.40])
    comp._cyrillic_reader = _Exploding()
    comp._load_image = lambda path: "sentinel"  # type: ignore[method-assign]

    entry = comp._ocr_image(tmp_path / "x.png", force_cyrillic=False)

    assert entry["language_model"] == PRIMARY_KEY
    assert "RuntimeError" in entry["by_model"][CYRILLIC_KEY]["error"]


def test_unreadable_image_becomes_an_error_entry(tmp_path: Path) -> None:
    comp = _component()

    entry = comp._ocr_image(tmp_path / "does-not-exist.png", force_cyrillic=False)

    assert "error" in entry
    assert entry_reliability(entry) == "error"


# --- metadata resolution ------------------------------------------------------


def test_metadata_is_taken_from_explicit_meta() -> None:
    comp = _component()
    stream = ByteStream(data=b"x", meta={"file_name": "a.jpg"})

    resolved = comp._resolve(stream, {"alert_id": "123", "folder": "restricted"})

    assert resolved == {
        "alert_id": "123",
        "folder": "restricted",
        "file_name": "a.jpg",
        "country": "",
    }


def test_metadata_falls_back_to_the_file_path_layout() -> None:
    """Keeps local round-trip testing working, where the directory layout is
    the only source of alert_id and folder."""
    comp = _component()
    stream = ByteStream(
        data=b"x", meta={"file_path": "/data/10099509/restricted/photo.png"}
    )

    resolved = comp._resolve(stream, {})

    assert resolved["alert_id"] == "10099509"
    assert resolved["folder"] == "restricted"
    assert resolved["file_name"] == "photo.png"


def test_metadata_defaults_when_nothing_is_available() -> None:
    comp = _component(default_folder="published")

    resolved = comp._resolve(ByteStream(data=b"x"), {})

    assert resolved["alert_id"] == "unknown"
    assert resolved["folder"] == "published"
    assert resolved["file_name"] == "image"


def test_filename_prefix_supplies_provenance_and_is_stripped() -> None:
    """`<alertId>__<folder>__<original>` — needed because .meta.json sidecars are
    silently Skipped on UI upload, so metadata cannot be assumed to arrive."""
    comp = _component()
    stream = ByteStream(
        data=b"x", meta={"file_name": "10099538__restricted__20251112_134414.jpg"}
    )

    resolved = comp._resolve(stream, {})

    assert resolved["alert_id"] == "10099538"
    assert resolved["folder"] == "restricted"
    # The prefix must NOT survive into title:/source: or reference parity breaks.
    assert resolved["file_name"] == "20251112_134414.jpg"


def test_prefix_is_stripped_even_when_metadata_wins_on_the_values() -> None:
    comp = _component()
    stream = ByteStream(data=b"x", meta={"file_name": "999__published__photo.png"})

    resolved = comp._resolve(stream, {"alert_id": "10099538", "folder": "restricted"})

    assert resolved["alert_id"] == "10099538"  # metadata is more specific
    assert resolved["folder"] == "restricted"
    assert resolved["file_name"] == "photo.png"  # but the name is still cleaned


@pytest.mark.parametrize(
    "name",
    [
        "20251112_134414.jpg",  # no prefix at all
        "report__final__v2.jpg",  # double underscores, but not the convention
        "abc__published__x.jpg",  # non-numeric alert id
        "123__archive__x.jpg",  # unknown folder name
    ],
)
def test_non_conforming_filenames_are_left_untouched(name: str) -> None:
    """The pattern is strict on purpose: real attachment names contain double
    underscores, and a loose match would invent bogus provenance."""
    alert_id, folder, original = parse_prefixed_filename(name)

    assert (alert_id, folder, original) == ("", "", name)


def test_default_alert_id_is_used_before_falling_back_to_unknown() -> None:
    comp = _component(default_alert_id="10099538")

    assert comp._resolve(ByteStream(data=b"x"), {})["alert_id"] == "10099538"


def test_alert_id_is_unknown_with_no_default() -> None:
    assert _component()._resolve(ByteStream(data=b"x"), {})["alert_id"] == "unknown"


def test_explicit_meta_overrides_the_path() -> None:
    comp = _component()
    stream = ByteStream(data=b"x", meta={"file_path": "/data/999/published/p.png"})

    resolved = comp._resolve(stream, {"alert_id": "111", "folder": "restricted"})

    assert resolved["alert_id"] == "111"
    assert resolved["folder"] == "restricted"


# --- document identity --------------------------------------------------------


def test_document_id_is_stable_and_content_independent() -> None:
    """Content carries a `generated` timestamp, so a content-derived id would
    change on every re-index and defeat DuplicatePolicy.SKIP."""
    first = document_id("10099509", "published", "a.png")
    second = document_id("10099509", "published", "a.png")

    assert first == second
    assert first != document_id("10099509", "restricted", "a.png")
    assert first != document_id("10099510", "published", "a.png")
    assert first != document_id("10099509", "published", "b.png")


# --- run() --------------------------------------------------------------------


def _png_bytes() -> bytes:
    """Opaque bytes standing in for an image.

    Deliberately not a real PNG and deliberately not built with Pillow: these
    tests monkeypatch `_ocr_image`, so the bytes are only ever staged to a temp
    file and never decoded. Keeping the suite free of Pillow means it runs in a
    bare env, before the paddle dependency tree is installed.
    """
    return b"\x89PNG\r\n\x1a\nnot-a-real-image"


def test_run_emits_one_document_per_image_with_frontmatter_in_both_places() -> None:
    comp = _component()
    comp._ocr_image = lambda path, force_cyrillic: {  # type: ignore[method-assign]
        "text": "HELLO",
        "language_model": PRIMARY_KEY,
        "mean_confidence": 0.9123,
        "lines": [{"text": "HELLO", "confidence": 0.9123}],
    }
    stream = ByteStream(data=_png_bytes(), meta={"file_name": "photo.png"})

    documents = comp.run(
        [stream], meta={"alert_id": "10099509", "folder": "published"}
    )["documents"]

    assert len(documents) == 1
    doc = documents[0]
    # frontmatter inside the content, for the prediction pipeline's bundler
    assert doc.content is not None
    assert doc.content.startswith('---\ntitle: "photo.png"')
    assert "artifact: image-ocr" in doc.content
    assert "| 0.91 | HELLO |" in doc.content
    # and mirrored into meta, for retriever filtering
    assert doc.meta["alert_id"] == "10099509"
    assert doc.meta["folder"] == "published"
    assert doc.meta["artifact"] == "image-ocr"
    assert doc.meta["reliability"] == "high"
    assert doc.meta["lines"] == 1
    assert doc.id == document_id("10099509", "published", "photo.png")


def test_run_skips_unreadable_sources_without_aborting_the_batch() -> None:
    comp = _component()
    comp._ocr_image = lambda path, force_cyrillic: {  # type: ignore[method-assign]
        "text": "ok",
        "language_model": PRIMARY_KEY,
        "mean_confidence": 0.9,
        "lines": [{"text": "ok", "confidence": 0.9}],
    }
    good = ByteStream(data=_png_bytes(), meta={"file_name": "good.png"})

    documents = comp.run(
        ["/nonexistent/missing.png", good],
        meta=[{"alert_id": "1"}, {"alert_id": "1"}],
    )["documents"]

    assert len(documents) == 1
    assert documents[0].meta["file_name"] == "good.png"


def test_run_returns_no_documents_for_no_sources() -> None:
    assert _component().run([])["documents"] == []


def test_serialization_roundtrip() -> None:
    comp = SafetyGatePaddleOCR(
        include_cyrillic=False,
        escalate_conf=0.5,
        max_side=1200,
        enable_mkldnn=False,
        default_folder="restricted",
        cyrillic_countries=["bg", "rs"],
    )

    restored = SafetyGatePaddleOCR.from_dict(comp.to_dict())

    assert restored.include_cyrillic is False
    assert restored.escalate_conf == 0.5
    assert restored.max_side == 1200
    assert restored.default_folder == "restricted"
    assert restored.cyrillic_countries == ["BG", "RS"]


def test_mkldnn_defaults_to_disabled() -> None:
    """Enabling it crashes text detection on deepset's platform — measured
    2026-07-28. The default must not regress."""
    assert SafetyGatePaddleOCR().enable_mkldnn is False


def test_warm_up_runs_a_preload_pass(monkeypatch: Any) -> None:
    """PP-LCNet_x1_0_textline_ori loads on first PREDICT, not at construction —
    without a throwaway pass the first real request paid 2m34s on deepset."""
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")

    reader = _FakeReader(["WARMUP 123"], [0.99])
    comp = SafetyGatePaddleOCR()
    monkeypatch.setattr(comp, "_build_reader", lambda rec: reader)
    monkeypatch.setattr(
        "dc_custom_component.components.safety_gate.paddle_ocr.paddleocr",
        None,
        raising=False,
    )
    monkeypatch.setitem(
        __import__("sys").modules, "paddleocr", type("M", (), {"__version__": "3.7.0"})
    )

    comp.warm_up()

    assert reader.calls == 1, "warm_up must force one predict pass"
    assert comp._engine == "paddleocr 3.7.0"


def test_warm_up_is_idempotent(monkeypatch: Any) -> None:
    reader = _FakeReader(["x"], [0.9])
    comp = SafetyGatePaddleOCR(preload_on_warm_up=False)
    monkeypatch.setattr(comp, "_build_reader", lambda rec: reader)
    monkeypatch.setitem(
        __import__("sys").modules, "paddleocr", type("M", (), {"__version__": "3.7.0"})
    )

    comp.warm_up()
    comp.warm_up()

    assert reader.calls == 0  # preload disabled
    assert comp._primary is reader


def test_failed_preload_does_not_break_warm_up(monkeypatch: Any) -> None:
    """A pre-load failure costs first-request latency, nothing more."""

    class _Exploding:
        def predict(self, _: Any) -> Any:
            raise RuntimeError("boom")

    comp = SafetyGatePaddleOCR()
    monkeypatch.setattr(comp, "_build_reader", lambda rec: _Exploding())
    monkeypatch.setitem(
        __import__("sys").modules, "paddleocr", type("M", (), {"__version__": "3.7.0"})
    )

    comp.warm_up()  # must not raise

    assert comp._primary is not None
