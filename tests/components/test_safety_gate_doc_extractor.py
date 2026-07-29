"""Tests for SafetyGateDocExtractor.

As with the OCR component, the markdown format is a contract: the issue-#37
acceptance check byte-diffs it against reference files produced by
`preprocessing/doc_extract.py`. The golden test below is taken from a real one
(`training_data/10099509/output/Rapport_SCL-2025-25921-1-v1.md`).

Two format details it pins down, both easy to "tidy up" by accident:
  * `generated:` is UNQUOTED here, unlike the image markdown
  * there is NO `artifact:` line in the content — the field lives in meta only

Nothing here needs paddle, pypdfium2 or openpyxl: the rendering and gate logic
are pure, and the extraction paths are driven with fakes.
"""

from pathlib import Path
from typing import Any, List, Tuple

import pytest
from haystack.dataclasses import ByteStream

from dc_custom_component.components.safety_gate.doc_extractor import (
    SafetyGateDocExtractor,
    cell_str,
    clean_page_text,
    document_id,
    looks_garbled,
    render_document_md,
    sanitize_name,
    usable_count,
    word_count,
)

# --- the markdown contract ----------------------------------------------------

EXPECTED_MD = """---
title: "Rapport_SCL-2025-25921-1-v1"
alert_id: 10099509
source: "restricted/Rapport_SCL-2025-25921-1-v1.pdf"
folder: restricted
file_type: pdf
pages: 4
extraction: embedded
generated: 2026-07-27T14:09:01
engine: "pypdfium2 5.11.0 + paddleocr 3.7.0 (scanned pages) +qgate"
---

# Rapport_SCL-2025-25921-1-v1

## Page 1

SERVICE COMMUN DES LABORATOIRES

## Page 2

RESULTATS
"""


def test_render_matches_the_reference_markdown() -> None:
    rendered = render_document_md(
        title="Rapport_SCL-2025-25921-1-v1",
        alert_id="10099509",
        folder="restricted",
        file_name="Rapport_SCL-2025-25921-1-v1.pdf",
        unit_name="pages",
        n_units=4,
        kind="embedded",
        note=None,
        sections=[
            ("Page 1", "SERVICE COMMUN DES LABORATOIRES"),
            ("Page 2", "RESULTATS"),
        ],
        engine="pypdfium2 5.11.0 + paddleocr 3.7.0 (scanned pages) +qgate",
        generated="2026-07-27T14:09:01",
    )

    assert rendered == EXPECTED_MD


def test_generated_is_unquoted_unlike_the_image_markdown() -> None:
    """doc_extract writes it bare; ocr_render_md quotes it. A real difference
    between the two reference formats, not an inconsistency to fix."""
    rendered = render_document_md(
        "t", "1", "published", "d.pdf", "pages", 1, "embedded", None, [], "e", "TS"
    )

    assert "\ngenerated: TS\n" in rendered
    assert 'generated: "TS"' not in rendered


def test_no_artifact_line_in_the_content() -> None:
    """Adding one would break byte-parity with the reference. The field belongs
    in meta, which the run() test covers."""
    rendered = render_document_md(
        "t", "1", "published", "d.pdf", "pages", 1, "embedded", None, [], "e", "TS"
    )

    assert "artifact:" not in rendered


def test_note_line_is_appended_only_when_present() -> None:
    with_note = render_document_md(
        "t",
        "1",
        "published",
        "d.pdf",
        "pages",
        2,
        "mixed",
        "quality-gate: garbled embedded text on page(s) 2 -> OCR fallback",
        [],
        "e",
        "TS",
    )

    assert (
        'note: "quality-gate: garbled embedded text on page(s) 2 -> OCR fallback"'
        in with_note
    )


def test_empty_section_renders_the_placeholder() -> None:
    rendered = render_document_md(
        "t",
        "1",
        "published",
        "d.pdf",
        "pages",
        1,
        "none",
        None,
        [("Page 1", "")],
        "e",
        "TS",
    )

    assert "## Page 1\n\n_(no text)_" in rendered


def test_page_headings_use_the_exact_form_data_py_splits_on() -> None:
    """optimization/data.py::_PAGE_SPLIT matches '^## Page \\d+'. A different
    heading silently disables verdict-aware truncation of long lab reports."""
    import re

    rendered = render_document_md(
        "t",
        "1",
        "published",
        "d.pdf",
        "pages",
        3,
        "embedded",
        None,
        [("Page 1", "a"), ("Page 2", "b"), ("Page 3", "c")],
        "e",
        "TS",
    )

    assert len(re.split(r"(?m)^(?=## Page \d+)", rendered)) == 4  # head + 3 pages


# --- the two gates ------------------------------------------------------------


def test_usable_count_counts_alphanumerics_in_any_script() -> None:
    assert usable_count("abc 123") == 6
    assert usable_count("Übergröße") == 9
    assert usable_count("!!! ...") == 0


def test_word_count_requires_three_or_more_letters() -> None:
    assert word_count("the cat sat") == 3
    assert word_count("a bc d") == 0
    assert word_count("Vizsgálati jegyzőkönyv") == 2


def test_clean_prose_is_not_garbled() -> None:
    prose = (
        "Questo lampadario e' stato concepito in modo accurato e precontrollato, "
        "per lasciarvi totalmente soddisfatti in ogni occasione."
    )

    assert looks_garbled(prose) is False


def test_broken_cid_font_output_is_garbled() -> None:
    """Plenty of alphanumerics, almost no real words — passes the quantity gate
    and must be caught by the quality gate."""
    mojibake = "a1b2c3d4 e5f6g7h8 i9j0k1l2 m3n4o5p6 q7r8s9t0 u1v2w3x4 y5z6a7b8"

    assert usable_count(mojibake) >= 50
    assert looks_garbled(mojibake) is True


def test_short_text_is_never_judged_garbled() -> None:
    """Below the quantity threshold the quantity gate decides, not this one."""
    assert looks_garbled("ab1 cd2") is False


def test_configurable_density_is_honoured() -> None:
    prose = "the cat sat on the mat and then the dog ran to the park quickly today"

    assert looks_garbled(prose) is False
    assert looks_garbled(prose, max_density=99.0) is True


# --- clean_page_text ----------------------------------------------------------


def test_private_use_area_text_is_decoded() -> None:
    """Some PDFs emit their whole text as 0xF000+ascii from symbolic cmaps."""
    encoded = "".join(chr(0xF000 + ord(c)) for c in "HELLO WORLD")

    assert clean_page_text(encoded) == "HELLO WORLD"


def test_control_characters_and_leftover_pua_are_dropped() -> None:
    assert clean_page_text("ab\x01cdef") == "abcdef"


def test_tabs_and_newlines_survive() -> None:
    assert clean_page_text("a\tb\nc\rd") == "a\tb\nc\rd"


def test_pua_decoding_is_skipped_when_it_does_not_help() -> None:
    """Only decode when PUA dominates AND decoding improves usable content."""
    mostly_normal = "normal text here" + chr(0xF000)

    assert "normal text here" in clean_page_text(mostly_normal)


# --- helpers ------------------------------------------------------------------


def test_sanitize_name_falls_back_to_document_not_image() -> None:
    """ocr_render_md falls back to 'image'; this one to 'document'."""
    assert sanitize_name("") == "document"
    assert sanitize_name("!!!") == "___"
    assert (
        sanitize_name("Vizsgálati jegyzőkönyv_6247-2-25")
        == "Vizsgálati jegyzőkönyv_6247-2-25"
    )
    assert sanitize_name("a/b:c") == "a_b_c"


def test_cell_str_escapes_pipes_and_flattens_newlines() -> None:
    assert cell_str(None) == ""
    assert cell_str("a|b") == "a\\|b"
    assert cell_str("a\nb") == "a b"
    assert cell_str(42) == "42"


def test_document_id_is_stable_and_distinct_from_the_image_namespace() -> None:
    from dc_custom_component.components.safety_gate.paddle_ocr import (
        document_id as image_document_id,
    )

    first = document_id("10099509", "restricted", "r.pdf")

    assert first == document_id("10099509", "restricted", "r.pdf")
    assert first != document_id("10099509", "published", "r.pdf")
    # A doc and an image of the same name must never collide.
    assert first != image_document_id("10099509", "restricted", "r.pdf")


# --- provenance and run() -----------------------------------------------------


def _component(**kwargs: Any) -> SafetyGateDocExtractor:
    comp = SafetyGateDocExtractor(**kwargs)
    comp._engines = {
        "pdf": "pypdfium2 5.11.0 + paddleocr 3.7.0 (scanned pages) +qgate",
        "xlsx": "openpyxl 3.1.5",
    }
    return comp


def test_filename_prefix_provenance_is_shared_with_the_ocr_component() -> None:
    comp = _component()
    stream = ByteStream(
        data=b"x", meta={"file_name": "10099509__restricted__Rapport_SCL.pdf"}
    )

    resolved = comp._resolve(stream, {})

    assert resolved == {
        "alert_id": "10099509",
        "folder": "restricted",
        "file_name": "Rapport_SCL.pdf",
    }


def test_run_puts_artifact_doc_in_meta_but_not_in_content() -> None:
    comp = _component()
    sections: List[Tuple[str, str]] = [("Page 1", "text")]
    comp._extract_pdf = lambda path: (sections, 1, "embedded", None)  # type: ignore[method-assign]
    stream = ByteStream(data=b"%PDF-1.4 fake", meta={"file_name": "r.pdf"})

    documents = comp.run(
        [stream], meta={"alert_id": "10099509", "folder": "restricted"}
    )["documents"]

    assert len(documents) == 1
    doc = documents[0]
    assert doc.meta["artifact"] == "doc"
    assert doc.content is not None and "artifact:" not in doc.content
    assert doc.meta["extraction"] == "embedded"
    assert doc.meta["pages"] == 1
    assert doc.meta["alert_id"] == "10099509"


def test_run_skips_non_document_extensions() -> None:
    comp = _component()
    image = ByteStream(data=b"x", meta={"file_name": "photo.jpg"})

    assert comp.run([image])["documents"] == []


def test_extraction_failure_produces_an_error_document_not_a_missing_one() -> None:
    """Mirrors doc_extract.process_alert: the failure stays visible downstream."""
    comp = _component()

    def _boom(path: Path) -> Any:
        raise ValueError("corrupt xref table")

    comp._extract_pdf = _boom  # type: ignore[method-assign]
    stream = ByteStream(data=b"broken", meta={"file_name": "r.pdf"})

    documents = comp.run([stream], meta={"alert_id": "1"})["documents"]

    assert len(documents) == 1
    doc = documents[0]
    assert doc.content is not None
    assert "extraction: error" in doc.content
    assert "ValueError: corrupt xref table" in doc.content
    assert doc.meta["extraction"] == "error"


def test_xlsx_routes_to_the_spreadsheet_path() -> None:
    comp = _component()
    comp._extract_xlsx = lambda path: (  # type: ignore[method-assign]
        [("Sheet: Results", "| a |\n| --- |\n| 1 |")],
        1,
        "spreadsheet",
        None,
    )
    stream = ByteStream(data=b"PK fake", meta={"file_name": "data.xlsx"})

    doc = comp.run([stream], meta={"alert_id": "1"})["documents"][0]

    assert doc.content is not None
    assert "sheets: 1" in doc.content
    assert "extraction: spreadsheet" in doc.content
    assert "## Sheet: Results" in doc.content
    assert doc.meta["engine"] == "openpyxl 3.1.5"


def test_serialization_roundtrip() -> None:
    comp = SafetyGateDocExtractor(
        min_text_chars=80,
        garbled_max_density=4.5,
        max_sheet_rows=100,
        default_folder="restricted",
        default_alert_id="10099509",
    )

    restored = SafetyGateDocExtractor.from_dict(comp.to_dict())

    assert restored.min_text_chars == 80
    assert restored.garbled_max_density == 4.5
    assert restored.max_sheet_rows == 100
    assert restored.default_folder == "restricted"
    assert restored.default_alert_id == "10099509"


def test_mkldnn_defaults_to_disabled() -> None:
    assert SafetyGateDocExtractor().enable_mkldnn is False


@pytest.mark.parametrize("kind", ["embedded", "ocr", "mixed", "none", "error"])
def test_every_extraction_kind_renders(kind: str) -> None:
    rendered = render_document_md(
        "t", "1", "published", "d.pdf", "pages", 1, kind, None, [], "e", "TS"
    )

    assert f"extraction: {kind}" in rendered
