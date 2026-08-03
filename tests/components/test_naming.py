"""Tests for the flat-namespace filename convention.

The point of these tests is the ROUND TRIP. `naming.build_prefixed_filename`
writes the form that `paddle_ocr.parse_prefixed_filename` reads, and a skew
between them produces `alert_id: "unknown"` on every Document rather than an
error — so the guard has to be mechanical, not a code-review habit.
"""

from typing import List, Tuple

import pytest

from dc_custom_component.components.safety_gate.naming import (
    FOLDERS,
    build_prefixed_filename,
    clean_attachment_name,
)
from dc_custom_component.components.safety_gate.paddle_ocr import (
    parse_prefixed_filename,
)

# (alert_id, folder, raw name, the name the parser must hand back). Includes the
# cases that have actually broken: an accented name, a name carrying its own
# double underscores, and filesystem-hostile characters.
ROUND_TRIP: List[Tuple[str, str, str, str]] = [
    ("10099538", "published", "photo.jpg", "photo.jpg"),
    ("10098308", "restricted", "APEX TATOO.jpg", "APEX TATOO.jpg"),
    ("10099110", "restricted", "report__final__v2.pdf", "report__final__v2.pdf"),
    (
        "10096892",
        "restricted",
        "Vizsgálati jegyzökönyv.pdf",
        "Vizsgálati jegyzökönyv.pdf",
    ),
    ("10098474", "published", "a/b?c.png", "a_b_c.png"),
]


@pytest.mark.parametrize("alert_id, folder, raw, expected_name", ROUND_TRIP)
def test_round_trips_through_the_parser(
    alert_id: str, folder: str, raw: str, expected_name: str
) -> None:
    built = build_prefixed_filename(alert_id, folder, raw)
    parsed_alert, parsed_folder, parsed_name = parse_prefixed_filename(built)

    assert parsed_alert == alert_id
    assert parsed_folder == folder
    assert parsed_name == expected_name


def test_nfd_is_normalised_at_naming_time() -> None:
    """Accents must be composed BEFORE `sanitize_name` sees them, or every
    combining mark becomes `_` (paddle_ocr.py:236-241)."""
    nfd = "Vizsgálati.pdf"  # 'a' + COMBINING ACUTE, as deepset delivered it

    cleaned = clean_attachment_name(nfd)

    assert cleaned == "Vizsgálati.pdf"
    assert "́" not in cleaned


def test_prefix_keeps_two_alerts_with_the_same_filename_apart() -> None:
    """The collision the prefix exists to prevent: with write_mode=OVERWRITE an
    unprefixed name would let one alert destroy the other's image."""
    first = build_prefixed_filename("10098474", "restricted", "1.jpg")
    second = build_prefixed_filename("10099110", "restricted", "1.jpg")

    assert first != second


@pytest.mark.parametrize("raw", ["", "   ", "///", "???", "..."])
def test_unusable_names_fall_back(raw: str) -> None:
    """A name must keep at least one alphanumeric character. `"///"` substitutes
    to a single `"_"`, which is truthy — so without the alnum check two unusable
    names in the same alert and folder would both become `"_"` and one would
    silently overwrite the other."""
    assert clean_attachment_name(raw) == "attachment"


def test_non_latin_names_survive() -> None:
    """`isalnum` is script-agnostic; a Chinese lab report is real content."""
    assert clean_attachment_name("检验报告.pdf") == "检验报告.pdf"


@pytest.mark.parametrize("alert_id", ["", "abc", "1009-538", "10099538x"])
def test_rejects_non_numeric_alert_id(alert_id: str) -> None:
    """Better to fail here than to emit a name the strict parser silently drops."""
    with pytest.raises(ValueError, match="numeric"):
        build_prefixed_filename(alert_id, "published", "photo.jpg")


@pytest.mark.parametrize("folder", ["", "Published", "other", "images"])
def test_rejects_unknown_folder(folder: str) -> None:
    with pytest.raises(ValueError, match="folder"):
        build_prefixed_filename("10099538", folder, "photo.jpg")


def test_folders_match_the_decoder_ordering() -> None:
    """`FOLDER_ORDER` in barcode_decoder drives published-before-restricted; if
    these two ever disagree the sort silently changes dedup attribution."""
    from dc_custom_component.components.safety_gate.barcode_decoder import (
        FOLDER_ORDER,
    )

    assert FOLDERS == FOLDER_ORDER
