"""Tests for the attachment inventory's name parsing and document ids."""

from dc_custom_component.components.safety_gate.attachment_inventory import (
    document_id,
    parse_name,
)


def test_parses_a_prefixed_name() -> None:
    assert parse_name("10099538__published__photo.jpg") == (
        "10099538",
        "published",
        "photo.jpg",
    )


def test_keeps_double_underscores_in_the_original_name() -> None:
    assert parse_name("10099110__restricted__report__final__v2.pdf") == (
        "10099110",
        "restricted",
        "report__final__v2.pdf",
    )


def test_an_unprefixed_name_is_kept_under_an_unknown_alert() -> None:
    assert parse_name("photo.jpg") == ("unknown", "published", "photo.jpg")


def test_document_id_is_stable_per_alert() -> None:
    assert document_id("10099538") == document_id("10099538")
    assert document_id("10099538") != document_id("10099539")
