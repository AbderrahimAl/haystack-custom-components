"""Tests for SafetyGateRapidOCR.

No rapidocr install required: the engine is only imported inside `warm_up()`, so
every pure function below is testable on its own. The tests concentrate on the
camera-watermark rule, which is the only logic here that decides what reaches the
prediction model — the rest of the component is layout.

The rule matches an OCR line against the device the image's own EXIF reports,
rather than a brand lexicon, so the cases below are written in terms of a
signature the file claims plus a line the recognizer produced.
"""


from dc_custom_component.components.safety_gate.rapid_ocr import (

    edit_distance,
    is_device_watermark,
    normalise_result,
    parse_prefixed_name,
    reliability,
    render_markdown,
    sort_key,
    suppress_watermarks,
)

OPPO = "OPPOA985G"


def test_compact_strips_separators_and_uppercases() -> None:
    # One physical overlay, four spellings from the same corpus.
    for spelling in ("OPPO A98 5G", "OPPO A985G", "OPPO A98-5G", "oppoa985g"):
        assert compact(spelling) == OPPO


def test_edit_distance_abandons_early_on_length() -> None:
    assert edit_distance("ABCDEF", "ABCDEF") == 0
    assert edit_distance("ABCDEF", "ABCDEG") == 1
    # A length gap wider than the budget short-circuits rather than computing.
    assert edit_distance("ABCDEF", "ABCDEFGHIJ") > 3


def test_device_signatures_reads_model_and_make() -> None:
    # Model alone is 5 chars, under the floor, so only make+model survives.
    assert device_signatures({272: "A98 5G", 271: "OPPO"}) == [OPPO]
    # Both survive when the model alone is long enough to be evidence.
    assert device_signatures({272: "Galaxy A14", 271: "Samsung"}) == [
        "GALAXYA14",
        "SAMSUNGGALAXYA14",
    ]
    # Model already carries the make: no redundant second signature.
    assert device_signatures({272: "OPPO A98 5G", 271: "OPPO"}) == [OPPO]
    # No model tag: the rule must be disabled, not guess.
    assert device_signatures({271: "OPPO"}) == []
    assert device_signatures(None) == []


def test_device_signatures_drops_signatures_too_short_to_be_evidence() -> None:
    # `Mi 9` compacts to `MI9`; any tolerance against that matches half a corpus.
    assert device_signatures({272: "Mi 9"}) == []
    assert len(compact("Mi 9")) < MIN_DEVICE_CHARS


def test_is_device_watermark_tolerates_misreads_but_not_real_product_text() -> None:
    assert is_device_watermark("OPPO A98 5G", [OPPO])
    assert is_device_watermark("OPPOA985G", [OPPO])  # no spaces at all
    assert is_device_watermark("OPPO A78 6G", [OPPO])  # two digits misread
    assert is_device_watermark("Shot on OPPO A98 5G", [OPPO])
    # Real label text from the same alerts, 7+ edits away.
    for real in ("GUARDIA", "FINANZA", "MILANO", "Arcobaleno parfumes"):
        assert not is_device_watermark(real, [OPPO])


def test_no_signature_disables_the_rule() -> None:
    pairs = [("Arcobaleno", 0.9), ("OPPO A98 5G", 1.0)]
    assert suppress_watermarks(pairs, []) == (pairs, 0)


def test_suppression_is_restricted_to_the_tail() -> None:
    tail = [("Arcobaleno", 0.9), ("TODAS", 0.9), ("OPPO A98 5G", 1.0)]
    kept, dropped = suppress_watermarks(tail, [OPPO])
    assert dropped == 1 and [t for t, _ in kept] == ["Arcobaleno", "TODAS"]

    # Same string in the middle of a label is product text, not an overlay.
    middle = [("OPPO A98 5G", 1.0), ("Arcobaleno", 0.9), ("TODAS", 0.9)]
    assert suppress_watermarks(middle, [OPPO]) == (middle, 0)


def test_trailing_timestamp_goes_only_with_a_watermark() -> None:
    stamped = [("Arcobaleno", 0.9), ("OPPO A98 5G", 1.0), ("29.4.2026 11:56", 0.98)]
    kept, dropped = suppress_watermarks(stamped, [OPPO])
    assert dropped == 2 and [t for t, _ in kept] == ["Arcobaleno"]

    # An unanchored date is a batch code or best-before: it must survive.
    alone = [("Arcobaleno", 0.9), ("29.4.2026 11:56", 0.98)]
    assert suppress_watermarks(alone, [OPPO]) == (alone, 0)


def test_frontmatter_carries_the_count_but_never_the_string() -> None:
    md = render_markdown(
        "p.jpg",
        "published",
        "10092373",
        [("Arcobaleno", 0.9)],
        "rapidocr 2.0",
        suppressed=1,
    )
    assert "watermarks_suppressed: 1" in md
    assert "1 line suppressed" in md
    assert "OPPO" not in md  # re-naming it would restore the bad input

    clean = render_markdown("p.jpg", "published", "1", [("x", 0.9)], "e", suppressed=0)
    assert "watermarks_suppressed" not in clean and "suppressed" not in clean


def test_prefixed_name_and_sort_order() -> None:
    assert parse_prefixed_name("10099538__published__photo.jpg") == (
        "10099538",
        "published",
        "photo.jpg",
    )
    assert parse_prefixed_name("photo.jpg") == ("", "", "photo.jpg")
    names = ["1__restricted__a.jpg", "1__published__b.jpg", "loose.jpg"]
    assert sorted(names, key=sort_key) == [
        "1__published__b.jpg",
        "1__restricted__a.jpg",
        "loose.jpg",
    ]


def test_reliability_bands_and_empty_image() -> None:
    assert reliability(0.9, 3) == "high"
    assert reliability(0.7, 3) == "mixed"
    assert reliability(0.4, 3) == "low"
    assert reliability(0.0, 0) == "none"


def test_normalise_result_handles_both_engine_shapes() -> None:
    class V2:
        txts = ("a", "b")
        scores = (0.9, 0.8)

    assert normalise_result(V2()) == [("a", 0.9), ("b", 0.8)]
    legacy = ([[None, "a", 0.9], [None, "b", 0.8]], 0.01)
    assert normalise_result(legacy) == [("a", 0.9), ("b", 0.8)]
    assert normalise_result(None) == []
