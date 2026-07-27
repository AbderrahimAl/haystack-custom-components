"""Tests for PaddleOcrProbe.

The point of this component is to run inside deepset, where paddle exists. The
tests here therefore cover the parts that must work *regardless* of whether
paddle is installed: the measurement scaffolding, the failure capture, and the
verdict thresholds. Anything needing paddle is skipped when it is absent.
"""

from typing import Any, Dict

import pytest

from dc_custom_component.components.diagnostics.paddle_ocr_probe import (
    PaddleOcrProbe,
    _make_test_image,
    _shape_result,
)


def _light() -> PaddleOcrProbe:
    """A probe that touches no models — exercises the scaffolding only."""
    return PaddleOcrProbe(
        load_models=False, include_cyrillic=False, run_inference=False
    )


def test_thread_env_is_set_at_import_time() -> None:
    """The env vars must be set by importing the module, not by constructing
    the component — paddle may already be imported by then."""
    import os

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_run_returns_report_and_facts() -> None:
    result = _light().run()

    assert set(result) == {"report", "facts"}
    assert "PaddleOCR feasibility probe" in result["report"]

    facts = result["facts"]
    for key in ("memory", "stages", "checkpoints", "config", "verdict"):
        assert key in facts


def test_baseline_checkpoint_recorded_before_stages() -> None:
    facts = _light().run()["facts"]

    assert facts["checkpoints"][0]["stage"] == "baseline"


def test_stage_failure_is_captured_not_raised() -> None:
    """A stage that explodes must land in the report, not abort the run —
    otherwise an OOM during model load costs us the memory numbers too."""
    probe = _light()

    def _boom() -> Dict[str, Any]:
        raise MemoryError("simulated OOM")

    record = probe._stage("explode", _boom)

    assert record["ok"] is False
    assert "MemoryError" in record["error"]
    assert record["seconds"] >= 0
    assert probe._stages[-1]["stage"] == "explode"


def test_verdict_flags_a_failed_stage() -> None:
    probe = _light()
    probe._stage("explode", lambda: (_ for _ in ()).throw(RuntimeError("nope")))

    assert "FAILED at: explode" in probe.run()["facts"]["verdict"]


@pytest.mark.parametrize(
    "limit, peak, expected",
    [
        (4000.0, 3900.0, "UNSAFE"),
        (4000.0, 3200.0, "TIGHT"),
        (4000.0, 1500.0, "COMFORTABLE"),
    ],
)
def test_verdict_thresholds(limit: float, peak: float, expected: str) -> None:
    facts = {
        "stages": [],
        "memory": {"limit_mb": limit, "peak_rss_mb": peak},
    }

    assert expected in PaddleOcrProbe._verdict(facts)


def test_verdict_survives_missing_memory_readings() -> None:
    """macOS has no cgroup files; the probe must degrade, not crash."""
    facts = {"stages": [], "memory": {"limit_mb": None, "peak_rss_mb": None}}

    assert "memory limit or peak unavailable" in PaddleOcrProbe._verdict(facts)


def test_serialization_roundtrip() -> None:
    probe = PaddleOcrProbe(
        include_cyrillic=False, rec_model="custom_rec", image_width=640
    )
    restored = PaddleOcrProbe.from_dict(probe.to_dict())

    assert restored.include_cyrillic is False
    assert restored.rec_model == "custom_rec"
    assert restored.image_width == 640


def test_test_image_is_bgr_and_correctly_sized() -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")

    image = _make_test_image("HELLO 123", 800, 200)

    assert image.shape == (200, 800, 3)
    # White background in BGR is still 255 across all channels.
    assert int(image[0][0][0]) == 255


def test_shape_result_matches_ocr_extract_contract() -> None:
    """Shaping must mirror preprocessing/ocr_extract.py::run_reader so the
    numbers are directly comparable with the local reference output."""
    raw = [{"rec_texts": ["ABC", "DEF"], "rec_scores": [0.9, 0.7]}]

    shaped = _shape_result(raw)

    assert shaped["text"] == "ABC\nDEF"
    assert shaped["lines"] == 2
    assert shaped["mean_confidence"] == 0.8
    assert shaped["score"] == 1.6
    assert shaped["per_line"][0] == {"text": "ABC", "conf": 0.9}


def test_shape_result_handles_empty_reading() -> None:
    assert _shape_result([{"rec_texts": [], "rec_scores": []}]) == {
        "text": "",
        "lines": 0,
        "mean_confidence": 0.0,
        "score": 0.0,
        "per_line": [],
    }


def test_full_run_when_paddle_available() -> None:
    """Only meaningful where paddle is installed — i.e. in deepset, or locally
    once the dependency is pulled in."""
    pytest.importorskip("paddle")
    pytest.importorskip("paddleocr")

    facts = PaddleOcrProbe(include_cyrillic=False).run()["facts"]

    assert facts["stages"][0]["stage"] == "import_paddle"
    assert facts["stages"][0]["ok"] is True
