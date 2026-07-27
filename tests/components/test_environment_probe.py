"""Tests for EnvironmentProbe.

Network probing is disabled throughout (`probe_urls=[]`) so the suite stays
offline and fast; the network path is exercised in deepset, not in CI.
"""

from pathlib import Path
from typing import Any, Dict

from dc_custom_component.components.diagnostics.environment_probe import (
    EnvironmentProbe,
)


def _probe(tmp_path: Path, **kwargs: Any) -> EnvironmentProbe:
    defaults: Dict[str, Any] = {
        "marker_dir": str(tmp_path / "markers"),
        "probe_dirs": [str(tmp_path)],
        "probe_urls": [],
        "probe_modules": ["json"],
    }
    defaults.update(kwargs)
    return EnvironmentProbe(**defaults)


def test_run_returns_report_and_facts(tmp_path: Path) -> None:
    result = _probe(tmp_path).run()

    assert set(result) == {"report", "facts"}
    assert isinstance(result["report"], str)
    assert "environment probe" in result["report"]

    facts = result["facts"]
    for key in ("runtime", "cpu", "memory", "directories", "modules", "persistence"):
        assert key in facts


def test_warm_up_is_counted_and_idempotent_per_call(tmp_path: Path) -> None:
    probe = _probe(tmp_path)
    probe.warm_up()
    probe.warm_up()

    assert probe.run()["facts"]["warm_up"]["calls"] == 2


def test_run_warms_up_when_pipeline_did_not(tmp_path: Path) -> None:
    """A component run without warm_up() must still produce a full report."""
    facts = _probe(tmp_path).run()["facts"]

    assert facts["warm_up"]["calls"] == 1
    assert facts["warm_up"]["at"] is not None


def test_markers_accumulate_across_instances(tmp_path: Path) -> None:
    """Second instance, same marker dir — simulates a restart on shared disk."""
    marker_dir = str(tmp_path / "shared")

    first = _probe(tmp_path, marker_dir=marker_dir)
    first.warm_up()
    assert first.run()["facts"]["persistence"]["previous_count"] == 0

    second = _probe(tmp_path, marker_dir=marker_dir)
    second.warm_up()
    persistence = second.run()["facts"]["persistence"]

    assert persistence["previous_count"] == 1
    assert "SAME PROCESS" in persistence["verdict"]  # same pid inside one test run


def test_writable_directory_is_detected(tmp_path: Path) -> None:
    entry = _probe(tmp_path).run()["facts"]["directories"][0]

    assert entry["writable"] is True
    assert entry["error"] is None
    assert entry["free_gb"] is not None


def test_missing_module_is_soft(tmp_path: Path) -> None:
    """A dependency that is not installed must be reported, never raised —
    this is what lets the component deploy before paddle is added."""
    entry = _probe(tmp_path, probe_modules=["definitely_not_a_real_module"])
    module = entry.run()["facts"]["modules"][0]

    assert module["importable"] is False
    assert "error" in module


def test_serialization_roundtrip(tmp_path: Path) -> None:
    probe = _probe(tmp_path, network_timeout=1.5)
    restored = EnvironmentProbe.from_dict(probe.to_dict())

    assert restored.marker_dir == probe.marker_dir
    assert restored.probe_urls == []
    assert restored.network_timeout == 1.5
