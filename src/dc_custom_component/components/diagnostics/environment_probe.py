"""Runtime environment probe for deepset custom components.

Answers the questions the deepset docs do not, and that decide whether a local
PaddleOCR component is viable:

  1. How much memory and CPU does a component actually get?
  2. Which directories are writable, and how much disk is free? (PaddleOCR
     downloads ~500 MB of weights on first use.)
  3. Can the container reach the model-hosting CDNs? ("Components have access
     to the internet" is documented; which hosts is not.)
  4. Does the local filesystem SURVIVE a container restart? This is the one
     that matters most: if it does not, every cold start re-downloads the
     models. Deployed pipelines idle out (dev after ~20 min), so cold starts
     are routine, not exceptional.
  5. Do the heavy dependencies import at all, and how slow is that import?

Every import of a heavy dependency is soft, so this component deploys and
reports usefully BEFORE `paddlepaddle` is added to pyproject.toml, and reports
on it afterwards. That gives a two-step rollout: probe the environment first,
then probe the dependency, without ever being left with a component that will
not deploy.

How to read the persistence answer
----------------------------------
`warm_up()` appends a record (timestamp, pid, boot_id) to a marker file under
`marker_dir`. `run()` reports every record it finds, including earlier ones.
Compare the newest record against the previous ones:

  * no previous records          -> fresh filesystem (or first ever run)
  * previous records, same pid   -> same process; proves nothing about restarts
  * previous records, new pid,
    SAME boot_id                 -> process restarted, disk survived
  * previous records, new pid,
    DIFFERENT boot_id            -> new container, and the disk STILL survived
                                    (this is the good outcome)

So: deploy, run once, let the pipeline go idle past its standby timeout, then
run again and compare.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from haystack import component, default_from_dict, default_to_dict

# Hosts worth checking: PaddleX pulls official model weights from the first,
# PyPI matters for the build, HF is the fallback if we end up self-hosting the
# weights somewhere we control.
DEFAULT_URLS = (
    "https://paddle-model-ecology.bj.bcebos.com",
    "https://pypi.org",
    "https://huggingface.co",
)

# Candidate cache locations, best-first. PaddleOCR defaults to ~/.paddlex,
# which is exactly the path most likely to be read-only in a container.
DEFAULT_DIRS = ("~", "/tmp", "/var/tmp", "/opt", ".")

# Dependencies the OCR component will need. Probed, never required.
PROBE_MODULES = ("numpy", "PIL", "cv2", "pypdfium2", "openpyxl", "paddle", "paddleocr")

_GB = 1024**3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_text(path: str) -> Optional[str]:
    """Read a small file, returning None instead of raising."""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _boot_id() -> Optional[str]:
    """Kernel boot id — changes when the container is replaced, not merely
    when the process restarts. The sharpest signal we have for 'new box'."""
    return _read_text("/proc/sys/kernel/random/boot_id")


def _memory_facts() -> Dict[str, Any]:
    """Container memory limit (cgroup v2, then v1) plus host total and current
    RSS. The limit is what actually matters: PaddleOCR budgets ~5 GB/worker."""
    facts: Dict[str, Any] = {}

    v2 = _read_text("/sys/fs/cgroup/memory.max")
    if v2 is not None:
        facts["cgroup_v2_max"] = v2
        if v2.isdigit():
            facts["limit_gb"] = round(int(v2) / _GB, 2)
        else:
            facts["limit_gb"] = None  # literal "max" == unlimited

    if "limit_gb" not in facts:
        v1 = _read_text("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if v1 is not None and v1.isdigit():
            facts["cgroup_v1_limit"] = v1
            # cgroup v1 reports a sentinel near 2^63 when unlimited
            facts["limit_gb"] = round(int(v1) / _GB, 2) if int(v1) < 2**62 else None

    meminfo = _read_text("/proc/meminfo") or ""
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            facts["host_total_gb"] = round(int(line.split()[1]) * 1024 / _GB, 2)
        elif line.startswith("MemAvailable:"):
            facts["host_available_gb"] = round(int(line.split()[1]) * 1024 / _GB, 2)

    status = _read_text("/proc/self/status") or ""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            facts["process_rss_mb"] = round(int(line.split()[1]) / 1024, 1)

    return facts


def _cpu_facts() -> Dict[str, Any]:
    facts: Dict[str, Any] = {"cpu_count": os.cpu_count()}
    try:
        # What we are actually allowed to run on, which can be far less than
        # cpu_count() under a cgroup.
        facts["affinity"] = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        facts["affinity"] = None

    quota = _read_text("/sys/fs/cgroup/cpu.max")
    if quota:
        facts["cgroup_v2_cpu_max"] = quota
        parts = quota.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            facts["cpu_limit"] = round(int(parts[0]) / int(parts[1]), 2)
    return facts


def _probe_dir(raw: str) -> Dict[str, Any]:
    """Can we actually write here, and how much room is there?

    Tests by writing a real file rather than trusting os.access, which lies
    under some overlay/read-only mount configurations.
    """
    path = Path(raw).expanduser().resolve()
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "writable": False,
        "free_gb": None,
        "error": None,
    }
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".dc_probe_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        info["writable"] = True
        info["exists"] = True
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"[:200]

    try:
        info["free_gb"] = round(shutil.disk_usage(str(path)).free / _GB, 2)
    except Exception:
        pass
    return info


def _probe_url(url: str, timeout: float) -> Dict[str, Any]:
    """One byte over the wire beats a DNS lookup — proves egress, not just
    name resolution. HEAD is avoided because several CDNs reject it.

    `reachable` is the answer we actually care about, and it is deliberately
    NOT the same as a 2xx. A 403 from a CDN root still means the request left
    the container, crossed the network and came back with an HTTP response —
    egress works. Only DNS failures, refused connections and timeouts mean the
    host is genuinely unreachable.
    """
    started = time.monotonic()

    def _elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    try:
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1)
            status = int(getattr(response, "status", 0))
        return {"url": url, "reachable": True, "status": status, "ms": _elapsed_ms()}
    except urllib.error.HTTPError as exc:
        # Server answered — that is proof of egress, whatever the status code.
        return {
            "url": url,
            "reachable": True,
            "status": int(exc.code),
            "note": "HTTP error, but the host responded",
            "ms": _elapsed_ms(),
        }
    except Exception as exc:
        return {
            "url": url,
            "reachable": False,
            "status": None,
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "ms": _elapsed_ms(),
        }


def _probe_module(name: str) -> Dict[str, Any]:
    """Import timing matters as much as success: a 30 s paddle import on every
    cold start is a very different proposition from a 2 s one."""
    started = time.monotonic()
    try:
        module = __import__(name)
        return {
            "module": name,
            "importable": True,
            "version": str(getattr(module, "__version__", "n/a")),
            "seconds": round(time.monotonic() - started, 2),
        }
    except Exception as exc:
        return {
            "module": name,
            "importable": False,
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "seconds": round(time.monotonic() - started, 2),
        }


@component
class EnvironmentProbe:
    """Reports the runtime environment of a deepset custom component.

    Deploy it in a one-component query pipeline and run it to get memory and
    CPU limits, writable paths, egress reachability, dependency import status
    and — across two runs separated by an idle period — whether the local
    filesystem survives a restart.

    :param marker_dir: where the persistence marker is written. Must be a path
        that could plausibly hold the model cache, so the answer transfers.
    :param probe_urls: hosts to test egress against. Empty list skips the check.
    :param probe_modules: modules to attempt importing. All failures are soft.
    :param network_timeout: per-URL timeout in seconds; keep it short so a
        blocked network cannot stall the pipeline.
    """

    def __init__(
        self,
        marker_dir: str = "/tmp/dc_env_probe",
        probe_dirs: Optional[List[str]] = None,
        probe_urls: Optional[List[str]] = None,
        probe_modules: Optional[List[str]] = None,
        network_timeout: float = 5.0,
    ) -> None:
        self.marker_dir = marker_dir
        self.probe_dirs = list(probe_dirs) if probe_dirs else list(DEFAULT_DIRS)
        self.probe_urls = (
            list(probe_urls) if probe_urls is not None else list(DEFAULT_URLS)
        )
        self.probe_modules = (
            list(probe_modules) if probe_modules else list(PROBE_MODULES)
        )
        self.network_timeout = network_timeout

        # Runtime state — deliberately not serialised.
        self._warm_up_calls = 0
        self._warm_up_at: Optional[str] = None
        self._previous_records: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            default_to_dict(
                self,
                marker_dir=self.marker_dir,
                probe_dirs=self.probe_dirs,
                probe_urls=self.probe_urls,
                probe_modules=self.probe_modules,
                network_timeout=self.network_timeout,
            ),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentProbe":
        return cast("EnvironmentProbe", default_from_dict(cls, data))

    @property
    def _marker_file(self) -> Path:
        return Path(self.marker_dir).expanduser() / "markers.jsonl"

    def warm_up(self) -> None:
        """Called once by Pipeline before the first run — the same hook the OCR
        component will use to load models. Reads any markers left by earlier
        runs, then appends its own.

        Counting the calls also verifies the 'warm_up runs once, not per
        request' assumption the whole design rests on.
        """
        self._warm_up_calls += 1
        self._warm_up_at = _utc_now()

        marker = self._marker_file
        records: List[Dict[str, Any]] = []
        try:
            if marker.exists():
                for line in marker.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as exc:  # a corrupt marker must never break warm-up
            records.append({"read_error": f"{type(exc).__name__}: {exc}"[:200]})
        self._previous_records = records

        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "utc": self._warm_up_at,
                "pid": os.getpid(),
                "boot_id": _boot_id(),
                "hostname": socket.gethostname(),
                "warm_up_call": self._warm_up_calls,
            }
            with marker.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except Exception:
            # Unwritable marker dir is itself a finding; run() surfaces it via
            # the writable-directory table.
            pass

    @component.output_types(report=str, facts=Dict[str, Any])
    def run(self, query: str = "") -> Dict[str, Any]:
        """Collect and return the environment report.

        :param query: ignored; present so the component can sit in a plain
            query pipeline and be triggered from the playground.
        """
        if self._warm_up_calls == 0:
            # Running un-warmed is itself worth knowing about.
            self.warm_up()

        facts: Dict[str, Any] = {
            "collected_utc": _utc_now(),
            "warm_up": {
                "calls": self._warm_up_calls,
                "at": self._warm_up_at,
            },
            "runtime": {
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "boot_id": _boot_id(),
                "cwd": os.getcwd(),
                "temp_dir": tempfile.gettempdir(),
            },
            "cpu": _cpu_facts(),
            "memory": _memory_facts(),
            "directories": [_probe_dir(d) for d in self.probe_dirs],
            "network": [_probe_url(u, self.network_timeout) for u in self.probe_urls],
            "modules": [_probe_module(m) for m in self.probe_modules],
            "env": {
                key: os.environ.get(key)
                for key in (
                    "HOME",
                    "TMPDIR",
                    "XDG_CACHE_HOME",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "HF_HOME",
                    "PADDLE_PDX_CACHE_HOME",
                    "PADDLE_HOME",
                )
                if os.environ.get(key) is not None
            },
            "persistence": {
                "marker_file": str(self._marker_file),
                "previous_records": self._previous_records,
                "previous_count": len(self._previous_records),
                "verdict": self._persistence_verdict(),
            },
        }
        return {"report": self._render(facts), "facts": facts}

    def _persistence_verdict(self) -> str:
        """Turn the marker history into a one-line answer."""
        if not self._previous_records:
            return (
                "NO PRIOR RECORDS — fresh filesystem, or this is the first run. "
                "Run again after the pipeline has been idle to get a real answer."
            )
        pid, boot = os.getpid(), _boot_id()
        prior_pids = {r.get("pid") for r in self._previous_records}
        prior_boots = {r.get("boot_id") for r in self._previous_records}

        if prior_pids == {pid}:
            return (
                f"SAME PROCESS (pid {pid}) — disk obviously intact, but this "
                "proves nothing about restarts. Let the pipeline go idle and re-run."
            )
        if boot is not None and boot not in prior_boots:
            return (
                "PERSISTS ACROSS CONTAINERS — new boot_id but earlier markers "
                "survived. The model cache would survive a cold start."
            )
        return (
            "PERSISTS ACROSS PROCESSES — new pid, same boot_id, earlier markers "
            "survived. Disk outlives the process; container replacement untested."
        )

    @staticmethod
    def _render(facts: Dict[str, Any]) -> str:
        """Human-readable rendering — this is what you read in the playground."""
        lines: List[str] = ["=== deepset custom-component environment probe ==="]

        runtime = facts["runtime"]
        lines += [
            "",
            f"collected      {facts['collected_utc']}",
            f"python         {runtime['python']}",
            f"platform       {runtime['platform']} ({runtime['machine']})",
            f"host / pid     {runtime['hostname']} / {runtime['pid']}",
            f"boot_id        {runtime['boot_id']}",
            f"cwd            {runtime['cwd']}",
            f"warm_up calls  {facts['warm_up']['calls']}  (expected: 1)",
        ]

        cpu, mem = facts["cpu"], facts["memory"]
        lines += [
            "",
            "-- resources --",
            f"cpu_count={cpu.get('cpu_count')} affinity={cpu.get('affinity')} "
            f"cpu_limit={cpu.get('cpu_limit')}",
            f"memory limit   {mem.get('limit_gb')} GB   "
            f"(host total {mem.get('host_total_gb')} GB, "
            f"available {mem.get('host_available_gb')} GB)",
            f"process RSS    {mem.get('process_rss_mb')} MB",
            "NOTE: PaddleOCR budgets ~5 GB per OCR worker.",
        ]

        lines += ["", "-- writable directories --"]
        for entry in facts["directories"]:
            flag = "WRITABLE" if entry["writable"] else "read-only"
            detail = f"  ({entry['error']})" if entry["error"] else ""
            lines.append(
                f"  {flag:9s} free={entry['free_gb']} GB  {entry['path']}{detail}"
            )

        lines += ["", "-- network egress --"]
        if not facts["network"]:
            lines.append("  (skipped)")
        for entry in facts["network"]:
            if entry["reachable"]:
                note = f"  [{entry['note']}]" if entry.get("note") else ""
                lines.append(
                    f"  REACHABLE   {entry['ms']:>5} ms  {entry['url']} "
                    f"(status {entry['status']}){note}"
                )
            else:
                lines.append(
                    f"  UNREACHABLE {entry['ms']:>5} ms  {entry['url']}  "
                    f"{entry['error']}"
                )

        lines += ["", "-- dependencies --"]
        for entry in facts["modules"]:
            if entry["importable"]:
                lines.append(
                    f"  OK    {entry['module']:<12} {entry['version']:<12} "
                    f"{entry['seconds']}s"
                )
            else:
                lines.append(f"  MISS  {entry['module']:<12} {entry['error']}")

        persistence = facts["persistence"]
        lines += [
            "",
            "-- filesystem persistence --",
            f"marker file    {persistence['marker_file']}",
            f"prior records  {persistence['previous_count']}",
            f"VERDICT        {persistence['verdict']}",
        ]
        for record in persistence["previous_records"][-5:]:
            lines.append(f"    {json.dumps(record)}")

        env = facts["env"]
        lines += ["", "-- relevant env vars --"]
        lines += [f"  {k}={v}" for k, v in env.items()] or ["  (none set)"]

        return "\n".join(lines)
