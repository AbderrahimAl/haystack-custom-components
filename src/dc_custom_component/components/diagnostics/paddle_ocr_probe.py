"""PaddleOCR feasibility probe for deepset custom components.

`EnvironmentProbe` measured the box: 3.91 GB cgroup memory limit with ~650 MB
already resident, writable `$HOME`, reachable model CDN, `warm_up()` called
once. Everything looked viable except the memory headroom — roughly 3.2 GB
against a documented ~5 GB-per-worker budget.

This component answers the question that leaves open, by actually doing it:

  1. Does `paddlepaddle` / `paddleocr` install and import in deepset's image?
  2. Do the ~500 MB of PP-OCRv6 weights download, from where, and how long?
  3. **What is the peak RSS?** Measured at four checkpoints via `VmHWM`, the
     kernel's high-water mark — a sampled `VmRSS` would miss a transient spike
     that is exactly what would trigger an OOM kill.
  4. Does inference actually produce correct text on this platform?

Every stage is individually switchable (`load_models`, `include_cyrillic`,
`run_inference`) and individually wrapped, so a stage that OOMs or fails still
returns a report naming the stage that died — and can be turned off for the
retry rather than having to guess.

The threading env vars are set at MODULE level, above any paddle import, which
is both correct (`preprocessing/ocr_extract.py:41-43` requires it) and a live
test of that approach inside a Haystack component, where `__init__` and
`warm_up()` both run far too late.
"""

from __future__ import annotations

import os

# MUST precede any paddle/numpy import anywhere in the process. Without it each
# worker spawns a thread pool across all cores; with 8 cores visible and no CPU
# quota, that is 8x oversubscription and a throughput collapse.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_THREAD_LIMIT",
):
    os.environ.setdefault(_var, "1")

# Quieten paddle's C++ logging and skip its per-run model-host reachability
# check (the weights are cached after the first warm-up).
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("FLAGS_call_stack_level", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import hashlib  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple, cast  # noqa: E402

from haystack import component, default_from_dict, default_to_dict  # noqa: E402

# Mirrors preprocessing/ocr_extract.py so the measurement reflects the real
# component, not a lighter stand-in.
DET_MODEL = "PP-OCRv6_medium_det"
PRIMARY_REC = "PP-OCRv6_medium_rec"
CYRILLIC_REC = "cyrillic_PP-OCRv5_mobile_rec"
MAX_SIDE = 2000

# Where PaddleOCR/PaddleX may put weights. Probed after warm-up to report what
# actually landed on disk and where.
CACHE_CANDIDATES = (
    "~/.paddlex",
    "~/.paddleocr",
    "~/.cache/paddle",
    "~/.cache/paddlex",
    "/tmp/paddlex",
)

# Drawn into the synthetic test image. Deliberately mixes case, digits and a
# EUR sign — enough to tell "OCR works" from "OCR returns noise".
DEFAULT_TEST_TEXT = "SAFETY GATE 12345 EUR"

# /proc/self/status values are in kB; cgroup limits are in bytes. Two
# constants, so a division never silently uses the wrong one.
_MB = 1024.0
_BYTES_PER_MB = 1024.0 * 1024.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _proc_status(field: str) -> Optional[float]:
    """Read one kB-valued field out of /proc/self/status, in MB."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{field}:"):
                return round(int(line.split()[1]) / _MB, 1)
    except Exception:
        pass
    return None


def _rss_mb() -> Optional[float]:
    """Current resident set size."""
    return _proc_status("VmRSS")


def _peak_mb() -> Optional[float]:
    """Peak RSS since process start (VmHWM). This is the number that decides
    whether we get OOM-killed — a spike during model load or inference counts
    even if memory is released immediately afterwards."""
    return _proc_status("VmHWM")


def _memory_limit_mb() -> Optional[float]:
    """Container memory limit in MB.

    NOTE the unit difference that bit us once already: `/proc/self/status`
    reports kB (so `/ _MB` is right there), but cgroup `memory.max` reports
    BYTES and needs dividing twice. Getting this wrong silently inflated the
    limit 1024x and made every headroom check meaningless.
    """
    for path, divisor in (
        ("/sys/fs/cgroup/memory.max", _BYTES_PER_MB),  # cgroup v2
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", _BYTES_PER_MB),  # cgroup v1
    ):
        try:
            raw = Path(path).read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if raw.isdigit():
            value = int(raw)
            # cgroup v1 uses a near-2^63 sentinel to mean "unlimited"
            if value < 2**62:
                return round(value / divisor, 1)
    return None


def _checkpoint(label: str) -> Dict[str, Any]:
    return {"stage": label, "rss_mb": _rss_mb(), "peak_mb": _peak_mb()}


def _dir_size(path: Path) -> Tuple[int, int]:
    """(bytes, file count) for a directory tree; (0, 0) if absent."""
    total = files = 0
    if not path.exists():
        return 0, 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
                files += 1
        except OSError:
            continue
    return total, files


def _cache_report(candidates: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in candidates:
        path = Path(raw).expanduser()
        size, files = _dir_size(path)
        if files:
            out.append(
                {
                    "path": str(path),
                    "mb": round(size / (1024 * 1024), 1),
                    "files": files,
                }
            )
    return out


def _make_test_image(text: str, width: int, height: int) -> Any:
    """Synthesise a BGR numpy array with known text.

    Generated rather than bundled: keeps binaries out of the component zip, and
    the expected string is known exactly, so the report can state whether OCR
    read it correctly. BGR channel order matches what `ocr_extract.load_image`
    hands to paddle.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=int(height * 0.28))
    except TypeError:  # Pillow < 10.1 has no size argument
        font = ImageFont.load_default()
    draw.text((int(width * 0.04), int(height * 0.32)), text, fill="black", font=font)
    return np.asarray(image)[:, :, ::-1].copy()


def _shape_result(raw: Any) -> Dict[str, Any]:
    """Same shaping as `preprocessing/ocr_extract.py::run_reader`, so numbers
    here are directly comparable with local output."""
    first = raw[0]
    texts = list(first.get("rec_texts") or [])
    scores = [float(s) for s in (first.get("rec_scores") or [])]
    mean_conf = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {
        "text": "\n".join(texts),
        "lines": len(texts),
        "mean_confidence": mean_conf,
        "score": round(len(texts) * mean_conf, 4),
        "per_line": [
            {"text": t, "conf": round(scores[i], 4) if i < len(scores) else None}
            for i, t in enumerate(texts)
        ],
    }


@component
class PaddleOcrProbe:
    """Installs, downloads, loads and runs PaddleOCR — and reports the cost.

    :param load_models: build the recognizer in `warm_up()`. Turn off to test
        import-only memory if a full load gets the pod OOM-killed.
    :param include_cyrillic: also build the Cyrillic escalation recognizer.
        Production may hold both simultaneously, so this is the honest
        worst-case memory figure.
    :param run_inference: OCR the synthetic image after loading.
    :param det_model / rec_model: overridable so cheaper `mobile` variants can
        be measured if `medium` does not fit.
    :param enable_mkldnn: matches `preprocessing/ocr_extract.py::build_reader`,
        which passes True. On macOS/arm64 that flag is effectively inert; on
        deepset's Linux x86_64 it is live, and the oneDNN kernel path under
        paddle's PIR executor raised
        `ConvertPirAttribute2RuntimeAttribute not support` during text
        detection (measured 2026-07-28). Kept True by default so the failure
        stays reproducible; `mkldnn_fallback` handles the retry.
    :param mkldnn_fallback: when inference fails and mkldnn was enabled,
        rebuild the recognizer with it disabled and try once more. Turns "did
        it break?" and "is mkldnn the cause?" into a single run.
    """

    def __init__(
        self,
        load_models: bool = True,
        include_cyrillic: bool = True,
        run_inference: bool = True,
        det_model: str = DET_MODEL,
        rec_model: str = PRIMARY_REC,
        cyrillic_model: str = CYRILLIC_REC,
        test_text: str = DEFAULT_TEST_TEXT,
        image_width: int = 900,
        image_height: int = 200,
        cache_candidates: Optional[List[str]] = None,
        enable_mkldnn: bool = True,
        mkldnn_fallback: bool = True,
    ) -> None:
        self.load_models = load_models
        self.include_cyrillic = include_cyrillic
        self.run_inference = run_inference
        self.det_model = det_model
        self.rec_model = rec_model
        self.cyrillic_model = cyrillic_model
        self.test_text = test_text
        self.image_width = image_width
        self.image_height = image_height
        self.cache_candidates = (
            list(cache_candidates) if cache_candidates else list(CACHE_CANDIDATES)
        )
        self.enable_mkldnn = enable_mkldnn
        self.mkldnn_fallback = mkldnn_fallback

        self._primary: Any = None
        self._cyrillic: Any = None
        self._stages: List[Dict[str, Any]] = []
        self._checkpoints: List[Dict[str, Any]] = []
        self._warm_up_calls = 0
        # Which mkldnn setting produced a working inference, once known.
        self._working_mkldnn: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            default_to_dict(
                self,
                load_models=self.load_models,
                include_cyrillic=self.include_cyrillic,
                run_inference=self.run_inference,
                det_model=self.det_model,
                rec_model=self.rec_model,
                cyrillic_model=self.cyrillic_model,
                test_text=self.test_text,
                image_width=self.image_width,
                image_height=self.image_height,
                cache_candidates=self.cache_candidates,
                enable_mkldnn=self.enable_mkldnn,
                mkldnn_fallback=self.mkldnn_fallback,
            ),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaddleOcrProbe":
        return cast("PaddleOcrProbe", default_from_dict(cls, data))

    def _stage(self, name: str, fn: Any) -> Dict[str, Any]:
        """Run one measured stage, recording duration, outcome and memory.

        Failures are captured rather than raised: a report that says which
        stage died is far more useful than a stack trace with no memory
        numbers attached.
        """
        started = time.monotonic()
        record: Dict[str, Any] = {"stage": name, "ok": True}
        try:
            result = fn()
            if isinstance(result, dict):
                record.update(result)
        except BaseException as exc:  # MemoryError is not an Exception subclass
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"[:400]
            record["traceback"] = traceback.format_exc()[-800:]
        record["seconds"] = round(time.monotonic() - started, 2)
        record["rss_mb"] = _rss_mb()
        record["peak_mb"] = _peak_mb()
        self._stages.append(record)
        self._checkpoints.append(_checkpoint(name))
        return record

    def warm_up(self) -> None:
        """Import paddle, download weights, build recognizers — measuring each.

        This is the same hook the production OCR component will use, so the
        timings here are the real cold-start cost.
        """
        self._warm_up_calls += 1
        if self._primary is not None:
            return  # idempotent, per the SentenceTransformers pattern

        self._checkpoints.append(_checkpoint("baseline"))

        def _import() -> Dict[str, Any]:
            import paddle
            import paddleocr

            return {
                "paddle_version": paddle.__version__,
                "paddleocr_version": paddleocr.__version__,
            }

        imported = self._stage("import_paddle", _import)
        if not imported["ok"] or not self.load_models:
            return

        def _build_primary() -> Dict[str, Any]:
            self._primary = self._build_reader(self.rec_model, self.enable_mkldnn)
            return {"model": self.rec_model, "mkldnn": self.enable_mkldnn}

        # First build pays the weight download; the timing is the cold start.
        self._stage("load_primary_model", _build_primary)

        if self.include_cyrillic:

            def _build_cyrillic() -> Dict[str, Any]:
                self._cyrillic = self._build_reader(
                    self.cyrillic_model, self.enable_mkldnn
                )
                return {"model": self.cyrillic_model, "mkldnn": self.enable_mkldnn}

            self._stage("load_cyrillic_model", _build_cyrillic)

    def _build_reader(self, rec_model: str, enable_mkldnn: bool) -> Any:
        """Construct a PaddleOCR reader.

        Deliberately identical to `preprocessing/ocr_extract.py::build_reader`
        apart from `enable_mkldnn` being a parameter — anything else diverging
        would make the measurements incomparable with the local reference.
        """
        from paddleocr import PaddleOCR

        return PaddleOCR(
            text_detection_model_name=self.det_model,
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="cpu",
            enable_mkldnn=enable_mkldnn,
            cpu_threads=1,
        )

    def _infer_once(self, reader: Any, recognized: Dict[str, Any]) -> Dict[str, Any]:
        """OCR the synthetic image with `reader`, recording the reading.

        The image SHA-256 is reported so a cross-platform comparison can PROVE
        the OCR input was identical rather than assume it. If the local and
        deepset hashes differ, Pillow rendered the font differently and any
        confidence comparison is meaningless.
        See `safety-gate-extraction/experiments/parity_check_ocr.py`.
        """
        image = _make_test_image(self.test_text, self.image_width, self.image_height)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        shaped = _shape_result(reader.predict(image))
        recognized.clear()
        recognized.update(shaped)
        recognized["image_sha256"] = digest
        return {
            "lines": shaped["lines"],
            "mean_confidence": shaped["mean_confidence"],
            "text_matches_expected": shaped["text"].strip() == self.test_text,
            "image_sha256": digest,
            "image_shape": list(image.shape),
        }

    @component.output_types(report=str, facts=Dict[str, Any])
    def run(self, query: str = "") -> Dict[str, Any]:
        """Run the probe. `query` is ignored; it exists so the component can be
        triggered from a plain query pipeline."""
        if self._warm_up_calls == 0:
            self.warm_up()

        recognized: Dict[str, Any] = {}
        if self.run_inference and self._primary is not None:
            attempt = self._stage(
                "inference", lambda: self._infer_once(self._primary, recognized)
            )
            if attempt["ok"]:
                self._working_mkldnn = self.enable_mkldnn
            elif self.mkldnn_fallback and self.enable_mkldnn:
                # oneDNN kernels are chosen at construction, so the reader has
                # to be rebuilt — the flag cannot be toggled on a live one.
                def _rebuild_and_infer() -> Dict[str, Any]:
                    reader = self._build_reader(self.rec_model, enable_mkldnn=False)
                    result = self._infer_once(reader, recognized)
                    self._primary = reader  # keep the one that works
                    return result

                retry = self._stage("inference_without_mkldnn", _rebuild_and_infer)
                if retry["ok"]:
                    self._working_mkldnn = False

        facts: Dict[str, Any] = {
            "collected_utc": _utc_now(),
            "warm_up_calls": self._warm_up_calls,
            "config": {
                "load_models": self.load_models,
                "include_cyrillic": self.include_cyrillic,
                "run_inference": self.run_inference,
                "det_model": self.det_model,
                "rec_model": self.rec_model,
                "max_side": MAX_SIDE,
            },
            "thread_env": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "memory": {
                "limit_mb": _memory_limit_mb(),
                "current_rss_mb": _rss_mb(),
                "peak_rss_mb": _peak_mb(),
            },
            "stages": self._stages,
            "checkpoints": self._checkpoints,
            "model_cache": _cache_report(self.cache_candidates),
            "expected_text": self.test_text,
            "recognized": recognized,
            "mkldnn": {
                "requested": self.enable_mkldnn,
                "fallback_allowed": self.mkldnn_fallback,
                "working": self._working_mkldnn,
            },
        }
        facts["verdict"] = self._verdict(facts)
        return {"report": self._render(facts), "facts": facts}

    @staticmethod
    def _verdict(facts: Dict[str, Any]) -> str:
        stages = facts["stages"]
        mkldnn = facts.get("mkldnn") or {}
        working = mkldnn.get("working")

        # A first-attempt inference failure that the fallback then rescued is a
        # RESULT, not a failure — it identifies mkldnn as the cause. Only treat
        # stages as fatal when nothing produced a working reading.
        rescued = (
            any(s["stage"] == "inference_without_mkldnn" and s["ok"] for s in stages)
            and working is False
        )
        failed = [
            s["stage"]
            for s in stages
            if not s["ok"] and not (rescued and s["stage"] == "inference")
        ]
        if failed:
            return f"FAILED at: {', '.join(failed)} — see the stage detail below."

        limit = facts["memory"]["limit_mb"]
        peak = facts["memory"]["peak_rss_mb"]
        prefix = "COMPLETED"
        if rescued:
            prefix = (
                "COMPLETED WITH enable_mkldnn=False — mkldnn confirmed as the "
                "cause of the detection failure; the port must disable it"
            )

        if limit is None or peak is None:
            return f"{prefix} — memory limit or peak unavailable on this platform."

        headroom = limit - peak
        pct = round(peak / limit * 100)
        if headroom < 256:
            return (
                f"{prefix}; UNSAFE — peak {peak:.0f} MB of a {limit:.0f} MB limit "
                f"({pct}%), only {headroom:.0f} MB spare. A larger image would OOM."
            )
        if headroom < 1024:
            return (
                f"{prefix}; TIGHT — peak {peak:.0f} MB of {limit:.0f} MB ({pct}%), "
                f"{headroom:.0f} MB spare. Real alert photos are larger than this "
                "synthetic test image; measure with a real one before committing."
            )
        return (
            f"{prefix}; COMFORTABLE — peak {peak:.0f} MB of {limit:.0f} MB ({pct}%), "
            f"{headroom:.0f} MB spare."
        )

    @staticmethod
    def _render(facts: Dict[str, Any]) -> str:
        lines: List[str] = [
            "=== PaddleOCR feasibility probe ===",
            "",
            f"collected      {facts['collected_utc']}",
            f"warm_up calls  {facts['warm_up_calls']}",
            f"config         {facts['config']['rec_model']} "
            f"(cyrillic={facts['config']['include_cyrillic']})",
            f"mkldnn         requested={facts['mkldnn']['requested']} "
            f"working={facts['mkldnn']['working']}",
            f"thread env     {facts['thread_env']}",
            "",
            "-- memory --",
        ]
        mem = facts["memory"]
        lines += [
            f"cgroup limit   {mem['limit_mb']} MB",
            f"current RSS    {mem['current_rss_mb']} MB",
            f"PEAK RSS       {mem['peak_rss_mb']} MB   <- the OOM-relevant number",
            "",
            "-- stages --",
        ]
        for stage in facts["stages"]:
            status = "OK  " if stage["ok"] else "FAIL"
            lines.append(
                f"  {status} {stage['stage']:<20} {stage['seconds']:>7.2f}s  "
                f"rss={stage['rss_mb']} MB  peak={stage['peak_mb']} MB"
            )
            if not stage["ok"]:
                lines.append(f"       {stage.get('error')}")

        lines += ["", "-- model cache on disk --"]
        if facts["model_cache"]:
            for entry in facts["model_cache"]:
                lines.append(
                    f"  {entry['mb']:>8.1f} MB  {entry['files']:>4} files  {entry['path']}"
                )
        else:
            lines.append("  (nothing found — models may live elsewhere)")

        recognized = facts["recognized"]
        lines += ["", "-- inference --"]
        if recognized:
            match = recognized["text"].strip() == facts["expected_text"]
            lines += [
                f"  expected    {facts['expected_text']!r}",
                f"  recognized  {recognized['text']!r}",
                f"  exact match {match}",
                f"  lines={recognized['lines']} "
                f"mean_confidence={recognized['mean_confidence']}",
                f"  image sha256 {recognized.get('image_sha256', 'n/a')[:32]}"
                "   <- must match the local parity check",
            ]
            for entry in recognized["per_line"]:
                lines.append(f"    {entry['conf']}  {entry['text']!r}")
        else:
            lines.append("  (not run)")

        lines += ["", f"VERDICT  {facts['verdict']}"]
        return "\n".join(lines)
