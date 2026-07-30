"""Safety Gate image OCR — deepset port of the local preprocessing scripts.

A faithful port of `preprocessing/ocr_extract.py` (per-image OCR, Cyrillic
escalation) plus `preprocessing/ocr_render_md.py` (markdown rendering), from the
`safety-gate-extraction` repository. One input image in, one Haystack Document
out whose `content` is byte-comparable with the reference `<alertId>/output/`
markdown.

Port decisions, all of them load-bearing
----------------------------------------

**`enable_mkldnn=False`.** The local `build_reader()` passes True, but on
deepset's Linux/x86_64 that routes text detection through oneDNN kernels which
raise `ConvertPirAttribute2RuntimeAttribute not support` and fail outright
(measured 2026-07-28 via PaddleOcrProbe; disabling it fixed inference and read
the test image exactly). Paddle's macOS/arm64 build almost certainly has no
oneDNN compiled in, so the local reference was already produced without it —
this likely *aligns* the platforms rather than diverging them.

**Images are staged to a temp file, not decoded in-process.** This looks like a
detour and is deliberate. `ocr_extract.load_image` hands paddle a *path* for
images within `MAX_SIDE`, and paddle then decodes with cv2 — which applies EXIF
orientation. Decoding here with PIL instead would skip EXIF, silently rotating
every phone photo relative to the reference. Staging the bytes and reusing the
exact local branch keeps both paths identical.

**Frontmatter is emitted twice.** It stays inside `content` because the
prediction pipeline's bundler parses it out of the text and the prompt shows
`reliability` to the model; it is also copied into `meta` so the retriever can
filter by `alert_id` and rank by reliability.

**Document ids are deterministic.** `content` embeds a `generated` timestamp, so
Haystack's content-derived id would change on every re-index and defeat
`DocumentWriter`'s duplicate policy. The id is therefore derived from
alert/folder/filename only — which is what makes re-indexing idempotent, the
deepset equivalent of the local skip-if-exists behaviour.

Deliberately NOT ported: the alert-level content-hash dedup of
`ocr_extract.process_alert`. It needs every image of an alert at once, which a
per-file index component never sees.
"""

from __future__ import annotations

import os

# MUST precede any paddle/numpy import. Mirrors ocr_extract.py:41-43 — without
# it each process spawns thread pools across every core and throughput
# collapses. __init__ and warm_up() both run far too late for this.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_THREAD_LIMIT",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("FLAGS_call_stack_level", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import hashlib  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import unicodedata  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple, Union, cast  # noqa: E402

from haystack import (  # noqa: E402
    Document,
    component,
    default_from_dict,
    default_to_dict,
)
from haystack.components.converters.utils import (  # noqa: E402
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream  # noqa: E402

logger = logging.getLogger(__name__)

# --- constants, all mirrored from preprocessing/ocr_extract.py ---------------
DET_MODEL = "PP-OCRv6_medium_det"
PRIMARY_REC = "PP-OCRv6_medium_rec"
PRIMARY_KEY = "PP-OCRv6"
CYRILLIC_REC = "cyrillic_PP-OCRv5_mobile_rec"
CYRILLIC_KEY = "cyrillic"
ESCALATE_CONF = 0.65
MAX_SIDE = 2000
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Verbatim from ocr_render_md.py — including the en-dash and the ≥/< signs.
# Any change here breaks byte-comparability with the reference markdown.
LEGEND = (
    "Confidence is the recognizer's probability per line (0–1): ≥0.85 reliable, "
    "0.65–0.85 mixed, <0.65 often misread — treat low-confidence lines as hints."
)


# --- rendering: pure functions, ported 1:1 from ocr_render_md.py -------------
# Kept at module level and paddle-free so they are testable without the models.


def yaml_str(value: Any) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def band(conf: float) -> str:
    if conf >= 0.85:
        return "high"
    if conf >= 0.65:
        return "medium"
    return "low"


def table_text(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def entry_reliability(entry: Dict[str, Any]) -> str:
    if "error" in entry and "text" not in entry:
        return "error"
    if not (entry.get("lines") or []):
        return "none"
    return band(entry.get("mean_confidence") or 0.0)


def render_body(entry: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if "error" in entry and "text" not in entry:
        out.append(f"**Extraction error:** {entry['error']}")
        return out

    lines = entry.get("lines") or []
    if not lines:
        out.append("_No text detected._")
        return out

    out.append("| Conf | Text |")
    out.append("|------|------|")
    for line in lines:
        conf = line.get("confidence")
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
        out.append(f"| {conf_s} | {table_text(line.get('text', ''))} |")

    by_model = entry.get("by_model")
    if by_model:
        winner = entry.get("language_model")
        for model, alt in by_model.items():
            if model == winner:
                continue
            out.append("")
            if "error" in alt:
                out.append(f"> Alternative model `{model}` failed: {alt['error']}")
            else:
                out.append(
                    f"> Alternative reading by `{model}` "
                    f"(mean confidence {alt.get('mean_confidence', 0):.2f}):"
                )
                alt_text = (alt.get("text") or "").replace("\n", " / ")
                out.append(f"> {alt_text if alt_text else '_(empty)_'}")
    return out


def render_image_md(
    entry: Dict[str, Any],
    alert_id: str,
    folder: str,
    image_name: str,
    engine: str,
    ocr_generated: str,
    generated: Optional[str] = None,
) -> str:
    """The standalone markdown for one image, matching ocr_render_md.py exactly.

    `ocr_generated` is a separate argument because locally the OCR pass and the
    markdown render are two scripts run at different times; here they happen
    microseconds apart, so the two timestamps will normally be equal. That is a
    documented, deliberate difference from the reference.
    """
    reliability = entry_reliability(entry)
    lines = entry.get("lines") or []
    mean = entry.get("mean_confidence")
    suffix = Path(image_name).suffix.lower().lstrip(".") or "unknown"

    frontmatter = [
        "---",
        f"title: {yaml_str(image_name)}",
        f"alert_id: {alert_id}",
        "artifact: image-ocr",
        f"source: {yaml_str(f'{folder}/{image_name}')}",
        f"folder: {folder}",
        f"file_type: {suffix}",
        f"language_model: {entry.get('language_model', 'n/a')}",
        f"mean_confidence: {f'{mean:.2f}' if isinstance(mean, (int, float)) else 'n/a'}",
        f"reliability: {reliability}",
        f"lines: {len(lines)}",
        f"engine: {yaml_str(engine)}",
        f"detection_model: {DET_MODEL}",
        f"generated: {yaml_str(generated or _now())}",
        f"ocr_generated: {yaml_str(ocr_generated)}",
        "---",
        "",
        f"# OCR — {folder}/{image_name} (alert {alert_id})",
        "",
        LEGEND,
        "",
    ]
    return "\n".join(frontmatter) + "\n" + "\n".join(render_body(entry)) + "\n"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# `<alertId>__<folder>__<original name>` — the self-describing upload convention.
# Deliberately strict: a numeric alert id AND one of the two known folder names.
# Real attachment names do contain double underscores, and a loose pattern would
# silently mangle them into bogus provenance.
_PREFIX_RX = re.compile(
    r"^(?P<alert>\d+)__(?P<folder>published|restricted)__(?P<name>.+)$"
)


def parse_prefixed_filename(name: str) -> Tuple[str, str, str]:
    """-> (alert_id, folder, original_name).

    Returns empty strings for alert_id/folder and the name unchanged when the
    filename does not follow the convention, so callers can treat it as a
    best-effort source and fall through to their other options.

    The name is normalised to **NFC** first, and that is not cosmetic. Filenames
    reached deepset in NFD (`a` + U+0301) where the local disk holds NFC (`á`).
    A combining accent is not `isalnum()`, so `sanitize_name` turned every one
    into `_` and "Vizsgálati jegyzőkönyv" rendered as "Vizsga_lati jegyzo_ko_nyv"
    — a byte-parity break on every accented filename (observed 2026-07-30).
    NFC is the canonical interchange form and is a no-op on already-NFC input.
    """
    name = unicodedata.normalize("NFC", name)
    match = _PREFIX_RX.match(name)
    if not match:
        return "", "", name
    return match.group("alert"), match.group("folder"), match.group("name")


def _preload_image() -> Any:
    """A tiny image WITH TEXT, for the warm-up pass.

    It has to contain text: the textline-orientation model is only invoked once
    detection finds a box, so a blank canvas would not trigger the download this
    pass exists to force.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (320, 96), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=40)
    except TypeError:  # Pillow < 10.1 takes no size argument
        font = ImageFont.load_default()
    draw.text((12, 24), "WARMUP 123", fill="black", font=font)
    return np.asarray(image)[:, :, ::-1].copy()


def document_id(alert_id: str, folder: str, image_name: str) -> str:
    """Stable id: the same image always maps to the same Document.

    Deliberately excludes content, which carries a timestamp — a content-derived
    id would change on every re-index and turn DuplicatePolicy.SKIP into an
    append-forever policy.
    """
    return hashlib.sha1(
        f"safety-gate-image-ocr|{alert_id}|{folder}|{image_name}".encode("utf-8")
    ).hexdigest()


@component
class SafetyGatePaddleOCR:
    """OCRs Safety Gate alert images into per-image markdown Documents.

    :param include_cyrillic: enable the Cyrillic escalation recognizer. The
        model is still built lazily on first use, as it is locally.
    :param escalate_conf: escalate when the primary reading's mean confidence
        falls below this. Mirrors `ESCALATE_CONF`.
    :param max_side: images longer than this on their longest edge are
        downscaled before OCR. Mirrors `MAX_SIDE`.
    :param enable_mkldnn: leave False. See the module docstring — True crashes
        text detection on deepset's platform.
    :param default_folder: used when a source carries no `folder` metadata.
    :param cyrillic_countries: alert countries that force escalation regardless
        of confidence. Locally this is read from `published.json`, which
        production does not have, so the country must arrive as metadata
        instead.
    """

    def __init__(
        self,
        include_cyrillic: bool = True,
        escalate_conf: float = ESCALATE_CONF,
        max_side: int = MAX_SIDE,
        enable_mkldnn: bool = False,
        default_folder: str = "published",
        cyrillic_countries: Optional[List[str]] = None,
        default_alert_id: str = "",
        preload_on_warm_up: bool = True,
    ) -> None:
        self.include_cyrillic = include_cyrillic
        self.escalate_conf = escalate_conf
        self.max_side = max_side
        self.enable_mkldnn = enable_mkldnn
        self.default_folder = default_folder
        self.default_alert_id = default_alert_id
        self.preload_on_warm_up = preload_on_warm_up
        self.cyrillic_countries = (
            [c.upper() for c in cyrillic_countries]
            if cyrillic_countries is not None
            else ["BG"]
        )

        self._primary: Any = None
        self._cyrillic_reader: Any = None
        self._engine: str = "paddleocr"

    def to_dict(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            default_to_dict(
                self,
                include_cyrillic=self.include_cyrillic,
                escalate_conf=self.escalate_conf,
                max_side=self.max_side,
                enable_mkldnn=self.enable_mkldnn,
                default_folder=self.default_folder,
                cyrillic_countries=self.cyrillic_countries,
                default_alert_id=self.default_alert_id,
                preload_on_warm_up=self.preload_on_warm_up,
            ),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyGatePaddleOCR":
        return cast("SafetyGatePaddleOCR", default_from_dict(cls, data))

    # --- model lifecycle ----------------------------------------------------

    def warm_up(self) -> None:
        """Build the primary recognizer once per process, and force a first pass.

        Measured on deepset 2026-07-28: ~6 s import + ~6 s model build, ~1.5 GB
        RSS with both recognizers loaded, weights cached at `~/.paddlex`
        (147 MB). The Cyrillic reader stays lazy, as it does locally — most
        alerts never trigger escalation.

        The dummy inference is not decoration. `PP-LCNet_x1_0_textline_ori` is
        built on FIRST PREDICT, not at construction, so without this the first
        real request pays that download: an observed 2m34s for one image on
        deepset, versus seconds locally where the model was already cached. A
        throwaway pass moves that cost into warm-up, where it belongs.
        """
        if self._primary is not None:
            return

        import paddleocr

        self._engine = f"paddleocr {paddleocr.__version__}"
        self._primary = self._build_reader(PRIMARY_REC)

        if self.preload_on_warm_up:
            try:
                self._run_reader(self._primary, _preload_image())
            except Exception as exc:
                # A failed pre-load costs latency on the first request, nothing
                # more — never let it take the component down.
                logger.warning("OCR pre-load pass failed (harmless): %s", exc)

    def _build_reader(self, rec_model: str) -> Any:
        """Identical to ocr_extract.py::build_reader apart from enable_mkldnn."""
        from paddleocr import PaddleOCR

        return PaddleOCR(
            text_detection_model_name=DET_MODEL,
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="cpu",
            enable_mkldnn=self.enable_mkldnn,
            cpu_threads=1,
        )

    @property
    def _cyrillic(self) -> Any:
        if self._cyrillic_reader is None:
            self._cyrillic_reader = self._build_reader(CYRILLIC_REC)
        return self._cyrillic_reader

    # --- OCR, ported from ocr_extract.py ------------------------------------

    def _load_image(self, path: Path) -> Any:
        """Port of ocr_extract.py::load_image.

        Returns the PATH for images within `max_side` so paddle decodes them
        itself (via cv2, which honours EXIF orientation), and a downscaled BGR
        array otherwise. Reproducing that branch exactly matters: decoding
        everything with PIL here would drop EXIF rotation on phone photos and
        silently diverge from the reference.
        """
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            if max(width, height) <= self.max_side:
                return str(path)
            scale = self.max_side / max(width, height)
            resized = image.convert("RGB").resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.LANCZOS,
            )
            return np.asarray(resized)[:, :, ::-1].copy()

    @staticmethod
    def _run_reader(reader: Any, ocr_input: Any) -> Dict[str, Any]:
        """Port of ocr_extract.py::run_reader.

        Bounding boxes are not collected: the markdown never renders them, and
        skipping them avoids carrying polygon arrays for every line.
        """
        result = reader.predict(ocr_input)[0]
        texts = list(result.get("rec_texts") or [])
        scores = [float(s) for s in (result.get("rec_scores") or [])]
        mean_conf = round(sum(scores) / len(scores), 4) if scores else 0.0
        return {
            "text": "\n".join(texts),
            "mean_confidence": mean_conf,
            "lines": [
                {
                    "text": text,
                    "confidence": round(scores[i], 4) if i < len(scores) else None,
                }
                for i, text in enumerate(texts)
            ],
            "score": len(texts) * mean_conf,
        }

    def _ocr_image(self, path: Path, force_cyrillic: bool) -> Dict[str, Any]:
        """Port of ocr_extract.py::ocr_image, escalation semantics included."""
        try:
            ocr_input = self._load_image(path)
            primary = self._run_reader(self._primary, ocr_input)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        escalate = self.include_cyrillic and (
            force_cyrillic
            or bool(primary["lines"])
            and primary["mean_confidence"] < self.escalate_conf
        )
        candidates: Dict[str, Dict[str, Any]] = {PRIMARY_KEY: primary}
        if escalate:
            try:
                candidates[CYRILLIC_KEY] = self._run_reader(self._cyrillic, ocr_input)
            except Exception as exc:
                candidates[CYRILLIC_KEY] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "score": -1,
                }

        best_key = max(candidates, key=lambda k: candidates[k].get("score", -1))
        best = candidates[best_key]
        entry: Dict[str, Any] = {
            "text": best["text"],
            "language_model": best_key,
            "mean_confidence": best["mean_confidence"],
            "lines": best["lines"],
        }
        if escalate:
            entry["by_model"] = {
                key: (
                    {"error": value["error"]}
                    if "error" in value
                    else {
                        "lines": len(value["lines"]),
                        "mean_confidence": value["mean_confidence"],
                        "text": value["text"],
                    }
                )
                for key, value in candidates.items()
            }
        return entry

    # --- metadata resolution -------------------------------------------------

    def _resolve(self, stream: ByteStream, extra: Dict[str, Any]) -> Dict[str, str]:
        """Work out alert_id, folder and file name for one source.

        Locally both come from the directory layout
        (`<alertId>/{published,restricted}/<file>`). deepset has no such path, so
        provenance is resolved from four sources, most specific first:

          1. explicit file metadata (`alert_id`, `folder`) — what the fetcher and
             the SDK upload path provide
          2. a filename prefix, `<alertId>__<folder>__<original>` — self-describing,
             and depends on nothing deepset has to do correctly. Needed because
             `.meta.json` sidecars are silently SKIPPED on UI upload (observed
             2026-07-28), so metadata cannot be assumed to arrive.
          3. inference from `file_path` — only fires on a real directory layout,
             which keeps local round-trip verification working
          4. init-parameter defaults, then "unknown"

        The prefix is ALWAYS stripped from the reported name, even when metadata
        supplied the values, so `title:` and `source:` in the markdown carry the
        original filename and stay byte-comparable with the reference.
        """
        meta: Dict[str, Any] = {**(stream.meta or {}), **extra}

        file_path = str(meta.get("file_path") or "")
        name = str(meta.get("file_name") or (Path(file_path).name if file_path else ""))
        if not name:
            name = "image"

        # (2) — parsed unconditionally, because the name must be stripped even
        # when metadata wins on the values.
        prefix_alert, prefix_folder, name = parse_prefixed_filename(name)

        folder = str(meta.get("folder") or "") or prefix_folder
        if not folder and file_path:
            parts = Path(file_path).parts
            for candidate in ("published", "restricted"):
                if candidate in parts:
                    folder = candidate
                    break
        if not folder:
            folder = self.default_folder

        alert_id = str(meta.get("alert_id") or "") or prefix_alert
        if not alert_id and file_path:
            parts = Path(file_path).parts
            for index, part in enumerate(parts):
                if part in ("published", "restricted") and index > 0:
                    alert_id = parts[index - 1]
                    break
        if not alert_id:
            alert_id = self.default_alert_id or "unknown"

        country = str(meta.get("country") or "").upper()
        return {
            "alert_id": alert_id,
            "folder": folder,
            "file_name": name,
            "country": country,
        }

    # --- run -----------------------------------------------------------------

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Document]]:
        """OCR each source image into one markdown Document.

        A source that fails to read is skipped with a warning rather than
        aborting the batch — one unreadable photo must not cost an entire
        alert's indexing run. OCR failures are different: those produce a
        Document with `reliability: error`, exactly as the local scripts do, so
        the failure is visible downstream instead of silently absent.
        """
        self.warm_up()

        documents: List[Document] = []
        metadata = normalize_metadata(meta, sources_count=len(sources))

        for source, extra in zip(sources, metadata):
            try:
                stream = get_bytestream_from_source(source)
            except Exception as exc:
                logger.warning("Could not read source %s: %s", source, exc)
                continue

            resolved = self._resolve(stream, extra)
            try:
                document = self._process(stream, resolved, extra)
            except Exception as exc:
                logger.warning(
                    "OCR failed for %s/%s: %s",
                    resolved["folder"],
                    resolved["file_name"],
                    exc,
                )
                continue
            documents.append(document)

        return {"documents": documents}

    def _process(
        self, stream: ByteStream, resolved: Dict[str, str], extra: Dict[str, Any]
    ) -> Document:
        """Stage the bytes, OCR them, render the markdown, build the Document."""
        name = resolved["file_name"]
        suffix = Path(name).suffix.lower() or ".png"

        # Staged to disk so `_load_image` can hand paddle a path for small
        # images — see the module docstring on EXIF orientation.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(stream.data)
            staged = Path(handle.name)
        try:
            ocr_generated = _now()
            entry = self._ocr_image(
                staged, force_cyrillic=resolved["country"] in self.cyrillic_countries
            )
        finally:
            staged.unlink(missing_ok=True)

        content = render_image_md(
            entry,
            alert_id=resolved["alert_id"],
            folder=resolved["folder"],
            image_name=name,
            engine=self._engine,
            ocr_generated=ocr_generated,
        )

        lines = entry.get("lines") or []
        document_meta: Dict[str, Any] = {
            **{k: v for k, v in (stream.meta or {}).items() if k != "file_path"},
            **extra,
            "alert_id": resolved["alert_id"],
            "artifact": "image-ocr",
            "source": f"{resolved['folder']}/{name}",
            "folder": resolved["folder"],
            "file_name": name,
            "file_type": Path(name).suffix.lower().lstrip(".") or "unknown",
            "language_model": entry.get("language_model", "n/a"),
            "mean_confidence": entry.get("mean_confidence"),
            "reliability": entry_reliability(entry),
            "lines": len(lines),
            "engine": self._engine,
            "detection_model": DET_MODEL,
        }
        return Document(
            id=document_id(resolved["alert_id"], resolved["folder"], name),
            content=content,
            meta=document_meta,
        )
