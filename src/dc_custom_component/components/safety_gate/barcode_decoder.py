"""Safety Gate barcode decoding — deepset port of `preprocessing/barcode_extract.py`.

An alert's images in, **exactly one** Document out per alert.

Why this component is alert-level, not per-file
-----------------------------------------------
The other two components map one input file to one output Document. This one does
not: the reference is a single `barcodes.md` per alert, aggregated across every
image, and four of its properties depend on seeing the whole alert at once:

  1. iteration order — `published` before `restricted`, each sorted by filename
  2. dedup on `(type, value)` across all images
  3. attribution of each value to the FIRST image that produced it, which
     follows from (1)
  4. emitted even when nothing decodes (`decoded: 0`), so "ran and found
     nothing" stays distinguishable from "never ran"

deepset's index processes files in **batches**, so `sources` normally contains
every file of an upload — which makes this achievable inside the index pipeline.
Sources are grouped by `alert_id` and each group yields one Document.

    THE CONSTRAINT THIS IMPOSES: an alert's images must arrive in ONE batch.
    Upload six now and four later and you get two batches; the second would
    overwrite the alert's barcode Document with only the later decodes, silently
    losing the earlier ones. Production satisfies this naturally — the fetcher
    downloads all of an alert's attachments and uploads them together.

Port decisions
--------------
**OpenCV, not pyzbar.** pyzbar with Homebrew's libzbar segfaults on arm64
(`barcode_extract.py:56-58`). `BARCODE_ENGINE=pyzbar` opts back in on a system
with a working libzbar. cv2 is already in deepset's base image, so this component
adds no dependency.

**Bytes are staged to a temp file.** `cv2.imread` takes a path, and using it keeps
the decode identical to the local script rather than introducing an
`imdecode` path with its own colour/EXIF behaviour.

**No check-digit validation, deliberately.** An EAN-13 checksum gate would be
cheap and would fit the "never emit a wrong identifier" philosophy — but it would
change the output and break byte-parity with the reference. Noted as a candidate
improvement, not applied here.

**Ordering is reconstructed, not inherited.** `sources` arrive in whatever order
deepset supplies. The local script iterates `sorted(d.iterdir())` — a
CASE-SENSITIVE sort, unlike `ocr_extract.list_images` which lowercases. That
inconsistency between the two originals is preserved, because dedup attribution
depends on it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from haystack import (
    Document,
    component,
    default_from_dict,
    default_to_dict,
)
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream

from dc_custom_component.components.safety_gate.paddle_ocr import (
    parse_prefixed_filename,
)

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
FOLDER_ORDER = ("published", "restricted")

# Verbatim from barcode_extract.py::render_md — any change breaks byte-parity.
_DECODES_PREAMBLE = (
    "Programmatically decoded symbols (decoder output, not OCR). "
    "`value` is exactly what the decoder read from the pixels."
)
_NO_DECODES = "_No barcodes decoded from any image of this alert._"


# --- decoders, ported 1:1 -----------------------------------------------------


def read_with_pyzbar(path: Path) -> List[Dict[str, str]]:
    from PIL import Image
    from pyzbar.pyzbar import decode

    results: List[Dict[str, str]] = []
    with Image.open(path) as image:
        for symbol in decode(image.convert("RGB")):
            try:
                value = symbol.data.decode("utf-8").strip()
            except UnicodeDecodeError:
                value = symbol.data.decode("latin-1").strip()
            if value:
                results.append({"type": symbol.type, "value": value})
    return results


def read_with_opencv(path: Path) -> List[Dict[str, str]]:
    """Barcodes then QR codes, in that order — the order affects dedup.

    `cv2.barcode` only exists in opencv-CONTRIB. On plain opencv the guard below
    skips it silently and you get QR codes only, which looks identical to an
    alert with no barcodes. `assert_engine_capable()` exists to surface that.
    """
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return []

    results: List[Dict[str, str]] = []
    if getattr(cv2, "barcode", None) is not None:
        try:
            detector = cv2.barcode.BarcodeDetector()
            # OpenCV 5.x GraphicalCodeDetector signature:
            # (retval, decoded_info, points, straight_code) — no type vector,
            # which is why every barcode is reported as "BARCODE".
            ok, decoded, _, _ = detector.detectAndDecodeMulti(image)
            if ok:
                for value in decoded:
                    value = (value or "").strip()
                    if value:
                        results.append({"type": "BARCODE", "value": value})
        except Exception:
            pass
    try:
        qr = cv2.QRCodeDetector()
        ok, decoded, _, _ = qr.detectAndDecodeMulti(image)
        if ok:
            for value in decoded:
                value = (value or "").strip()
                if value:
                    results.append({"type": "QRCODE", "value": value})
    except Exception:
        pass
    return results


def detect_engine(preference: str = "auto") -> str:
    """Pick the decoder once. OpenCV first — libzbar segfaults on arm64."""
    if preference == "pyzbar" or os.environ.get("BARCODE_ENGINE") == "pyzbar":
        import PIL  # noqa: F401
        import pyzbar.pyzbar  # noqa: F401

        return "pyzbar"
    if preference == "opencv":
        import cv2  # noqa: F401

        return "opencv"
    try:
        import cv2  # noqa: F401

        return "opencv"
    except Exception:
        pass
    try:
        import PIL  # noqa: F401
        import pyzbar.pyzbar  # noqa: F401

        return "pyzbar"
    except Exception as exc:
        raise RuntimeError(
            "No barcode decoder available: install opencv-contrib-python "
            f"(preferred) or pyzbar with a working libzbar ({exc})"
        ) from exc


def assert_engine_capable(engine: str) -> Optional[str]:
    """-> a warning string when the engine can only do half the job.

    Plain `opencv-python` has no `cv2.barcode`, so 1-D barcodes are silently
    skipped and the output is indistinguishable from an alert that genuinely has
    none. Given the whole point of this component is trustworthy identifiers,
    that degradation must be visible.
    """
    if engine != "opencv":
        return None
    import cv2

    if getattr(cv2, "barcode", None) is None:
        return (
            "cv2.barcode is missing (plain opencv-python instead of "
            "opencv-contrib-python): 1-D barcodes will NOT be decoded, only QR codes"
        )
    return None


# --- rendering, ported 1:1 from barcode_extract.py::render_md ------------------


def render_barcode_md(
    alert_id: str,
    engine: str,
    decodes: List[Dict[str, Any]],
    generated: str,
) -> str:
    lines = [
        "---",
        'title: "Barcode decodes"',
        f"alert_id: {alert_id}",
        "artifact: barcode-decode",
        f"engine: {engine}",
        f"decoded: {len(decodes)}",
        f'generated: "{generated}"',
        "---",
        "",
        f"# Barcode decodes (alert {alert_id})",
        "",
    ]
    if decodes:
        lines.append(_DECODES_PREAMBLE)
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(decodes, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append(_NO_DECODES)
    lines.append("")
    return "\n".join(lines)


def document_id(alert_id: str) -> str:
    """One barcode Document per alert — no folder or filename in the key."""
    return hashlib.sha1(f"safety-gate-barcode|{alert_id}".encode("utf-8")).hexdigest()


def sort_key(folder: str, file_name: str) -> Tuple[int, str]:
    """Reproduces `alert_images`: published before restricted, then a
    CASE-SENSITIVE filename sort (`sorted(d.iterdir())`). Note this differs from
    `ocr_extract.list_images`, which lowercases — an inconsistency in the
    originals that dedup attribution depends on."""
    try:
        folder_rank = FOLDER_ORDER.index(folder)
    except ValueError:
        folder_rank = len(FOLDER_ORDER)  # unknown folders sort last, stably
    return folder_rank, file_name


@component
class SafetyGateBarcodeDecoder:
    """Decodes barcodes from an alert's images into one markdown Document.

    :param engine: `auto` (OpenCV, falling back to pyzbar), or force `opencv` /
        `pyzbar`. `BARCODE_ENGINE=pyzbar` in the environment also forces pyzbar.
    :param default_folder / default_alert_id: used when a source carries neither
        metadata nor a `<alertId>__<folder>__` filename prefix.
    """

    def __init__(
        self,
        engine: str = "auto",
        default_folder: str = "published",
        default_alert_id: str = "",
    ) -> None:
        self.engine = engine
        self.default_folder = default_folder
        self.default_alert_id = default_alert_id
        self._resolved_engine: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            default_to_dict(
                self,
                engine=self.engine,
                default_folder=self.default_folder,
                default_alert_id=self.default_alert_id,
            ),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyGateBarcodeDecoder":
        return cast("SafetyGateBarcodeDecoder", default_from_dict(cls, data))

    def warm_up(self) -> None:
        """Resolve the decoder once, and warn loudly if it is only half-capable."""
        if self._resolved_engine:
            return
        self._resolved_engine = detect_engine(self.engine)
        warning = assert_engine_capable(self._resolved_engine)
        if warning:
            logger.warning("SafetyGateBarcodeDecoder: %s", warning)

    def _read(self, path: Path) -> List[Dict[str, str]]:
        try:
            if self._resolved_engine == "pyzbar":
                return read_with_pyzbar(path)
            return read_with_opencv(path)
        except Exception as exc:
            logger.warning("Barcode read failed for %s: %s", path.name, exc)
            return []

    # NOTE: a near-duplicate of the same method on the OCR and document
    # components. Worth extracting into one shared helper once all three are
    # verified — doing it now would mean re-verifying two components that are
    # already byte-exact against their references.
    def _resolve(self, stream: ByteStream, extra: Dict[str, Any]) -> Dict[str, str]:
        meta: Dict[str, Any] = {**(stream.meta or {}), **extra}

        file_path = str(meta.get("file_path") or "")
        name = str(meta.get("file_name") or (Path(file_path).name if file_path else ""))
        if not name:
            name = "image"

        prefix_alert, prefix_folder, name = parse_prefixed_filename(name)

        folder = str(meta.get("folder") or "") or prefix_folder
        if not folder and file_path:
            parts = Path(file_path).parts
            for candidate in FOLDER_ORDER:
                if candidate in parts:
                    folder = candidate
                    break
        folder = folder or self.default_folder

        alert_id = str(meta.get("alert_id") or "") or prefix_alert
        if not alert_id and file_path:
            parts = Path(file_path).parts
            for index, part in enumerate(parts):
                if part in FOLDER_ORDER and index > 0:
                    alert_id = parts[index - 1]
                    break
        alert_id = alert_id or self.default_alert_id or "unknown"

        return {"alert_id": alert_id, "folder": folder, "file_name": name}

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Document]]:
        """Decode every image, grouped by alert. One Document per alert.

        Alerts appear in the output in the order their first image appeared in
        `sources`, so a single-alert batch behaves identically to the local
        script.
        """
        self.warm_up()

        # 1. group the batch by alert, keeping only images
        grouped: Dict[str, List[Tuple[str, str, bytes]]] = {}
        metadata = normalize_metadata(meta, sources_count=len(sources))
        for source, extra in zip(sources, metadata):
            try:
                stream = get_bytestream_from_source(source)
            except Exception as exc:
                logger.warning("Could not read source %s: %s", source, exc)
                continue

            resolved = self._resolve(stream, extra)
            if Path(resolved["file_name"]).suffix.lower() not in IMAGE_EXTS:
                continue
            grouped.setdefault(resolved["alert_id"], []).append(
                (resolved["folder"], resolved["file_name"], stream.data)
            )

        # 2. one Document per alert, images in the reference order
        documents: List[Document] = []
        for alert_id, items in grouped.items():
            items.sort(key=lambda item: sort_key(item[0], item[1]))
            documents.append(self._decode_alert(alert_id, items))
        return {"documents": documents}

    def _decode_alert(
        self, alert_id: str, items: List[Tuple[str, str, bytes]]
    ) -> Document:
        """Dedup on (type, value) with first-occurrence attribution."""
        decodes: List[Dict[str, Any]] = []
        seen: set = set()

        for folder, file_name, data in items:
            suffix = Path(file_name).suffix.lower() or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(data)
                staged = Path(handle.name)
            try:
                for result in self._read(staged):
                    key = (result["type"], result["value"])
                    if key not in seen:
                        seen.add(key)
                        decodes.append({"file": file_name, "folder": folder, **result})
            finally:
                staged.unlink(missing_ok=True)

        generated = datetime.now().isoformat(timespec="seconds")
        content = render_barcode_md(alert_id, self._resolved_engine, decodes, generated)
        return Document(
            id=document_id(alert_id),
            content=content,
            meta={
                "alert_id": alert_id,
                "artifact": "barcode-decode",
                "engine": self._resolved_engine,
                "decoded": len(decodes),
                "images_scanned": len(items),
            },
        )
