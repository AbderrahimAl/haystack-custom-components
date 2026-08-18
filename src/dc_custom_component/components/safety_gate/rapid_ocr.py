"""Image OCR with RapidOCR, called directly.

RapidOCR is PaddleOCR's detection and recognition models exported to ONNX, so this is
the same model family as `SafetyGatePaddleOCR` in `paddle_ocr.py` without the paddle
runtime. Use this component where the platform image cannot carry paddlepaddle; use
`SafetyGatePaddleOCR` where it can and byte-parity with the local reference matters.

Docling is deliberately not in the path. Routing images through Docling scored 0.444
identifier recall against 0.682 for calling RapidOCR directly (measured on prod,
2026-08-06), but that measured Docling's document pipeline rather than the recognizer:
it emitted `## MADE IN CHINA` as a section heading on a product photo, reported every
image as `application/pdf`, and returned empty content for six of ten images -
consistent with a page-layout model finding no text regions and never invoking OCR at
all. Docling's own OCR route uses this same RapidOCR underneath, so the loss is the
layout stage discarding text regions, not a better recognizer being unavailable.

Calling the engine directly also restores the per-line confidences Docling's markdown
export drops, which is what `mean_confidence`, `reliability` and the watermark rule
below are built on.

The prefixed-filename parser is local to this module rather than imported from
`naming.py`, following the note at the top of that module: the parsers are
deliberately not shared.
"""

import hashlib
import importlib
import io
import json
import re
from typing import Any, Dict, List, Tuple

from haystack import Document, component
from haystack.dataclasses import ByteStream

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
FOLDER_ORDER = ("published", "restricted")
PREFIXED = re.compile(r"^(?P<alert>\d+)__(?P<folder>published|restricted)__(?P<name>.+)$")

# Matches the reference OCR pipeline. Full-resolution phone photos are what made
# the Docling run return nothing at all.
MAX_SIDE = 2000

RELIABILITY = ((0.85, "high"), (0.65, "mixed"), (0.0, "low"))
HEADER = (
    "Confidence is the recognizer's probability per line (0-1): >=0.85 reliable, "
    "0.65-0.85 mixed, <0.65 often misread - treat low-confidence lines as hints."
)

# --- camera watermark suppression (#26) --------------------------------------
# Phone cameras burn their own model name onto the bottom of every frame. The
# recognizer reads it at ~1.00 confidence, it lands in each photo of the alert,
# and the prediction model concludes the product is a phone: alert 10092373 is an
# Arcobaleno perfume that all three models answered as an OPPO A98 5G smartphone.
#
# The camera writes that same string into the file's EXIF `Model` tag, so the
# watermark is not guessed from a brand list - it is matched against the device
# the image itself reports. That generalises to brands nobody has enumerated,
# which is the one thing a lexicon cannot do.
#
# Measured over the six affected alerts (50 images, 341 OCR lines): the nine
# spellings the recognizer produced for three physical watermarks all land within
# 2 edits of the EXIF model - including `OPPOA985G` with no spaces at all, which
# defeats any token-based rule, and `OPPO A78 6G` with two digits misread, which
# defeats exact matching. The nearest real product line is 7 edits away (`ora`,
# `MILANO`, `AB.MDEA`), so the band from 3 to 6 is empty and the threshold has
# room for a worse read than any observed.
#
# Position is the second half of the rule: 37 of 41 occurrences are the last OCR
# line and the remaining 4 are second to last, because the recognizer emits
# roughly top to bottom and the overlay sits on the bottom edge. Restricting to
# the tail preserves a genuine mention of a phone model inside a label, and buys
# back the protection lost by ignoring spaces.
WATERMARK_TAIL_LINES = 2
# Below this a model string is too generic to be evidence: a phone reporting
# `Mi 9` compacts to `MI9`, where any tolerance at all matches half the corpus.
MIN_DEVICE_CHARS = 6
MAX_DEVICE_EDITS = 3

EXIF_MAKE, EXIF_MODEL = 271, 272

# `29.4.2026 11:56` - the same overlay, on the line after the model name. Only
# ever suppressed when it trails a watermark: an unanchored date rule would also
# match genuine batch codes and best-before dates, which feed the `batches` field.
STAMP = re.compile(r"^\d{1,4}[.\-/]\d{1,2}[.\-/]\d{1,4}[\s,]+\d{1,2}[:.]\d{2}([:.]\d{2})?$")


def parse_prefixed_name(file_name: str) -> Tuple[str, str, str]:
    match = PREFIXED.match(file_name)
    if not match:
        return "", "", file_name
    return match.group("alert"), match.group("folder"), match.group("name")


def sort_key(file_name: str) -> Tuple[int, str]:
    _, folder, name = parse_prefixed_name(file_name)
    order = FOLDER_ORDER.index(folder) if folder in FOLDER_ORDER else len(FOLDER_ORDER)
    return order, name


def reliability(mean_confidence: float, lines: int) -> str:
    if not lines:
        return "none"
    for floor, label in RELIABILITY:
        if mean_confidence >= floor:
            return label
    return "low"


def normalise_result(result: Any) -> List[Tuple[str, float]]:
    """(text, confidence) pairs from whichever RapidOCR result shape this build returns.

    v2 returns an object carrying `txts` and `scores`; the older
    `rapidocr_onnxruntime` returned `(list_of_[box, text, score], elapse)`. Each deploy
    is a slow round trip, so both are handled rather than guessed at.
    """
    if result is None:
        return []

    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if texts is not None:
        scored = scores if scores is not None else [None] * len(texts)
        return [(str(text), float(score) if score is not None else 0.0)
                for text, score in zip(texts, scored)]

    rows = result[0] if isinstance(result, tuple) else result
    if not rows:
        return []
    pairs: List[Tuple[str, float]] = []
    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            pairs.append((str(row[1]), float(row[2])))
    return pairs


def compact(text: str) -> str:
    """Spacing is not evidence. One physical overlay was read as `OPPO A98 5G`,
    `OPPO A985G`, `OPPO A98-5G` and `OPPOA985G` within a single corpus, so the
    separators are stripped before anything is compared and the edit budget is
    spent entirely on character errors (`O`->`Q`, `9`->`7`, `G`->`0`)."""
    return re.sub(r"[^0-9A-Za-z]", "", text).upper()


def edit_distance(left: str, right: str) -> int:
    """Levenshtein, abandoned early once the strings cannot possibly match."""
    if abs(len(left) - len(right)) > MAX_DEVICE_EDITS:
        return MAX_DEVICE_EDITS + 1
    previous = list(range(len(right) + 1))
    for i, source in enumerate(left, 1):
        current = [i]
        for j, target in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (source != target)))
        previous = current
    return previous[-1]


def device_signatures(exif: Any) -> List[str]:
    """What this camera stamps on its own photos, per the file's own metadata.

    Both the model alone (`Galaxy A14`, which is exactly what Samsung burns in)
    and make + model, since which of the two a vendor writes varies. Returns
    nothing when the tags are absent - some alerts arrive re-saved through an
    editor that dropped them, and a missing signature must disable the rule
    rather than fall back to guessing.
    """
    def tag(number: int) -> str:
        value = exif.get(number) if exif else None
        if isinstance(value, bytes):
            value = value.decode("utf-8", "ignore")
        return str(value or "").replace("\x00", "").strip()

    make, model = tag(EXIF_MAKE), tag(EXIF_MODEL)
    if not model:
        return []
    signatures = [compact(model)]
    if make and not signatures[0].startswith(compact(make)):
        signatures.append(compact(make + model))
    return [s for s in signatures if len(s) >= MIN_DEVICE_CHARS]


def is_device_watermark(text: str, signatures: List[str]) -> bool:
    """Tolerance scales with the model string rather than being flat: 3 edits
    against `OPPOA985G` is a misread, 3 edits against a 6-character model is a
    wildcard."""
    candidate = compact(text)
    # "Shot on <device>" is the other common overlay format. Untested against our
    # corpus, which has none, but it costs one branch and would otherwise blow the
    # edit budget on the prefix alone.
    if candidate.startswith("SHOTON"):
        candidate = candidate[len("SHOTON"):]
    if len(candidate) < MIN_DEVICE_CHARS:
        return False
    return any(
        edit_distance(candidate, signature)
        <= min(MAX_DEVICE_EDITS, max(1, len(signature) // 3))
        for signature in signatures
    )


def suppress_watermarks(
    pairs: List[Tuple[str, float]], signatures: List[str]
) -> Tuple[List[Tuple[str, float]], int]:
    """-> (kept lines, number suppressed)."""
    if not signatures or not pairs:
        return pairs, 0
    dropped = set()
    for index in range(max(len(pairs) - WATERMARK_TAIL_LINES, 0), len(pairs)):
        if is_device_watermark(pairs[index][0], signatures):
            dropped.add(index)
            if index + 1 < len(pairs) and STAMP.match(pairs[index + 1][0].strip()):
                dropped.add(index + 1)
    return [p for i, p in enumerate(pairs) if i not in dropped], len(dropped)


def render_markdown(
    name: str,
    folder: str,
    alert_id: str,
    pairs: List[Tuple[str, float]],
    engine: str,
    suppressed: int = 0,
) -> str:
    """The reference OCR markdown layout, minus `generated:`.

    No timestamp: it would be the only line that differs on every run, and the point of
    this component is to be diffable against the reference output.
    """
    mean = round(sum(score for _, score in pairs) / len(pairs), 2) if pairs else 0.0
    source = f"{folder}/{name}" if folder else name
    lines = [
        "---",
        f'title: "{name}"',
        f"alert_id: {alert_id}",
        "artifact: image-ocr",
        f'source: "{source}"',
        f"folder: {folder}",
        f"mean_confidence: {mean:.2f}",
        f"reliability: {reliability(mean, len(pairs))}",
        f"lines: {len(pairs)}",
        # Only when it fired, so an image with no watermark renders exactly as it
        # did before the rule existed and the diff over a corpus stays readable.
        *([f"watermarks_suppressed: {suppressed}"] if suppressed else []),
        f'engine: "{engine}"',
        "---",
        "",
        f"# OCR - {source} (alert {alert_id})",
        "",
        HEADER,
        "",
    ]
    # The count, never the string. Naming the suppressed text here would put the
    # brand-shaped token back into the prompt once per photo, which is the exact
    # input that produced the misprediction; it stays recoverable from the report.
    note = (
        ["", f"_{suppressed} line{'s' if suppressed > 1 else ''} suppressed as a "
             "camera device watermark (overlay burned in by the phone, not "
             "product text)._"]
        if suppressed else []
    )
    if not pairs:
        lines.append("_No text detected._")
        return "\n".join(lines + note)

    lines += ["| Conf | Text |", "|------|------|"]
    for text, score in pairs:
        lines.append(f"| {score:.2f} | {text.replace('|', chr(92) + '|')} |")
    return "\n".join(lines + note)


def document_id(alert_id: str, folder: str, name: str) -> str:
    key = f"safety-gate-image-ocr|{alert_id}|{folder}|{name}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


@component
class SafetyGateRapidOCR:
    """One Document per image, matching the granularity of the reference output."""

    def __init__(
        self,
        max_side: int = MAX_SIDE,
        det_limit_side_len: int = 1600,
        det_limit_type: str = "max",
        text_score: float = 0.5,
        extra_params: str = "",
        languages: str = "",
    ) -> None:
        """:param det_limit_side_len: the detector's own resize limit, which defaults to
        around 736 and is applied *after* our downscale. Small print then arrives too
        small to detect: on a pack shot where an EAN spans 3% of the width, 736px leaves
        it roughly 20px wide. The measured symptom is that large text reads as well as the
        reference while small text is missed entirely. Raising it to 1600 was verified to
        take effect (it appears in the engine's resolved config) and did **not** close the
        gap: 300 alerts still scored 0.635 identifier recall against 0.739 detection, so
        the remaining loss is the detector's model rather than its input size. Left
        exposed because it costs nothing and the failure mode is resolution-shaped.
        :param extra_params: JSON object merged over the resolved parameters, so tuning
        does not need the source redeployed.
        """
        self.max_side = max_side
        self.det_limit_side_len = det_limit_side_len
        self.det_limit_type = det_limit_type
        self.text_score = text_score
        self.extra_params = extra_params
        self.languages = languages
        self._engine: Any = None
        self._version = ""
        self._configured_by = ""
        self._params: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}

    @staticmethod
    def _describe(value: Any, depth: int = 0) -> Any:
        """A JSON-safe sketch of an object, two levels deep.

        Model objects and numpy arrays hang off the same tree as the settings we want,
        so anything past the depth limit or without a `__dict__` collapses to its type
        name rather than being serialised.
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if depth >= 2:
            return type(value).__name__
        if isinstance(value, dict):
            return {str(k): SafetyGateRapidOCR._describe(v, depth + 1)
                    for k, v in list(value.items())[:60]}
        if isinstance(value, (list, tuple)):
            return [SafetyGateRapidOCR._describe(v, depth + 1) for v in list(value)[:12]]
        inner = getattr(value, "__dict__", None)
        if inner:
            return {k: SafetyGateRapidOCR._describe(v, depth + 1)
                    for k, v in list(inner.items())[:60] if not k.startswith("_")}
        return type(value).__name__

    def _engine_config(self, engine: Any) -> Dict[str, Any]:
        """What the engine actually resolved to.

        The parameter names are guesses until this reports back: whether the detection
        limit is `Det.limit_side_len`, `det_limit_side_len` or something else, what its
        real default is, and which recognition model got loaded. Tuning without this is
        guesswork, and one partially-applied config already produced a run where some
        alerts improved and the two known failures did not.
        """
        found: Dict[str, Any] = {}
        for name in ("config", "cfg", "params", "text_det", "text_rec", "text_cls"):
            attribute = getattr(engine, name, None)
            if attribute is not None:
                found[name] = self._describe(attribute)
        found["public_attributes"] = sorted(
            n for n in dir(engine) if not n.startswith("_"))[:60]
        found["instance"] = self._describe(getattr(engine, "__dict__", {}) or {})
        return found

    def _version_of(self, module: Any) -> str:
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
        try:
            from importlib.metadata import version as metadata_version

            for name in ("rapidocr", "rapidocr-onnxruntime", "rapidocr_onnxruntime"):
                try:
                    return str(metadata_version(name))
                except Exception:
                    continue
        except Exception:
            pass
        return "unknown"

    def _construct(self, engine_class: Any) -> Any:
        """Try each configuration API, recording which one took.

        v2 takes a dotted `params` mapping, older builds took flat keyword arguments, and
        the shapes are not interchangeable. Silently falling back to an unconfigured
        engine would be the worst outcome: the run would look tuned and measure nothing,
        so `_configured_by` is reported in the output.
        """
        dotted = {
            "Det.limit_side_len": self.det_limit_side_len,
            "Det.limit_type": self.det_limit_type,
            "Global.text_score": self.text_score,
        }
        flat = {
            "det_limit_side_len": self.det_limit_side_len,
            "det_limit_type": self.det_limit_type,
            "text_score": self.text_score,
        }
        if self.extra_params:
            overrides = json.loads(self.extra_params)
            dotted.update(overrides)
            flat.update(overrides)

        errors = []
        for label, attempt, params in (
            ("params", lambda: engine_class(params=dotted), dotted),
            ("kwargs", lambda: engine_class(**flat), flat),
        ):
            try:
                engine = attempt()
                self._configured_by, self._params = label, params
                return engine
            except Exception as error:
                errors.append(f"{label}: {type(error).__name__}: {error}")

        self._configured_by = "DEFAULTS (tuning not applied): " + "; ".join(errors)
        self._params = {}
        return engine_class()

    def warm_up(self) -> None:
        """Construct here so any model resolution happens at deployment validation
        rather than inside a request, where it would present as a timeout."""
        if self._engine is not None:
            return
        module = importlib.import_module("rapidocr")
        self._version = self._version_of(module)
        engine_class = getattr(module, "RapidOCR", None)
        if engine_class is None:
            raise RuntimeError(
                f"rapidocr {self._version} exposes no RapidOCR class; "
                f"available: {sorted(n for n in dir(module) if not n.startswith('_'))}"
            )
        self._engine = self._construct(engine_class)
        try:
            self._config = self._engine_config(self._engine)
        except Exception as error:
            self._config = {"error": f"{type(error).__name__}: {error}"}

    def _prepare(self, data: bytes) -> Tuple[Any, List[str]]:
        """-> (RGB array, device signatures).

        The EXIF is read from the image already open for the downscale, so
        identifying the camera costs no second decode. It is read *before*
        `convert`, which returns a new image carrying no metadata.
        """
        import numpy
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            try:
                signatures = device_signatures(image.getexif())
            except Exception:
                # A malformed EXIF block must cost the watermark rule, not the OCR.
                signatures = []
            image = image.convert("RGB")
            longest = max(image.size)
            if longest > self.max_side:
                scale = self.max_side / float(longest)
                image = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                    Image.LANCZOS,
                )
            return numpy.asarray(image), signatures

    @component.output_types(documents=List[Document], report=Dict[str, Any])
    def run(self, sources: List[ByteStream]) -> Dict[str, Any]:
        self.warm_up()
        engine = f"rapidocr {self._version}"

        named = sorted(
            ((stream.meta.get("file_name") or "", stream) for stream in sources),
            key=lambda pair: sort_key(pair[0]),
        )

        documents: List[Document] = []
        summary: List[Dict[str, Any]] = []
        errors: List[str] = []

        for file_name, stream in named:
            if not file_name.lower().endswith(IMAGE_EXTS):
                continue
            alert_id, folder, name = parse_prefixed_name(file_name)
            try:
                array, signatures = self._prepare(stream.data)
                pairs = normalise_result(self._engine(array))
            except Exception as error:
                errors.append(f"{file_name}: {type(error).__name__}: {error}")
                continue

            # Before the mean: the overlay reads at ~1.00 and is usually the
            # highest-confidence line in the image, so leaving it in would report
            # a reliability that describes the camera rather than the product.
            pairs, suppressed = suppress_watermarks(pairs, signatures)

            mean = round(sum(s for _, s in pairs) / len(pairs), 2) if pairs else 0.0
            documents.append(
                Document(
                    id=document_id(alert_id, folder, name),
                    content=render_markdown(
                        name, folder, alert_id, pairs, engine, suppressed
                    ),
                    meta={
                        "alert_id": alert_id,
                        "artifact": "image-ocr",
                        "file": name,
                        "folder": folder,
                        "lines": len(pairs),
                        "mean_confidence": mean,
                        "reliability": reliability(mean, len(pairs)),
                        # Always present, unlike the frontmatter: meta is queried
                        # rather than prompted, and a stable key can be aggregated
                        # across a corpus to see how often the rule fires.
                        "watermarks_suppressed": suppressed,
                        "device": signatures[0] if signatures else "",
                        "engine": engine,
                    },
                )
            )
            summary.append({"file": name, "folder": folder,
                            "lines": len(pairs), "mean_confidence": mean,
                            "watermarks_suppressed": suppressed,
                            "device": signatures[0] if signatures else ""})

        return {
            "documents": documents,
            "report": {
                "engine": engine,
                "configured_by": self._configured_by,
                "params": self._params,
                "engine_config": self._config,
                "max_side": self.max_side,
                "images": len(summary),
                "watermarks_suppressed": sum(f["watermarks_suppressed"] for f in summary),
                "files": summary,
                "errors": errors,
            },
        }
