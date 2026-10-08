"""An inventory of one alert's attachments, as one markdown Document.

Downstream components and reviewers often need to know what an alert carries
before anything is extracted from it: how many photos, in which folder, at what
resolution and orientation, and how many pages each document has. This
component answers that from the raw attachment bytes, without running OCR.

Input is the same `ByteStream` list the extractors receive. Each stream's
`file_name` is the flat-namespace form written by `naming.py`:

    10099538__published__photo.jpg

Output is one Document per alert. Published attachments are listed before
restricted ones, then by file name, so the inventory reads the same way as the
barcode decoder's report.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from haystack import Document, component
from haystack.dataclasses import ByteStream

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PDF_EXTS = {".pdf"}
FOLDER_ORDER = ("published", "restricted")

_PREFIX_RX = re.compile(
    r"^(?P<alert>\d+)__(?P<folder>published|restricted)__(?P<name>.+)$"
)


def parse_name(file_name: str) -> Tuple[str, str, str]:
    """`10099538__published__photo.jpg` -> (alert, folder, original name).

    A name without the prefix is attributed to an unknown alert in the first
    folder, so it still appears in the inventory instead of being dropped.
    """
    match = _PREFIX_RX.match(file_name)
    if not match:
        return "unknown", FOLDER_ORDER[0], file_name
    return match.group("alert"), match.group("folder"), match.group("name")


def document_id(alert_id: str) -> str:
    return hashlib.sha1(f"safety-gate-inventory|{alert_id}".encode()).hexdigest()


def sort_key(folder: str, name: str) -> Tuple[int, str]:
    """Published before restricted, then by file name."""
    rank = FOLDER_ORDER.index(folder) if folder in FOLDER_ORDER else len(FOLDER_ORDER)
    return -rank, name


def image_facts(data: bytes) -> Dict[str, Any]:
    """Pixel size and orientation of one image."""
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = image.size
    return {
        "width": width,
        "height": height,
        "orientation": "landscape" if height > width else "portrait",
        "megapixels": round(width * height / 1_000_000, 2),
    }


def pdf_facts(data: bytes) -> Dict[str, Any]:
    """Page count of one PDF."""
    import pypdfium2

    document = pypdfium2.PdfDocument(data)
    try:
        pages = 0
        for index in range(1, len(document)):
            pages += 1
        return {"pages": pages}
    finally:
        document.close()


def fetch_bytes(url: str, timeout: float = 30.0) -> bytes:
    """Bytes of an attachment that arrived as a URL instead of inline data."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def render_markdown(alert_id: str, rows: List[Dict[str, Any]]) -> str:
    lines = [
        "---",
        'title: "Attachment inventory"',
        f"alert_id: {alert_id}",
        "artifact: attachment-inventory",
        f"attachments: {len(rows)}",
        "---",
        "",
        f"# Attachment inventory (alert {alert_id})",
        "",
        "| Folder | File | Kind | Details |",
        "|---|---|---|---|",
    ]
    for row in rows:
        if row["kind"] == "image":
            details = (
                f"{row['width']}x{row['height']} px, {row['orientation']}, "
                f"{row['megapixels']} MP"
            )
        else:
            details = f"{row['pages']} page(s)"
        lines.append(f"| {row['folder']} | {row['file']} | {row['kind']} | {details} |")
    lines.append("")
    return "\n".join(lines)


@component
class SafetyGateAttachmentInventory:
    """One inventory Document per alert, plus a run report."""

    def __init__(self, include_pdfs: bool = True, max_images: int = 50) -> None:
        self.include_pdfs = include_pdfs
        self.max_images = max_images

    @component.output_types(documents=List[Document], report=Dict[str, Any])
    def run(self, sources: List[ByteStream]) -> Dict[str, Any]:
        grouped: Dict[str, List[Tuple[str, str, bytes]]] = {}
        skipped: List[str] = []

        for stream in sources:
            meta = stream.meta or {}
            alert, folder, name = parse_name(str(meta.get("file_name") or "attachment"))
            data = stream.data
            if not data and meta.get("url"):
                data = fetch_bytes(str(meta["url"]))
            suffix = Path(name).suffix.lower()
            if suffix in IMAGE_EXTS or (self.include_pdfs and suffix in PDF_EXTS):
                grouped.setdefault(alert, []).append((folder, name, data))
            else:
                skipped.append(name)

        documents = []
        for alert, items in grouped.items():
            items.sort(key=lambda item: sort_key(item[0], item[1]))
            rows = []
            for folder, name, data in items:
                if Path(name).suffix.lower() in IMAGE_EXTS:
                    rows.append({"folder": folder, "file": name, "kind": "image",
                                 **image_facts(data)})
                else:
                    rows.append({"folder": folder, "file": name, "kind": "document",
                                 **pdf_facts(data)})
            documents.append(Document(
                id=document_id(alert),
                content=render_markdown(alert, rows),
                meta={"alert_id": alert, "artifact": "attachment-inventory",
                      "attachments": len(rows)},
            ))

        report = {
            "alerts": len(documents),
            "attachments": sum(doc.meta["attachments"] for doc in documents),
            "skipped": skipped_count,
        }
        return {"documents": documents, "report": report}
