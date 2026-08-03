"""Filenames for deepset's flat file store.

deepset's file store is one flat namespace per workspace — there are no
directories, so the `<alertId>/<folder>/` layout of the local reference cannot be
expressed as a path. It is carried in the filename instead:

    10099538__published__photo.jpg

`parse_prefixed_filename` in `paddle_ocr.py` reads that form back; this module
writes it. The two live in the same package and ship in the same bundle version
deliberately: a fetcher that writes a form the parsers cannot read yields
`alert_id: "unknown"` on every Document rather than an error, so a skew between
them would be silent. `tests/components/test_naming.py` asserts the round trip.

The parser is NOT moved or re-exported here. Doing so would touch three
components that are already byte-exact against their local references (see the
note at `barcode_decoder.py:305`), so this module is additive only.
"""

from __future__ import annotations

import re
import unicodedata

FOLDERS = ("published", "restricted")

# Only the characters a name cannot carry through a filesystem or a multipart
# upload. This is deliberately the SAME rule as the local staging script
# (`ai-alert-assistant/src/utils/prepare_training_data.py:101`) and deliberately
# NOT `doc_extractor.sanitize_name`, which is far more aggressive because it
# builds a markdown *title*: it would strip accents, brackets and commas out of
# the filename itself. Every Document's `title:` and `source:` derive from the
# name chosen here, so any divergence from what the reference held on disk breaks
# byte-parity for a reason that has nothing to do with the pipeline.
_UNSAFE_RX = re.compile(r'[<>:"/\\|?*\n\r\t]+')


def clean_attachment_name(file_name: str, fallback: str = "attachment") -> str:
    """Normalise to NFC, then replace only filesystem-hostile characters.

    NFC first, and that is not cosmetic. Attachment names arrive in NFD
    (`a` + U+0301) where the local disk holds NFC (`á`). A combining accent is
    not `isalnum()`, so the components' own `sanitize_name` later turns each one
    into `_` and "Vizsgálati jegyzőkönyv" renders as
    "Vizsga_lati jegyzo_ko_nyv" (`paddle_ocr.py:236-241`). Normalising at the
    point the name is *chosen* fixes it once for every downstream consumer
    instead of defensively in each one.
    """
    name = unicodedata.normalize("NFC", str(file_name))
    name = _UNSAFE_RX.sub("_", name).strip()
    # A name has to keep at least one alphanumeric character to count as a name.
    # `"///"` substitutes to a single `"_"` — truthy, so a plain `or fallback`
    # would accept it, and two differently-named unusable attachments in the same
    # alert and folder would both become `"_"` and silently overwrite each other.
    # `isalnum` rather than ASCII: CJK and Cyrillic names are real content.
    if not any(char.isalnum() for char in name):
        return fallback
    return name


def build_prefixed_filename(alert_id: str, folder: str, file_name: str) -> str:
    """-> `<alertId>__<folder>__<cleaned name>` for deepset's flat namespace.

    Raises on an alert id or folder the parser would reject rather than emitting
    a name that silently resolves to `alert_id: "unknown"`. `_PREFIX_RX` requires
    a numeric id and one of exactly two folder names, and a name that fails to
    parse is indistinguishable downstream from an attachment that never had any
    provenance.

    The prefix also does work metadata cannot: it keeps two alerts that both
    contain `1.jpg` from colliding in the flat namespace. Combined with
    `write_mode=OVERWRITE` an unprefixed name would let one alert's image
    silently destroy another's, and a count-based readiness check would still
    pass because *a* file is there.
    """
    alert = str(alert_id).strip()
    if not alert.isdigit():
        raise ValueError(f"alert_id must be numeric, got {alert_id!r}")
    if folder not in FOLDERS:
        raise ValueError(f"folder must be one of {FOLDERS}, got {folder!r}")
    return f"{alert}__{folder}__{clean_attachment_name(file_name)}"
