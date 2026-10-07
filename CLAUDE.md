# dc-custom-component

Custom Haystack components deployed to the deepset AI Platform as an uploaded
package. Components live under `src/dc_custom_component/components/`.

## Invariants

Each rule below is a deployment break or a silent data defect, not a style
preference. All of them have been violated at least once.

### 1. Declared dependencies only

Components may import only stdlib, haystack, and packages declared in
`pyproject.toml` `dependencies`. The dependency list is deliberately minimal —
deepset's base image already provides numpy, PIL, cv2, pypdfium2 and openpyxl, and
they are intentionally not listed.

An undeclared third-party import installs cleanly in local dev and in CI (the
paddleocr stack pulls a lot in transitively) and then fails at deploy time on the
platform, which installs only the declared set. CI cannot see this class of bug.

`httpx` is this package's HTTP client (see `components/safety_gate/sgrg_client.py`).
`requests` is not a dependency and must not be imported.

### 2. Apply EXIF orientation before OCR

Any component that decodes an image with PIL must call
`PIL.ImageOps.exif_transpose(image)` before `.convert()`, `.resize()`, or
`numpy.asarray()`.

PIL does **not** auto-apply the EXIF `Orientation` tag. A phone photo saved
rotated-via-tag is otherwise handed to the recognizer in its raw, unrotated pixel
layout, which degrades detection and silently diverges from any reference produced
by a cv2-based path (cv2 *does* apply orientation). `paddle_ocr.py` avoids this by
handing paddle a file path so cv2 decodes; a component that decodes with PIL
itself must do the transpose explicitly.

This is load-bearing beyond OCR quality: a rotated frame invalidates any rule that
assumes the recognizer emits lines top-to-bottom, such as the bottom-edge
watermark suppression in `rapid_ocr.py`.

### 3. Normalise filenames to NFC before parsing them

Any function that parses or matches on an attachment filename must normalise it
with `unicodedata.normalize("NFC", name)` first.

Filenames reach deepset in NFD while local disk holds NFC. Without the
normalisation, decomposed accents flow straight into `title:`, `source:` and the
sha1 `document_id` — so the same attachment yields two different document ids
depending on which component parsed it. `paddle_ocr.py`'s
`parse_prefixed_filename` normalises; a new parser that omits it reintroduces a
bug already fixed next door.

`naming.py` documents why these parsers are deliberately duplicated per component
rather than shared. The duplication is accepted; skipping the normalisation is not.

### 4. No dead constructor parameters

A parameter accepted by `__init__` and stored on `self` must be read somewhere in
the component. A stored-but-never-read parameter looks configurable, changes
nothing, and is indistinguishable from a wiring bug until someone tries to use it.
