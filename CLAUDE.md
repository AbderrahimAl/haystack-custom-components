# dc-custom-component

Custom Haystack components deployed to the deepset AI Platform as an uploaded
package. Components live under `src/dc_custom_component/components/`.

## Invariants

**Components may import only stdlib, haystack, and packages declared in
`pyproject.toml` `dependencies`.** The dependency list is deliberately minimal —
deepset's base image already provides numpy, PIL, cv2, pypdfium2 and openpyxl, and
they are intentionally not listed. An import of an undeclared third-party package
installs cleanly in local dev and then fails at deploy time on the platform, so it
is a deployment break rather than a style issue.

`httpx` is the HTTP client for this package (see
`components/safety_gate/sgrg_client.py`). `requests` is not a dependency and must
not be imported.
