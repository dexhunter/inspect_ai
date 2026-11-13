"""Peak traced bytes for one provider_image_data_uri call.

Workload: a 3 MiB canonical PNG data URI (MIME already image/png), the shape
of an inline image a provider returns. The URI is built before tracing.
"""

from __future__ import annotations

import base64
import tracemalloc

from inspect_ai._util.images import provider_image_data_uri

_RAW_BYTES = 3 * 1024 * 1024


def _canonical_png_uri() -> str:
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * (_RAW_BYTES - 8)
    payload = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{payload}"


def main() -> None:
    image = _canonical_png_uri()
    tracemalloc.start()
    result = provider_image_data_uri(image)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if result != image:
        raise SystemExit("canonical provider image URI was rewritten")
    print(peak)


if __name__ == "__main__":
    main()
