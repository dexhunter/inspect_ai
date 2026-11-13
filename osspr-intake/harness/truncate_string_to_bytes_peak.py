"""Peak traced bytes for one truncate_string_to_bytes call.

Workload: an 8 MiB tool/log string with non-ASCII characters, truncated to the
default 16 KiB max_tool_output. The input is built before tracing, so the
printed value is memory allocated by the call itself.
"""

from __future__ import annotations

import tracemalloc

from inspect_ai._util.text import truncate_string_to_bytes

_MAX_BYTES = 16 * 1024
_LINE = "2026-11-13T12:00:00Z INFO worker café processed record — ok\n"


def _tool_output() -> str:
    repeat = (8 * 1024 * 1024) // len(_LINE.encode("utf-8"))
    return _LINE * repeat


def main() -> None:
    text = _tool_output()
    tracemalloc.start()
    truncated = truncate_string_to_bytes(text, _MAX_BYTES)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if truncated is None:
        raise SystemExit("expected truncation of an over-long tool output")
    if truncated.original_bytes != len(text.encode("utf-8")):
        raise SystemExit("original_bytes does not match the UTF-8 length")
    if len(truncated.output.encode("utf-8")) > _MAX_BYTES:
        raise SystemExit("truncated output exceeded max_bytes")
    print(peak)


if __name__ == "__main__":
    main()
