#!/usr/bin/env bash
# Create the CPU-only development environment from a clean checkout.
# Matches CONTRIBUTING.md (`uv sync`) and CI (uv 0.13.5, locked uv.lock).
# No GPU extras, no API keys, no model downloads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.13.5/install.sh | sh
fi

# The installer drops uv in ~/.local/bin, which a non-login shell may not search.
if ! command -v uv >/dev/null 2>&1; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv --version
uv sync --frozen
