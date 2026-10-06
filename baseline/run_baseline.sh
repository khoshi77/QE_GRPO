#!/usr/bin/env bash
set -euo pipefail
BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$BASELINE_DIR/.venv/bin/python" ]]; then
    DEFAULT_PYTHON="$BASELINE_DIR/.venv/bin/python"
else
    DEFAULT_PYTHON=python3
fi
exec "${PYTHON:-$DEFAULT_PYTHON}" "$BASELINE_DIR/run_baseline.py" "$@"
