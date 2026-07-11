#!/bin/bash
# GSM8K を verl 形式の parquet に変換し ${PROJECT_DIR}/data/gsm8k/ へ保存。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

OUT_DIR="${PROJECT_DIR}/data/gsm8k"
mkdir -p "${OUT_DIR}"

python "${PROJECT_DIR}/verl/examples/data_preprocess/gsm8k.py" \
    --local_save_dir "${OUT_DIR}"

echo "[done] $(ls -la "${OUT_DIR}")"
