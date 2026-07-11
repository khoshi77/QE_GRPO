#!/bin/bash
# WMT21 en-ja QE jsonl を verl 形式 parquet に変換し ${PROJECT_DIR}/data/qe_wmt21_en_ja/ へ保存。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

INPUT_DIR=/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT/data/WMT21_en-ja/en-ja-test21
OUT_DIR="${PROJECT_DIR}/data/qe_wmt21_en_ja"
mkdir -p "${OUT_DIR}"

python "${SCRIPT_DIR}/preprocess_qe_wmt21.py" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUT_DIR}"

echo "[done] $(ls -la "${OUT_DIR}")"
