#!/bin/bash
# WMT22 en-de QE jsonl を verl 形式 parquet に変換する。
# QE_FORMAT=labels (default) または QE_FORMAT=xml_mt。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

INPUT_DIR=/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/data/WMT22_en-de_mqm
QE_FORMAT="${QE_FORMAT:-labels}"
case "${QE_FORMAT}" in
    labels) OUT_DIR="${PROJECT_DIR}/data/qe_wmt22_en_de" ;;
    xml_mt) OUT_DIR="${PROJECT_DIR}/data/qe_wmt22_en_de_xml_mt" ;;
    *) echo "ERROR: QE_FORMAT must be labels or xml_mt: ${QE_FORMAT}" >&2; exit 2 ;;
esac
mkdir -p "${OUT_DIR}"

python "${SCRIPT_DIR}/preprocess_qe_wmt21.py" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUT_DIR}" \
    --format "${QE_FORMAT}"

echo "[done] $(ls -la "${OUT_DIR}")"
