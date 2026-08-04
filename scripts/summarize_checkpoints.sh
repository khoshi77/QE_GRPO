#!/bin/bash
set -euo pipefail

# ==============================================================================
# summarize_checkpoints.sh — GRPO の global_step_* checkpoint 評価結果を集計
# Usage:
#   bash summarize_checkpoints.sh                            # デフォルトの MODEL_DIR を使用
#   bash summarize_checkpoints.sh <MODEL_DIR>                # 任意の GRPO 実験ディレクトリを指定
#   bash summarize_checkpoints.sh <MODEL_DIR> _constrained   # 語彙制約デコーディングの結果を集計
#   bash summarize_checkpoints.sh <MODEL_DIR> _calibrated    # 閾値校正済みの結果を集計
#
# 期待するディレクトリ構造:
#   <MODEL_DIR>/
#     global_step_50/result/result.txt
#     global_step_100/result/result.txt
#     ...
# (run_all_checkpoints.sh が出力する形式)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GRPO 後のモデル配置ルート (global_step_* を含むディレクトリ)
DEFAULT_MODEL_DIR="/work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende/0_868943_nqsv"
MODEL_DIR="${1:-$DEFAULT_MODEL_DIR}"
SUFFIX="${2:-}"
OUTPUT_FILE="$MODEL_DIR/summary${SUFFIX}.tsv"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: ディレクトリが存在しません: $MODEL_DIR"
    exit 1
fi

echo "=============================="
echo " Checkpoint Summary"
echo " Model dir: $MODEL_DIR"
echo " Output   : $OUTPUT_FILE"
echo "=============================="

# uv の Python 環境は work_SFT/QE_SFT_4B 側にしかないため、そこで実行する
WORK_DIR="/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_4B"
cd "$WORK_DIR"
uv run python "$SCRIPT_DIR/summarize_checkpoints.py" "$MODEL_DIR" --suffix "$SUFFIX" --output "$OUTPUT_FILE"
