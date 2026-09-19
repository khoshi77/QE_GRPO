#!/bin/bash
#PBS -b 1
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=5.0.10/intel2023.0.0-cuda12.9.1
#PBS -q gpu
#PBS -A UTSUROLB
#PBS -l elapstim_req=24:00:00
#PBS -j o
#PBS -N inf1
set -euo pipefail

# ==============================================================================
# run_all_checkpoints.sh — GRPO の全 global_step checkpoint を順番に推論・評価
# Usage:
#   bash run_all_checkpoints.sh                          # デフォルト MODEL_DIR, INFER_MODE=all
#   bash run_all_checkpoints.sh <MODEL_DIR>              # INFER_MODE=all
#   bash run_all_checkpoints.sh <MODEL_DIR> generate     # 通常の自由生成のみ
#   bash run_all_checkpoints.sh <MODEL_DIR> constrained  # 語彙制約デコーディング (+閾値校正)
#   bash run_all_checkpoints.sh <MODEL_DIR> all labels   # labels: 上記すべて
#   bash run_all_checkpoints.sh <MODEL_DIR> generate xml_mt
#
# 期待するディレクトリ構造:
#   <MODEL_DIR>/
#     global_step_50/
#       actor/
#         huggingface/      <- ここを model_path として渡す
#     ...
#
# 結果ディレクトリ (global_step_* 直下):
#   result_dev/ result/                          ... 通常生成 (dev / test)
#   result_constrained_dev/ result_constrained/  ... 語彙制約デコーディング
#   result_calibrated_dev/  result_calibrated/   ... dev で校正した閾値を適用
# dev/test のファイルパスは config.yaml の data.eval_file / data.test_file を使う。
#
# 推論・評価のスクリプトと config は work_SFT/QE_SFT を流用します。
# config.yaml は実行前にバックアップし、終了時に自動復元します。
# ==============================================================================

# 推論・評価に使う既存パイプライン (work_SFT/QE_SFT_8B)
WORK_DIR="/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B"
QE_FORMAT="${3:-${QE_FORMAT:-labels}}"
case "${QE_FORMAT}" in
    labels)
        CONFIG="$WORK_DIR/src/config2.yaml"
        DEFAULT_INFER_MODE=all
        DEFAULT_MAX_NEW_TOKENS=256
        ;;
    xml_mt)
        CONFIG="$WORK_DIR/src/config2.yaml"
        DEFAULT_INFER_MODE=generate
        DEFAULT_MAX_NEW_TOKENS=448
        ;;
    *) echo "ERROR: QE_FORMAT must be labels or xml_mt: ${QE_FORMAT}"; exit 2 ;;
esac
CONFIG_BAK="${CONFIG}.bak"
MAX_NEW_TOKENS="${QE_MAX_NEW_TOKENS:-${DEFAULT_MAX_NEW_TOKENS}}"

# GRPO 後のモデル配置ルート (global_step_* を含むディレクトリ)
DEFAULT_MODEL_DIR="/work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende/0_868935_nqsv"
MODEL_DIR="${1:-$DEFAULT_MODEL_DIR}"
INFER_MODE="${2:-${DEFAULT_INFER_MODE}}"   # generate | constrained | all

if [ "${QE_FORMAT}" = "xml_mt" ] && [ "$#" -lt 1 ]; then
    echo "ERROR: xml_mt requires MODEL_DIR as the first argument" >&2
    exit 2
fi
if [ "${QE_FORMAT}" = "xml_mt" ] && [ "${INFER_MODE}" != "generate" ]; then
    echo "ERROR: xml_mt supports generate only; constrained decoding is labels-only" >&2
    exit 2
fi

case "$INFER_MODE" in
    generate)    decodings=(generate) ;;
    constrained) decodings=(constrained) ;;
    all)         decodings=(generate constrained) ;;
    *) echo "ERROR: INFER_MODE は generate / constrained / all のいずれか: $INFER_MODE"; exit 1 ;;
esac

cd "$WORK_DIR"

# 終了時（正常・異常・Ctrl+C問わず）にconfig.yamlを元に戻す
restore_config() {
    if [ -f "$CONFIG_BAK" ]; then
        cp "$CONFIG_BAK" "$CONFIG"
        rm -f "$CONFIG_BAK"
        echo "$(basename "$CONFIG") を元の状態に復元しました。"
    fi
}
trap restore_config EXIT

# config.yaml をバックアップ
cp "$CONFIG" "$CONFIG_BAK"
echo "$(basename "$CONFIG") をバックアップしました: $CONFIG_BAK"

# 推論+評価用に config を書き換える (dev/test のパスは data セクションから取得)
write_infer_config() {
    local model_path="$1" split="$2" decoding="$3" pred_file="$4" result_file="$5"
    uv run python - "$CONFIG_BAK" "$CONFIG" "$model_path" "$split" "$decoding" "$pred_file" "$result_file" "$QE_FORMAT" "$MAX_NEW_TOKENS" <<'PYEOF'
import sys
import yaml

bak, cfg_path, model_path, split, decoding, pred_file, result_file, qe_format, max_new_tokens = sys.argv[1:10]
with open(bak) as f:
    config = yaml.safe_load(f)

split_file = config["data"]["eval_file" if split == "dev" else "test_file"]
config["data"]["format"] = qe_format
config["inference"]["model_path"] = model_path
config["inference"]["decoding"] = decoding
config["inference"]["max_new_tokens"] = int(max_new_tokens)
config["inference"]["input_file"] = split_file
config["inference"]["output_file"] = pred_file
config["evaluate"]["gold_file"] = split_file
config["evaluate"]["pred_file"] = pred_file
config["evaluate"]["result_file"] = result_file

with open(cfg_path, "w") as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
print(f"config 更新: split={split} decoding={decoding} format={qe_format}")
print(f"  model_path  : {model_path}")
print(f"  output_file : {pred_file}")
PYEOF
}

# 評価のみ実行するための config 書き換え (校正済み予測の評価用)
write_eval_config() {
    local split="$1" pred_file="$2" result_file="$3"
    uv run python - "$CONFIG_BAK" "$CONFIG" "$split" "$pred_file" "$result_file" <<'PYEOF'
import sys
import yaml

bak, cfg_path, split, pred_file, result_file = sys.argv[1:6]
with open(bak) as f:
    config = yaml.safe_load(f)

split_file = config["data"]["eval_file" if split == "dev" else "test_file"]
config["evaluate"]["gold_file"] = split_file
config["evaluate"]["pred_file"] = pred_file
config["evaluate"]["result_file"] = result_file

with open(cfg_path, "w") as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
PYEOF
}

echo "=============================="
echo " Batch Evaluation: All GRPO Checkpoints (dev + test, mode=$INFER_MODE, format=$QE_FORMAT)"
echo " Model dir: $MODEL_DIR"
echo "=============================="

# global_step_* を数値順にソートして取得
checkpoints=($(ls -d "$MODEL_DIR"/global_step_* 2>/dev/null | sort -V))

if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "ERROR: global_step_* が見つかりません: $MODEL_DIR"
    exit 1
fi

echo "対象checkpoint数: ${#checkpoints[@]}"
for cp in "${checkpoints[@]}"; do
    echo "  - $(basename "$cp")"
done
echo ""

# 各checkpointを処理
for checkpoint_path in "${checkpoints[@]}"; do
    checkpoint_name=$(basename "$checkpoint_path")
    hf_model_path="$checkpoint_path/actor/huggingface"

    # huggingface 形式のモデルが存在するかチェック
    if [ ! -d "$hf_model_path" ]; then
        echo "WARNING: huggingface モデルが見つかりません。スキップします: $hf_model_path"
        continue
    fi

    for decoding in "${decodings[@]}"; do
        if [ "$decoding" = "generate" ]; then
            suffix=""
        else
            suffix="_constrained"
        fi

        for split in dev test; do
            if [ "$split" = "dev" ]; then
                result_dir="$checkpoint_path/result${suffix}_dev"
            else
                result_dir="$checkpoint_path/result${suffix}"
            fi

            echo "=============================="
            echo " Processing: $checkpoint_name [$split, $decoding]"
            echo "=============================="

            mkdir -p "$result_dir"
            pred_file="$result_dir/predictions.tsv"
            result_file="$result_dir/result.txt"

            write_infer_config "$hf_model_path" "$split" "$decoding" "$pred_file" "$result_file"

            echo "--- 推論開始: $checkpoint_name [$split, $decoding] ---"
            bash inference.sh "$CONFIG"

            echo "--- 評価開始: $checkpoint_name [$split, $decoding] ---"
            bash evaluate.sh "$CONFIG"

            echo "--- 完了: $checkpoint_name [$split, $decoding] ---"
            echo ""
        done
    done

    # 語彙制約の結果があれば、dev で閾値校正 → dev/test に適用して評価
    if [[ " ${decodings[*]} " == *" constrained "* ]]; then
        echo "=============================="
        echo " Calibration: $checkpoint_name (dev で閾値フィット → dev/test に適用)"
        echo "=============================="
        cal_dev_dir="$checkpoint_path/result_calibrated_dev"
        cal_test_dir="$checkpoint_path/result_calibrated"

        uv run python src/models/calibrate_threshold.py \
            --dev-pred  "$checkpoint_path/result_constrained_dev/predictions.tsv" \
            --test-pred "$checkpoint_path/result_constrained/predictions.tsv" \
            --out-dev   "$cal_dev_dir" \
            --out-test  "$cal_test_dir"

        write_eval_config dev "$cal_dev_dir/predictions.tsv" "$cal_dev_dir/result.txt"
        bash evaluate.sh "$CONFIG"
        write_eval_config test "$cal_test_dir/predictions.tsv" "$cal_test_dir/result.txt"
        bash evaluate.sh "$CONFIG"
        echo "--- 校正完了: $checkpoint_name ---"
        echo ""
    fi
done

echo "=============================="
echo " 全checkpoint の評価が完了しました (dev + test, mode=$INFER_MODE)"
echo "=============================="

SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:?SLACK_WEBHOOK_URL is not set}"
curl -X POST -H 'Content-type: application/json' --data '{"text":"終了しました！"}' "$SLACK_WEBHOOK_URL"
