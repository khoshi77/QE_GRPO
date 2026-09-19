#!/bin/bash
# Nノード(各1 H100 80GB) で SFT済 Qwen3-8B + WMT22 en-de word-level QE + GRPO。
# QE_FORMAT=labels (default) / xml_mt を選択できる。
# Ray クラスタが立ち上がっている前提で、rank0 (head) からのみ呼ばれる。
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PROJECT_NAME は setup_env.sh より前に export する必要あり
# (EXPERIMENT_NAME / TENSORBOARD_DIR がこれを参照するため)
QE_FORMAT="${QE_FORMAT:-labels}"
case "${QE_FORMAT}" in
    labels) DEFAULT_PROJECT_NAME=verl_grpo_qwen3_8b_qe_wmt22_ende ;;
    xml_mt) DEFAULT_PROJECT_NAME=verl_grpo_qwen3_8b_qe_wmt22_ende_xml_mt ;;
    *) echo "ERROR: QE_FORMAT must be labels or xml_mt: ${QE_FORMAT}" >&2; exit 2 ;;
esac
export PROJECT_NAME="${PROJECT_NAME:-${DEFAULT_PROJECT_NAME}}"

source "${SCRIPT_DIR}/setup_env.sh"

# ============================================================================
# 学習設定（このブロックを編集して挙動を変える）
# ============================================================================

# モデル・データ・応答長。QE_MODEL_PATH / QE_MAX_RESPONSE_LEN で上書き可。
case "${QE_FORMAT}" in
    labels)
        DEFAULT_MODEL_PATH=/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/output/Qwen3-8B-Base_labels_5epoch/checkpoint-4250
        DATA_DIR=${PROJECT_DIR}/data/qe_wmt22_en_de
        DEFAULT_MAX_RESPONSE_LEN=256
        ;;
    xml_mt)
        # dev MCC が最大の SFT checkpoint (summary_best_dev.tsv, 2026-08-05 時点)
        DEFAULT_MODEL_PATH=/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/output/Qwen3-8B-Base_xml_mt_5epoch/checkpoint-3500
        DATA_DIR=${PROJECT_DIR}/data/qe_wmt22_en_de_xml_mt
        # 実測 max=413 token。448 なら train/dev/test の正解XMLを全件収容できる。
        DEFAULT_MAX_RESPONSE_LEN=448
        ;;
esac
MODEL_PATH="${QE_MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
TRAIN_FILE=${DATA_DIR}/train.parquet
VAL_FILE=${DATA_DIR}/dev.parquet

# 学習量
TOTAL_STEPS="${QE_TOTAL_STEPS:-null}"  # null = epoch ベース / 正の整数で step 数固定
TOTAL_EPOCHS="${QE_TOTAL_EPOCHS:-1}"  # 59,855件 / batch=16 = 約3,741 step/epoch

# 保存・評価
SAVE_FREQ=200                # 最終stepは必ず保存 / -1 で保存しない
TEST_FREQ=100                # -1 で評価しない
# SAVE_CONTENTS=[model,optimizer,extra,hf_model]
SAVE_CONTENTS=[hf_model]

# 報酬関数の挙動 (qe_reward.py の compute_score に渡る reward_kwargs)
REWARD_METRIC="${REWARD_METRIC:-f1_macro}"     # token_mix / weighted_token_accuracy / bad_f1_safe / mcc / f1_bad / f1_ok / f1_product / f1_macro
REWARD_LENGTH_MISMATCH="${REWARD_LENGTH_MISMATCH:-pad_bad}"    # pad_bad / penalize / zero
REWARD_INVALID_TOKEN="${REWARD_INVALID_TOKEN:-as_bad}"         # as_bad / penalize / zero
REWARD_LENGTH_PENALTY=0.5
REWARD_INVALID_PENALTY=0.5
REWARD_W_BAD=2.5
REWARD_W_OK=1.0
REWARD_TOKEN_MIX_WEIGHT=0.8
REWARD_BAD_F1_WEIGHT=0.2
REWARD_LENGTH_MISMATCH_PENALTY=0.2
REWARD_INVALID_RATIO_PENALTY=0.1
REWARD_EXACT_FORMAT_BONUS=0.0
REWARD_XML_COPY_MISMATCH_FACTOR=0.25
REWARD_XML_UNBALANCED_FACTOR=0.25

# シーケンス長 (QE は src+mt が長め、ラベル列は短め)
MAX_PROMPT_LEN=768
MAX_RESPONSE_LEN="${QE_MAX_RESPONSE_LEN:-${DEFAULT_MAX_RESPONSE_LEN}}"

# rollout sampling 設定
ROLLOUT_TEMPERATURE=0.5
ROLLOUT_TOP_P=0.9
ROLLOUT_TOP_K=-1

# ============================================================================

for required_path in "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}"; do
    if [ ! -e "${required_path}" ]; then
        echo "ERROR: required path does not exist: ${required_path}" >&2
        if [ "${QE_FORMAT}" = "xml_mt" ] && [ "${required_path}" = "${TRAIN_FILE}" ]; then
            echo "Run: QE_FORMAT=xml_mt bash scripts/preprocess_qe.sh" >&2
        fi
        exit 2
    fi
done

echo "QE_FORMAT=${QE_FORMAT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "MAX_PROMPT_LEN=${MAX_PROMPT_LEN} MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN}"
echo "REWARD_METRIC=${REWARD_METRIC} REWARD_LENGTH_MISMATCH=${REWARD_LENGTH_MISMATCH} REWARD_INVALID_TOKEN=${REWARD_INVALID_TOKEN}"

python "${SCRIPT_DIR}/validate_qe_parquet.py" \
    --format "${QE_FORMAT}" \
    "${TRAIN_FILE}" "${VAL_FILE}"

CKPTS_DIR=${PROJECT_DIR}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}
LOG_DIR=${PROJECT_DIR}/logs/${PROJECT_NAME}
mkdir -p "${CKPTS_DIR}" "${LOG_DIR}" "${TENSORBOARD_DIR}"

# Match generation EOS to the SFT assistant terminator. The original checkpoint
# is never edited; already-aligned chat checkpoints are used as-is. A fresh
# directory per invocation also allows restarting the same experiment safely.
EOS_PREP_DIR=$(mktemp -d "${CKPTS_DIR}/eos_prepare.XXXXXX")
MODEL_PATH=$(python "${SCRIPT_DIR}/prepare_qe_eos.py" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${EOS_PREP_DIR}/model")
echo "EOS-validated MODEL_PATH=${MODEL_PATH}"

# 既存 Ray クラスタへ接続
export RAY_ADDRESS="${HEAD_IP}:${HEAD_PORT}"

python -m verl.trainer.main_ppo \
    hydra.run.dir="${PROJECT_DIR}/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=16 \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers=2 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents="${SAVE_CONTENTS}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="${SCRIPT_DIR}/qe_reward.py" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.metric="${REWARD_METRIC}" \
    +custom_reward_function.reward_kwargs.length_mismatch="${REWARD_LENGTH_MISMATCH}" \
    +custom_reward_function.reward_kwargs.invalid_token="${REWARD_INVALID_TOKEN}" \
    +custom_reward_function.reward_kwargs.length_penalty="${REWARD_LENGTH_PENALTY}" \
    +custom_reward_function.reward_kwargs.invalid_penalty="${REWARD_INVALID_PENALTY}" \
    +custom_reward_function.reward_kwargs.w_bad="${REWARD_W_BAD}" \
    +custom_reward_function.reward_kwargs.w_ok="${REWARD_W_OK}" \
    +custom_reward_function.reward_kwargs.token_mix_weight="${REWARD_TOKEN_MIX_WEIGHT}" \
    +custom_reward_function.reward_kwargs.bad_f1_weight="${REWARD_BAD_F1_WEIGHT}" \
    +custom_reward_function.reward_kwargs.length_mismatch_penalty="${REWARD_LENGTH_MISMATCH_PENALTY}" \
    +custom_reward_function.reward_kwargs.invalid_ratio_penalty="${REWARD_INVALID_RATIO_PENALTY}" \
    +custom_reward_function.reward_kwargs.exact_format_bonus="${REWARD_EXACT_FORMAT_BONUS}" \
    +custom_reward_function.reward_kwargs.output_format="${QE_FORMAT}" \
    +custom_reward_function.reward_kwargs.xml_copy_mismatch_factor="${REWARD_XML_COPY_MISMATCH_FACTOR}" \
    +custom_reward_function.reward_kwargs.xml_unbalanced_factor="${REWARD_XML_UNBALANCED_FACTOR}" \
    trainer.critic_warmup=0 \
    trainer.logger='[console,tensorboard]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes="${NNODES:-2}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.val_before_train=True 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"
