#!/bin/bash
# Nノード(各1 H100 80GB) で Qwen3-4B(SFT済) + WMT21 en-ja word-level QE + GRPO。
# Ray クラスタが立ち上がっている前提で、rank0 (head) からのみ呼ばれる。
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PROJECT_NAME は setup_env.sh より前に export する必要あり
# (EXPERIMENT_NAME / TENSORBOARD_DIR がこれを参照するため)
export PROJECT_NAME="${PROJECT_NAME:-verl_grpo_qwen3_4b_qe_wmt21_enja}"

source "${SCRIPT_DIR}/setup_env.sh"

# ============================================================================
# 学習設定（このブロックを編集して挙動を変える）
# ============================================================================

# モデル
MODEL_PATH=/work/UTSUROLB/utlb_buma2/work_grpo/sft_ckpts/Qwen3-8B-SFT/checkpoint-500
# MODEL_PATH=/work/UTSUROLB/utlb_buma2/models/Qwen3-4B-Base

# データ
TRAIN_FILE=${PROJECT_DIR}/data/qe_wmt21_en_ja/train.parquet
VAL_FILE=${PROJECT_DIR}/data/qe_wmt21_en_ja/dev.parquet

# 学習量
TOTAL_STEPS=null            # null = epoch ベース (total_epochs に従う) / 正の整数で step 数固定
TOTAL_EPOCHS=10             # 800 件 / batch=16 = 50 step/epoch -> 500 step

# 保存・評価
SAVE_FREQ=50                # 50 step ごと(+最終stepは必ず保存) / -1 で保存しない
TEST_FREQ=50                # 50 step ごとに dev 評価 / -1 で評価しない
SAVE_CONTENTS=[model,optimizer,extra,hf_model]

# 報酬関数の挙動 (qe_reward.py の compute_score に渡る reward_kwargs)
REWARD_METRIC=token_mix     # token_mix / weighted_token_accuracy / bad_f1_safe / mcc / f1_bad / f1_ok / f1_product / f1_macro
REWARD_LENGTH_MISMATCH=pad_bad    # pad_bad / penalize / zero
REWARD_INVALID_TOKEN=as_bad       # as_bad / penalize / zero
REWARD_LENGTH_PENALTY=0.5
REWARD_INVALID_PENALTY=0.5
REWARD_W_BAD=2.5
REWARD_W_OK=1.0
REWARD_TOKEN_MIX_WEIGHT=0.8
REWARD_BAD_F1_WEIGHT=0.2
REWARD_LENGTH_MISMATCH_PENALTY=0.2
REWARD_INVALID_RATIO_PENALTY=0.1
REWARD_EXACT_FORMAT_BONUS=0.0

# シーケンス長 (QE は src+mt が長め、ラベル列は短め)
MAX_PROMPT_LEN=768
MAX_RESPONSE_LEN=256

# rollout sampling 設定
ROLLOUT_TEMPERATURE=0.5
ROLLOUT_TOP_P=0.9
ROLLOUT_TOP_K=-1

# ============================================================================

CKPTS_DIR=${PROJECT_DIR}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}
LOG_DIR=${PROJECT_DIR}/logs/${PROJECT_NAME}
mkdir -p "${CKPTS_DIR}" "${LOG_DIR}" "${TENSORBOARD_DIR}"

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
