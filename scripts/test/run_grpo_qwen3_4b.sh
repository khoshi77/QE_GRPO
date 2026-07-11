#!/bin/bash
# Nノード(各1 H100 80GB) で Qwen3-4B + GSM8K + GRPO。NNODES は環境変数で受け取る。
# Ray クラスタが立ち上がっている前提で、rank0 (head) からのみ呼ばれる。
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

# PROJECT_NAME / EXPERIMENT_NAME / TENSORBOARD_DIR は setup_env.sh で定義済み

# ============================================================================
# 学習設定（このブロックを編集して挙動を変える）
# ============================================================================

# モデル
MODEL_PATH=/work/UTSUROLB/utlb_buma2/models/Qwen3-4B

# データ
TRAIN_FILE=${PROJECT_DIR}/data/gsm8k/train.parquet
TEST_FILE=${PROJECT_DIR}/data/gsm8k/test.parquet

# 学習量
TOTAL_STEPS=1               # 総ステップ数。-1 で epoch ベース（total_epochs に従う）
TOTAL_EPOCHS=1

# 保存・評価
SAVE_FREQ=-1                # -1=保存しない / N=N step ごと(+最終stepは必ず保存)
TEST_FREQ=-1                # -1=評価しない / N=N step ごとに評価
# checkpoint に保存する内容。'hf_model' を含めると actor/huggingface/ 配下に
# from_pretrained() でロードできる HF 形式の weights が一緒に保存される。
SAVE_CONTENTS=[model,optimizer,extra,hf_model]

# ============================================================================

CKPTS_DIR=${PROJECT_DIR}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}
LOG_DIR=${PROJECT_DIR}/logs/${PROJECT_NAME}
mkdir -p "${CKPTS_DIR}" "${LOG_DIR}" "${TENSORBOARD_DIR}"

# 既存の Ray クラスタ（sub_grpo.sh が起動済）に接続する。
# main_ppo は ray.init() 時に環境変数 RAY_ADDRESS を見るので明示。
export RAY_ADDRESS="${HEAD_IP}:${HEAD_PORT}"

python -m verl.trainer.main_ppo \
    hydra.run.dir="${PROJECT_DIR}/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
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
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
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
    trainer.val_before_train=False 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"
