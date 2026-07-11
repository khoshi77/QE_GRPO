#!/bin/bash
# 共通環境セットアップ。各ジョブスクリプトの先頭で `source` する想定。

USER_NAME=$(whoami)
BASE_DIR=/work/UTSUROLB/${USER_NAME}
PROJECT_DIR=${BASE_DIR}/work_grpo

# Job ID。PBS 配下なら PBS_JOBID（"0:710009.nqsv" のような形式）をサニタイズして利用。
# qlogin / 純ローカル時はタイムスタンプにフォールバック。
_RAW_JOB_ID="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
export JOB_ID="${_RAW_JOB_ID//[:.\/]/_}"

# 実験を識別する名前（呼び出し側で上書き可）。ray start 前に決まる必要がある。
export PROJECT_NAME="${PROJECT_NAME:-verl_grpo_qwen3_4b_gsm8k}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${JOB_ID}}"
# verl の tracking は TENSORBOARD_DIR (env var) 経由で受け取り、ここに subpath を勝手に足さない。
# ray start より前に export する必要がある（actor へ env を伝播させるため）。
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_DIR}/tensorboard/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

export HF_HOME="${HF_HOME:-${BASE_DIR}/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
# GPUノードは外向きDNS不通。ローカル/キャッシュのみ使う。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
# Ray の preemptive OOM kill を緩める（デフォ 0.95 だと CPU 80% 程度で worker kill される）
export RAY_memory_usage_threshold=0.99
# Ray の memory monitor 自体を停止（spawn-mode worker 大量起動の瞬間スパイクで誤kill されないため）
export RAY_memory_monitor_refresh_ms=0

# torch wheel(cu126) と整合する CUDA toolkit。計算ノードで module が無ければ無視。
module load cuda/12.6.3 2>/dev/null || true

source "${PROJECT_DIR}/.venv/bin/activate"
