#!/bin/bash
#PBS -b 8
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=5.0.10/intel2023.0.0-cuda12.9.1
#PBS -q gpu
#PBS -A UTSUROLB
#PBS -l elapstim_req=24:00:00
#PBS -j o
#PBS -N grpo_qe

# Nノード(各1 H100) で Ray クラスタを立て、rank0 から WMT22 en-de word-level QE GRPO を実行する。
# default: `qsub scripts/submit_grpo_qe.sh` (8ノード)
# 任意Nノード: `qsub -b N scripts/submit_grpo_qe.sh`
# XML MT: `qsub -v QE_FORMAT=xml_mt scripts/submit_grpo_qe.sh`
# Reward: `qsub -v REWARD_METRIC=mcc,REWARD_LENGTH_MISMATCH=pad_bad,REWARD_INVALID_TOKEN=as_bad scripts/submit_grpo_qe.sh`
# Reward: `qsub -v QE_FORMAT=xml_mt,REWARD_METRIC=mcc,REWARD_LENGTH_MISMATCH=pad_bad,REWARD_INVALID_TOKEN=as_bad scripts/submit_grpo_qe.sh`

# Reward: `qsub -v QE_MODEL_PATH=/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/output/Qwen3-8B_labels_5epoch/checkpoint-2750,REWARD_METRIC=token_mix scripts/submit_grpo_qe.sh`
# Reward: `qsub -v QE_MODEL_PATH=/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/output/Qwen3-8B_xml_mt_5epoch/checkpoint-4000,QE_FORMAT=xml_mt,REWARD_METRIC=token_mix scripts/submit_grpo_qe.sh`
set -euo pipefail

USER_NAME=$(whoami)
PROJECT_DIR=/work/UTSUROLB/${USER_NAME}/work_grpo

# qsub stdout は終了までバッファされるため、共有FSにもリアルタイム出力を残す。
_RAW_JOB_ID="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
JOB_ID="${_RAW_JOB_ID//[:.\/]/_}"
mkdir -p "${PROJECT_DIR}/logs/jobs"
JOB_LOG="${PROJECT_DIR}/logs/jobs/${JOB_ID}.log"
exec > >(tee -a "${JOB_LOG}") 2>&1
echo "[$(date '+%F %T')] job log: ${JOB_LOG}  (JOB_ID=${JOB_ID})"

# rank0 ノード自身の IB IP を head IP として使う
export HEAD_IP=$(ip a show dev ibp106s0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export HEAD_PORT=$((RANDOM % 10000 + 30000))

# sub_grpo.sh が起動するスクリプトを QE 用に切り替え
export RUN_SCRIPT=run_grpo_qe.sh

QE_FORMAT="${QE_FORMAT:-labels}"
case "${QE_FORMAT}" in
    labels) DEFAULT_PROJECT_NAME=verl_grpo_qwen3_8b_qe_wmt22_ende ;;
    xml_mt) DEFAULT_PROJECT_NAME=verl_grpo_qwen3_8b_qe_wmt22_ende_xml_mt ;;
    *) echo "ERROR: QE_FORMAT must be labels or xml_mt: ${QE_FORMAT}" >&2; exit 2 ;;
esac
export QE_FORMAT

# PROJECT_NAME を ここで export → mpirun -x で全 rank に伝播させる。
# こうしないと sub_grpo.sh が setup_env.sh を source した時点で
# default の "verl_grpo_qwen3_4b_gsm8k" に固定されてしまう。
export PROJECT_NAME="${PROJECT_NAME:-${DEFAULT_PROJECT_NAME}}"

# qsub -v で任意指定できる上書き値。空なら run_grpo_qe.sh の既定値を使う。
export QE_MODEL_PATH="${QE_MODEL_PATH:-}"
export QE_MAX_RESPONSE_LEN="${QE_MAX_RESPONSE_LEN:-}"
export QE_TOTAL_STEPS="${QE_TOTAL_STEPS:-}"
export QE_TOTAL_EPOCHS="${QE_TOTAL_EPOCHS:-}"

# Defaults are included in the submitted script, then forwarded to every rank.
# This keeps these settings fixed even if run_grpo_qe.sh defaults change in QUE.
export REWARD_METRIC="${REWARD_METRIC:-f1_macro}"
export REWARD_LENGTH_MISMATCH="${REWARD_LENGTH_MISMATCH:-pad_bad}"
export REWARD_INVALID_TOKEN="${REWARD_INVALID_TOKEN:-as_bad}"

NQSV_MPI_VER="${NQSV_MPI_VER:-5.0.10/intel2023.0.0-cuda12.9.1}"
module load openmpi/${NQSV_MPI_VER}

cd "${PROJECT_DIR}"

# ノード数は PBS_NODEFILE から動的に決定 (`#PBS -b N` または `qsub -b N` で変更)
export NNODES=$(wc -l < "${PBS_NODEFILE}")
echo "[$(date '+%F %T')] NNODES=${NNODES}  RUN_SCRIPT=${RUN_SCRIPT}"

mpirun -hostfile "${PBS_NODEFILE}" -np "${NNODES}" \
    -npernode 1 \
    -x HEAD_IP \
    -x HEAD_PORT \
    -x NNODES \
    -x RUN_SCRIPT \
    -x PROJECT_NAME \
    -x QE_FORMAT \
    -x QE_MODEL_PATH \
    -x QE_MAX_RESPONSE_LEN \
    -x QE_TOTAL_STEPS \
    -x QE_TOTAL_EPOCHS \
    -x REWARD_METRIC \
    -x REWARD_LENGTH_MISMATCH \
    -x REWARD_INVALID_TOKEN \
    -x PATH \
    -x LD_LIBRARY_PATH \
    bash "${PROJECT_DIR}/scripts/sub_grpo.sh"
