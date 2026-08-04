#!/bin/bash
#PBS -b 8
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=5.0.10/intel2023.0.0-cuda12.9.1
#PBS -q gpu
#PBS -A UTSUROLB
#PBS -l elapstim_req=24:00:00
#PBS -j o
#PBS -N grpo_qe

# Nノード(各1 H100) で Ray クラスタを立て、rank0 から WMT21 en-ja word-level QE GRPO を実行する。
# default: `qsub scripts/submit_grpo_qe.sh` (2ノード)
# 任意Nノード: `qsub -b N scripts/submit_grpo_qe.sh`
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

# PROJECT_NAME を ここで export → mpirun -x で全 rank に伝播させる。
# こうしないと sub_grpo.sh が setup_env.sh を source した時点で
# default の "verl_grpo_qwen3_4b_gsm8k" に固定されてしまう。
export PROJECT_NAME=verl_grpo_qwen3_8b_qe_wmt22_ende

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
    -x PATH \
    -x LD_LIBRARY_PATH \
    bash "${PROJECT_DIR}/scripts/sub_grpo.sh"
