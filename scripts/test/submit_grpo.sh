#!/bin/bash
#PBS -b 2
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=5.0.10/intel2023.0.0-cuda12.9.1
#PBS -q gpu
#PBS -A UTSUROLB
#PBS -l elapstim_req=12:00:00
#PBS -j o
#PBS -N grpo_qwen3_4b

# Nノード(各1 H100) で Ray クラスタを立て、rank0 から GRPO を実行する。
# デフォルト: `qsub scripts/submit_grpo.sh` (2ノード)
# 任意Nノード:  `qsub -b N scripts/submit_grpo.sh`
set -euo pipefail

USER_NAME=$(whoami)
PROJECT_DIR=/work/UTSUROLB/${USER_NAME}/work_grpo

# qsub stdout は終了までバッファされるため、共有FSにもリアルタイム出力を残す。
# JOB_ID は他スクリプトと揃えるため setup_env.sh と同じサニタイズを適用。
_RAW_JOB_ID="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
JOB_ID="${_RAW_JOB_ID//[:.\/]/_}"
mkdir -p "${PROJECT_DIR}/logs/jobs"
JOB_LOG="${PROJECT_DIR}/logs/jobs/${JOB_ID}.log"
exec > >(tee -a "${JOB_LOG}") 2>&1
echo "[$(date '+%F %T')] job log: ${JOB_LOG}  (JOB_ID=${JOB_ID})"

# rank0 ノード自身の IB IP を head IP として使う（SFT スクリプトと同じ I/F）
export HEAD_IP=$(ip a show dev ibp106s0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export HEAD_PORT=$((RANDOM % 10000 + 30000))

module load openmpi/${NQSV_MPI_VER}

cd "${PROJECT_DIR}"

# ノード数は PBS_NODEFILE から動的に決定（`#PBS -b N` または `qsub -b N` で変更）
export NNODES=$(wc -l < "${PBS_NODEFILE}")
echo "[$(date '+%F %T')] NNODES=${NNODES}"

mpirun -hostfile "${PBS_NODEFILE}" -np "${NNODES}" \
    -npernode 1 \
    -x HEAD_IP \
    -x HEAD_PORT \
    -x NNODES \
    -x PATH \
    -x LD_LIBRARY_PATH \
    bash "${PROJECT_DIR}/scripts/sub_grpo.sh"
