#!/bin/bash
#PBS -b 1
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=5.0.10/intel2023.0.0-cuda12.9.1
#PBS -q gpu
#PBS -A UTSUROLB
#PBS -l elapstim_req=04:00:00
#PBS -j o
#PBS -N sft_eos
set -euo pipefail

# Prepare first with reevaluate_sft_qe_eos.py --prepare-only, then submit:
# qsub -v QE_EVAL_RUN_DIR=/absolute/prepared/run -o /absolute/prepared/run/job.log scripts/submit_sft_qe_eos_eval.sh
: "${QE_EVAL_RUN_DIR:?Set QE_EVAL_RUN_DIR to a prepared evaluation directory}"
PYTHON=/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/.venv/bin/python
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${QE_EVAL_RUN_DIR}/cache/huggingface"
export HF_DATASETS_CACHE="${QE_EVAL_RUN_DIR}/cache/datasets"
module load cuda/12.6.3 2>/dev/null || true
cd "${QE_EVAL_RUN_DIR}"
"${PYTHON}" "${QE_EVAL_RUN_DIR}/code/reevaluate_sft_qe_eos.py" --run-dir "${QE_EVAL_RUN_DIR}" --execute
