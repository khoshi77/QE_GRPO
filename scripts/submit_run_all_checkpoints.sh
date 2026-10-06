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

# bash /work/UTSUROLB/utlb_buma2/work_grpo/scripts/run_all_checkpoints.sh /work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende/0_5919_nqsv all labels
# bash /work/UTSUROLB/utlb_buma2/work_grpo/scripts/run_all_checkpoints2.sh /work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende_xml_mt/0_5923_nqsv generate xml_mt
# bash /work/UTSUROLB/utlb_buma2/work_grpo/scripts/run_all_checkpoints3.sh /work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende/0_5925_nqsv all labels
# bash /work/UTSUROLB/utlb_buma2/work_grpo/scripts/run_all_checkpoints4.sh /work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende_xml_mt/0_5924_nqsv generate xml_mt

# bash /work/UTSUROLB/utlb_buma2/work_grpo/scripts/run_all_checkpoints.sh /work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende/0_16101_nqsv all labels
# bash /work/UTSUROLB/utlb_buma2/work_grpo/scripts/run_all_checkpoints2.sh /work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende/0_16110_nqsv all labels
# bash /work/UTSUROLB/utlb_buma2/work_grpo/scripts/run_all_checkpoints3.sh /work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende/0_16111_nqsv all labels
# bash /work/UTSUROLB/utlb_buma2/work_grpo/scripts/run_all_checkpoints4.sh /work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende_xml_mt/0_16112_nqsv generate xml_mt
# bash /work/UTSUROLB/utlb_buma2/work_grpo/scripts/run_all_checkpoints5.sh /work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende_xml_mt/0_16113_nqsv generate xml_mt
bash /work/UTSUROLB/utlb_buma2/work_grpo/scripts/run_all_checkpoints6.sh /work/UTSUROLB/utlb_buma2/work_grpo/ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende_xml_mt/0_16114_nqsv generate xml_mt

