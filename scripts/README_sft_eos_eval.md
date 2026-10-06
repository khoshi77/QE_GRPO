# Base SFT EOS reevaluation

This reevaluates the existing `checkpoint-3250` for each of
`Qwen3-8B-Base_labels_5epoch` and `Qwen3-8B-Base_xml_mt_5epoch` on WMT22 en–de
dev and test. No training or checkpoint reselection is performed.

The prepared model directories symlink the original weights. Their model and
tokenizer EOS is `<|im_end|>` (151645), while PAD remains 151643. The saved
generation configuration retains both EOS alternatives, as in GRPO; the actual
HF inference call explicitly uses **only 151645**, as GRPO's evaluator does.
The textual `<EOS>` position in the MT data remains part of evaluation.

Inference and parsing use archived, unchanged copies of the SFT scripts also
called by `run_all_checkpoints.sh`, and the SFT Python environment. Settings:
greedy free generation, batch size 64, thinking disabled, left padding,
prompt truncation at 1024, and maximum new tokens 256 (labels) or 448 (xml_mt).
These are the offline evaluation settings, not the sampled GRPO training rollout.
Shared YAML, original model metadata, and old result files are never modified.

Prepare a fresh output directory on the login node:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/.venv/bin/python \
  scripts/reevaluate_sft_qe_eos.py \
  --run-dir /work/UTSUROLB/utlb_buma2/work_grpo/analysis/sft_eos_RUN_NAME \
  --prepare-only
```

Then submit one GPU job:

```bash
qsub -v QE_EVAL_RUN_DIR=/work/UTSUROLB/utlb_buma2/work_grpo/analysis/sft_eos_RUN_NAME \
  -o /work/UTSUROLB/utlb_buma2/work_grpo/analysis/sft_eos_RUN_NAME/job.log \
  scripts/submit_sft_qe_eos_eval.sh
```

Each `<format>/<split>/` contains the resolved YAML, `predictions.tsv`, the
original evaluator's `result.txt`, and full precision `metrics.json`. The latter
includes MCC, BAD/OK F1, macro F1, F1 product, BAD precision/recall, BAD ratios,
confusion counts, and raw output format statistics. Strict checks reject
sentence or label-length mismatches rather than silently truncating them.
`manifest.json` records source metadata hashes, evaluator hashes, dataset hashes,
package versions, prompt lengths, and old results. `runtime.json` records the
GPU and PBS job. `summary.md` and `summary.json` compare all four evaluations
with their original results; completion is recorded in `status.json`.

There is no notification or webhook in this job. Existing paper/audit artifacts
are not automatically replaced; the revised SFT–GRPO comparison must use the
new baseline results explicitly.
