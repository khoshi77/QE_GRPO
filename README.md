# QE_GRPO

WMT22 English–German word-level QE supports two response formats:

- `labels` (default): `OK BAD OK ...`
- `xml_mt`: copy the MT and wrap BAD spans in `<e> ... </e>`

`xml_src_mt` is not supported.

## Prepare `xml_mt` data

```bash
QE_FORMAT=xml_mt bash scripts/preprocess_qe.sh

python scripts/validate_qe_parquet.py \
  --format xml_mt \
  data/qe_wmt22_en_de_xml_mt/train.parquet \
  data/qe_wmt22_en_de_xml_mt/dev.parquet \
  data/qe_wmt22_en_de_xml_mt/test.parquet
```

The XML parquet files use the same prompt as
`work_SFT/QE_SFT_8B` with `data.format: xml_mt`. They are stored separately
from the existing labels parquet files.

## Submit GRPO

```bash
qsub -v QE_FORMAT=xml_mt scripts/submit_grpo_qe.sh
```

The default XML SFT model is the dev-selected checkpoint:

```text
/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/output/Qwen3-8B_xml_mt_5epoch/checkpoint-2000
```

Optional overrides can be passed with `qsub -v`, for example:

```bash
qsub -v QE_FORMAT=xml_mt,QE_TOTAL_STEPS=20 scripts/submit_grpo_qe.sh
qsub -v QE_FORMAT=xml_mt,QE_MODEL_PATH=/path/to/checkpoint scripts/submit_grpo_qe.sh
```

Reward settings can also be passed at submission time:

```bash
qsub -v QE_FORMAT=labels,REWARD_METRIC=f1_product,REWARD_LENGTH_MISMATCH=zero,REWARD_INVALID_TOKEN=zero scripts/submit_grpo_qe.sh
qsub -v QE_FORMAT=xml_mt,REWARD_METRIC=f1_macro scripts/submit_grpo_qe.sh
```

| Variable | Default if omitted or empty | Accepted values |
|---|---|---|
| `REWARD_METRIC` | `f1_macro` | `token_mix`, `weighted_token_accuracy`, `bad_f1_safe`, `mcc`, `f1_bad`, `f1_ok`, `f1_product`, `f1_macro` |
| `REWARD_LENGTH_MISMATCH` | `pad_bad` | `pad_bad`, `penalize`, `zero` |
| `REWARD_INVALID_TOKEN` | `as_bad` | `as_bad`, `penalize`, `zero` |

The submitted script resolves defaults and forwards all three variables through
MPI to the training script. Their effective values are printed in the job log
and passed to Hydra's `custom_reward_function.reward_kwargs`. The two error
policies apply to `labels`; `xml_mt` uses its XML copy/balance penalties instead.
This protects these parameter values for newly submitted jobs, but does not
snapshot the shared Python/shell code or retrofit already-queued jobs.

The default XML response limit is 448 tokenizer tokens. The measured WMT22
maximum is 413; prompts have a maximum of 681 against a limit of 768.

## Qwen3 Base EOS alignment

Before GRPO, `run_grpo_qe.sh` validates the SFT assistant terminator against
the tokenizer, model config, and generation config. If they disagree, it
creates `eos_prepare.*/model` inside the new run directory. This model view
uses `<|im_end|>` as primary EOS, retains existing alternative EOS IDs in
`generation_config.json`, and links the original weights. The original
checkpoint and PAD token are unchanged. Already-aligned chat models are
used directly. The prepared path is recorded in the run configuration;
new exported checkpoints inherit the corrected metadata.

Inspect any Qwen3 QE checkpoint without loading its weights:

```bash
python scripts/prepare_qe_eos.py --check-only --model-path /path/to/checkpoint
```

For standalone inference with an existing SFT or GRPO checkpoint, prepare
a separate model view and pass the returned path to the inference program:

```bash
python scripts/prepare_qe_eos.py \
  --model-path /path/to/checkpoint \
  --output-dir /path/to/new_eos_model
```

The output directory must be new. Keep the original checkpoint available
because the prepared directory links its weights. Save trained weights to
the usual new checkpoint directory, not into this input model view.
See [eos_fix_notes.md](eos_fix_notes.md) for evidence, GPU smoke commands,
and the distinction between completed CPU tests and pending 8B validation.

## Evaluate XML checkpoints

XML uses free generation. Labels-only constrained decoding is intentionally
rejected.

```bash
bash scripts/run_all_checkpoints.sh /path/to/grpo/run generate xml_mt
```

The resulting `predictions.tsv` contains `pred_labels`, so
`scripts/significance_test.py` can be used unchanged.

## Tests

```bash
python -m unittest discover -s tests -v
bash -n scripts/preprocess_qe.sh scripts/run_grpo_qe.sh \
  scripts/submit_grpo_qe.sh scripts/run_all_checkpoints.sh
```
