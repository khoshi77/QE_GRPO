"""
QE 用 parquet データの prompt / response トークン長分布を測る。

run_grpo_qe.sh の MAX_PROMPT_LEN / MAX_RESPONSE_LEN を決めるための事前調査。
chat template は学習時と同じ enable_thinking=False, add_generation_prompt=True で適用する。

使い方:
    python scripts/measure_qe_token_lengths.py
    python scripts/measure_qe_token_lengths.py \
        --model /work/UTSUROLB/utlb_buma2/work_grpo/sft_ckpts/Qwen3-4B-Base-SFT/checkpoint-500 \
        --data-dir /work/UTSUROLB/utlb_buma2/work_grpo/data/qe_wmt21_en_ja \
        --splits train dev test \
        --prompt-thresholds 768 1024 \
        --response-thresholds 256 384
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


DEFAULT_MODEL = "/work/UTSUROLB/utlb_buma2/work_grpo/sft_ckpts/Qwen3-4B-Base-SFT/checkpoint-500"
DEFAULT_DATA_DIR = "/work/UTSUROLB/utlb_buma2/work_grpo/data/qe_wmt21_en_ja"


def measure(parquet_path: Path, tok) -> dict[str, np.ndarray]:
    table = pq.read_table(str(parquet_path)).to_pandas()
    prompt_lens, resp_lens, word_counts = [], [], []
    for _, row in table.iterrows():
        msgs = list(row["prompt"])
        text = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_lens.append(len(tok(text, add_special_tokens=False).input_ids))

        gt = row["reward_model"]["ground_truth"]
        # gold ラベル列 + EOS 1 トークン分を response 長と見なす
        resp_lens.append(len(tok(gt, add_special_tokens=False).input_ids) + 1)
        word_counts.append(int(row["extra_info"]["num_words"]))

    return {
        "prompt": np.asarray(prompt_lens),
        "response": np.asarray(resp_lens),
        "words": np.asarray(word_counts),
    }


def print_stats(name: str, arr: np.ndarray) -> None:
    qs = [50, 90, 95, 99, 100]
    pct = " ".join(f"{int(np.percentile(arr, q)):5d}" for q in qs)
    print(f"{name:<10} {arr.min():5d} {arr.mean():6.1f} {pct}")


def print_overflow(name: str, arr: np.ndarray, thresholds: list[int]) -> None:
    n = len(arr)
    for th in thresholds:
        over = int((arr > th).sum())
        print(f"  {name} > {th:>4}: {over:4d} / {n}  ({100 * over / n:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Tokenizer path (HF model dir)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing {train,dev,test}.parquet")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--prompt-thresholds", nargs="+", type=int,
                        default=[256, 384, 512, 768, 1024])
    parser.add_argument("--response-thresholds", nargs="+", type=int,
                        default=[64, 96, 128, 256])
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"tokenizer: {tok.__class__.__name__}  vocab={tok.vocab_size}")
    print(f"model:     {args.model}")
    print(f"data_dir:  {args.data_dir}")

    data_dir = Path(args.data_dir)
    for split in args.splits:
        path = data_dir / f"{split}.parquet"
        if not path.exists():
            print(f"\n[skip] {path} not found")
            continue
        stats = measure(path, tok)
        n = len(stats["prompt"])
        print(f"\n=== {split} (n={n}) ===")
        print(f"{'':<10} {'min':>5} {'mean':>6} {'p50':>5} {'p90':>5} {'p95':>5} {'p99':>5} {'max':>5}")
        for key in ("prompt", "response", "words"):
            print_stats(key, stats[key])
        print_overflow("prompt  ", stats["prompt"], args.prompt_thresholds)
        print_overflow("response", stats["response"], args.response_thresholds)


if __name__ == "__main__":
    main()
