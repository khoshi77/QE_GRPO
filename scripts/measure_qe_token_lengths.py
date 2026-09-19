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
        --response-thresholds 256 384 448 512
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
    prompt_lens, resp_lens, word_counts = [], [], []
    parquet = pq.ParquetFile(str(parquet_path))
    # 大きい train parquet を pandas に全展開せず、一定量ずつ処理する。
    for batch in parquet.iter_batches(
        batch_size=256,
        columns=["prompt", "reward_model", "extra_info"],
    ):
        rows = batch.to_pylist()
        prompt_texts = [
            tok.apply_chat_template(
                row["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in rows
        ]
        ground_truths = [row["reward_model"]["ground_truth"] for row in rows]
        prompt_lens.extend(
            tok(
                prompt_texts,
                add_special_tokens=False,
                return_length=True,
            )["length"]
        )
        resp_lens.extend(
            length + 1
            for length in tok(
                ground_truths,
                add_special_tokens=False,
                return_length=True,
            )["length"]
        )
        word_counts.extend(int(row["extra_info"]["num_words"]) for row in rows)

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
                        default=[64, 96, 128, 256, 384, 448, 512])
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
