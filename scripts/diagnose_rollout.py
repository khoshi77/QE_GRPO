"""
verl rollout で SFT モデルが大量の invalid token (非 OK/BAD) を吐く現象の切り分け。

学習を回さずに以下を比較する:
  (A) SFT 推論パイプライン相当: HF transformers + greedy
  (B) verl 検証相当            : vLLM + greedy
  (C) verl 学習ロールアウト相当: vLLM + sampling (temperature)

dev.parquet の prompt をそのまま使うので、verl が読むのと完全に同じ入力。
各出力について生テキスト・OK 数・BAD 数・invalid 数・長さを表示する。

Usage:
    python scripts/diagnose_rollout.py \\
        --model-path /work/UTSUROLB/utlb_buma2/work_grpo/sft_ckpts/Qwen3-4B-SFT/checkpoint-450 \\
        --n-examples 3
"""

import argparse
import os
from pathlib import Path

# offline (GPU ノード対応)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_V1", "1")

import pandas as pd


def parse_pred(text: str, num_words: int):
    raw = text.strip().split()
    n_ok = n_bad = n_invalid = 0
    for tok in raw:
        cleaned = tok.upper().rstrip(".,;:")
        if cleaned == "OK":
            n_ok += 1
        elif cleaned == "BAD":
            n_bad += 1
        else:
            n_invalid += 1
    return {
        "n_raw_tokens": len(raw),
        "n_ok": n_ok,
        "n_bad": n_bad,
        "n_invalid": n_invalid,
        "expected_words": num_words,
        "length_mismatch": len(raw) != num_words,
    }


def show(label: str, raw_text: str, num_words: int):
    stats = parse_pred(raw_text, num_words)
    print(f"--- [{label}] ---")
    print(f"  raw_tokens={stats['n_raw_tokens']:4d}  expected={stats['expected_words']:4d}  "
          f"OK={stats['n_ok']:3d}  BAD={stats['n_bad']:3d}  invalid={stats['n_invalid']:3d}  "
          f"len_mismatch={stats['length_mismatch']}")
    preview = raw_text[:400].replace("\n", "\\n")
    if len(raw_text) > 400:
        preview += " ..."
    print(f"  raw_text: {preview!r}")


def run_hf(model_path: str, prompts_text: list[str], do_sample: bool, temperature: float):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[HF] Loading {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model.eval()

    inputs = tok(prompts_text, return_tensors="pt", padding=True, truncation=True,
                 max_length=1024).to(next(model.parameters()).device)
    gen_kwargs = dict(
        max_new_tokens=256,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)
    in_len = inputs["input_ids"].shape[1]
    decoded = tok.batch_decode(out_ids[:, in_len:], skip_special_tokens=True)

    del model
    torch.cuda.empty_cache()
    return decoded


def run_vllm(model_path: str, prompts_text: list[str], do_sample: bool, temperature: float):
    from vllm import LLM, SamplingParams

    print(f"\n[vLLM] Loading {model_path}", flush=True)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        enforce_eager=False,
        trust_remote_code=True,
    )
    sp_kwargs = dict(max_tokens=256, n=1)
    if do_sample:
        sp_kwargs.update(temperature=temperature, top_p=1.0)
    else:
        sp_kwargs.update(temperature=0.0)
    sp = SamplingParams(**sp_kwargs)
    outs = llm.generate(prompts_text, sp)
    return [o.outputs[0].text for o in outs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="SFT model dir (same one passed to GRPO actor_rollout_ref.model.path)")
    parser.add_argument("--parquet",
                        default="/work/UTSUROLB/utlb_buma2/work_grpo/data/qe_wmt21_en_ja/dev.parquet")
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for rollout-mode comparison")
    parser.add_argument("--engines", default="hf_greedy,vllm_greedy,vllm_sample",
                        help="Comma-separated subset of {hf_greedy, vllm_greedy, vllm_sample}")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df.head(args.n_examples)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompts_text = []
    metas = []
    for _, row in df.iterrows():
        messages = list(row["prompt"])
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts_text.append(text)
        metas.append({
            "num_words": int(row["extra_info"]["num_words"]),
            "gold": row["reward_model"]["ground_truth"],
        })

    print("=" * 80)
    print("Prompt sample (first example, after apply_chat_template):")
    print("-" * 80)
    print(prompts_text[0])
    print("=" * 80)

    engines = set(s.strip() for s in args.engines.split(",") if s.strip())

    results: dict[str, list[str]] = {}

    if "hf_greedy" in engines:
        results["HF greedy"] = run_hf(args.model_path, prompts_text, False, 0.0)
    if "vllm_greedy" in engines or "vllm_sample" in engines:
        # one vLLM load for both greedy and sampling
        if "vllm_greedy" in engines:
            results["vLLM greedy"] = run_vllm(args.model_path, prompts_text, False, 0.0)
        if "vllm_sample" in engines:
            results[f"vLLM sample T={args.temperature}"] = run_vllm(
                args.model_path, prompts_text, True, args.temperature
            )

    # Per-example side-by-side
    for i, meta in enumerate(metas):
        print()
        print("=" * 80)
        print(f"Example {i}  (num_words={meta['num_words']})")
        print(f"  gold: {meta['gold']}")
        print("=" * 80)
        for label, outs in results.items():
            show(label, outs[i], meta["num_words"])


if __name__ == "__main__":
    main()
