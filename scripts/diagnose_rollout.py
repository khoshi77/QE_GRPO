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
import json
import os
from pathlib import Path

# offline (GPU ノード対応)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_V1", "1")

import pandas as pd


def generation_record(token_ids, tokenizer, max_new_tokens, finish_reason=None):
    ids = [int(token) for token in token_ids]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    positions = [i for i, token in enumerate(ids) if token == im_end]
    return {
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "raw_text": tokenizer.decode(ids, skip_special_tokens=False),
        "token_ids": ids,
        "generated_tokens": len(ids),
        "reached_length_limit": len(ids) >= max_new_tokens,
        "finish_reason": finish_reason,
        "im_end_positions": positions,
        "tokens_after_first_im_end": len(ids) - positions[0] - 1 if positions else None,
    }


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


def show(label: str, record: dict, meta: dict, output_format: str):
    print(f"--- [{label}] ---")
    print(f"  generated_tokens={record['generated_tokens']}  "
          f"finish_reason={record['finish_reason']}  "
          f"im_end_positions={record['im_end_positions']}  "
          f"tokens_after_first_im_end={record['tokens_after_first_im_end']}")
    if output_format == "xml_mt":
        from qe_xml_utils import first_nonempty_line, xml_to_labels
        _, stats = xml_to_labels(first_nonempty_line(record["text"]), meta["mt"].split())
        print(f"  XML: {stats}")
    else:
        stats = parse_pred(record["text"], meta["num_words"])
        print(f"  labels: {stats}")
    raw_text = record["raw_text"]
    preview = raw_text[:400].replace("\n", "\\n")
    if len(raw_text) > 400:
        preview += " ..."
    print(f"  raw_text: {preview!r}")


def run_hf(model_path: str, prompts_text: list[str], do_sample: bool, temperature: float,
           max_new_tokens: int = 256, top_p: float = 0.9):
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
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)
    in_len = inputs["input_ids"].shape[1]
    records = []
    for row in out_ids[:, in_len:].tolist():
        # Remove post-EOS batch padding, but retain an ignored <|im_end|> so
        # the original Base stopping bug remains visible in diagnostics.
        stopped = tok.eos_token_id in row
        if stopped:
            row = row[:row.index(tok.eos_token_id) + 1]
        records.append(generation_record(row, tok, max_new_tokens, "stop" if stopped else "length"))

    del model
    torch.cuda.empty_cache()
    return records


def run_vllm(model_path: str, prompts_text: list[str], do_sample: bool, temperature: float,
             max_new_tokens: int = 256, top_p: float = 0.9, seed: int = 42):
    from vllm import LLM, SamplingParams

    print(f"\n[vLLM] Loading {model_path}", flush=True)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        enforce_eager=False,
        trust_remote_code=True,
        seed=seed,
    )
    sp_kwargs = dict(max_tokens=max_new_tokens, n=1, seed=seed)
    if do_sample:
        sp_kwargs.update(temperature=temperature, top_p=top_p)
    else:
        sp_kwargs.update(temperature=0.0)
    sp = SamplingParams(**sp_kwargs)
    outs = llm.generate(prompts_text, sp)
    tokenizer = llm.get_tokenizer()
    return [generation_record(o.outputs[0].token_ids, tokenizer, max_new_tokens,
                              o.outputs[0].finish_reason) for o in outs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="SFT model dir (same one passed to GRPO actor_rollout_ref.model.path)")
    parser.add_argument("--parquet",
                        default="/work/UTSUROLB/utlb_buma2/work_grpo/data/qe_wmt21_en_ja/dev.parquet")
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature for rollout-mode comparison")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--format", choices=["labels", "xml_mt"], default="labels")
    parser.add_argument("--output-json", type=Path,
                        help="Save raw token IDs and EOS diagnostics to a new JSON file")
    parser.add_argument("--engines", default="hf_greedy,vllm_greedy,vllm_sample",
                        help="Comma-separated subset of {hf_greedy, vllm_greedy, vllm_sample}")
    args = parser.parse_args()
    if args.output_json is not None and args.output_json.exists():
        parser.error("--output-json must be a new file")
    if args.n_examples < 1 or args.max_new_tokens < 1:
        parser.error("--n-examples and --max-new-tokens must be positive")

    df = pd.read_parquet(args.parquet)
    df = df.head(args.n_examples)
    if df.empty:
        parser.error("parquet is empty")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"EOS: token={tok.eos_token!r} id={tok.eos_token_id}; PAD={tok.pad_token_id}")

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
            "mt": str(row["extra_info"]["mt"]),
        })

    print("=" * 80)
    print("Prompt sample (first example, after apply_chat_template):")
    print("-" * 80)
    print(prompts_text[0])
    print("=" * 80)

    engines = set(s.strip() for s in args.engines.split(",") if s.strip())
    if not engines or engines - {"hf_greedy", "vllm_greedy", "vllm_sample"}:
        parser.error("unknown or empty --engines")

    results: dict[str, list[dict]] = {}

    if "hf_greedy" in engines:
        results["HF greedy"] = run_hf(args.model_path, prompts_text, False, 0.0,
                                      args.max_new_tokens, args.top_p)
    if "vllm_greedy" in engines or "vllm_sample" in engines:
        # Prefer one --engines value per process for isolated GPU smoke tests.
        if "vllm_greedy" in engines:
            results["vLLM greedy"] = run_vllm(args.model_path, prompts_text, False, 0.0,
                                             args.max_new_tokens, args.top_p, args.seed)
        if "vllm_sample" in engines:
            results[f"vLLM sample T={args.temperature}"] = run_vllm(
                args.model_path, prompts_text, True, args.temperature,
                args.max_new_tokens, args.top_p, args.seed
            )

    # Per-example side-by-side
    for i, meta in enumerate(metas):
        print()
        print("=" * 80)
        print(f"Example {i}  (num_words={meta['num_words']})")
        print(f"  gold: {meta['gold']}")
        print("=" * 80)
        for label, outs in results.items():
            show(label, outs[i], meta, args.format)

    if args.output_json is not None:
        with args.output_json.open("x", encoding="utf-8") as handle:
            json.dump({"model_path": args.model_path, "parquet": args.parquet,
                       "format": args.format, "seed": args.seed,
                       "temperature": args.temperature, "top_p": args.top_p,
                       "max_new_tokens": args.max_new_tokens,
                       "examples": metas, "results": results}, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
