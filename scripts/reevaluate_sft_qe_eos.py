"""Reevaluate the two selected Base SFT checkpoints using GRPO's HF evaluator.

Prepare on the login node, then execute on one GPU. Original checkpoints and
shared SFT YAML are read-only. Each run archives its evaluator, data, and settings.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from datetime import datetime, timezone


SFT_ROOT = Path("/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B")
PACKAGES = ("torch", "transformers", "peft", "accelerate", "datasets", "pandas", "scikit-learn", "PyYAML")
FORMATS = {"labels": 256, "xml_mt": 448}
EOS = 151645
BATCH_SIZE = 64


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def native_modules(run_dir):
    sys.path.insert(0, str(run_dir / "code"))
    import inference
    import evaluate
    import data_utils
    return inference, evaluate, data_utils


def checked_metrics(evaluator, data_utils, gold_df, pred_df, fmt):
    """Reject any row/label mismatch before calling the existing evaluator."""
    if len(gold_df) != len(pred_df):
        raise ValueError("gold/prediction row counts differ")
    gold, pred = [], []
    for row, (g, p) in enumerate(zip(gold_df.to_dict("records"), pred_df.to_dict("records"))):
        if (g["src"], g["mt"]) != (p["src"], p["mt"]):
            raise ValueError(f"source/translation differ at row {row}")
        gl, pl = g["labels"].split(), p["pred_labels"].split()
        if not gl or set(gl + pl) - {"OK", "BAD"}:
            raise ValueError(f"invalid label at row {row}")
        if len(gl) != len(pl) or len(gl) != data_utils.count_words(g["mt"]):
            raise ValueError(f"label length mismatch at row {row}")
        gold.append([int(x == "BAD") for x in gl])
        pred.append([int(x == "BAD") for x in pl])
    report = evaluator.evaluate(gold, pred)
    gf, pf = report["gold_flat"], report["pred_flat"]
    tp = sum(g == 1 and p == 1 for g, p in zip(gf, pf))
    fp = sum(g == 0 and p == 1 for g, p in zip(gf, pf))
    fn = sum(g == 1 and p == 0 for g, p in zip(gf, pf))
    tn = len(gf) - tp - fp - fn
    metrics = {key: float(value) for key, value in report.items() if key not in ("gold_flat", "pred_flat")}
    metrics.update({
        "precision_bad": tp / (tp + fp) if tp + fp else 0.0,
        "recall_bad": tp / (tp + fn) if tp + fn else 0.0,
        "predicted_bad_ratio": (tp + fp) / len(gf),
        "gold_bad_ratio": (tp + fn) / len(gf),
        "n_sentences": len(gold), "n_tokens": len(gf),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "format_stats": evaluator.compute_format_stats(pred_df, fmt),
    })
    return metrics, report


def prepare(run_dir, sft_root):
    import yaml
    from transformers import AutoTokenizer
    from prepare_qe_eos import prepare_model

    if run_dir.exists():
        raise FileExistsError(f"use a new run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    code_dir = run_dir / "code"
    code_dir.mkdir()
    for name in ("inference.py", "evaluate.py", "data_utils.py"):
        shutil.copy2(sft_root / "src/models" / name, code_dir / name)
    for name in (Path(__file__).name, "prepare_qe_eos.py"):
        shutil.copy2(Path(__file__).parent / name, code_dir / name)
    shutil.copy2(sft_root / "src/config.yaml", run_dir / "original_sft_config.yaml")
    cfg = yaml.safe_load((run_dir / "original_sft_config.yaml").read_text())
    if cfg["inference"].get("batch_size") != BATCH_SIZE:
        raise ValueError("shared GRPO inference batch size changed; inspect before proceeding")
    manifest = {
        "created_at": now(), "sft_root": str(sft_root),
        "python_executable": sys.executable, "python_version": sys.version,
        "package_versions": versions(),
        "settings": {"batch_size": BATCH_SIZE, "decoding": "generate", "do_sample": False,
                     "enable_thinking": False, "prompt_max_length": 1024,
                     "padding_side": "left", "eos_token_id": EOS, "pad_token_id": 151643,
                     "temperature": None, "top_p": None, "top_k": None,
                     "max_new_tokens": FORMATS, "checkpoint_selection": "fixed original checkpoint-3250; no reselection"},
        "code_sha256": {p.name: sha256(p) for p in code_dir.iterdir()},
        "datasets": {}, "models": {}, "original_results": {},
    }
    (run_dir / "data").mkdir()
    for split in ("dev", "test"):
        original = Path(cfg["data"]["eval_file" if split == "dev" else "test_file"])
        if not original.is_absolute():
            original = sft_root / original
        copied = run_dir / "data" / f"{split}.jsonl"
        shutil.copy2(original, copied)
        rows = [json.loads(line) for line in copied.read_text().splitlines() if line.strip()]
        for i, row in enumerate(rows):
            labels = row["labels"].split()
            if not labels or set(labels) - {"OK", "BAD"} or len(labels) != len(row["mt"].split()):
                raise ValueError(f"invalid gold labels in {split} row {i}")
        manifest["datasets"][split] = {
            "original": str(original), "snapshot": str(copied), "sha256": sha256(copied),
            "n_sentences": len(rows), "n_tokens": sum(len(r["labels"].split()) for r in rows),
            "n_bad": sum(r["labels"].split().count("BAD") for r in rows),
        }
    native, evaluator, utils = native_modules(run_dir)
    for fmt, limit in FORMATS.items():
        source = sft_root / "output" / f"Qwen3-8B-Base_{fmt}_5epoch/checkpoint-3250"
        model_view = prepare_model(source, run_dir / "models" / fmt)
        tokenizer = AutoTokenizer.from_pretrained(model_view, local_files_only=True)
        if tokenizer.eos_token_id != EOS or tokenizer.pad_token_id != 151643:
            raise ValueError("prepared EOS/PAD mismatch")
        weights = list(source.glob("*.safetensors"))
        if not weights or any((model_view / p.name).resolve() != p.resolve() for p in weights):
            raise ValueError("model view does not reference original weights")
        metadata = ("config.json", "tokenizer_config.json", "generation_config.json")
        manifest["models"][fmt] = {
            "original": str(source), "prepared": str(model_view),
            "original_metadata_sha256": {name: sha256(source / name) for name in metadata},
            "prepared_metadata_sha256": {name: sha256(model_view / name) for name in metadata},
            "weights": [{"path": str(p), "size_bytes": p.stat().st_size} for p in weights],
            "prompt_stats": {},
        }
        for split in ("dev", "test"):
            dataset = utils.load_dataset_file(manifest["datasets"][split]["snapshot"])
            prompts = [tokenizer.apply_chat_template(
                utils.build_messages_any(fmt, row["src"], row["mt"], labels=None),
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) for row in dataset]
            lengths = [len(ids) for ids in tokenizer(prompts, truncation=False)["input_ids"]]
            manifest["models"][fmt]["prompt_stats"][split] = {
                "max_tokens": max(lengths), "n_truncated_at_1024": sum(n > 1024 for n in lengths),
            }
            result_dir = run_dir / fmt / split
            result_dir.mkdir(parents=True)
            resolved = {
                "model": {"name_or_path": cfg["model"]["name_or_path"]},
                "data": {"format": fmt},
                "inference": {"model_path": str(model_view), "batch_size": BATCH_SIZE,
                              "max_new_tokens": limit, "decoding": "generate",
                              "input_file": manifest["datasets"][split]["snapshot"],
                              "output_file": str(result_dir / "predictions.tsv")},
                "evaluate": {"gold_file": manifest["datasets"][split]["snapshot"],
                             "pred_file": str(result_dir / "predictions.tsv"),
                             "gold_col": "labels", "pred_col": "pred_labels",
                             "result_file": str(result_dir / "result.txt")},
            }
            (result_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
            old_path = source / ("result_dev" if split == "dev" else "result") / "predictions.tsv"
            if old_path.exists():
                old_dir = run_dir / "original_results" / fmt / split
                old_dir.mkdir(parents=True)
                shutil.copy2(old_path, old_dir / old_path.name)
                if (old_path.parent / "result.txt").exists():
                    shutil.copy2(old_path.parent / "result.txt", old_dir / "result.txt")
                old_metrics, _ = checked_metrics(evaluator, utils, dataset.to_pandas(), evaluator.load_file(str(old_dir / old_path.name)), fmt)
                manifest["original_results"][f"{fmt}/{split}"] = {
                    "original": str(old_path), "sha256": sha256(old_dir / old_path.name), "metrics": old_metrics,
                }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "status.json", {"phase": "prepared", "updated_at": now()})
    print(json.dumps({"run_dir": str(run_dir), "datasets": manifest["datasets"], "settings": manifest["settings"]}, indent=2))


def execute(run_dir):
    import torch

    manifest = json.loads((run_dir / "manifest.json").read_text())
    if versions() != manifest["package_versions"]:
        raise RuntimeError("Python package versions changed after preparation")
    for name, expected in manifest["code_sha256"].items():
        if sha256(run_dir / "code" / name) != expected:
            raise RuntimeError(f"archived evaluator changed: {name}")
    for info in manifest["datasets"].values():
        if sha256(info["snapshot"]) != info["sha256"]:
            raise RuntimeError("archived gold data changed")
    for info in manifest["models"].values():
        for key, root in (("original_metadata_sha256", info["original"]), ("prepared_metadata_sha256", info["prepared"])):
            for name, expected in info[key].items():
                if sha256(Path(root) / name) != expected:
                    raise RuntimeError(f"checkpoint metadata changed: {root}/{name}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution required; submit the prepared run to the gpu queue")
    write_json(run_dir / "runtime.json", {
        "started_at": now(), "hostname": socket.gethostname(), "pbs_job_id": os.environ.get("PBS_JOBID"),
        "python_executable": sys.executable, "python_version": sys.version, "package_versions": versions(),
        "torch_cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    })
    native, evaluator, utils = native_modules(run_dir)
    summary = {}
    for fmt, limit in FORMATS.items():
        if any((run_dir / fmt / split / "predictions.tsv").exists() for split in ("dev", "test")):
            raise FileExistsError("predictions already exist; use a fresh run to avoid overwriting results")
        model, tokenizer = native.load_model_and_tokenizer(manifest["models"][fmt]["prepared"])
        if tokenizer.eos_token_id != EOS or model.config.eos_token_id != EOS or tokenizer.pad_token_id != 151643:
            raise ValueError("loaded model EOS/PAD mismatch")
        for split in ("dev", "test"):
            write_json(run_dir / "status.json", {"phase": "running", "format": fmt, "split": split, "updated_at": now()})
            print(f"\nEvaluating {fmt}/{split}: EOS={tokenizer.eos_token_id}, batch={BATCH_SIZE}, max_new_tokens={limit}", flush=True)
            dataset = utils.load_dataset_file(manifest["datasets"][split]["snapshot"])
            labels, raw, _ = native.run_inference(model, tokenizer, dataset, BATCH_SIZE, limit, fmt=fmt)
            gold_df = dataset.to_pandas()
            pred_df = gold_df.copy()
            pred_df["pred_labels"] = [" ".join(row) for row in labels]
            pred_df["raw_output"] = [utils.escape_raw_output(text) for text in raw]
            metrics, report = checked_metrics(evaluator, utils, gold_df, pred_df, fmt)
            result_dir = run_dir / fmt / split
            pred_df.to_csv(result_dir / "predictions.tsv", sep="\t", index=False)
            # Validate the serialized predictions as consumed by the original CLI.
            serialized, _ = checked_metrics(evaluator, utils, gold_df, evaluator.load_file(str(result_dir / "predictions.tsv")), fmt)
            if serialized != metrics:
                raise RuntimeError("TSV round-trip changed evaluation")
            with (result_dir / "result.txt").open("w", encoding="utf-8") as output:
                evaluator.print_report(report, file=output, format_stats=metrics["format_stats"])
            write_json(result_dir / "metrics.json", metrics)
            old = manifest["original_results"].get(f"{fmt}/{split}")
            summary[f"{fmt}/{split}"] = {
                "result_dir": str(result_dir), "metrics": metrics,
                "original_metrics": old["metrics"] if old else None,
                "delta_mcc": metrics["mcc"] - old["metrics"]["mcc"] if old else None,
            }
            write_json(run_dir / "summary.json", summary)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    lines = ["# Base SFT reevaluation (EOS 151645)", "", "Greedy generation; batch 64; fixed checkpoint-3250 for each format.", "",
             "| Format | Split | Original MCC | New MCC | Delta | F1-BAD | P-BAD | R-BAD | Pred. BAD % | Gold BAD % |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, result in summary.items():
        fmt, split = key.split("/")
        m, old = result["metrics"], result["original_metrics"]
        original = f"{old['mcc']:.4f}" if old else "n/a"
        delta = f"{result['delta_mcc']:+.4f}" if old else "n/a"
        lines.append(f"| {fmt} | {split} | {original} | {m['mcc']:.4f} | {delta} | {m['f1_bad']:.4f} | {m['precision_bad']:.4f} | {m['recall_bad']:.4f} | {100*m['predicted_bad_ratio']:.2f} | {100*m['gold_bad_ratio']:.2f} |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(run_dir / "status.json", {"phase": "complete", "updated_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, default=SFT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.prepare_only:
        prepare(run_dir, args.sft_root.resolve())
    else:
        try:
            execute(run_dir)
        except Exception as exc:
            write_json(run_dir / "status.json", {"phase": "failed", "updated_at": now(), "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()
