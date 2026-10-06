"""OpenAI Responses API baseline for the existing word-level QE evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from qe_xml_utils import first_nonempty_line, xml_to_labels

HERE = Path(__file__).resolve().parent
PROFILES = {"eval": {"temperature": 0.0, "top_p": 1.0, "samples": 1},
            "rollout": {"temperature": 0.5, "top_p": 0.9, "samples": 8}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else HERE / path


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_records(path, fmt):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, row in enumerate(rows):
        extra = row["extra_info"]
        # The original labels parquet predates the output_format metadata field.
        if extra.get("output_format", "labels") != fmt:
            raise ValueError(f"row {i}: dataset format differs from --format {fmt}")
        if not extra.get("mt") or not isinstance(extra.get("src"), str):
            raise ValueError(f"row {i}: src/mt missing")
        if int(extra["num_words"]) != len(extra["mt"].split()):
            raise ValueError(f"row {i}: MT word count mismatch")
        prompt = row["prompt"]
        if not prompt or any(m.get("role") not in {"system", "user", "developer"}
                             or not isinstance(m.get("content"), str) for m in prompt):
            raise ValueError(f"row {i}: invalid prompt")
        gold_labels(row, fmt)  # Validate all gold before the first paid request.
    if not rows:
        raise ValueError(f"No records: {path}")
    return rows


def gold_labels(row, fmt):
    gold = row["reward_model"]["ground_truth"]
    if fmt == "xml_mt":
        labels, stats = xml_to_labels(gold, row["extra_info"]["mt"].split())
        if not stats["copy_exact"] or not stats["tags_balanced"]:
            raise ValueError("Invalid XML gold")
    else:
        labels = gold.strip().upper().split()
    if len(labels) != row["extra_info"]["num_words"] or set(labels) - {"OK", "BAD"}:
        raise ValueError("Invalid gold labels")
    return labels


def normalize(text, row, fmt):
    if fmt == "xml_mt":
        return xml_to_labels(first_nonempty_line(text), row["extra_info"]["mt"].split())[0]
    from qe_reward import _parse_pred_labels
    return _parse_pred_labels(text, row["extra_info"]["num_words"])[0]


def request_parameters(cfg):
    params = {"model": cfg["model"], "store": False, "truncation": "disabled"}
    if cfg["max_output_tokens"] != "omit":
        params["max_output_tokens"] = cfg["max_output_tokens"]
    if cfg["reasoning_effort"] != "omit":
        params["reasoning"] = {"effort": cfg["reasoning_effort"]}
    for name in ("temperature", "top_p"):
        if cfg[name] is not None:
            params[name] = cfg[name]
    return params


def resolve_config(args):
    cfg = json.loads(local_path(args.config).read_text(encoding="utf-8"))
    for key in ("model", "format", "split", "profile", "reasoning_effort", "samples",
                "max_output_tokens", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg["format"] = {"xml": "xml_mt"}.get(cfg["format"], cfg["format"])
    if cfg["format"] not in {"labels", "xml_mt"} or cfg["profile"] not in PROFILES:
        raise ValueError("format must be labels/xml_mt; profile must be eval/rollout")
    if cfg["split"] not in {"dev", "test", "both"}:
        raise ValueError("split must be dev/test/both")
    for name in ("temperature", "top_p"):
        value = getattr(args, name)
        value = cfg[name] if value is None else value
        cfg[name] = (PROFILES[cfg["profile"]][name] if value == "profile" else
                     None if value is None or value == "omit" else float(value))
    if cfg["samples"] is None:
        cfg["samples"] = PROFILES[cfg["profile"]]["samples"]
    if cfg["max_output_tokens"] is None:
        cfg["max_output_tokens"] = 256 if cfg["format"] == "labels" else 448
    if cfg["samples"] < 1:
        raise ValueError("samples >= 1 required")
    limit = cfg["max_output_tokens"]
    if limit != "omit" and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 16):
        raise ValueError("max_output_tokens must be an integer >= 16, null, or 'omit'")
    if cfg["temperature"] is not None and not 0 <= cfg["temperature"] <= 2:
        raise ValueError("temperature must be in [0, 2]")
    if cfg["top_p"] is not None and not 0 < cfg["top_p"] <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.input and cfg["split"] == "both":
        raise ValueError("--input requires --split dev or test")
    if "output_format" in cfg["reward_kwargs"]:
        raise ValueError("Use format instead of reward_kwargs.output_format")
    return cfg


def get_client(cfg):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        key_path = local_path(cfg["api_key_file"])
        if key_path.exists():
            key = key_path.read_text(encoding="utf-8").strip()
    if not key or "REPLACE" in key:
        raise ValueError("Set OPENAI_API_KEY or replace the placeholder in baseline/api_key.txt")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url="https://api.openai.com/v1",
                  timeout=cfg["timeout"], max_retries=cfg["max_retries"])


def read_cache(path):
    if not path.exists():
        return []
    records = []
    with path.open("rb+") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            # A killed process may leave a partial final record. Only that tail is removed.
            if not line.endswith(b"\n"):
                stream.truncate(position)
                break
            records.append(json.loads(line))
    return records


def summarize(rows, records, cfg, directory):
    from qe_reward import _all_metrics, compute_score
    fmt = cfg["format"]
    predictions = []
    scores = []
    by_sample = {}
    for item in sorted(records, key=lambda x: (x["row_id"], x["sample_id"])):
        row = rows[item["row_id"]]
        raw = item["raw_output"]
        pred = normalize(raw, row, fmt)
        gold = gold_labels(row, fmt)
        score = compute_score(row.get("data_source", "qe"), raw, row["reward_model"]["ground_truth"],
                              row["extra_info"], output_format=fmt, **cfg["reward_kwargs"])
        scores.append(score)
        gold_int, pred_int = by_sample.setdefault(item["sample_id"], ([], []))
        gold_int.extend(int(x == "BAD") for x in gold)
        pred_int.extend(int(x == "BAD") for x in pred)
        escaped = raw.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
        predictions.append({"row_id": item["row_id"], "sample_id": item["sample_id"],
                            "src": row["extra_info"]["src"], "mt": row["extra_info"]["mt"],
                            "labels": " ".join(gold), "pred_labels": " ".join(pred),
                            "raw_output": escaped, "status": item["status"]})
    def write_tsv(path, values):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(values)
    write_tsv(directory / "predictions.tsv", predictions)
    if cfg["samples"] > 1:
        for sample_id in by_sample:
            write_tsv(directory / f"predictions.sample_{sample_id:03d}.tsv",
                      [p for p in predictions if p["sample_id"] == sample_id])
    gold_all = [x for gold, _ in by_sample.values() for x in gold]
    pred_all = [x for _, pred in by_sample.values() for x in pred]
    usage = Counter()
    for item in records:
        u = item.get("usage") or {}
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            usage[name] += u.get(name, 0) or 0
        usage["reasoning_tokens"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    summary = {
        "n_sentences": len(rows), "n_outputs": len(records), "samples_per_sentence": cfg["samples"],
        "corpus_metrics": _all_metrics(gold_all, pred_all),
        "per_sample_corpus_metrics": {str(k): _all_metrics(*v) for k, v in by_sample.items()},
        "sentence_mean_reward_metrics": {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]},
        "status_counts": dict(Counter(x["status"] for x in records)),
        "incomplete_reason_counts": dict(Counter((x.get("incomplete_details") or {}).get("reason", "unknown")
                                                 for x in records if x["status"] == "incomplete")),
        "refusal_count": sum(bool(x.get("refusals")) for x in records),
        "empty_output_count": sum(not x["raw_output"].strip() for x in records),
        "usage": dict(usage),
    }
    write_json(directory / "metrics.json", summary)
    report = [f"Sentences    : {len(rows)}", f"Outputs      : {len(records)}", f"Total tokens : {len(gold_all)}"]
    for title, key in (("MCC", "mcc"), ("F1-BAD", "f1_bad"), ("F1-OK", "f1_ok"),
                       ("F1-macro", "f1_macro"), ("F1-product", "f1_product")):
        report.append(f"{title:<13}: {summary['corpus_metrics'][key]:.6f}")
    report.append(f"Mean reward  : {summary['sentence_mean_reward_metrics']['score']:.6f}")
    report.append(f"Statuses     : {summary['status_counts']}")
    (directory / "result.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


def run_split(rows, cfg, split, args, client=None):
    parameters = request_parameters(cfg)
    # Fingerprint inputs, API parameters, scoring code and reward settings to prevent mixed experiments.
    metadata = {"schema_version": 1, "format": cfg["format"], "split": split,
                "profile": cfg["profile"], "samples": cfg["samples"], "n_rows": len(rows),
                "input_sha256": digest(rows), "request_parameters": parameters,
                "reward_kwargs": cfg["reward_kwargs"],
                "code_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ("run_baseline.py", "qe_reward.py", "qe_xml_utils.py")}}
    model_folder = re.sub(r"[^a-zA-Z0-9_.-]", "_", cfg["model"])
    directory = local_path(cfg["output_dir"]) / model_folder / cfg["format"] / cfg["profile"] / split
    if args.dry_run:
        print(json.dumps({"output_dir": str(directory), "run": metadata,
                          "first_request": {**parameters, "input": rows[0]["prompt"]},
                          "api_requests": len(rows) * cfg["samples"]}, ensure_ascii=False, indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    # Prevent two processes from submitting the same experiment concurrently.
    import fcntl
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another process is using {directory}") from exc
        config_path = directory / "run_config.json"
        if config_path.exists():
            if not (args.resume or args.evaluate_only):
                raise ValueError(f"Output exists: {directory}. Use --resume or a new --output-dir")
            if json.loads(config_path.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Resume configuration/data/code mismatch. Use a new --output-dir")
        elif args.evaluate_only:
            raise ValueError(f"No saved run: {directory}")
        else:
            if (directory / "responses.jsonl").exists():
                raise ValueError("responses.jsonl exists without run_config.json")
            write_json(config_path, metadata)
            with (directory / "inputs.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        cache_path = directory / "responses.jsonl"
        records = read_cache(cache_path)
        done = set()
        for item in records:
            key = (item["row_id"], item["sample_id"])
            if (key in done or not 0 <= key[0] < len(rows) or not 0 <= key[1] < cfg["samples"]
                    or item["status"] not in {"completed", "incomplete"}):
                raise ValueError("Invalid/duplicate saved response")
            done.add(key)
        total = len(rows) * cfg["samples"]
        if args.evaluate_only and len(records) != total:
            raise ValueError(f"Incomplete run: {len(records)}/{total}. Resume inference first")
        if not args.evaluate_only and len(records) < total:
            client = client or get_client(cfg)
            with cache_path.open("a", encoding="utf-8") as stream:
                for row_id, row in enumerate(rows):
                    for sample_id in range(cfg["samples"]):
                        if (row_id, sample_id) in done:
                            continue
                        start = time.monotonic()
                        response = client.responses.create(input=row["prompt"], **parameters)
                        body = response.model_dump(mode="json")
                        if body["status"] not in {"completed", "incomplete"}:
                            raise RuntimeError(f"Unexpected API response status: {body['status']}")
                        refusals = [part["refusal"] for item in body.get("output", [])
                                    if item.get("type") == "message" for part in item.get("content", [])
                                    if part.get("type") == "refusal"]
                        record = {"row_id": row_id, "sample_id": sample_id, "raw_output": response.output_text,
                                  "status": body["status"], "incomplete_details": body.get("incomplete_details"),
                                  "refusals": refusals, "usage": body.get("usage"),
                                  "model": body.get("model"), "response_id": body.get("id"),
                                  "elapsed_seconds": time.monotonic() - start, "response": body}
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        records.append(record)
                        print(f"[{split}] {len(records)}/{total} status={record['status']}", flush=True)
        summarize(rows, records, cfg, directory)
        print(f"Saved: {directory}", flush=True)


def output_token_limit(value):
    if value == "omit":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer or 'omit'") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model")
    parser.add_argument("--format", choices=["labels", "xml_mt", "xml"])
    parser.add_argument("--split", choices=["dev", "test", "both"])
    parser.add_argument("--profile", choices=list(PROFILES))
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", "omit"])
    parser.add_argument("--temperature", help="number, profile, or omit")
    parser.add_argument("--top-p", help="number, profile, or omit")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--max-output-tokens", type=output_token_limit, help="integer or omit (API default)")
    parser.add_argument("--input", help="verl parquet/JSONL; paths relative to baseline")
    parser.add_argument("--output-dir", help="paths relative to baseline")
    parser.add_argument("--limit", type=int, help="first N examples (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="validate and display request without calling API")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="rebuild reports without calling API")
    args = parser.parse_args()
    cfg = resolve_config(args)
    splits = ["dev", "test"] if cfg["split"] == "both" else [cfg["split"]]
    datasets = {}
    for split in splits:
        path = (local_path(args.input) if args.input else
                local_path(cfg["data_dir"]) / cfg["format"] / f"{split}.parquet")
        datasets[split] = load_records(path, cfg["format"])[:args.limit]
    if not args.dry_run:
        # Check scoring dependencies/config before making a paid request.
        from qe_reward import compute_score
        for rows in datasets.values():
            row = rows[0]
            compute_score(row.get("data_source", "qe"), row["reward_model"]["ground_truth"],
                          row["reward_model"]["ground_truth"], row["extra_info"],
                          output_format=cfg["format"], **cfg["reward_kwargs"])
    for split, rows in datasets.items():
        run_split(rows, cfg, split, args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
