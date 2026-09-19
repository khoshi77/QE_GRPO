"""
WMT word-level QE jsonl → verl 形式 parquet 変換。

入力:  src / mt / labels (空白区切りの OK/BAD 列、1単語1ラベル)
出力:  verl の RLHFDataset が読める parquet。各レコード:
    prompt:       chat list ([{role:system,...}, {role:user,...}])
    data_source:  "wmt22_en_de_qe"
    ability:      "qe"
    reward_model: {style:"rule", ground_truth:<format-specific gold response>}
    extra_info:   {split, index, mt, num_words, src, output_format}

``--format labels`` と ``--format xml_mt`` をサポートする。
prompt と教師出力は SFT 側 (work_SFT/QE_SFT_8B/src/models/data_utils.py) と
**完全に同一のテキスト**になるよう揃える(SFT 後 weights からそのまま GRPO 開始するため)。
"""

import argparse
import json
import os
from pathlib import Path

import datasets

from qe_xml_utils import build_messages_xml_mt, labels_to_xml


# SFT (work_SFT/QE_SFT_8B/src/models/data_utils.py) と同一にする
SYSTEM_PROMPT = (
    "You are a machine translation quality estimator. "
    "Given a source sentence and its translation, label each word in the translation "
    "as OK (no error) or BAD (translation error). "
    "Output only the labels as a single space-separated sequence, one label per word, "
    "with no extra text."
)

VALID_FORMATS = {"labels", "xml_mt"}


def build_messages(src: str, mt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Source: {src.strip()}\n"
                f"Translation: {mt.strip()}\n\n"
                "Labels:"
            ),
        },
    ]


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_verl_record(row: dict, split: str, idx: int, output_format: str = "labels") -> dict:
    if output_format not in VALID_FORMATS:
        raise ValueError(f"format must be one of {sorted(VALID_FORMATS)}, got {output_format!r}")

    src = str(row["src"]).strip()
    mt = str(row["mt"]).strip()
    mt_tokens = mt.split()
    label_tokens = str(row["labels"]).strip().upper().split()
    where = f"{split}[{idx}]"
    if len(label_tokens) != len(mt_tokens):
        raise ValueError(
            f"{where}: MT/label length mismatch: mt={len(mt_tokens)} labels={len(label_tokens)}"
        )
    invalid = sorted(set(label_tokens) - {"OK", "BAD"})
    if invalid:
        raise ValueError(f"{where}: labels contain invalid values: {invalid}")

    labels = " ".join(label_tokens)
    if output_format == "xml_mt":
        prompt = build_messages_xml_mt(src, mt)
        ground_truth = labels_to_xml(mt_tokens, label_tokens)
    else:
        prompt = build_messages(src, mt)
        ground_truth = labels

    return {
        "prompt": prompt,
        "data_source": "wmt22_en_de_qe",
        "ability": "qe",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": idx,
            "src": src,
            "mt": mt,
            "num_words": len(mt_tokens),
            "output_format": output_format,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/data/WMT22_en-de_mqm",
        help="train.jsonl / dev.jsonl / test.jsonl が置かれているディレクトリ",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="parquet 出力先",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=sorted(VALID_FORMATS),
        default="labels",
        help="model response format (default: labels)",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": in_dir / "train.jsonl",
        "dev":   in_dir / "dev.jsonl",
        "test":  in_dir / "test.jsonl",
    }

    for split, path in splits.items():
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        rows = load_jsonl(str(path))
        records = [
            to_verl_record(r, split, i, output_format=args.output_format)
            for i, r in enumerate(rows)
        ]
        ds = datasets.Dataset.from_list(records)
        out_path = out_dir / f"{split}.parquet"
        ds.to_parquet(str(out_path))
        print(f"[{split}] format={args.output_format} {len(records)} rows -> {out_path}")


if __name__ == "__main__":
    main()
