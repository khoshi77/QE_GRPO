"""
WMT21 en-ja word-level QE jsonl → verl 形式 parquet 変換。

入力:  src / mt / labels (空白区切りの OK/BAD 列、1単語1ラベル)
出力:  verl の RLHFDataset が読める parquet。各レコード:
    prompt:       chat list ([{role:system,...}, {role:user,...}])
    data_source:  "wmt21_en_ja_qe"
    ability:      "qe"
    reward_model: {style:"rule", ground_truth:<gold labels string>}
    extra_info:   {split, index, mt, num_words, src}

prompt は SFT 側 (work_SFT/QE_SFT/src/models/data_utils.py) と
**完全に同一のテキスト**になるよう揃える(SFT 後 weights からそのまま GRPO 開始するため)。
"""

import argparse
import json
import os
from pathlib import Path

import datasets


# SFT (work_SFT/QE_SFT/src/models/data_utils.py) と同一にする
SYSTEM_PROMPT = (
    "You are a machine translation quality estimator. "
    "Given a source sentence and its translation, label each word in the translation "
    "as OK (no error) or BAD (translation error). "
    "Output only the labels as a single space-separated sequence, one label per word, "
    "with no extra text."
)


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


def to_verl_record(row: dict, split: str, idx: int) -> dict:
    src = row["src"]
    mt = row["mt"]
    labels = row["labels"].strip()
    num_words = len(mt.strip().split())
    return {
        "prompt": build_messages(src, mt),
        "data_source": "wmt21_en_ja_qe",
        "ability": "qe",
        "reward_model": {"style": "rule", "ground_truth": labels},
        "extra_info": {
            "split": split,
            "index": idx,
            "src": src,
            "mt": mt,
            "num_words": num_words,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT/data/WMT21_en-ja/en-ja-test21",
        help="train.jsonl / dev.jsonl / test.jsonl が置かれているディレクトリ",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="parquet 出力先",
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
        records = [to_verl_record(r, split, i) for i, r in enumerate(rows)]
        ds = datasets.Dataset.from_list(records)
        out_path = out_dir / f"{split}.parquet"
        ds.to_parquet(str(out_path))
        print(f"[{split}] {len(records)} rows -> {out_path}")


if __name__ == "__main__":
    main()
