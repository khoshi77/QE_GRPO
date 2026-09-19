"""Fail-fast validation for labels/xml_mt verl QE parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from preprocess_qe_wmt21 import build_messages
from qe_xml_utils import build_messages_xml_mt, xml_to_labels


VALID_FORMATS = {"labels", "xml_mt"}


def validate_file(path: Path, expected_format: str) -> int:
    parquet = pq.ParquetFile(path)
    count = 0
    for batch in parquet.iter_batches(
        batch_size=1024,
        columns=["prompt", "reward_model", "extra_info"],
    ):
        for record in batch.to_pylist():
            index = count
            count += 1
            extra = record["extra_info"]
            actual_format = str(extra.get("output_format") or "labels")
            if actual_format != expected_format:
                raise ValueError(
                    f"{path}[{index}]: expected format={expected_format}, got {actual_format}"
                )

            mt = str(extra["mt"]).strip()
            src = str(extra["src"]).strip()
            num_words = int(extra["num_words"])
            mt_tokens = mt.split()
            if num_words != len(mt_tokens):
                raise ValueError(
                    f"{path}[{index}]: num_words={num_words}, mt_words={len(mt_tokens)}"
                )

            ground_truth = str(record["reward_model"]["ground_truth"])
            if expected_format == "xml_mt":
                expected_prompt = build_messages_xml_mt(src, mt)
                labels, stats = xml_to_labels(ground_truth, mt_tokens)
                if not stats["copy_exact"] or not stats["tags_balanced"]:
                    raise ValueError(f"{path}[{index}]: invalid XML ground truth: {stats}")
                if len(labels) != num_words:
                    raise ValueError(f"{path}[{index}]: XML label length mismatch")
            else:
                expected_prompt = build_messages(src, mt)
                labels = ground_truth.upper().split()
                invalid = sorted(set(labels) - {"OK", "BAD"})
                if len(labels) != num_words or invalid:
                    raise ValueError(
                        f"{path}[{index}]: invalid labels ground truth: "
                        f"labels={len(labels)}, invalid={invalid}"
                    )

            if record["prompt"] != expected_prompt:
                raise ValueError(f"{path}[{index}]: prompt does not match {expected_format}")

    if count == 0:
        raise ValueError(f"{path}: empty parquet")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=sorted(VALID_FORMATS), required=True)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.files:
        if not path.is_file():
            raise FileNotFoundError(path)
        count = validate_file(path, args.format)
        print(f"[valid] format={args.format} rows={count} path={path}")


if __name__ == "__main__":
    main()
