#!/usr/bin/env python3
"""Compute BAD-label precision/recall statistics from prediction TSV files.

The expected TSV columns are:
  - labels: gold OK/BAD sequence
  - pred_labels: predicted OK/BAD sequence

By default, this script uses the prediction files reported in the paper draft.
It prints a compact LaTeX table, plus optional Markdown/TSV output.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODELS: list[tuple[str, Path]] = [
    (
        "mDeBERTa",
        Path("/work/UTSUROLB/utlb_buma2/work_grpo/sft_ckpts/mDeBERTa/predictions.tsv"),
    ),
    (
        "Qwen3-4B-Base (SFT)",
        Path(
            "/work/UTSUROLB/utlb_buma2/work_grpo/sft_ckpts/"
            "Qwen3-4B-Base-SFT/checkpoint-500/result/predictions.tsv"
        ),
    ),
    (
        "Qwen3-4B (SFT)",
        Path(
            "/work/UTSUROLB/utlb_buma2/work_grpo/sft_ckpts/"
            "Qwen3-4B-SFT/checkpoint-450/result/predictions.tsv"
        ),
    ),
    (
        "Qwen3-4B-Base (GRPO)",
        Path(
            "/work/UTSUROLB/utlb_buma2/work_grpo/ckpts/"
            "Qwen3-4B-Base-SFT_wmt21_enja/"
            "718511_mcc-penalize=50-penalize=50/"
            "global_step_250/result/predictions.tsv"
        ),
    ),
    (
        "Qwen3-4B (GRPO)",
        Path(
            "/work/UTSUROLB/utlb_buma2/work_grpo/ckpts/"
            "Qwen3-4B-SFT_wmt21_enja/"
            "717784_f1-multi-pad_bad-as_bad/"
            "global_step_50/result/predictions.tsv"
        ),
    ),
    (
        r"Qwen3-4B-Base (GRPO-\texttt{token\_mix})",
        Path(
            "/work/UTSUROLB/utlb_buma2/work_grpo/ckpts/"
            "verl_grpo_qwen3_4b_qe_wmt21_enja/"
            "0_748997_nqsv/global_step_150/result/predictions.tsv"
        ),
    ),
    (
        r"Qwen3-4B (GRPO-\texttt{token\_mix})",
        Path(
            "/work/UTSUROLB/utlb_buma2/work_grpo/ckpts/"
            "verl_grpo_qwen3_4b_qe_wmt21_enja/"
            "0_749050_nqsv/global_step_500/result/predictions.tsv"
        ),
    ),
]


@dataclass
class BadStats:
    model: str
    sentences: int = 0
    tokens: int = 0
    gold_bad: int = 0
    pred_bad: int = 0
    tp_bad: int = 0
    fp_bad: int = 0
    fn_bad: int = 0
    length_mismatches: int = 0

    @property
    def pred_bad_ratio(self) -> float:
        return self.pred_bad / self.tokens if self.tokens else 0.0

    @property
    def precision_bad(self) -> float:
        denom = self.tp_bad + self.fp_bad
        return self.tp_bad / denom if denom else 0.0

    @property
    def recall_bad(self) -> float:
        denom = self.tp_bad + self.fn_bad
        return self.tp_bad / denom if denom else 0.0

    @property
    def f1_bad(self) -> float:
        p = self.precision_bad
        r = self.recall_bad
        return 2 * p * r / (p + r) if (p + r) else 0.0


def normalize_labels(text: str) -> list[str]:
    labels: list[str] = []
    for token in text.strip().split():
        cleaned = token.upper().rstrip(".,;:")
        labels.append(cleaned if cleaned in {"OK", "BAD"} else "BAD")
    return labels


def pad_or_truncate(labels: list[str], target_len: int) -> list[str]:
    if len(labels) > target_len:
        return labels[:target_len]
    if len(labels) < target_len:
        return labels + ["BAD"] * (target_len - len(labels))
    return labels


def compute_stats(model: str, path: Path) -> BadStats:
    if not path.exists():
        raise FileNotFoundError(f"missing prediction file: {path}")

    stats = BadStats(model=model)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"labels", "pred_labels"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")

        for row_no, row in enumerate(reader, start=2):
            gold = normalize_labels(row["labels"])
            pred_raw = normalize_labels(row["pred_labels"])
            if len(pred_raw) != len(gold):
                stats.length_mismatches += 1
            pred = pad_or_truncate(pred_raw, len(gold))

            stats.sentences += 1
            stats.tokens += len(gold)
            for g, p in zip(gold, pred):
                if g == "BAD":
                    stats.gold_bad += 1
                if p == "BAD":
                    stats.pred_bad += 1
                if g == "BAD" and p == "BAD":
                    stats.tp_bad += 1
                elif g != "BAD" and p == "BAD":
                    stats.fp_bad += 1
                elif g == "BAD" and p != "BAD":
                    stats.fn_bad += 1

    return stats


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_latex(rows: list[BadStats]) -> None:
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Model & Pred. BAD & P-BAD & R-BAD & F1-BAD \\")
    print(r"\midrule")
    for row in rows:
        print(
            f"{row.model} & {pct(row.pred_bad_ratio)} & "
            f"{row.precision_bad:.4f} & {row.recall_bad:.4f} & {row.f1_bad:.4f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{BAD-label prediction analysis on the test set.}")
    print(r"\label{tab:bad-analysis}")
    print(r"\end{table*}")


def print_markdown(rows: list[BadStats]) -> None:
    print("| Model | Pred. BAD | P-BAD | R-BAD | F1-BAD | Length mismatches |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.model} | {pct(row.pred_bad_ratio)} | "
            f"{row.precision_bad:.4f} | {row.recall_bad:.4f} | {row.f1_bad:.4f} | "
            f"{row.length_mismatches} |"
        )


def print_tsv(rows: list[BadStats]) -> None:
    print(
        "\t".join(
            [
                "model",
                "sentences",
                "tokens",
                "gold_bad",
                "pred_bad",
                "pred_bad_ratio",
                "precision_bad",
                "recall_bad",
                "f1_bad",
                "tp_bad",
                "fp_bad",
                "fn_bad",
                "length_mismatches",
            ]
        )
    )
    for row in rows:
        print(
            "\t".join(
                [
                    row.model,
                    str(row.sentences),
                    str(row.tokens),
                    str(row.gold_bad),
                    str(row.pred_bad),
                    f"{row.pred_bad_ratio:.6f}",
                    f"{row.precision_bad:.6f}",
                    f"{row.recall_bad:.6f}",
                    f"{row.f1_bad:.6f}",
                    str(row.tp_bad),
                    str(row.fp_bad),
                    str(row.fn_bad),
                    str(row.length_mismatches),
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("latex", "markdown", "tsv", "all"),
        default="all",
        help="Output format (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [compute_stats(model, path) for model, path in DEFAULT_MODELS]

    if args.format in {"latex", "all"}:
        print_latex(rows)
    if args.format == "all":
        print()
    if args.format in {"markdown", "all"}:
        print_markdown(rows)
    if args.format == "all":
        print()
    if args.format in {"tsv", "all"}:
        print_tsv(rows)


if __name__ == "__main__":
    main()

