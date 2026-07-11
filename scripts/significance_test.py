"""
Paired bootstrap significance test for word-level QE predictions.

Compares two systems' predictions on the same test set and reports, for each
of MCC / F1-BAD / F1-OK / F1-product / F1-macro:
  - the observed metric for each system
  - the observed difference (A - B)
  - a 95% bootstrap percentile CI on the difference
  - a two-sided bootstrap p-value (H0: no difference)

Inputs are predictions.tsv files as produced by work_SFT/QE_SFT, which include
both the gold `labels` column and the predicted `pred_labels` column. Inference
is NOT re-run; the labels are simply resampled from the existing files.

Usage:
    python scripts/significance_test.py \\
        --pred-a path/to/A/predictions.tsv \\
        --pred-b path/to/B/predictions.tsv \\
        --n-bootstrap 10000 --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_NAMES = ["MCC", "F1-BAD", "F1-OK", "F1-product", "F1-macro"]


def parse_label_sequence(label_str: str) -> list[int]:
    result = []
    for tok in label_str.strip().split():
        normalized = tok.upper().rstrip(".,;:")
        result.append(0 if normalized == "OK" else 1)
    return result


def load_file(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl") or path.endswith(".json"):
        return pd.read_json(path, lines=True, dtype=str).fillna("")
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def per_sentence_confusion(gold: list[list[int]],
                           pred: list[list[int]]) -> tuple[np.ndarray, int]:
    """
    Returns (N, 4) int array of per-sentence [TN, FP, FN, TP] (label 1 = BAD = positive),
    plus a count of sentences whose gold/pred lengths mismatched (truncated).
    """
    n = len(gold)
    counts = np.zeros((n, 4), dtype=np.int64)
    skipped = 0
    for i, (g, p) in enumerate(zip(gold, pred)):
        if len(g) != len(p):
            skipped += 1
            m = min(len(g), len(p))
            g, p = g[:m], p[:m]
        g_arr = np.asarray(g, dtype=np.int8)
        p_arr = np.asarray(p, dtype=np.int8)
        counts[i, 0] = int(((g_arr == 0) & (p_arr == 0)).sum())  # TN
        counts[i, 1] = int(((g_arr == 0) & (p_arr == 1)).sum())  # FP
        counts[i, 2] = int(((g_arr == 1) & (p_arr == 0)).sum())  # FN
        counts[i, 3] = int(((g_arr == 1) & (p_arr == 1)).sum())  # TP
    return counts, skipped


def metrics_from_aggregated(c: np.ndarray) -> np.ndarray:
    """
    c shape (..., 4) where last dim is [TN, FP, FN, TP].
    Returns array with same leading shape and last dim of size 5:
        [MCC, F1-BAD, F1-OK, F1-product, F1-macro]
    Computed analytically from token counts (no sklearn loop needed).
    """
    c = c.astype(np.float64)
    tn = c[..., 0]
    fp = c[..., 1]
    fn = c[..., 2]
    tp = c[..., 3]

    denom_bad = 2 * tp + fp + fn
    f1_bad = np.where(denom_bad > 0, 2 * tp / np.where(denom_bad > 0, denom_bad, 1), 0.0)

    denom_ok = 2 * tn + fn + fp
    f1_ok = np.where(denom_ok > 0, 2 * tn / np.where(denom_ok > 0, denom_ok, 1), 0.0)

    mcc_denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc_denom = np.sqrt(np.maximum(mcc_denom_sq, 0))
    mcc = np.where(mcc_denom > 0,
                   (tp * tn - fp * fn) / np.where(mcc_denom > 0, mcc_denom, 1),
                   0.0)

    f1_product = f1_ok * f1_bad
    f1_macro = (f1_ok + f1_bad) / 2

    return np.stack([mcc, f1_bad, f1_ok, f1_product, f1_macro], axis=-1)


def paired_bootstrap(counts_a: np.ndarray,
                     counts_b: np.ndarray,
                     n_boot: int,
                     seed: int,
                     chunk: int = 1000) -> np.ndarray:
    """
    Returns (n_boot, 5) bootstrap distribution of (metric_A - metric_B), where
    each bootstrap iteration resamples sentence indices with replacement and
    applies the SAME index set to both systems (paired).
    """
    n = counts_a.shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, 5), dtype=np.float64)

    for start in range(0, n_boot, chunk):
        end = min(start + chunk, n_boot)
        b = end - start
        idx = rng.integers(0, n, size=(b, n))                 # (b, n)
        sum_a = counts_a[idx].sum(axis=1)                     # (b, 4)
        sum_b = counts_b[idx].sum(axis=1)                     # (b, 4)
        m_a = metrics_from_aggregated(sum_a)                  # (b, 5)
        m_b = metrics_from_aggregated(sum_b)                  # (b, 5)
        out[start:end] = m_a - m_b

    return out


def format_report(args, n_sentences: int, n_skipped_a: int, n_skipped_b: int,
                  observed_a: np.ndarray, observed_b: np.ndarray,
                  observed_diff: np.ndarray, ci_lower: np.ndarray,
                  ci_upper: np.ndarray, p_values: np.ndarray,
                  gold_mismatch: int) -> str:
    ci_pct = int(round((1 - args.alpha) * 100))

    lines = []
    lines.append(f"System A ({args.name_a}): {args.pred_a}")
    lines.append(f"System B ({args.name_b}): {args.pred_b}")
    lines.append(f"N sentences: {n_sentences}   "
                 f"N bootstrap: {args.n_bootstrap}   "
                 f"seed: {args.seed}   alpha: {args.alpha}")
    if n_skipped_a or n_skipped_b:
        lines.append(f"Length-mismatched sentences (truncated to shorter): "
                     f"A={n_skipped_a}, B={n_skipped_b}")
    if gold_mismatch:
        lines.append(f"WARNING: {gold_mismatch} sentences have differing gold labels "
                     f"between A and B (different test sets?)")
    lines.append("")

    header = (f"{'Metric':<12} {args.name_a[:10]:>10} {args.name_b[:10]:>10} "
              f"{'Δ(A-B)':>10} {f'{ci_pct}% CI':>22} {'p-value':>10}  sig")
    lines.append(header)
    lines.append("─" * len(header))
    for i, name in enumerate(METRIC_NAMES):
        p = p_values[i]
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = ""
        ci_str = f"[{ci_lower[i]:+.4f},{ci_upper[i]:+.4f}]"
        lines.append(
            f"{name:<12} {observed_a[i]:>10.4f} {observed_b[i]:>10.4f} "
            f"{observed_diff[i]:>+10.4f} {ci_str:>22} {p:>10.4f}  {sig}"
        )
    lines.append("")
    lines.append("p-value: two-sided paired bootstrap, null = no difference "
                 "(recentered bootstrap distribution)")
    lines.append("CI    : percentile bootstrap on Δ(A-B)")
    lines.append("sig   : *** p<0.001   ** p<0.01   * p<0.05")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap significance test on word-level QE predictions."
    )
    parser.add_argument("--pred-a", required=True, help="Path to System A predictions.tsv/jsonl")
    parser.add_argument("--pred-b", required=True, help="Path to System B predictions.tsv/jsonl")
    parser.add_argument("--gold-col", default="labels")
    parser.add_argument("--pred-col", default="pred_labels")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="1 - confidence level for CI (default 0.05 → 95%% CI)")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--output", default=None,
                        help="Optional path to save the report (text). Stdout always printed.")
    args = parser.parse_args()

    df_a = load_file(args.pred_a)
    df_b = load_file(args.pred_b)

    for col in (args.gold_col, args.pred_col):
        if col not in df_a.columns:
            raise ValueError(f"Column '{col}' not found in {args.pred_a}. "
                             f"Available: {list(df_a.columns)}")
        if col not in df_b.columns:
            raise ValueError(f"Column '{col}' not found in {args.pred_b}. "
                             f"Available: {list(df_b.columns)}")

    if len(df_a) != len(df_b):
        raise ValueError(f"Number of examples mismatch: A={len(df_a)}, B={len(df_b)}")

    gold_a = [parse_label_sequence(s) for s in df_a[args.gold_col]]
    gold_b = [parse_label_sequence(s) for s in df_b[args.gold_col]]
    pred_a = [parse_label_sequence(s) for s in df_a[args.pred_col]]
    pred_b = [parse_label_sequence(s) for s in df_b[args.pred_col]]

    gold_mismatch = sum(1 for g_a, g_b in zip(gold_a, gold_b) if g_a != g_b)

    counts_a, skipped_a = per_sentence_confusion(gold_a, pred_a)
    counts_b, skipped_b = per_sentence_confusion(gold_b, pred_b)

    n = counts_a.shape[0]

    observed_a = metrics_from_aggregated(counts_a.sum(axis=0))
    observed_b = metrics_from_aggregated(counts_b.sum(axis=0))
    observed_diff = observed_a - observed_b

    boot_diffs = paired_bootstrap(counts_a, counts_b, args.n_bootstrap, args.seed)

    # Two-sided bootstrap p-value via recentering at the null Δ=0:
    #   null_dist = boot_diffs - observed_diff
    #   p = P(|null_dist| >= |observed_diff|)
    shifted = boot_diffs - observed_diff[None, :]
    p_values = np.mean(np.abs(shifted) >= np.abs(observed_diff)[None, :], axis=0)

    lo_pct = 100 * args.alpha / 2
    hi_pct = 100 * (1 - args.alpha / 2)
    ci_lower = np.percentile(boot_diffs, lo_pct, axis=0)
    ci_upper = np.percentile(boot_diffs, hi_pct, axis=0)

    report = format_report(
        args, n, skipped_a, skipped_b,
        observed_a, observed_b, observed_diff,
        ci_lower, ci_upper, p_values, gold_mismatch,
    )
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"\nReport saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
