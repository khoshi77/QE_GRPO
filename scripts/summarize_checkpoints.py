"""
summarize_checkpoints.py — checkpoint間の評価結果を集計して表示するスクリプト

期待するディレクトリ構造 (checkpoint-NNN / global_step_NNN のどちらでも可):
    <model_dir>/
      checkpoint-50/
        result/result.txt        ... test の評価結果（最終報告用）
        result_dev/result.txt    ... dev の評価結果（checkpoint 選択用、任意）
      checkpoint-100/
        ...

dev の結果 (result_dev) がある場合は、指標ごとに dev スコア最大の checkpoint と
その checkpoint の test スコアを出力する（dev 選択 → test 報告のプロトコル）。

--suffix で推論モードごとの結果を集計できる:
    ""             : result/ result_dev/                         (通常生成)
    "_constrained" : result_constrained/ result_constrained_dev/ (語彙制約)
    "_calibrated"  : result_calibrated/ result_calibrated_dev/   (閾値校正済み)

Usage:
    python summarize_checkpoints.py <model_dir>
    python summarize_checkpoints.py <model_dir> --output summary.tsv
    python summarize_checkpoints.py <model_dir> --suffix _calibrated --output summary_calibrated.tsv
"""

import argparse
import json
import re
import sys
from pathlib import Path


METRICS = ["MCC", "F1-BAD", "F1-OK", "F1-macro", "F1-product"]

PATTERNS = {
    "MCC":        re.compile(r"^MCC\s*:\s*([+-]?\d+\.\d+)"),
    "F1-BAD":     re.compile(r"^F1-BAD\s*:\s*([+-]?\d+\.\d+)"),
    "F1-OK":      re.compile(r"^F1-OK\s*:\s*([+-]?\d+\.\d+)"),
    "F1-macro":   re.compile(r"^F1-macro\s*:\s*([+-]?\d+\.\d+)"),
    "F1-product": re.compile(r"^F1-product\s*:\s*([+-]?\d+\.\d+)"),
}

# SFT (checkpoint-NNN) と GRPO (global_step_NNN) の両方に対応
CKPT_RE = re.compile(r"(?:checkpoint-|global_step_)(\d+)")


def parse_result(path: Path) -> dict | None:
    """result.txt から指標を抽出する。"""
    found = {}
    try:
        for line in path.read_text().splitlines():
            for name, pat in PATTERNS.items():
                if name not in found:
                    m = pat.match(line.strip())
                    if m:
                        found[name] = float(m.group(1))
    except Exception as e:
        print(f"  [WARN] {path} の読み込みに失敗: {e}", file=sys.stderr)
        return None
    return found if found else None


def checkpoint_number(path: Path) -> int:
    """checkpoint-NNN / global_step_NNN から数値を取り出す（ソート用）。"""
    m = CKPT_RE.search(path.name)
    return int(m.group(1)) if m else -1


def print_table(rows: list[dict], key: str, title: str) -> None:
    """rows[i][key] が {metric: value} の表を表示する。"""
    col_w = max(len(r["checkpoint"]) for r in rows) + 2
    header = f"{'Checkpoint':<{col_w}}" + "".join(f"{m:>12}" for m in METRICS)
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        metrics = r.get(key)
        line = f"{r['checkpoint']:<{col_w}}"
        for m in METRICS:
            val = metrics.get(m) if metrics else None
            line += f"{val:>12.4f}" if val is not None else f"{'N/A':>12}"
        print(line)
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="checkpoint間の評価結果を集計")
    parser.add_argument("model_dir", help="checkpoint-* / global_step_* を含むディレクトリ")
    parser.add_argument("--output", "-o", help="TSV形式で保存するファイルパス（省略時は表示のみ）")
    parser.add_argument("--suffix", default="",
                        help='result ディレクトリの suffix ("", "_constrained", "_calibrated")')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: ディレクトリが存在しません: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # checkpoint ディレクトリを数値順に収集
    checkpoints = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and CKPT_RE.match(d.name)],
        key=checkpoint_number,
    )

    if not checkpoints:
        print(f"ERROR: checkpoint-* / global_step_* が見つかりません: {model_dir}", file=sys.stderr)
        sys.exit(1)

    suffix = args.suffix
    rows = []
    for cp in checkpoints:
        test_file = cp / f"result{suffix}" / "result.txt"
        dev_file = cp / f"result{suffix}_dev" / "result.txt"
        test_metrics = parse_result(test_file) if test_file.exists() else None
        dev_metrics = parse_result(dev_file) if dev_file.exists() else None
        if test_metrics is None and dev_metrics is None:
            print(f"  [SKIP] result{suffix}/result.txt なし: {cp.name}", file=sys.stderr)
            continue

        # 校正済みモードでは採用閾値も拾う
        threshold = None
        thr_file = cp / f"result{suffix}" / "threshold.json"
        if thr_file.exists():
            try:
                threshold = json.loads(thr_file.read_text()).get("threshold")
            except Exception:
                pass

        rows.append({
            "checkpoint": cp.name, "test": test_metrics, "dev": dev_metrics,
            "threshold": threshold,
        })

    if not rows:
        print("集計できる結果がありませんでした。", file=sys.stderr)
        sys.exit(1)

    has_test = any(r["test"] for r in rows)
    has_dev = any(r["dev"] for r in rows)

    mode_label = suffix if suffix else " (generate)"

    # ---- 表示 ----
    if has_test:
        print_table(rows, "test", f"{model_dir}  [test{mode_label}]")
    if has_dev:
        print_table(rows, "dev", f"{model_dir}  [dev{mode_label}]")

    # 校正済みモード: checkpoint ごとの採用閾値を表示
    if any(r["threshold"] is not None for r in rows):
        print("\n[Calibrated thresholds (dev でフィット)]")
        for r in rows:
            if r["threshold"] is not None:
                print(f"  {r['checkpoint']:<20}: {r['threshold']:.2f}")

    # ---- dev 選択 → test 報告（正式プロトコル） ----
    best_dev_rows = []
    if has_dev:
        print("\n[Checkpoint selection by dev (指標ごとに dev 最大 → その checkpoint の test スコアを報告)]")
        for m in METRICS:
            vals = [r for r in rows if r["dev"] and m in r["dev"]]
            if not vals:
                continue
            best = max(vals, key=lambda r: r["dev"][m])
            test_val = best["test"].get(m) if best["test"] else None
            test_str = f"{test_val:.4f}" if test_val is not None else "N/A"
            print(f"  {m:<12}: dev={best['dev'][m]:.4f}  test={test_str}  ({best['checkpoint']})")
            best_dev_rows.append({
                "metric": m,
                "checkpoint": best["checkpoint"],
                "dev": best["dev"][m],
                "test": test_val,
            })
    else:
        print(f"\n[WARN] result{suffix}_dev が見つかりません。dev 選択には run_all_checkpoints.sh の"
              " dev+test 版で再評価してください。", file=sys.stderr)

    # ---- test ベスト（参考値: checkpoint 選択には使わないこと） ----
    if has_test:
        print("\n[Best per metric on test (参考値。test での checkpoint 選択は選択バイアスになる)]")
        for m in METRICS:
            vals = [(r["checkpoint"], r["test"][m]) for r in rows if r["test"] and m in r["test"]]
            if not vals:
                continue
            best_cp, best_val = max(vals, key=lambda x: x[1])
            print(f"  {m:<12}: {best_val:.4f}  ({best_cp})")

    # ---- TSV 出力 ----
    if args.output:
        out_path = Path(args.output)
        with out_path.open("w") as f:
            header = ["checkpoint"] + METRICS + [f"dev_{m}" for m in METRICS]
            f.write("\t".join(header) + "\n")
            for r in rows:
                cells = [r["checkpoint"]]
                for key in ("test", "dev"):
                    metrics = r.get(key)
                    cells += [
                        f"{metrics[m]:.4f}" if metrics and m in metrics else "N/A"
                        for m in METRICS
                    ]
                f.write("\t".join(cells) + "\n")
        print(f"\nTSV を保存しました: {out_path}")

        if best_dev_rows:
            best_path = out_path.with_name(out_path.stem + "_best_dev.tsv")
            with best_path.open("w") as f:
                f.write("metric\tbest_checkpoint\tdev\ttest\n")
                for b in best_dev_rows:
                    test_str = f"{b['test']:.4f}" if b["test"] is not None else "N/A"
                    f.write(f"{b['metric']}\t{b['checkpoint']}\t{b['dev']:.4f}\t{test_str}\n")
            print(f"dev 選択サマリを保存しました: {best_path}")


if __name__ == "__main__":
    main()
