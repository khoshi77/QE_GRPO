"""
単語レベル QE のための切り替え可能な GRPO 報酬関数。

``output_format=labels`` と ``output_format=xml_mt`` をサポートする。
xml_mt では生成XMLを MT 語ごとの OK/BAD に戻して意味品質を計算し、
MT の copy-exact とタグの開閉整合性を別の format factor として掛ける。

verl の custom_reward_function 規約:
    compute_score(data_source, solution_str, ground_truth, extra_info, **reward_kwargs)
        -> float | dict

reward_kwargs は run_grpo_qe.sh 内で Hydra CLI override
    +custom_reward_function.reward_kwargs.<key>=<value>
で指定する。verl 0.8 の verl/trainer/ppo/reward.py:get_custom_reward_fn が
partial(_call_with_kwargs, raw_fn, reward_kwargs) で wrap して呼ぶ。

引数:
    metric:           "f1_product" | "f1_bad" | "f1_ok" | "f1_macro" | "mcc"
                      | "weighted_token_accuracy" | "bad_f1_safe" | "token_mix"
                      mcc は (mcc+1)/2 で 0..1 に正規化される。
                      token_mix は
                        token_mix_weight * weighted_token_accuracy
                        + bad_f1_weight * bad_f1_safe
                      に format penalty を引く。
    length_mismatch:  生成ラベル数が num_words と一致しない場合の扱い
                      "pad_bad"  : SFT 流。短ければ BAD で埋め、長ければ truncate (ペナルティなし)
                      "penalize" : pad_bad と同じ調整 + reward に length_penalty を掛ける
                      "zero"     : 不一致なら reward=0
    invalid_token:    生成トークン中に OK/BAD 以外が含まれた場合の扱い
                      "as_bad"   : SFT 流。OK 以外はすべて BAD 扱い (ペナルティなし)
                      "penalize" : as_bad と同じ + invalid トークンが 1 つ以上あれば
                                   reward に invalid_penalty を掛ける
                      "zero"     : 1 つでもあれば reward=0
    length_penalty:   length_mismatch="penalize" 時の倍率 (default 0.5)
    invalid_penalty:  invalid_token="penalize"  時の倍率 (default 0.5)
    w_bad:            weighted_token_accuracy で BAD 正解に掛ける重み
    w_ok:             weighted_token_accuracy で OK 正解に掛ける重み
    token_mix_weight: token_mix 内の weighted_token_accuracy の重み
    bad_f1_weight:    token_mix 内の bad_f1_safe の重み
    length_mismatch_penalty:
                      token_mix 系で長さ不一致のとき score から引く値
    invalid_ratio_penalty:
                      token_mix 系で invalid_ratio に掛けて score から引く値
    xml_copy_mismatch_factor:
                      xml_mt で MT の復唱が崩れた場合に reward に掛ける倍率
    xml_unbalanced_factor:
                      xml_mt で <e> タグが不均衡な場合に reward に掛ける倍率

返り値: dict
    score:        最終的に学習に使う scalar (上記 metric 選択 + ペナルティ適用後)
    metric_*:     副次メトリック (毎ステップ tensorboard に出る)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# sklearn は重いが reward は CPU 側で 1 回/step なので OK。
from sklearn.metrics import f1_score, matthews_corrcoef

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qe_xml_utils import first_nonempty_line, xml_to_labels  # noqa: E402


VALID_METRICS = {
    "f1_product",
    "f1_bad",
    "f1_ok",
    "f1_macro",
    "mcc",
    "weighted_token_accuracy",
    "bad_f1_safe",
    "token_mix",
}
VALID_LENGTH_POLICY = {"pad_bad", "penalize", "zero"}
VALID_INVALID_POLICY = {"as_bad", "penalize", "zero"}
VALID_OUTPUT_FORMATS = {"labels", "xml_mt"}


def _parse_pred_labels(text: str, num_words: int) -> tuple[list[str], int, int, bool]:
    """
    生成テキストを OK/BAD ラベル列に正規化。

    Returns:
        labels:        長さ num_words に揃えた OK/BAD のリスト
        n_invalid:     OK/BAD 以外だった raw トークン数 (pad は数えない)
        raw_len:       生成された raw token 数
        length_mismatch: raw トークン長が num_words と一致しなかったか
    """
    raw_tokens = text.strip().split()
    n_invalid = 0
    norm: list[str] = []
    for tok in raw_tokens:
        # 末尾の句読点だけ落とす (SFT 側 parse_labels と揃える)
        cleaned = tok.upper().rstrip(".,;:")
        if cleaned == "OK":
            norm.append("OK")
        elif cleaned == "BAD":
            norm.append("BAD")
        else:
            n_invalid += 1
            norm.append("BAD")  # SFT 流: 不正トークンは BAD 扱い

    raw_len = len(norm)
    length_mismatch = raw_len != num_words
    if len(norm) > num_words:
        norm = norm[:num_words]
    elif len(norm) < num_words:
        norm.extend(["BAD"] * (num_words - len(norm)))
    return norm, n_invalid, raw_len, length_mismatch


def _labels_to_int(labels: list[str]) -> list[int]:
    """OK -> 0, BAD -> 1"""
    return [0 if x == "OK" else 1 for x in labels]


def _all_metrics(gold: list[int], pred: list[int]) -> dict[str, float]:
    """全メトリックを計算 (sklearn は両クラス揃ってないと 0 を返す/警告)"""
    if len(gold) == 0:
        return {"f1_bad": 0.0, "f1_ok": 0.0, "f1_product": 0.0, "f1_macro": 0.0, "mcc": 0.0}
    f1_bad = f1_score(gold, pred, pos_label=1, zero_division=0)
    f1_ok = f1_score(gold, pred, pos_label=0, zero_division=0)
    # MCC は gold/pred のどちらかが単一クラスだと未定義なので 0 にする。
    if len(set(gold)) < 2 or len(set(pred)) < 2:
        mcc = 0.0
    else:
        mcc = matthews_corrcoef(gold, pred)
    return {
        "f1_bad": float(f1_bad),
        "f1_ok": float(f1_ok),
        "f1_product": float(f1_ok * f1_bad),
        "f1_macro": float((f1_ok + f1_bad) / 2.0),
        "mcc": float(mcc),
    }


def _select_score(metrics: dict[str, float], metric: str) -> float:
    if metric == "mcc":
        return (metrics["mcc"] + 1.0) / 2.0  # 0..1 に正規化
    return metrics[metric]


def _weighted_token_accuracy(gold: list[int], pred: list[int], w_bad: float, w_ok: float) -> float:
    """Gold class weight 付き token accuracy。完全正解なら単一クラス文でも 1.0。"""
    denom = 0.0
    correct = 0.0
    for g, p in zip(gold, pred):
        weight = w_bad if g == 1 else w_ok
        denom += weight
        if g == p:
            correct += weight
    return float(correct / denom) if denom > 0.0 else 0.0


def _bad_f1_safe(gold: list[int], pred: list[int], f1_bad: float) -> float:
    """
    全OK文で F1-BAD が 0 になる問題を避ける。
    gold に BAD がなければ「BAD を出さない」ことを正解として 1.0 にする。
    """
    if not any(g == 1 for g in gold):
        return 1.0 if not any(p == 1 for p in pred) else 0.0
    return f1_bad


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _resolve_output_format(requested: str, extra_info: dict[str, Any]) -> str:
    if requested not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"output_format must be one of {sorted(VALID_OUTPUT_FORMATS)}, got {requested!r}"
        )
    dataset_format = str(extra_info.get("output_format", requested))
    if dataset_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(f"unsupported dataset output_format: {dataset_format!r}")
    if dataset_format != requested:
        raise ValueError(
            "reward/data format mismatch: "
            f"reward output_format={requested!r}, dataset output_format={dataset_format!r}"
        )
    return requested


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    metric: str = "f1_product",
    length_mismatch: str = "pad_bad",
    invalid_token: str = "as_bad",
    length_penalty: float = 0.5,
    invalid_penalty: float = 0.5,
    w_bad: float = 2.5,
    w_ok: float = 1.0,
    token_mix_weight: float = 0.8,
    bad_f1_weight: float = 0.2,
    length_mismatch_penalty: float = 0.2,
    invalid_ratio_penalty: float = 0.1,
    exact_format_bonus: float = 0.0,
    output_format: str = "labels",
    xml_copy_mismatch_factor: float = 0.25,
    xml_unbalanced_factor: float = 0.25,
) -> dict[str, float]:
    """verl から呼ばれる main entry。"""
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS}, got {metric!r}")
    if length_mismatch not in VALID_LENGTH_POLICY:
        raise ValueError(f"length_mismatch must be one of {VALID_LENGTH_POLICY}, got {length_mismatch!r}")
    if invalid_token not in VALID_INVALID_POLICY:
        raise ValueError(f"invalid_token must be one of {VALID_INVALID_POLICY}, got {invalid_token!r}")
    for name, factor in (
        ("xml_copy_mismatch_factor", xml_copy_mismatch_factor),
        ("xml_unbalanced_factor", xml_unbalanced_factor),
    ):
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {factor}")

    extra_info = extra_info or {}
    output_format = _resolve_output_format(output_format, extra_info)

    xml_stats: dict[str, int | bool] | None = None
    if output_format == "xml_mt":
        mt = str(extra_info.get("mt", "")).strip()
        if not mt:
            raise ValueError("format=xml_mt requires non-empty extra_info['mt']")
        ref_tokens = mt.split()
        num_words = int(extra_info.get("num_words", len(ref_tokens)))
        if num_words != len(ref_tokens):
            raise ValueError(
                f"extra_info num_words/mt mismatch: num_words={num_words}, mt={len(ref_tokens)}"
            )

        gold_tokens, gold_stats = xml_to_labels(ground_truth, ref_tokens)
        if not gold_stats["copy_exact"] or not gold_stats["tags_balanced"]:
            raise ValueError(f"invalid xml_mt ground_truth: {gold_stats}")

        annotated = first_nonempty_line(solution_str)
        pred_labels, xml_stats = xml_to_labels(annotated, ref_tokens)
        raw_len = int(xml_stats["n_output_words"])
        len_mismatch = raw_len != num_words
        n_invalid = 0
        invalid_ratio = 0.0
        exact_format = bool(xml_stats["copy_exact"] and xml_stats["tags_balanced"])
    else:
        gold_tokens = ground_truth.strip().upper().split()
        num_words = int(extra_info.get("num_words", len(gold_tokens)))
        invalid_gold = sorted(set(gold_tokens) - {"OK", "BAD"})
        if invalid_gold or len(gold_tokens) != num_words:
            raise ValueError(
                "invalid labels ground_truth: "
                f"num_words={num_words}, labels={len(gold_tokens)}, invalid={invalid_gold}"
            )
        pred_labels, n_invalid, raw_len, len_mismatch = _parse_pred_labels(
            solution_str, num_words
        )
        invalid_ratio = float(n_invalid / max(raw_len, num_words, 1))
        exact_format = not len_mismatch and n_invalid == 0

    gold_int = _labels_to_int(gold_tokens)
    pred_int = _labels_to_int(pred_labels)

    metrics = _all_metrics(gold_int, pred_int)
    weighted_acc = _weighted_token_accuracy(gold_int, pred_int, w_bad=w_bad, w_ok=w_ok)
    bad_f1_safe = _bad_f1_safe(gold_int, pred_int, metrics["f1_bad"])

    if metric == "weighted_token_accuracy":
        base_score = weighted_acc
    elif metric == "bad_f1_safe":
        base_score = bad_f1_safe
    elif metric == "token_mix":
        base_score = token_mix_weight * weighted_acc + bad_f1_weight * bad_f1_safe
        if exact_format:
            base_score += exact_format_bonus
        if output_format == "labels":
            base_score -= length_mismatch_penalty if len_mismatch else 0.0
            base_score -= invalid_ratio_penalty * invalid_ratio
        base_score = _clamp01(base_score)
    else:
        base_score = _select_score(metrics, metric)

    semantic_score = base_score
    format_factor = 1.0
    score = base_score
    if output_format == "xml_mt":
        assert xml_stats is not None
        if not xml_stats["copy_exact"]:
            format_factor *= xml_copy_mismatch_factor
        if not xml_stats["tags_balanced"]:
            format_factor *= xml_unbalanced_factor
        score *= format_factor
    else:
        if len_mismatch:
            if length_mismatch == "zero":
                score = 0.0
            elif length_mismatch == "penalize":
                score = score * length_penalty
            # "pad_bad" は no-op (上の正規化のみ)

        if n_invalid > 0:
            if invalid_token == "zero":
                score = 0.0
            elif invalid_token == "penalize":
                score = score * invalid_penalty
            # "as_bad" は no-op

    # 副次メトリックも返す: NaiveRewardManager が tensorboard に流してくれる。
    result = {
        "score": float(score),
        "metric_f1_bad": metrics["f1_bad"],
        "metric_f1_ok": metrics["f1_ok"],
        "metric_f1_product": metrics["f1_product"],
        "metric_f1_macro": metrics["f1_macro"],
        "metric_mcc": metrics["mcc"],
        "metric_weighted_token_accuracy": weighted_acc,
        "metric_bad_f1_safe": bad_f1_safe,
        "metric_semantic_score": float(semantic_score),
        "metric_format_factor": float(format_factor),
        "metric_invalid_ratio": invalid_ratio,
        "metric_n_invalid_tokens": float(n_invalid),
        "metric_length_mismatch": 1.0 if len_mismatch else 0.0,
    }
    if xml_stats is not None:
        result.update(
            {
                "metric_xml_copy_exact": 1.0 if xml_stats["copy_exact"] else 0.0,
                "metric_xml_tags_balanced": 1.0 if xml_stats["tags_balanced"] else 0.0,
                "metric_xml_word_count_match": 0.0 if len_mismatch else 1.0,
                "metric_xml_n_output_words": float(xml_stats["n_output_words"]),
            }
        )
    return result
