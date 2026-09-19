# WMT22 en-de Word-level QE: SFT → GRPO 現状レポート

作成日: 2026-08-13  
対象: WMT22 en-de MQM の token-level OK/BAD prediction

## 1. このレポートの位置づけ

この文書は、既存の `report.md` 以降に実施した WMT22 en-de の GRPO 4 run を、SFT と mDeBERTa の結果を含めて整理した現状報告である。数値は各 run の `summary.tsv`、`summary_best_dev.tsv`、Hydra 設定、学習ログ、および `predictions.tsv` から確認した。

本レポートでは、論文で使用できるモデル選択規則として、**各 run 内で dev MCC が最大の GRPO checkpoint を選び、その checkpoint の test 結果を報告する**。全 checkpoint の test 結果を見て選んだ最大値は、分析用の test-oracle 値として別表に分離し、主結果には用いない。

## 2. エグゼクティブサマリ

1. dev MCC で checkpoint を選んだ場合、現在の最良結果は **Qwen3-8B + GRPO Token-mix XML の test MCC 0.2521** である。次点は Qwen3-8B-Base + XML の 0.2431。
2. 実際に GRPO を開始した SFT checkpoint と比較すると、XML の MCC 改善は Base で +0.0796（p=0.0177）、Qwen3-8B で +0.0902（p=0.0001）。labels の改善は +0.0232 / +0.0251 で、MCC では有意ではない。
3. XML と labels の比較では、Qwen3-8B で XML が +0.0395（p=0.0377）と未補正の 5% 水準で有意。Base では +0.0439（p=0.0953）で有意ではない。
4. mDeBERTa（dev で epoch 4 を選択、test MCC 0.2092）に対し、Qwen3-8B + XML は +0.0429（p=0.0424）で未補正の 5% 水準では有意。ほかの3条件と mDeBERTa の MCC 差は有意ではない。
5. ただし4 run はいずれも設定した 3 epoch を完走しておらず、進捗は 26～34%。正式に選ばれた checkpoint もすべて第1 epoch 内にある。現在値を最終的な収束性能とはみなせない。
6. labels の SFT 初期 checkpoint は dev MCC 最大ではなく test MCC 最大の checkpoint と一致する一方、XML は dev MCC 最大 checkpoint を使用している。初期モデル選択規則が統一されていないため、現状の labels 対 XML は完全に統制された比較ではない。
7. Qwen3-8B-Base の2 runでは、学習 rollout が毎回最大応答長まで達し、`response_length/clip_ratio=1.0` となっている。NaN や発散はないが、Base系の終了トークン・生成停止挙動は再実験前に修正または検証が必要である。

## 3. データと評価プロトコル

### 3.1 データ

labels と XML は同一の WMT22 en-de 文集合を使用している。

| split | 文数 |
|---|---:|
| train | 59,855 |
| dev | 1,005 |
| test | 511 |

データ:

- labels: `data/qe_wmt22_en_de/{train,dev,test}.parquet`
- XML: `data/qe_wmt22_en_de_xml_mt/{train,dev,test}.parquet`

XML 条件は MT 文を復唱し、誤り箇所だけを `<bad>...</bad>` で囲む `xml_mt` 形式である。src へのタグ付けは行わない。

### 3.2 主指標とモデル選択

- 主指標: corpus-level MCC
- 補助指標: F1-BAD、F1-OK、F1-macro、F1-product
- GRPO checkpoint 選択: dev MCC 最大
- test: 選択後の1回の報告に相当する値として扱う
- 有意差検定: test 511文を単位とする two-sided paired bootstrap
- bootstrap: 10,000回、seed 42、95% CI

`scripts/significance_test.py` は同一文の予測を対にしてリサンプリングし、帰無仮説を差 0 とする再中心化 bootstrap p-value を計算する。以下の p-value は多重比較補正を行っていないため、p<0.05 は「名目上有意」と解釈する。

## 4. GRPO の実験設定

4 run に共通する主要設定は次のとおり。

| 項目 | 設定 |
|---|---|
| algorithm | GRPO |
| reward | `token_mix` |
| token mix | 0.8 × weighted token accuracy + 0.2 × BAD F1 |
| token weight | BAD 2.5、OK 1.0 |
| rollout数 | 8 |
| rollout temperature / top-p | 0.5 / 0.9 |
| learning rate | 1e-6、constant、warmupなし |
| KL loss | 使用、係数 0.001、`low_var_kl` |
| entropy coefficient | 0 |
| train batch / PPO mini-batch | 16 / 16 |
| checkpoint間隔 | 200 step |
| 設定epoch数 | 3 |
| 総予定step | 11,220（約3,740 step/epoch） |
| thinking | 無効 |

形式別の相違点:

- labels: `max_response_length=256`
- XML: `max_response_length=448`、`output_format=xml_mt`
- XML reward には copy mismatch と unbalanced tag の係数 0.25 が追加されている

## 5. 対象runとSFT初期モデル

| backbone | 形式 | run ID | SFT初期checkpoint | SFT dev MCC | SFT test MCC | 初期checkpointの性質 |
|---|---|---|---|---:|---:|---|
| Qwen3-8B-Base | labels | `0_868935_nqsv` | `Qwen3-8B-Base_labels_5epoch/checkpoint-4250` | 0.2127 | 0.1760 | test MCC 最大。dev MCC 最大は checkpoint-3250（dev 0.2348） |
| Qwen3-8B | labels | `0_868943_nqsv` | `Qwen3-8B_labels_5epoch/checkpoint-4250` | 0.1832 | 0.1875 | test MCC 最大。dev MCC 最大は checkpoint-2750（dev 0.2096） |
| Qwen3-8B-Base | XML | `0_889081_nqsv` | `Qwen3-8B-Base_xml_mt_5epoch/checkpoint-3250` | 0.1867 | 0.1635 | dev MCC 最大 |
| Qwen3-8B | XML | `0_889207_nqsv` | `Qwen3-8B_xml_mt_5epoch/checkpoint-2000` | 0.1789 | 0.1620 | dev MCC 最大 |

この不統一は重要である。labels の2つは test を見た選択と同じ結果になっているため、現在の4 runをそのまま「dev のみで選択したSFTから始めた公平な形式比較」とは記述できない。一方、各 run 内の **GRPO checkpoint 選択そのもの**は、本レポートでは dev MCC に統一している。

## 6. 学習の到達状況

| backbone | 形式 | run ID | 最終学習step | 予定11,220に対する進捗 | ログ上のepoch | dev選択checkpoint | 選択位置の概算epoch |
|---|---|---|---:|---:|---:|---|---:|
| Qwen3-8B-Base | labels | `0_868935_nqsv` | 3,328 | 30% | 0 | step 2,200 | 0.59 |
| Qwen3-8B | labels | `0_868943_nqsv` | 3,794 | 34% | 1 | step 1,400 | 0.37 |
| Qwen3-8B-Base | XML | `0_889081_nqsv` | 2,917 | 26% | 0 | step 2,800 | 0.75 |
| Qwen3-8B | XML | `0_889207_nqsv` | 3,514 | 31% | 0 | step 2,600 | 0.70 |

全runが約24時間で SIGTERM を受けて終了している。Qwen3-8B labels だけが第2 epochの冒頭に入ったが、それ以外は1 epoch未満であり、どのrunも3 epochを完了していない。

## 7. 主結果: dev MCC 選択 → test 報告

### 7.1 mDeBERTa ベースライン

mDeBERTa は epoch 1～5 の dev MCC がそれぞれ 0.2124、0.2079、0.2160、0.2218、0.2157 であり、dev 最大の epoch 4 を選択する。epoch 4 では dev で決めた閾値 0.30 を test に適用し、test MCC は 0.2092。

### 7.2 GRPO 4条件

| 順位 | backbone / 形式 | dev選択checkpoint | dev MCC | test MCC | F1-BAD | F1-OK | F1-macro | F1-product |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Qwen3-8B / XML | step 2,600 | **0.2323** | **0.2521** | **0.2818** | 0.9643 | **0.6230** | **0.2717** |
| 2 | Qwen3-8B-Base / XML | step 2,800 | 0.2219 | 0.2431 | 0.2740 | 0.9666 | 0.6203 | 0.2648 |
| 3 | Qwen3-8B / labels | step 1,400 | 0.2070 | 0.2126 | 0.2450 | 0.9661 | 0.6055 | 0.2367 |
| 4 | Qwen3-8B-Base / labels | step 2,200 | 0.2123 | 0.1992 | 0.2275 | **0.9710** | 0.5993 | 0.2210 |

参考として、mDeBERTa epoch 4 は test MCC 0.2092、F1-BAD 0.2398、F1-OK 0.9575、F1-macro 0.5986、F1-product 0.2296 である。

現在の点推定では XML 2条件が labels 2条件より高い。ただし、形式以外にSFT初期checkpointの選択規則、最大応答長、XML固有ペナルティが異なるため、差のすべてを「XML表現そのもの」の効果とは帰属できない。

## 8. SFT初期値からのGRPO改善

以下は、各runが実際に初期化されたSFT checkpointと、devで選んだGRPO checkpointの test MCC を比較した結果である。

| backbone / 形式 | SFT test MCC | GRPO test MCC | Δ(GRPO−SFT) | 95% CI | p-value | MCC有意差 |
|---|---:|---:|---:|---|---:|---|
| Qwen3-8B-Base / labels | 0.1760 | 0.1992 | +0.0232 | [-0.0205, +0.0687] | 0.3069 | なし |
| Qwen3-8B / labels | 0.1875 | 0.2126 | +0.0251 | [-0.0224, +0.0756] | 0.3089 | なし |
| Qwen3-8B-Base / XML | 0.1635 | 0.2431 | +0.0796 | [+0.0171, +0.1468] | 0.0177 | 名目上あり |
| Qwen3-8B / XML | 0.1620 | 0.2521 | +0.0902 | [+0.0477, +0.1364] | 0.0001 | あり |

解釈:

- labels でも点推定は改善しているが、test 511文では MCC の改善を統計的に確認できない。
- XML では両backboneで MCC が有意に改善しており、特に Qwen3-8B/XML の改善は大きい。
- Base labels、Qwen3-8B labels、Base XML では、GRPO後に F1-OK が有意に低下する一方、F1-BADが上昇する。主に BAD をより多く検出する方向へのトレードオフである。
- Qwen3-8B/XML は F1-OK を維持しながら、MCC、F1-BAD、F1-product、F1-macroがすべて改善しており、現在の4条件では最も良好な改善パターンである。

## 9. システム間のpaired bootstrap比較

すべて dev MCC で選択した GRPO checkpoint を比較している。差は表の A−B。

| 比較 A−B | ΔMCC | 95% CI | p-value | 解釈 |
|---|---:|---|---:|---|
| Qwen3-8B/XML − Qwen3-8B/labels | +0.0395 | [+0.0025, +0.0774] | 0.0377 | XMLが名目上有意に高い |
| Base/XML − Base/labels | +0.0439 | [-0.0051, +0.0979] | 0.0953 | XML優位の傾向、有意ではない |
| Qwen3-8B/labels − Base/labels | +0.0134 | [-0.0207, +0.0478] | 0.4468 | 有意差なし |
| Qwen3-8B/XML − Base/XML | +0.0090 | [-0.0192, +0.0378] | 0.5297 | 有意差なし |
| Qwen3-8B/XML − mDeBERTa | +0.0429 | [+0.0038, +0.0861] | 0.0424 | Qwen3-8B/XMLが名目上有意に高い |
| Base/XML − mDeBERTa | +0.0339 | [-0.0090, +0.0773] | 0.1238 | 有意差なし |
| Qwen3-8B/labels − mDeBERTa | +0.0034 | [-0.0447, +0.0536] | 0.8933 | 有意差なし |
| Base/labels − mDeBERTa | -0.0100 | [-0.0645, +0.0464] | 0.7268 | 有意差なし |

Qwen3-8B/XML の p=0.0377 と p=0.0424 は、多数の比較から事後的に得た未補正値である。複数の比較を同時に研究上の主張とする場合は補正後に有意ではなくなる可能性が高い。論文では主仮説と主比較を事前に1つへ絞るか、Holm法などの多重比較補正を適用する。

## 10. 参考値: test MCC 最大checkpoint

次は全checkpointの test MCCを見て選んだ値であり、モデル選択にtestを使用している。探索的な学習曲線の確認には使えるが、論文の主結果には使わない。

| backbone / 形式 | test最大checkpoint | そのcheckpointのdev MCC | test最大MCC | dev選択時との差 |
|---|---|---:|---:|---:|
| Qwen3-8B-Base / labels | step 3,000 | 0.1979 | 0.2152 | +0.0160 |
| Qwen3-8B / labels | step 1,200 | 0.1950 | 0.2389 | +0.0263 |
| Qwen3-8B-Base / XML | step 2,800 | 0.2219 | 0.2431 | +0.0000 |
| Qwen3-8B / XML | step 2,000 | 0.2074 | 0.2788 | +0.0267 |

したがって、0.2152、0.2389、0.2431、0.2788 という4値はファイルから確認できる正しい数値だが、前2つと最後の1つは test-oracle 値である。論文用の値は原則として 0.1992、0.2126、0.2431、0.2521 を使用する。

## 11. 学習状態の診断

### 11.1 正常に見える点

- 4 run とも学習ログに NaN はない。
- actor loss、gradient norm、KL loss、reward は有限値で推移している。
- 定期checkpointは保存され、dev/test推論と集計が完了している。
- SIGTERMまで学習stepは進行しており、数値発散による停止ではない。
- 4条件すべてで、少なくともSFT初期値より高い test MCC のcheckpointが得られている。

### 11.2 未完走とcheckpoint変動

- Base labels: dev MCC 最大は step 2,200 の 0.2123。test最大は step 3,000であり、devとtestのピークが一致しない。
- Qwen3-8B labels: dev MCC 最大は step 1,400 の 0.2070。その後は変動し、step 3,400で 0.1694まで低下している。
- Base XML: dev MCC は後半に上昇し、最後に評価できた step 2,800が最大 0.2219。未完走の影響が最も大きい可能性がある。
- Qwen3-8B XML: dev MCC 最大は step 2,600 の 0.2323。その後は低下しており、単調改善ではない。

checkpoint間の変動が大きいため、単一checkpointの偶然のスパイクを拾っている可能性がある。複数seed、隣接checkpointの平滑化、または事前に固定した評価間隔・early stopping規則が望ましい。

### 11.3 生成多様性とBaseモデルの応答切り詰め

学習後半の actor entropy はおおむね 0.0003～0.006 程度と低い。n=8 でも同一または非常に近いrolloutが増え、GRPOの相対報酬信号が弱くなる可能性がある。

さらに、Qwen3-8B-Base の labels と XML の両runでは、確認した step 1 から最終stepまで、平均応答長が設定上限（labels 256、XML 448）と一致し、`response_length/clip_ratio=1.0` だった。対して Qwen3-8B の2 runでは clip ratio は 0.0。この現象は出力形式よりも Base/chat の差に対応している。

Base系のoffline推論結果そのものが直ちに無効になるわけではないが、学習時rolloutの末尾、EOS、余分な反復、reward parserへの入力をサンプル単位で調べる必要がある。修正せずに単純再開すると、計算量の浪費と不正確なrewardが続く可能性がある。

## 12. 現時点で言えること・言えないこと

### 言えること

- 現在保存されているcheckpointの中では、dev選択した Qwen3-8B/XML が最良である。
- 実際のSFT初期checkpointに対するGRPOの改善は、XML 2条件で統計的に確認できる。
- labels より XML の点推定が両backboneで高く、Qwen3-8Bでは差が未補正の5%水準で有意である。
- Qwen3-8B/XML は mDeBERTa epoch 4 より高い点推定を持ち、未補正の5%水準では有意である。

### まだ言えないこと

- XML が一般に labels より優れている、またはQwen3-8BがBaseより優れているという確定的な結論。
- 3 epoch完走時の最終性能または収束性能。
- test最大値 0.2788 を、偏りのない論文上の主結果として提示すること。
- 現在の結果がseedを変えても再現すること。
- XMLの効果と、初期checkpoint、最大応答長、形式固有rewardの効果を完全に分離すること。

## 13. 論文での暫定的な報告方法

主表には次を用いる。

- mDeBERTa: devで選んだ epoch 4、test MCC 0.2092
- Qwen3-8B-Base GRPO labels: dev選択 step 2,200、test MCC 0.1992
- Qwen3-8B GRPO labels: dev選択 step 1,400、test MCC 0.2126
- Qwen3-8B-Base GRPO XML: dev選択 step 2,800、test MCC 0.2431
- Qwen3-8B GRPO XML: dev選択 step 2,600、test MCC 0.2521

ただし、現段階では「暫定結果」と明記する。test最大値は学習曲線分析または付録でのみ示し、「test-oracle」と明示する。また、現在の名目上有意な差は多重比較未補正であることを書く。

## 14. 次に行うべき作業

1. Qwen3-8B-Base のrollout原文を保存・確認し、EOSまたは反復生成により100%切り詰められる原因を修正する。
2. SFT初期checkpointの選択規則を全条件でdev MCCに統一する。公平な形式比較を主張するなら、labels側はdev選択checkpointからの再実験が必要。
3. 修正後、同一のstep予算または同一epoch数で4条件を完走させる。少なくとも1 epoch、可能なら設定どおり3 epochまで確認する。
4. 最終checkpoint群をdev MCCのみで選択し、test推論は選択後に行う運用へ固定する。
5. 主比較を事前に決めてpaired bootstrapを行い、複数比較を報告する場合はHolm法などで補正する。
6. 計算資源が許せば複数seedを実行し、seed間平均と分散を報告する。
7. 完走後の確定値で既存の `report.md` のエグゼクティブサマリと WMT22 節を更新する。

## 15. 根拠ファイル

GRPO labels:

- `ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende/0_868935_nqsv/summary.tsv`
- `ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende/0_868935_nqsv/summary_best_dev.tsv`
- `ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende/0_868943_nqsv/summary.tsv`
- `ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende/0_868943_nqsv/summary_best_dev.tsv`

GRPO XML:

- `ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende_xml_mt/0_889081_nqsv/summary.tsv`
- `ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende_xml_mt/0_889081_nqsv/summary_best_dev.tsv`
- `ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende_xml_mt/0_889207_nqsv/summary.tsv`
- `ckpts/verl_grpo_qwen3_8b_qe_wmt22_ende_xml_mt/0_889207_nqsv/summary_best_dev.tsv`

SFT:

- `/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/output/Qwen3-8B-Base_labels_5epoch/summary.tsv`
- `/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/output/Qwen3-8B_labels_5epoch/summary.tsv`
- `/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/output/Qwen3-8B-Base_xml_mt_5epoch/summary.tsv`
- `/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/output/Qwen3-8B_xml_mt_5epoch/summary.tsv`

mDeBERTa:

- `/work/UTSUROLB/utlb_buma2/work_SFT/mDeBERTa/outputs/epoch_4/dev/result`
- `/work/UTSUROLB/utlb_buma2/work_SFT/mDeBERTa/outputs/epoch_4/test/result`
- `/work/UTSUROLB/utlb_buma2/work_SFT/mDeBERTa/outputs/epoch_4/test/predictions.tsv`

検定:

- `scripts/significance_test.py`

