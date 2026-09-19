# WMT22 En-De QE GRPO 改善計画

作成日: 2026-08-13

## 1. 目的

SFT後にGRPOを適用した以下の4系列について、testデータをモデル選択やハイパーパラメータ調整に使用せず、dev MCCで選択したモデルのtest MCCを改善する。

1. Qwen3-8B-Base / labels
2. Qwen3-8B / labels
3. Qwen3-8B-Base / xml_mt
4. Qwen3-8B / xml_mt

現在の結果と詳細な事実確認は `report_wmt22_grpo.md` を参照する。

## 2. 現在の基準値

論文で報告する主結果は、各runのtest最大値ではなく、dev MCCで選択したcheckpointのtest MCCとする。

| モデル | 出力形式 | dev選択step | dev MCC | test MCC | test上の最大MCC（参考のみ） |
|---|---|---:|---:|---:|---:|
| Qwen3-8B-Base | labels | 2200 | 0.2123 | 0.1992 | 0.2152 |
| Qwen3-8B | labels | 1400 | 0.2070 | 0.2126 | 0.2389 |
| Qwen3-8B-Base | xml_mt | 2800 | 0.2219 | 0.2431 | 0.2431 |
| Qwen3-8B | xml_mt | 2600 | 0.2323 | 0.2521 | 0.2788 |

test上の最大値は分析用の参考値に限定し、設定選択、checkpoint選択、閾値調整には使用しない。

## 3. 実験上の共通ルール

- 変更は原則として一度に1要因とし、現在の設定をcontrolとして残す。
- pilotの選択はdev MCCを主指標とする。
- 同点に近い場合は、dev F1-BAD、複数checkpointでの安定性、生成形式成功率、KLの順に確認する。
- 最終test評価は設定とcheckpointをdevで固定した後に行う。
- 本実験では可能なら3 seedを実行し、平均、標準偏差、各seedの値を報告する。
- testの最大checkpointを選ぶ結果は、主結果に含めない。
- 全runで、設定ファイル、Git commit、seed、元SFT checkpoint、ジョブIDを記録する。

## 4. 優先度P0: 現checkpointのdev較正

再学習の前に、現在のcheckpointから得られる改善幅を確認する。

### labels

- [ ] dev上でOK/BADのlogit biasまたはBAD判定閾値を探索する。
- [ ] 目的関数はdev MCCとする。
- [ ] devで固定した値をtestに一度だけ適用する。
- [ ] 通常推論と較正推論について、MCC、F1-BAD、F1-OK、予測BAD率を比較する。

### xml_mt

- [ ] dev上で`<e>`開始確率に対するbiasまたはpenaltyを探索する。
- [ ] 必要なら一文当たりの予測エラー数に対するpenaltyも比較する。
- [ ] XMLのcopy exact率とタグ対応率を低下させないことを確認する。

### 背景

trainのBAD token率は約10.02%だが、dev/testは約3.86%である。GRPO後はBAD recallが上がる一方、全OK文に対するfalse positiveも増えている。再学習を伴わないdev較正は、この事前分布差を補正する最も低コストな候補である。

## 5. 優先度P0: BaseモデルのEOS修正

2026-09-06更新: EOS整合処理を `scripts/prepare_qe_eos.py` に実装し、
`scripts/run_grpo_qe.sh` の学習起動前に組み込んだ。既存checkpointは保存したまま、
必要な場合だけ修正版メタデータと重みへのリンクを持つ別ディレクトリを使用する。
CPUでの停止動作テストと実checkpointのメタデータ確認は完了した。
実8BモデルのGPU rolloutと精度の再評価は未実施。詳細は [eos_fix_notes.md](eos_fix_notes.md)。

Base 2系列では、rollout応答長がlabelsで256、xml_mtで448の上限にほぼ常時張り付いている。

原因候補は次の不整合である。

- Base tokenizer/modelのEOS: `<|endoftext|>`、ID 151643
- SFT chat templateのassistant終端: `<|im_end|>`、ID 151645

### 実施項目

- [x] Baseモデルの設定に`151645`を主EOSとして登録する（実GPU rolloutは未検証）。
- [x] generation設定で`151645`に加えて既存の`151643`も終了候補に保持する。
- [x] 修正版モデルのtokenizer/model/generation設定と新規GRPO checkpointへEOS設定を引き継ぐ経路を整備する。
- [ ] 実GPUで学習rolloutとgreedy dev推論の停止動作を比較する。
- [ ] 16～64 promptの小規模rolloutで、末尾の繰り返しが消えることを確認する。
- [ ] 平均・中央値・最大response lengthを記録する。
- [ ] labelsでは予測ラベル数、xml_mtではcopy exact率とタグ対応率を確認する。

### 合格条件

- Base出力が常時最大長になる状態が解消される。
- `<|im_end|>`より後ろの不要生成が消える。
- 形式成功率が現状から大きく低下しない。
- NaN、異常なKL、極端なreward低下がない。

### その後の扱い

Base labelsとBase xml_mtは、旧GRPO checkpointをそのまま長く学習するより、EOS修正版を用いて元SFT checkpointからcontrolと同条件の短い再実験を行う。EOS修正前後を同じstep、seed、推論条件で比較する。

## 6. 優先度P1: 報酬とdev MCCの整合

現在の`token_mix`は次の構成である。

```text
0.8 * weighted_token_accuracy(w_bad=2.5, w_ok=1.0)
+ 0.2 * bad_f1_safe
```

保存checkpointを比較すると、現在の報酬とdev MCCの相関はrunによって弱く、Base xml_mtでは逆相関になっている。一方、`f1_macro`は4系列で比較的安定して正の相関を示した。

### Pilot A: 報酬関数

各形式について、400～800 step、保存間隔200 step以下で以下を比較する。

| 設定 | reward |
|---|---|
| A0: control | 現在の`token_mix`（0.8/0.2） |
| A1 | `metric=f1_macro` |
| A2 | `token_mix`（0.6/0.4） |

- [ ] BaseについてはEOS修正後に実施する。
- [ ] 各保存checkpointでdev MCCを計算する。
- [ ] 平均rewardだけでなく、dev MCCとのcheckpoint間相関を確認する。
- [ ] 全OK文の完全正解率、BADを含む文の検出率、予測BAD率を記録する。
- [ ] XMLではcopy exact率とタグ対応率を記録する。

### 判定

- dev MCCの最大値だけでなく、隣接する3 checkpointの平均も比較する。
- 単一checkpointだけが高い設定は、安定性が低いものとして扱う。
- sentence-level MCC単独報酬は、全OK文での定義が不安定なため、そのまま採用しない。

## 7. 優先度P1: 探索量の改善

全runでentropyが学習開始時から終盤にかけて大きく低下している。現在の主設定はtemperature 0.5、top-p 0.9、entropy coefficient 0、rollout n=8である。

### 先に追加するログ

- [ ] promptごとのunique completion数
- [ ] promptごとのreward標準偏差
- [ ] advantageがほぼゼロとなるpromptの割合
- [ ] response lengthの分布
- [ ] promptごとのOK/BAD予測列の重複率

### Pilot B: sampling

報酬をPilot Aの勝者に固定し、以下を一度に1要因ずつ比較する。

| 設定 | temperature | top-p | entropy coefficient |
|---|---:|---:|---:|
| B0: control | 0.5 | 0.9 | 0 |
| B1 | 0.7 | 0.9 | 0 |
| B2 | 0.7 | 0.95 | 0 |
| B3 | B1またはB2の勝者 | 同左 | 小さい正値 |

- [ ] 形式成功率が保たれる範囲で、group内の報酬分散が増えるか確認する。
- [ ] n=8のcompletionがほぼ重複する場合のみ、n=4を比較する。
- [ ] nを下げた場合は、同じ計算量でより多くのpromptを処理した結果も比較する。

## 8. 優先度P1: PPO更新効率

現在は`ppo_epochs=1`、`shuffle=false`、`ppo_mini_batch_size=16`、rollout n=8である。VERL内部ではmini-batch sizeにnが掛かるため、実効mini-batchは128となり、1 rollout batchに対して全体更新を1回だけ行っている。

### Pilot C

| 設定 | ppo_epochs | ppo_mini_batch_size | shuffle |
|---|---:|---:|---|
| C0: control | 1 | 16 | false |
| C1 | 2 | 8 | true |

- [ ] 同じ生成prompt数または同じwall timeでdev MCCを比較する。
- [ ] KL、gradient norm、clip fraction、entropyを監視する。
- [ ] 更新が強すぎる場合はLRを下げるか、ppo_epochsを1に戻す。

PPO設定は報酬とsamplingの勝者を決めた後に比較し、複数要因を同時変更しない。

## 9. 優先度P2: LR、停止条件、checkpoint選択

現在はconstant LR 1e-6、warmupなしである。Qwen labelsとQwen xml_mtはdev MCCが途中でピークに達した後に低下しており、現設定のまま長く回す根拠は弱い。

- [ ] constant LR 1e-6をcontrolとして残す。
- [ ] 後半に5e-7以下へ下げるscheduleを比較する。
- [ ] 100～200 stepごとにdev評価する。
- [ ] 隣接3 checkpointのdev MCC移動平均を記録する。
- [ ] checkpoint averagingまたはEMAを比較し、採否をdevで決める。
- [ ] early stoppingのpatienceを事前に固定する。

Base xml_mtはstep 2800までdevが改善しているため、4系列の中では追加学習の余地が最も大きい。ただし、EOS修正後、かつ低いLRで確認する。

## 10. 優先度P2: SFTデータと初期checkpointの改善

現在の主な誤りには次が含まれる。

- 句読点のBADをほとんど検出できない。
- 固有名詞・大文字始まりの正しいtokenをBADと誤検出する。
- `&quot;`、`&apos;`などの表記を誤検出する。
- 軽微な文法誤りや機能語のBADを見逃す。
- labels系とBase xml_mtのSFT初期モデルは、BADを過少予測する傾向がある。

### 実施項目

- [ ] train内から句読点BADを含む文をhard positiveとして抽出する。
- [ ] 正しい句読点、固有名詞、HTML entityをhard negativeとして追加する。
- [ ] EOS欠落例と軽微な文法誤りを分類して追加する。
- [ ] dev分布に近いサンプリングまたはloss weightingを比較する。
- [ ] labelsとBase xml_mtでは`bad_loss_weight=1.5～2.0`をpilotする。
- [ ] Qwen xml_mtは既にBADを多めに予測するため、同じ重みを一律に適用しない。

BAD例の単純なoversamplingは、trainとdev/testのBAD率の差をさらに広げる可能性がある。hard positiveと同時にhard negativeを入れ、dev MCCと予測BAD率の両方で判断する。

## 11. run別の優先順位

### Qwen3-8B-Base / labels

1. EOS修正
2. 元SFT checkpointから短い再学習
3. `f1_macro`または0.6/0.4の`token_mix`
4. BAD較正
5. 句読点BADと正しい固有名詞のhard example追加

旧runを同じ設定で延長することは優先しない。

### Qwen3-8B / labels

1. devによるBAD閾値・logit bias較正
2. `f1_macro`報酬
3. LR decayと早期停止
4. 全OK文に対するfalse positive抑制

devピークが早いため、現設定のままstep数だけ増やさない。

### Qwen3-8B-Base / xml_mt

1. EOS修正
2. `f1_macro`報酬
3. 低LRで追加学習を比較
4. KLを継続監視

4系列の中では追加学習候補だが、現在の終盤KLが相対的に大きいため、EOS修正とLR低下を先に行う。

### Qwen3-8B / xml_mt

1. dev較正
2. step 2000～2800周辺のcheckpoint averaging
3. `f1_macro`または0.6/0.4の`token_mix`
4. LR decayと早期停止

現在の正式なtest MCCが最も高い系列であり、単純な学習延長より、選択安定化と誤検出較正を優先する。

## 12. 本実験の実行条件

現在の3 epochsは約11,220 stepであり、24時間制限では完走できない。実測では約2,900～3,800 stepで停止している。

- [ ] 本実験は`total_steps=2500～3000`を基本にする。
- [ ] 3 epochsを完走する場合は、必要時間を確保するか正確なresumeを使用する。
- [ ] `SAVE_CONTENTS=[hf_model]`だけでなく、optimizerとextra stateも保存する。
- [ ] 最新のfull checkpointは最低1つ残す。
- [ ] 保存・評価時間を含めてwall time内に終了するstep数を設定する。

## 13. 推奨する実行順

1. 現checkpointのdev較正を実施する。
2. BaseのEOS停止条件を修正し、小規模生成で検証する。
3. Pilot Aで報酬を選ぶ。
4. Pilot Bでsampling設定を選ぶ。
5. Pilot CでPPO更新回数を比較する。
6. LR scheduleとcheckpoint averagingを比較する。
7. 各系列の最良設定を2,500～3,000 step、3 seedで実行する。
8. dev MCC平均と安定性で設定を固定する。
9. 固定した各seed/checkpointをtestで一度だけ評価する。
10. paired bootstrapでSFT、mDeBERTa、GRPO間の有意差を検定する。

## 14. 各実験で保存する記録

| 項目 | 内容 |
|---|---|
| 実験識別 | project、experiment name、job ID、seed |
| 初期モデル | SFT形式、checkpoint、dev選択方法 |
| GRPO設定 | reward、LR、temperature、top-p、n、PPO設定、EOS |
| 学習状況 | 最終step、wall time、NaN有無、停止理由 |
| 学習指標 | reward、entropy、KL、gradient norm、response length |
| 生成多様性 | unique completion率、group reward標準偏差 |
| dev | MCC、F1-BAD、F1-OK、F1-product、F1-macro |
| エラー傾向 | 予測BAD率、全OK文false positive率、BAD文検出率 |
| 形式 | labels完全一致率、XML copy exact率、タグ対応率 |
| test | devで全設定を固定した後の最終値のみ |

## 15. 完了条件

- [ ] Baseの最大長生成問題が解消されている。
- [ ] 報酬とdev MCCが少なくとも負の相関ではない。
- [ ] 選択した設定が複数checkpointまたは複数seedで安定している。
- [ ] XML/labelsの形式成功率を維持している。
- [ ] 各モデルの選択方法がtest情報から独立している。
- [ ] 3 seedのdev選択済みtest MCCと有意差検定を報告できる。

## 16. 優先しないこと

- 現在の設定のまま3 epochsまで延長する。
- test MCC最大のcheckpointを採用する。
- 既にほぼ100%の出力形式だけをさらに強く報酬化する。
- BAD例だけを無条件にoversamplingする。
- reward、sampling、PPO、LRを同時に変更して原因を判別不能にする。
