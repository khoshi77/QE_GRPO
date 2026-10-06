**継続SFT対照実験の計画（2026-10-06）**

論文・保存済み実験設定・現行コードの確認に基づく提案。以下の学習量、学習率候補、seed、実行本数は新実験の提案値であり、実験結果ではない。今回は計画の作成のみ。学習・推論ジョブは投入していない。

**時間制約を反映した縮小案（現在の推奨）**

ユーザーから、学習率探索と複数seedを行う時間がないとの指定があった。以下の縮小案を現在の提案とする。後続の1–10節に残す24/48 runsの設計は、予算がある場合の拡張案であり、実施の必須条件ではない。

学習率候補を手法ごとに1つ、seedを1つに固定しても、継続SFT対照を追加する価値はある。固定した学習設定において、追加SFTとGRPOのどちらがよいかを比較できる。学習率の適否とrun間変動は未検証として残り、「最適化されたSFTよりGRPOが一般に優れる」とは主張しない。ここでの学習率固定は探索候補を1つにする意味であり、linear/constant等のscheduleを同一にする意味ではない。

| 選択肢 | 新規C-SFT | 新規GRPO | 合計の追加学習 |
|---|---:|---:|---:|
| 時間優先の推奨: Qwen3-8Bのlabels/XML、既存GRPOを再利用 | 2 runs | 0 | **2 runs** |
| 同じ条件で両手法を新規実行 | 2 runs | 2 runs | **4 runs** |

前者は新しいC-SFTを単一seedで実行し、既存の単一GRPO runと比較する。既存GRPOのdata.seedはnullなので、「両手法で同一seed・同一入力順序を固定した実験」とは記述しない。後者ならseedと入力順序を明示して揃えられるが、seed間の頑健性は評価しない。どちらも元の第1段階SFTを学習し直す必要はない。

既存のQwen3-8B/F1-macroは、labels/XMLともに `global_step_3200/actor/huggingface` のsafetensors 7ファイルと、同stepのdev/test結果が存在することを追加確認した。したがって51,200入力提示時点のGRPOとの比較には既存artifactを利用できる。重みのロードと新しい共通evaluatorでの必要な再評価は、実行時に確認する。

縮小案でも維持する条件:

- 同じSFT₀・同じtrain/dev/test・同じ出力形式から比較し、EOSと自由生成・parserを揃える。
- C-SFTのLRは元のSFT設定 `2e-5` を出発案とし、採用理由とscheduleを実行前に固定する。これは最適値の保証ではない。GRPOは既存runの `1e-6`。両手法のLRを同じ数値にすること自体は公平性を保証しない。C-SFTのseed案は42。
- 追加入力提示数を51,200に揃える。主比較は双方のこの時点のモデル。C-SFTのglobal batch 16なら3200 steps、元の64を維持するなら800 stepsとなる。時間優先で64を使う場合は、更新回数も同じだとは記述しない。
- 副比較としてdev選択を示す場合は、同じ入力提示間隔3200件・同じ16候補に制限する。既存GRPOの3200 steps超を選択候補に混ぜない。主比較のGRPO値は、現在の論文のdev-selected値とは異なり得る。
- testを見てLRや停止位置を変更しない。MCC、BAD precision/recall、形式遵守率、tokens、GPU時間を記録する。
- paired sentence bootstrapは固定したモデル対に対するtest標本の不確実性として使えるが、単一seedの制約を解消するものではない。

2 runs案が答えるのは、同じデータ集合・追加入力提示数で、事前固定したC-SFT設定が既存GRPOにどこまで近づくかである。入力順序や歴史的実行環境まで完全統制した因果比較ではない。4 runs案はそれらの統制を改善する。まず2 runsでも、継続SFT対照が全くない現状から論文の根拠を増やせる。Baseモデルへの拡張は、必要ならC-SFTをさらに2 runs追加する案として別途扱う。

**1. 必要性と、この実験で答える問い**

強く推奨する。「GRPOを追加すると通常のSFTを続ける以上の改善が得られる」と主張するなら、実質的に必要な対照である。現論文の限定された観測結果が、この対照の欠如だけで無効になるわけではない。

現論文は、16条件すべてでGRPO後のtest MCCがSFT初期値を上回ること、報酬によって性能・誤検出傾向が異なることを報告している。一方、継続SFTとの比較がないため、追加学習による改善と、GRPOを選ぶことによる上乗せを区別できないとLimitationsに記載している。

同じ初期重み θ₀ について、次の3条件を比較する。

| 条件 | 第2段階の学習 | 役割 |
|---|---|---|
| SFT₀ | 追加学習なし | 共通の出発点 |
| SFT₀ → continued SFT（C-SFT） | 元の正解応答への教師あり学習 | 通常の追加学習でどこまで改善するか |
| SFT₀ → GRPO | 同じ入力・gold annotationを使った報酬学習 | C-SFTに対する追加の利得があるか |

主な比較量は `MCC(GRPO) − MCC(C-SFT)`。併せて各手法の `MCC − MCC(SFT₀)` を報告する。gold annotationを新たに追加する実験ではなく、既存の学習データを第2段階でも使う実験である。

入力提示数を揃えた比較が答えるのは「同じ追加入力予算で、どちらの学習手続が有効か」。生成数や計算量まで同じになるわけではない。GRPOが優位でも、GRPOの数式だけの効果、他のRL手法に対する優位、all-OK文の報酬識別力が改善の原因であることまでは証明しない。後者には別の報酬ablationが必要。

**2. 確認できた現状**

| 項目 | 保存資料・コードで確認した内容 | 計画への影響 |
|---|---|---|
| データ | train 59,855 annotation rows、dev 1,005、test 511。trainは28,330の異なるsource–MTペアを含む | annotation row単位の提示数と、異なるペア数を区別する |
| SFT | full parameter、global batch 64、初期LR 2e-5、linear decay、warmup 5%、weight decay 0、seed 42 | 継続段階の設定を独立して記録する |
| GRPO | prompt batch 16、8 rollouts/prompt、LR 1e-6、constant、weight decay 0.01、PPO epochs 1 | step数やresponse数だけを合わせない |
| 過去の学習量 | 保存済み最終stepが3200/3400/3740などで異なる | 新実験は同じ上限・同じ評価間隔にする |
| 選択基準 | offline自由生成のdev corpus MCC。GRPOのonline validationとは異なる | 全手法を同じoffline evaluatorで選ぶ |
| seed | 保存GRPO設定の `data.seed` はnull | 過去runの入力順序を同じseedで再現できると仮定しない |
| EOS | Base SFTはEOSを151645に揃えて再評価済み。予測ラベルは不変 | 新しい両分岐も同じEOS・tokenizerから始める |
| 既存SFTの後続モデル | 4条件とも選択checkpoint後から4680までのcheckpointが残る | 旧スケジュールの参考曲線には使えるが、新しい対照の代用にはならない |

たとえばQwen3-8B/XMLでは、現在の表のSFT MCCは0.1620、GRPO/F1-macroは0.2632である。ただし、この差0.1012がC-SFTに対して残るかは未確認。Base/labelsでは、既存SFTのdev MCC 0.2348が、既存GRPOの最良dev MCC 0.2324を上回る。このため、「学習後の比較」と「devで追加学習自体を採用するか」は分けて検討する。

根拠: [設定節](paper/sections/setup.tex)、[Limitations](paper/sections/limitations.tex)、[主結果表](paper/tables/main_results.tex)、[選択checkpoint表](paper/tables/checkpoints.tex)、[監査所見](paper/audit/findings.md)、[保存GRPO設定](paper/audit/grpo_settings.json)、[保存SFT設定](paper/audit/sft_training_args.json)。保存設定は過去runの資料、現行コードは今後の実装計画の資料として区別した。

**3. 実験対象と優先順位**

論文全体を補強する推奨範囲は、2モデル × 2出力形式の4組。各組にC-SFTを1種類設ければ、同じ組の複数報酬に対して対照を共有できる。報酬ごとに同じC-SFTを重複学習する必要はない。

| モデル | 形式 | 共通の出発checkpoint |
|---|---|---|
| Qwen3-8B | labels | `Qwen3-8B_labels_5epoch/checkpoint-2750` |
| Qwen3-8B | xml_mt | `Qwen3-8B_xml_mt_5epoch/checkpoint-2000` |
| Qwen3-8B-Base | labels | `Qwen3-8B-Base_labels_5epoch/checkpoint-3250` |
| Qwen3-8B-Base | xml_mt | `Qwen3-8B-Base_xml_mt_5epoch/checkpoint-3250` |

上記は `/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/output/` 以下にあり、ディレクトリとsafetensorsファイルの存在を確認した。今回は重みのロード検証までは行っていない。

最初のGRPO報酬は全組でF1-macroに固定する。これは報酬設計の中心的な比較を絞って検証する選択であり、既存test結果を見た後の追実験であることは明示する。途中でtest成績がよい報酬へ切り替えない。

予算が限られる場合はQwen3-8Bのlabels/XMLの2組から着手する。この範囲だけなら結論もそのモデル・形式に限定する。4組への拡張は計算予算と論文の主張範囲で決め、都合のよいtest結果を得るまで条件を追加しない。

Token-mixは次の優先度。両方の強い報酬についてC-SFTとの差を示したい場合に追加する。F1-product/MCCの全条件再実行は、この対照実験の第一段階には要求しない。F1-macroだけの再比較から、16条件すべてのGRPO固有の利得を主張することはできない。

**4. 揃える条件と追加学習量**

両分岐で、初期重み、tokenizer/chat template、出力形式、train/dev/test、gold annotation、入力順序、推論・正規化処理を揃える。full parameter、bf16、BAD/XML tag loss weight 1.0を維持する。SFTではpromptをlossからmaskし、正解応答とassistant終端を教師にする。XMLの正解には元のMTのコピーを含める。

初期checkpointから重みを読み込み、両分岐でoptimizer・scheduler・第2段階stepを新規作成する。C-SFTだけ旧optimizer状態を引き継ぐ比較にはしない。これは「重みを継続して学習する」定義であり、元の5epochジョブを中断位置から完全再開する定義とは区別する。

Transformersの `resume_from_checkpoint` はmodelに加えてoptimizer/scheduler状態も復元するため、分岐開始ではこの機能を使わず、重みのロードと新規Trainerを使う。同じ第2段階runが中断した際の再開には、当該runの完全な状態復元を使う。[公式Trainer仕様](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.train)

提案する共通上限は **51,200回の追加入力提示**。元のtrainの約0.855周分であり、GRPOの3200 steps × 16 promptsに相当する。既存の全GRPO条件の保存範囲に収まる長さとして設定し、testで最もよかったstepに合わせたものではない。この予算での結論を、任意の長さの継続学習へ一般化しない。

| 項目 | C-SFT | GRPO |
|---|---:|---:|
| 1更新に供給するannotation rows | 16 | 16 |
| 第2段階の予定steps | 3200 | 3200 |
| 入力提示数 E | 51,200 | 51,200 |
| 1入力あたりの応答 | gold target 1本 | sampled response 8本 |
| target/generated sequences総数 | 51,200 | 409,600 |
| 保存・offline dev評価 | 200 stepsごと | 200 stepsごと |
| 学習後の評価候補 | 16 checkpoints | 16 checkpoints |

GRPOは既存のPPO mini-batch 16 prompts・PPO epochs 1を維持する。trainer stepと実optimizer更新回数が一致することは短時間検証で確認し、双方の実更新回数も記録する。

C-SFTのglobal batchは元の64から16に変更し、主比較では入力batchと更新機会も揃える。8 GPUsならper-device batch 2、gradient accumulation 1が候補。既存batch 64を保持する補助比較なら、同じEに対応するのは800 steps、評価間隔は50 stepsになる。この補助比較では更新回数が違うことを明記する。

Eはdataloaderから取り出したannotation rowの延べ数で、GRPOの8応答への展開前に数える。同じsource–MTペアに複数のannotationがあるため、51,200種類の異なる文を見たという意味ではない。seedごとのrow ID列を事前生成し、各手法・各LR試行で同じ列を使う。異なるframeworkに同じseedを渡すだけでは同一順序の保証にならないため、実際に消費したrow IDを照合する。

GRPOで8本生成することを理由にC-SFTの入力提示数を8倍にすると、この主比較の定義から外れる。8倍反復SFTは必要なら別の補助条件として扱い、それ自体を計算量一致と呼ばない。

**5. 学習率探索とseed**

双方に同じ3候補・同じ入力予算・同じ16回のdev評価を与える。学習率を同じ値にすることよりも、各手法が適切な学習率を選ぶ機会を揃えることを優先する。

| 項目 | C-SFT | GRPO |
|---|---|---|
| 初期LR候補 | `{1e-6, 5e-6, 2e-5}` | `{3e-7, 1e-6, 3e-6}` |
| schedule | 元SFTに合わせlinear、warmup 5% | 既存GRPOに合わせconstant、warmupなし |
| optimizer | AdamW、β=(0.9, 0.999)、clip norm 1.0 | 同左 |
| weight decay | 元SFTに合わせ0 | 既存GRPOに合わせ0.01 |
| KL | なし | 元のSFT₀参照に対し0.001 |
| 探索用seed | 42 | 42 |
| 確認用seed | 101, 102, 103 | 101, 102, 103 |

LR候補は提案値。候補端が最良となった際に探索範囲を広げるなら、確認用runへ進む前に、testを使わず、両手法に同数の追加試行枠を与える。追加探索の回数・GPU時間も公開する。scheduleとweight decayを各手法の既存設定にするため、結論は第2段階の学習手続の比較であり、lossだけを変えたablationではない。loss単独に近い効果を論じるなら、schedule・weight decayも共通化する補助比較を追加する。

探索seedでは、主評価用のLRを **E=51,200時点のdev corpus MCC** で選ぶ。これにより主比較は固定入力数で揃う。同点はfull precisionの値で判定し、それでも同じなら小さいLRを選ぶ。途中checkpointの最高値をLR選択に使う補助分析を行う場合は、主評価とは別の選択規則として記録する。

選んだLRを固定し、独立した3つの確認用seedで両手法を再実行する。探索seedの結果は最終の3-seed平均に混ぜない。全seedで同じSFT₀を使うため、測るのはその初期checkpointを条件とした第2段階のばらつきである。第1段階SFT自体のseed変動まで評価したという主張はしない。

**6. 評価とcheckpoint選択**

主評価は確認用3 seedsの **E=51,200時点** のtest MCC差。これが「実際の追加入力提示数を揃えた比較」に対応する。副評価として、同じ16 checkpointsからdev MCC最大を選んだモデルも評価し、「同じ上限・探索機会の中で得られる性能」として報告する。副評価の選択stepは手法ごとに異なり得るので、実際の入力数まで一致したと記述しない。

さらにSFT₀（E=0）のdev結果を共通で測り、追加学習なしを選択肢に含めた採用判断も示す。SFT₀が最高なら「dev基準では追加学習を採用しなかった」と報告する。これは固定予算で学習したモデル同士の比較を置き換えるものではない。checkpoint同点では早いstepを採用する。

評価は次を固定する。

- greedy自由生成、`do_sample=False`、thinking無効。学習rolloutのtemperature 0.5/top-p 0.9とは区別する。
- labelsは最大256 new tokens、XMLは448。推論で指定するEOSは151645。データ末尾の文字列 `<EOS>` は評価位置として維持する。
- promptのtoken ID、mask、PAD/EOS、長さ上限を両分岐で照合する。正解応答が学習前処理で切れていないことも確認し、片方だけの行除外やsilent truncationを認めない。
- 全手法で同じparser・gold alignmentを使い、corpus全体の混同行列からMCCを再計算する。teacher-forced MCC、XML validation loss、GRPOの文平均rewardで最終モデルを選ばない。
- 主指標MCCに加え、BAD F1、BAD precision/recall、予測BAD率、raw形式遵守率を保存する。all-OK文のfalse-positive率は報酬分析の補助指標とする。
- LR探索とcheckpoint選択はdevのみ。testは設定と選択規則を固定してから、主評価の最終checkpointと副評価の選択checkpointに限定して評価する。全stepのtest sweepは行わない。

最終報告ではseedごとのスコア・対応する差、3-seed平均とSDを示す。10,000回の文単位paired bootstrapは既存の集計方法を再利用し、同じ文IDを両手法で再標本化する。3-seed平均差の文bootstrapを計算する場合も、同じ再標本化文集合を全seed対に適用し、各seedのcorpus MCC差を求めて平均する。これは固定した3組のモデルに対するtest標本の不確実性であり、training-seedの母集団に対する精密な区間ではない。seed差は別途示し、3 runsだけで高精度のseed推定を主張しない。

「有意差なし」と「同等」を区別する。実用上の同等性まで論じたい場合は、許容MCC差δを結果を見る前に理由とともに決め、対応する同等性評価を行う。例として±0.01を検討できるが、確定した閾値ではない。区間が広ければ「同等」ではなく「結論不十分」とする。4組すべての有意な優位を主張する場合は、事前指定した比較群の多重比較補正も行う。

**7. 計算コストの記録と、計算予算を揃える拡張**

最初の比較から、以下をrun・checkpoint単位で記録する。

| 記録 | 定義・注意 |
|---|---|
| 入力提示数 | row IDの延べ数、異なるrow数・source–MTペア数も別記 |
| training tokens | paddingを除くprompt/response tokens。SFT gold targetとGRPO生成応答を区別 |
| loss-bearing tokens | 実際にloss maskへ入ったtarget/response tokens。再利用があるならその回数も記録 |
| GRPO処理 | rollout生成tokens、actor更新tokens、reference/log-prob計算を分けて記録 |
| GPU時間 | GPU台数×実測経過時間。学習本体、初期化、保存、dev/test推論を区別 |
| ハードウェア等 | GPU型・台数・VRAM、ソフトウェア版、job ID、コード・設定・データhash |
| 探索全体 | 採用runだけでなく、全LR候補、失敗run、再実行を含むコスト |

GRPO学習時間にはrollout、reward計算待ち、reference/old-policy log-prob、actor更新を含める。GPU確保中の待ち時間を都合よく除外しない。queue待機時間はGPU時間に含めず別記する。token数だけからFLOPs同等や計算効率を推論しない。

計算効率を論文で主張しないなら、コストの記録・開示まででよい。同じH100環境での計算予算当たり性能も主張する場合は、次を追加する。

1. 短時間pilotから、各形式で共通のGPU-hour上限Hと評価時点を事前設定する。test性能からHを決めない。
2. C-SFTとGRPOを同じHまで学習する。C-SFTが速ければ、より多くの入力を反復できる。両者の実Eも併記する。
3. 主比較と同様にLR探索機会を揃え、H時点のモデルとH以内のdev選択モデルを区別する。必要ならその予算で再探索し、短い予算で不利になったSFT設定だけを使わない。
4. MCC対GPU-hoursの曲線と、同一予算での性能差を報告する。単一予算しか測らないなら、その予算での結論に限定する。

これは実装・ハードウェアに依存したGPU時間による比較である。理論的なFLOPs効率まで主張するには、各処理を含むFLOPs見積もりまたは計測を追加する。初期SFTの共通費用は第2段階の比較では除外できるが、モデル構築全体の費用を示す場合には含める。

**8. 実行順序と規模**

| 段階 | 内容 | 学習run数（F1-macroのみ） |
|---|---|---:|
| 0 | 設定snapshot、データ・初期checkpoint確認、入力順序と評価実装の準備 | 0 |
| 1 | 各形式・手法の短時間pilot。50–100 steps程度でthroughput・メモリ・保存/再開を確認 | 本実験と別計上 |
| 2 | Qwen3-8Bの2形式 × 2手法 × 3 LRs、探索seed 42 | 12 |
| 3 | 同じ2形式 × 2手法 × 独立3 seeds、選択LR固定 | 12（累計24） |
| 4 | Baseの2形式にも同じ探索・確認を実施 | 24（累計48） |
| 5 | 必要なら4組でToken-mixを追加。C-SFTは共有 | 24追加（累計72） |

pilotだけでは性能を判断しない。2形式の探索12 runsは動向把握用であり、独立seedによる確認を伴う本結果とは区別する。論文の4組全体を対象とする推奨完了範囲は段階4。費用優先なら段階3までの範囲を事前に選び、その範囲で主張する。

実所要GPU時間は未計測なので固定値を置かない。段階4の概算は、1 run平均をC-SFTでh_S GPU-hours、GRPOでh_G GPU-hoursとすれば、`24 h_S + 24 h_G + 評価・pilot費用`。形式間で速度が違う場合は形式別に積算する。1 runあたり16 checkpointsの8B重みを保存すると容量も大きいため、保存・dev評価の所要時間と使用量をpilotで見積もる。モデル選択に必要な評価を済ませる前にcheckpointを削除しない。

**9. 実装時に必要な作業**

現行SFTの入口は `work_SFT/QE_SFT_8B/src/models/train.py`。checkpointを `model.name_or_path` に指定して新規Trainerを作る構造は利用できる。一方、現コードは `max_steps`、明示的seed/data_seed、scheduler、weight decay等をすべて設定からTrainingArgumentsへ渡してはいない。XMLでは内部の選択基準をeval_lossへ変更するため、そのままの `best` を最終比較に使わない。

GRPOは [run_grpo_qe.sh](scripts/run_grpo_qe.sh) を基にする。現状LRは1e-6固定、保存間隔も固定なので、実験専用の設定・runnerでLR、seed、入力順序、学習上限を明示する。分散worker、rollout、dataloaderのseedと、実際のrow ID列を保存する。SFT₀の重みをKL referenceとして固定する。

既存 [run_all_checkpoints.sh](scripts/run_all_checkpoints.sh) は共有SFT YAMLを書き換え、dev/testを両方評価するため、新実験の一括評価にはそのまま使わない。[EOS再評価runner](scripts/reevaluate_sft_qe_eos.py) の設定snapshot方式を参考に、run固有YAMLでdevのみ評価し、選択確定後にtestを評価できるrunnerを用意する。

実行用コード・設定は `work_grpo` 内に追加し、元のSFT重みや共有設定は保持する。SFTの外部コードを利用する場合はsnapshotと依存環境を記録する。新規runには衝突しないIDと出力先を割り当てる。中断再開に備え、評価用HF重みだけでなくoptimizer/scheduler/RNG/dataloader状態を適切な間隔で保存する。重みだけから再開したrunは完全継続として合算しない。

学習前の受入確認は、(a) 両分岐のstep 0重み・tokenizer・dev推論条件の一致、(b) 同じ入力列と正しいloss mask、(c) 予定Eと実Eの一致、(d) dev-onlyの選択、(e) short runの保存・再開とtoken/GPU時間カウンタの整合性。既存論文表へ反映するのは、最終predictionと評価の監査後とする。

**10. 結果によって論文で言えること**

| 観測結果 | 支持される結論 | 解釈上の範囲 |
|---|---|---|
| GRPO > C-SFT > SFT₀ | 通常の追加SFTにも効果があり、GRPOにさらに利得がある | 検証したモデル・形式・入力予算での結果 |
| GRPO > C-SFT、C-SFTはSFT₀付近 | 選んだGRPO手続は同量の通常追加SFTより有効 | LR探索・seedの不確実性を伴う |
| GRPOとC-SFTが近く、両方がSFT₀を上回る | 元の改善は通常の追加SFTでも相当部分を得られる可能性 | 差が不明確なら「同等」と断定しない |
| C-SFT > GRPO | この設定でGRPOを優先する根拠は弱い | 報酬間の比較や報酬の数学的性質は別の知見として残る |
| 形式・モデル・seedで結果が変わる | GRPOの追加利得には条件依存性がある | 都合のよい組だけで一般化しない |
| 入力数一致ではGRPO優位、GPU時間一致ではC-SFTが追いつく | 入力提示予算ではGRPOが有利だが、その計算予算での優位は確認できない | データ提示効率と計算効率を区別 |
| 区間が広い・seed差が大きい | 現在の規模では優劣が決まらない | 同等性や一貫した改善を主張しない |

論文には、各組のSFT₀/C-SFT/GRPOを並べた表、固定EでのΔMCCとseed変動、選択stepと実E、dev MCC対Eの曲線、費用表を追加する。既存の16条件表の過去runを新しい統制比較と混同しないよう、新実験の表として分ける。計算予算を揃えた実験を実施した場合だけ、MCC対GPU時間も追加する。

今回確認した範囲は論文本文・付録・監査JSON、選択checkpointの所在、SFT/GRPOの学習と前処理・評価コード。学習再現、GPU benchmark、全weight shardのロード、過去runの完全な入力順序復元は行っていない。
