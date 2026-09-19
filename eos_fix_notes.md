# Qwen3 QE: BaseモデルのEOS修正

更新日: 2026-09-06

## 確認できた事実

対象はlabelsのrun `0_868935_nqsv` / `0_868943_nqsv` と、xml_mtのrun
`0_889081_nqsv` / `0_889207_nqsv`。

元SFT checkpointのtokenizerを実際に読み込み、SFTと同じ
`apply_chat_template(..., enable_thinking=False)` を適用した。
4モデルともassistant回答末尾は `<|im_end|>\n` となる。
`src/models/data_utils.py:prepare_dataset` は回答部分を教師ラベルに残すため、
長さ上限で切られていない例ではこの `<|im_end|>` もSFTの学習対象になる。

| 系列 | SFT初期checkpoint | tokenizer / model EOS | generation EOS |
|---|---:|---|---|
| Base labels | 4250 | 151643 (`<\|endoftext\|>`) | 151643 |
| Chat labels | 4250 | 151645 (`<\|im_end\|>`) | [151645, 151643] |
| Base xml_mt | 3250 | 151643 (`<\|endoftext\|>`) | 151643 |
| Chat xml_mt | 2000 | 151645 (`<\|im_end\|>`) | [151645, 151643] |

Baseでは「SFTが回答終端として教えたtoken」と「生成側が停止に使うtoken」が異なる。
元のBaseモデル用のEOSメタデータを、chat templateでSFTした後も引き継いだことが原因である。
今回のchat 2系列にはこの設定不整合は見つからなかった。

学習ログの `response_length/mean` と `response_length/clip_ratio` も確認した。

| 系列 | ログに残るstep数 | step 1の平均生成長 | 最終stepの平均生成長 | step 1 / 最終stepのclip ratio |
|---|---:|---:|---:|---|
| Base labels | 3328 | 256.000 | 256.000 | 1.0 / 1.0 |
| Chat labels | 3794 | 39.625 | 33.375 | 0.0 / 0.0 |
| Base xml_mt | 2917 | 448.000 | 448.000 | 1.0 / 1.0 |
| Chat xml_mt | 3514 | 75.719 | 70.047 | 0.0 / 0.0 |

Baseは全ログstepで平均生成長が上限に一致していた。
ただし、設定と長さログだけでは「各生成列が実際に何token目で `<|im_end|>` を出したか」
までは分からない。今回の環境にはCUDA GPUがなく、元8Bモデルのtoken列を比較する
GPU実験はまだ行っていない。EOS不整合は確認済みだが、修正後の生成長と精度の改善幅は未測定。

データ中のMT語としての `<EOS>` は、ここで扱うモデルの特殊終了tokenとは別物である。

## 実装

- `scripts/prepare_qe_eos.py`: 実tokenizerとchat templateを検証する。修正が必要な場合、
  新しいディレクトリへ `config.json`、`tokenizer_config.json`、
  `generation_config.json`（存在する場合は `special_tokens_map.json` も）の修正版を作る。
  重み等は元checkpointへのsymlinkで参照する。
- 主EOSを `<|im_end|>` に変更し、generation設定には従来のEOS候補も保持する。
  token IDの追加、重みの変更、PADの変更は行わない。
- `scripts/run_grpo_qe.sh`: GRPO起動前に上記処理を自動実行し、返されたパスを
  `actor_rollout_ref.model.path` に渡す。chatモデルが既に整合していれば元パスを使う。
- 修正版のtokenizer/configをVERLと推論側が読むため、主EOSと応答マスクの基準が揃う。
  VERLのHF checkpoint保存処理もこの設定を参照する。実際のGPU保存・再読込はsmoke testで確認する。
- `scripts/diagnose_rollout.py`: `--max-new-tokens`、`--top-p`、`--seed`、`--format`、
  `--output-json` を追加。生token ID、特殊tokenを含む生成文、停止理由、最初の
  `<|im_end|>` の後に何token続いたかを記録できる。

既存のSFT/GRPO checkpointや過去の予測結果は更新しない。
既存モデルを再評価する場合も別のモデルビューと新しい出力先を使う。

## 検証

`python -m unittest discover -s tests -v` は21件すべて成功した。
実際の4つのSFT checkpointも、一時ディレクトリで修正版を読み込んで確認した。
各形式の短いサンプルをSFTの `prepare_dataset` に通し、修正前後でprompt・教師ラベルが
同一であること、教師ラベルに151645が含まれること、PADと元メタデータ・参照重みが
変わらないことを確認した。これは全学習例の切り詰め検査ではない。

CPU回帰テストでは、小さいQwen3モデルで出力tokenを固定し、重みと出力候補を同じにして
EOS設定だけを比較した。旧設定は `<|im_end|>` を通過して上限まで生成し、修正設定は
同tokenで終了した。また、以下を確認するテストを追加した。

- 修正版をAutoTokenizer / AutoConfig / GenerationConfigで再読込できる。
- labelsとxml_mtのprompt・教師出力のtoken列が変化しない。
- 重みは元ファイルを参照し、元メタデータとPADは変わらない。
- chatモデルへの適用と修正版の再利用は変更不要として扱う。
- 既存出力先の上書き、別モデル系列、異なるassistant終端を拒否する。
- EOS後の余分なtokenを診断出力から見落とさない。

実8BのGPU rollout、dev MCCの再評価、修正設定でのGRPO学習は未実施。

## SFTを最初からやり直す必要があるか

EOSの設定不整合を直すためだけなら、SFTのやり直しは必須ではない。
既存SFTは `<|im_end|>` を教師出力に含めているため、まずそのcheckpointの停止設定を
修正し、生成を確認するのが先である。修正しても終端を出せない例が多い場合は、
SFTでの長さ切り詰めや終端tokenの学習状況を追加調査する。

一方、旧GRPOではEOS後の不要生成にも学習信号がかかった可能性がある。
EOSだけの効果を比較する本実験は、同じSFT checkpointから修正版でGRPOを再実行する。
旧GRPO checkpointの停止設定を修正して推論することもできるが、過去の更新は元に戻らない。
初期SFT checkpointの選択自体はdevに基づいて別途固定する。
停止条件が変わるとSFTのdev評価値やcheckpointの順位も変わり得るため、
本実験用の初期モデルはEOS修正後のdev評価で選び直す。

## 影響の考え方

- Baseでは全生成が上限に達し、生成とlog probability計算のコストが増えている。
  解消できるtoken数やwall time短縮率はGPU比較後に測定する。
- 回答後のtokenもresponseに含まれれば、それらにもpolicy lossやKLがかかる。
  現在のtoken-mean集約では、その分だけ必要な回答tokenへの相対的な寄与が薄まる。
- labels報酬は生成全体を解析するため、余分なラベルや文章によって長さ不一致・無効tokenの
  penaltyやラベル解釈が変わり得る。
- XML報酬は最初の非空行を評価するため、後続行の繰り返しを無視して高い報酬を返し得る。
  その場合でも後続tokenの生成・学習コストは残る。
- したがってBaseの旧labels/XML比較には停止挙動と報酬parserの違いも影響し得る。
  MCCが何点変わるか、XML優位の何割を説明するかは、現時点では断定できない。

## GPUでの確認手順（未実行）

計算ノードで `work_grpo` の仮想環境を有効にして実行する。以下はBase XMLの例。
`--output-dir` と `--output-json` は未使用のパスを指定する。

```bash
source .venv/bin/activate

python scripts/prepare_qe_eos.py \
  --model-path ../work_SFT/QE_SFT_8B/output/Qwen3-8B-Base_xml_mt_5epoch/checkpoint-3250 \
  --output-dir sft_ckpts/Qwen3-8B-Base_xml_mt_cp3250_eos_fixed

python scripts/diagnose_rollout.py \
  --model-path ../work_SFT/QE_SFT_8B/output/Qwen3-8B-Base_xml_mt_5epoch/checkpoint-3250 \
  --parquet data/qe_wmt22_en_de_xml_mt/dev.parquet \
  --format xml_mt --max-new-tokens 448 --n-examples 32 \
  --engines vllm_sample --temperature 0.5 --top-p 0.9 --seed 42 \
  --output-json eos_base_xml_before.json

python scripts/diagnose_rollout.py \
  --model-path sft_ckpts/Qwen3-8B-Base_xml_mt_cp3250_eos_fixed \
  --parquet data/qe_wmt22_en_de_xml_mt/dev.parquet \
  --format xml_mt --max-new-tokens 448 --n-examples 32 \
  --engines vllm_sample --temperature 0.5 --top-p 0.9 --seed 42 \
  --output-json eos_base_xml_after.json
```

同じ比較を `--engines hf_greedy` と `--engines vllm_greedy` でも、各々新しいJSON出力先で行う。
Base labelsでは元checkpointを `Qwen3-8B-Base_labels_5epoch/checkpoint-4250`、
parquetを `data/qe_wmt22_en_de/dev.parquet`、形式を `labels`、上限を256に変更する。

合格の目安は、`<|im_end|>` を生成した後の余分なtokenがなくなり、常時上限に達する状態が
解消されること。すべての文が必ずEOSを出すことや、clip ratioが厳密に0になることは前提にしない。
意味的な予測精度、labels数、XML copy exact率・タグ対応率もdevで確認する。

その後、通常の `qsub` 起動で短いGRPOを実行すればEOS整合処理が自動適用される。
例として、以下の20 stepジョブは最終保存したモデルのEOS設定を確認するために使える。

```bash
qsub -v QE_FORMAT=xml_mt,QE_MODEL_PATH=/work/UTSUROLB/utlb_buma2/work_SFT/QE_SFT_8B/output/Qwen3-8B-Base_xml_mt_5epoch/checkpoint-3250,QE_TOTAL_STEPS=20 scripts/submit_grpo_qe.sh
```

元checkpointの重みを参照するため、モデルビューを使用している間は元checkpointを保持する。
新しいGRPOの重みは通常の `global_step_*/actor/huggingface` に保存する。
