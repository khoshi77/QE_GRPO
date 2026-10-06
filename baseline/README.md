# GPT APIによるword-level QEベースライン

`labels` と `xml_mt`（`xml` も別名として使用可）で、既存のGRPOと同じ評価データ・プロンプト・ラベル変換を使用します。既定モデルは `gpt-5.6-luna`。GPU、verl、SFTプロジェクトは実行時に不要です。コード、設定、APIキー、データ、出力はこのディレクトリ内に置きます。

## セットアップ

```bash
cd /work/UTSUROLB/utlb_buma2/work_grpo/baseline
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

現在の作業環境では `api_key.txt` に仮のキーを配置済みです。その内容を自分のAPIキーに置き換えてください。新しいチェックアウトでは `cp api_key.example.txt api_key.txt` で作成します。環境変数 `OPENAI_API_KEY` が設定されていればそちらを優先します。キーは実験設定・ログに保存しません。`api_key.txt` はgit管理対象外です。

現在の作業環境には、既存の両形式のdev/test parquetを `data/labels/` と `data/xml_mt/` にコピー済みです。新しいチェックアウトでコピーする場合のみ、次を実行します（API呼び出しなし）。元ファイルのパスとSHA-256は `data/manifest.json` に保存します。

```bash
python3 prepare_data.py
# 別のGRPOデータディレクトリからコピーする場合
python3 prepare_data.py --source-dir /path/to/work_grpo/data
```

同名の異なるデータがある場合は上書きせず停止します。`data/`、`results/`、`.venv/` はgit管理対象外です。別の場所へ移す場合は `data/` も一緒にコピーし、そこで仮想環境を再作成してください。

## 実行

```bash
# 入力・リクエスト設定の確認のみ（APIキー不要、課金なし）
bash run_baseline.sh --format labels --dry-run
bash run_baseline.sh --format xml --dry-run

# 動作確認: dev先頭3件。通常の全件実験とは別の出力先にする
bash run_baseline.sh --format labels --split dev --limit 3 --output-dir results/smoke

# dev + test を推論して評価
bash run_baseline.sh --format labels
bash run_baseline.sh --format xml_mt

# モデル変更: 利用可能なAPIモデルIDを指定
bash run_baseline.sh --model gpt-5.6-terra --format labels

# 学習rolloutに近い設定: temperature=0.5, top_p=0.9, 8出力/文
bash run_baseline.sh --format xml_mt --profile rollout

# 同一設定での途中再開（元のオプションをすべて付ける）
bash run_baseline.sh --format labels --resume

# 保存済み応答から評価ファイルを再生成（API呼び出しなし）
bash run_baseline.sh --format labels --evaluate-only

# 実行コマンド
bash run_baseline.sh --format labels --split test
bash run_baseline.sh --format xml_mt --split test
```

`config.json` の値はCLIで上書きできます。`--split dev|test|both`、`--samples`、`--max-output-tokens`、`--temperature`、`--top-p`、`--reasoning-effort` に対応しています。パスは作業ディレクトリによらず `baseline/` 基準です（絶対パスも可）。`PYTHON=/path/to/python bash run_baseline.sh ...` でPythonを指定できます。

`samples: null` はプロファイルの既定値（eval=1、rollout=8）を使います。`max_output_tokens: null` は形式の既定値（labels=256、xml_mt=448）を使い、APIにはその数値を送ります。API側に出力上限の設定を任せる場合は `max_output_tokens: "omit"` または `--max-output-tokens omit` を指定します。この場合、リクエストに `max_output_tokens` を含めません。モデルの制約は引き続き適用され、無制限になるわけではありません。

モデルによってはreasoningやsampling設定が非対応です。その場合はAPIエラーで停止し、設定を黙って変更しません。対応状況を確認して、例えば `--reasoning-effort omit --temperature omit --top-p omit` でパラメータを省略できます。省略時はAPI既定値が適用されるので、比較条件が変わります。再実験には新しい `--output-dir` を指定してください。

## 比較条件

| 項目 | 既存設定 | このベースライン |
|---|---|---|
| 評価時の生成 | greedy / `do_sample=False`、1出力 | `eval`（既定）: temperature=0、top_p=1、1出力 |
| 学習rollout | temperature=0.5、top_p=0.9、n=8 | `rollout`: 同じ値、独立API呼び出し8回 |
| 出力上限 | labels=256、xml_mt=448 | 同じ数値の `max_output_tokens` |
| 思考 | `enable_thinking=False` | `reasoning.effort=none` |
| プロンプト | parquet内のsystem/userメッセージ | 文字列とroleをそのまま送信 |
| ラベル正規化 | 不正トークンはBAD、短い列はBAD補完、長い列は切り詰め | 同じ処理 |
| XML | 最初の非空行、`<e>`をBADへ変換、コピー崩れは単語をアライン | 同じ処理 |
| GRPO報酬 | 既定f1_macro、XMLコピー/タグ不整合は各0.25倍 | 同じ関数・設定（configのreward_kwargsで変更可） |

`run_grpo_qe.sh` が設定する0.5/0.9は**学習rollout**用です。学習中validationは `verl/verl/trainer/config/rollout/rollout.yaml` の `val_kwargs`（temperature=0、top_p=1、n=1、do_sample=False）を使い、`run_all_checkpoints.sh` が呼ぶSFTの自由生成もgreedyです。このため、評価比較には既定の `eval` を使います。

APIではローカルモデルとtokenizer・chat template・EOS・数値計算が異なり、temperature=0でも完全な再現性は保証されません。`max_output_tokens` は可視出力とreasoningトークンの合計上限です。数値が同じでも生成できる文字数は同じではありません。reasoningを有効にした実験では必要に応じて上限を増やし、別条件として保存してください。

GRPOの `max_prompt_length=768` はQwen tokenizerでのフィルタです。このベースラインは同一の文を比較するため、コピーした評価データ全件を切り詰めず送信します（API側の `truncation=disabled`）。Qwenの768 tokenフィルタは再現しません。独自データや長文を使う際は、比較対象側で除外された文がないか確認し、同じ行集合の入力を用意してください。`--input` には既存と同じverlレコード形式のparquet/JSONLを渡せます（`--split dev` または `test` も指定）。

top-k、独自EOS、ラベル語彙制約、閾値校正はAPIに適用しません。比較対象は `run_all_checkpoints.sh` の `generate` です。JSON schemaなどの追加の形式制約や追加指示も使用しません。

## 保存結果

既定の出力先は `results/<model>/<format>/<profile>/<split>/` です。

- `inputs.jsonl`: 使用した評価レコード（gold含む。goldをAPIには送りません）
- `run_config.json`: 実際のAPI設定、報酬設定、データ・コードのハッシュ
- `responses.jsonl`: 1出力ごとの生の応答、モデル名、応答ID、usage、status、所要時間
- `predictions.tsv`: `src`, `mt`, `labels`, `pred_labels`, `raw_output` 等。既存の評価・有意差検定で使用可能
- `metrics.json`: corpus MCC/F1、出力ごとの報酬の平均、形式統計、トークン使用量、不完全出力・拒否の件数
- `result.txt`: corpus MCC、F1-BAD、F1-OK、F1-macro、F1-productと平均報酬

corpus全体のMCC/F1と、GRPO validationの文単位メトリックの平均は別の値です。それぞれ `corpus_metrics` と `sentence_mean_reward_metrics` を比較してください。サンプル数が複数の場合は全出力を含むcorpus値に加え、`per_sample_corpus_metrics` と `predictions.sample_000.tsv` 等も保存します。既存の1文1行の評価・有意差検定には各sampleのTSVを使います。best-of-n選択は行いません。

APIエラーはSDKで一時的エラーを再試行し、失敗が続けば停止します。保存済みの応答は `--resume` で再利用されます。設定・データ・コードが違う場合は混在を防ぐため停止します。API成功からディスク保存までの間にプロセスが落ちた1件は、再開時に再リクエストされる可能性があります。

出力上限に達した応答や拒否も除外せず、既存の正規化規則で評価して件数を記録します。空のXMLは既存処理と同じく全OKに変換され、コピー不一致ペナルティがかかります。品質指標と併せて `empty_output_count`、`status_counts`、XML形式指標も確認してください。

## 検証と実装の出典

```bash
.venv/bin/python -m unittest discover -s . -p 'test_baseline.py' -v
```

API呼び出しをモックし、両形式、出力上限、途中再開、データ不整合、複数サンプル、元の報酬関数との一致を検証します。`qe_reward.py` と `qe_xml_utils.py` は作成時点の `../scripts/` から変更せずコピーしたものです。推論時には外部のコードをimportしません。既存GRPO側を変更した場合は比較条件も確認してください。

OpenAI公式仕様: [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)、[Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create)。実API接続・アカウントでのモデル利用可否は、APIキー設定後に確認してください。
