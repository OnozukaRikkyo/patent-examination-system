# プロジェクト構成

```
patent-similarity-search/
│
├── patent_similarity_search.py    # メインスクリプト
├── test_utils.py                  # テスト・デバッグユーティリティ
│
├── requirements.txt               # Python依存パッケージ
├── .env.example                   # 環境変数のサンプル
├── .gitignore                     # Git除外設定
│
├── sample_patent.xml              # サンプルXMLファイル
│
├── README.md                      # 詳細ドキュメント
├── QUICKSTART.md                  # クイックスタートガイド
├── TECHNICAL.md                   # 技術詳細ドキュメント
└── PROJECT_STRUCTURE.md           # このファイル
```

## ファイル説明

### 実行ファイル

#### `patent_similarity_search.py`
メインプログラム。以下のクラスを含む：

- **PatentXMLParser**: 特許XMLをパースして情報を抽出
- **BigQueryPatentSearcher**: BigQueryから特許データとembeddingを取得
- **SimilarityCalculator**: コサイン類似度を計算
- **PatentSimilaritySearchSystem**: 全体を統合するメインクラス

**使用例:**
```bash
python patent_similarity_search.py sample_patent.xml \
  --project-id YOUR_PROJECT_ID \
  --top-k 1000 \
  --output results.csv
```

#### `test_utils.py`
デバッグ・テスト用のユーティリティスクリプト

**機能:**
- XMLパーサーのテスト
- BigQuery接続テスト
- Embedding取得テスト
- 候補検索テスト
- サンプル特許の検索

**使用例:**
```bash
python test_utils.py --test all --project-id YOUR_PROJECT_ID
```

### 設定ファイル

#### `requirements.txt`
必要なPythonパッケージのリスト

#### `.env.example`
環境変数の設定例。`.env`にコピーして使用

#### `.gitignore`
Gitで管理しないファイルのパターン

### サンプルデータ

#### `sample_patent.xml`
動作確認用の日本特許XMLサンプル

### ドキュメント

#### `README.md`
詳細なドキュメント。以下を含む：
- セットアップ手順
- 使用方法
- 入出力フォーマット
- カスタマイズ方法
- トラブルシューティング

#### `QUICKSTART.md`
5分で始めるクイックスタートガイド

#### `TECHNICAL.md`
技術詳細：
- アーキテクチャ
- データフロー
- パフォーマンス最適化
- エラーハンドリング

#### `PROJECT_STRUCTURE.md`
このファイル。プロジェクト構成の説明

## セットアップから実行までの流れ

```bash
# 1. プロジェクトのクローン
git clone <repository-url>
cd patent-similarity-search

# 2. 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 依存パッケージのインストール
pip install -r requirements.txt

# 4. 環境変数の設定
cp .env.example .env
# .env を編集して GCP_PROJECT_ID を設定

# 5. Google Cloud認証
gcloud auth application-default login

# 6. テスト実行
python test_utils.py --test all --project-id YOUR_PROJECT_ID

# 7. サンプルで実行
python patent_similarity_search.py sample_patent.xml \
  --project-id YOUR_PROJECT_ID

# 8. 実際のファイルで実行
python patent_similarity_search.py your_patent.xml \
  --project-id YOUR_PROJECT_ID \
  --output my_results.csv
```

## 出力ファイル

実行すると以下のファイルが生成されます：

```
patent-similarity-search/
│
├── similar_patents.csv   # 検索結果（デフォルト名）
├── results.csv           # カスタム出力名
│
└── cache/                # キャッシュディレクトリ（将来実装）
    └── *.pkl
```

## カスタマイズポイント

### XMLパーサー
`PatentXMLParser` クラスの以下のメソッドをカスタマイズ：
- `_get_publication_number()`: 公開番号の抽出方法
- `_get_classification_codes()`: 分類コードの抽出方法
- `_get_theme_codes()`: テーマコードの抽出方法

### BigQuery検索
`BigQueryPatentSearcher.get_candidate_embeddings()` 内で：
- 分類コードのマッチング方法（先頭2文字 → より詳細）
- 候補数の制限
- 日付範囲のフィルタ

### 類似度計算
`SimilarityCalculator` で：
- 類似度の閾値設定
- バッチサイズの調整
- 並列処理の追加

## トラブルシューティング

問題が発生した場合：

1. **まず試すこと:**
   ```bash
   python test_utils.py --test all --project-id YOUR_PROJECT_ID
   ```

2. **ログを確認:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **ドキュメントを確認:**
   - [README.md](README.md) - 一般的な使い方
   - [TECHNICAL.md](TECHNICAL.md) - 技術詳細
   - [QUICKSTART.md](QUICKSTART.md) - クイックスタート

## 開発環境

推奨スペック：
- Python 3.8以上
- メモリ 8GB以上
- Google Cloud Project（BigQuery有効）

## ライセンス

MIT License

## 貢献

Issue や Pull Request を歓迎します。
