# 特許類似検索システム

Google Patents Public Data (BigQuery) を使用して、日本特許文献から類似特許を検索するシステムです。

## 📋 概要

このシステムは以下の処理を実行します：

1. **XMLパース**: 日本特許文献（XML形式）から公開番号、分類コード、テーマコードを抽出
2. **Embedding取得**: 対象特許のembedding_v1ベクトル（64次元）をBigQueryから取得
3. **候補抽出**: 分類コード・テーマコードの先頭2文字に基づいて候補特許を抽出
4. **類似度計算**: コサイン類似度を計算し、Top 1000の類似特許を出力

## 🚀 セットアップ

### 1. 前提条件

- Python 3.8以上
- Google Cloud Project（BigQueryアクセス権限付き）
- Google Cloud認証情報

### 2. インストール

```bash
# リポジトリのクローン（または必要ファイルのダウンロード）
cd /path/to/project

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 3. Google Cloud認証

#### 方法A: サービスアカウントキーを使用

```bash
# サービスアカウントキー（JSON）を配置
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

#### 方法B: gcloudコマンドで認証

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

## 📖 使用方法

### 基本的な使い方

```bash
python patent_similarity_search.py sample_patent.xml \
  --project-id YOUR_PROJECT_ID \
  --top-k 1000 \
  --output results.csv
```

### パラメータ説明

| パラメータ | 必須 | 説明 | デフォルト |
|-----------|------|------|-----------|
| `xml_path` | ✓ | 特許XMLファイルのパス | - |
| `--project-id` | ✓ | Google Cloud Project ID | - |
| `--top-k` | | 取得する類似特許の件数 | 1000 |
| `--output` | | 出力CSVファイル名 | similar_patents.csv |

### 実行例

```bash
# サンプルXMLで実行
python patent_similarity_search.py sample_patent.xml \
  --project-id my-gcp-project \
  --top-k 500 \
  --output qr_similar_patents.csv

# 結果の確認
head -20 qr_similar_patents.csv
```

## 📁 入力XMLフォーマット

システムは以下の日本特許XML要素を抽出します：

### 必須要素

```xml
<publication-reference>
  <document-id>
    <country>JP</country>
    <doc-number>2023123456</doc-number>
    <kind>A</kind>
  </document-id>
</publication-reference>
```

### 分類情報（いずれか）

```xml
<!-- IPC分類 -->
<classification-ipcr>
  <text>H04L9/00</text>
</classification-ipcr>

<!-- CPC分類 -->
<patent-classification>
  <classification-scheme scheme="CPC">
    <section>H</section>
    <class>04</class>
    <subclass>L</subclass>
  </classification-scheme>
</patent-classification>

<!-- FI分類 -->
<classifications-national>
  <classification-national>
    <text>H04L9/00 301</text>
  </classification-national>
</classifications-national>

<!-- Fターム -->
<f-terms>
  <f-term>5B058 KA02</f-term>
</f-terms>
```

## 📊 出力フォーマット

出力CSVファイルには以下の列が含まれます：

| 列名 | 説明 |
|------|------|
| `publication_number` | 公開番号（例: JP-2023123456-A） |
| `title` | 発明の名称 |
| `filing_date` | 出願日 |
| `country_code` | 国コード |
| `similarity_score` | 類似度スコア（0.0〜1.0） |

### 出力例

```csv
publication_number,title,filing_date,country_code,similarity_score
JP-2022123456-A,セキュアQR通信システム,2021-03-15,JP,0.9234
JP-2021098765-A,暗号化二次元コード,2020-11-20,JP,0.8976
JP-2023234567-A,モバイル認証方法,2022-05-10,JP,0.8543
...
```

## 🏗️ システム構成

### クラス構造

```
PatentSimilaritySearchSystem
├── PatentXMLParser          # XMLパース
├── BigQueryPatentSearcher   # BigQuery検索
└── SimilarityCalculator     # 類似度計算
```

### 処理フロー

```
[XML入力]
    ↓
[PatentXMLParser]
    ├─ publication_number 抽出
    ├─ 分類コード抽出
    └─ テーマコード抽出
    ↓
[BigQueryPatentSearcher]
    ├─ 対象特許のembedding取得 (1件)
    └─ 候補特許のembedding取得 (分類コードフィルタ)
    ↓
[SimilarityCalculator]
    ├─ バッチコサイン類似度計算
    └─ Top-K抽出
    ↓
[CSV出力]
```

## ⚙️ カスタマイズ

### XMLパーサーの拡張

実際のXML構造に合わせて `PatentXMLParser` クラスを修正：

```python
# patent_similarity_search.py の PatentXMLParser クラス内

def _get_publication_number(self) -> str:
    """貴社のXML構造に合わせて実装"""
    # カスタムロジックを追加
    pass
```

### 分類コードフィルタの調整

先頭2文字ではなく、より詳細なフィルタを使用する場合：

```python
# BigQueryPatentSearcher.get_candidate_embeddings() 内で変更

# 先頭3文字にする場合
prefix_3chars = set()
for code in classification_codes + theme_codes:
    if len(code) >= 3:
        prefix_3chars.add(code[:3])
```

### 類似度閾値の追加

最小類似度を設定して結果を絞り込む：

```python
# SimilarityCalculator.find_top_similar() 内で追加

# 類似度0.7以上のみ取得
result_df = result_df[result_df['similarity_score'] >= 0.7]
top_k_df = result_df.nlargest(min(top_k, len(result_df)), 'similarity_score')
```

## 🔍 トラブルシューティング

### 1. BigQueryエラー: "Permission denied"

**原因**: BigQueryへのアクセス権限がない

**解決策**:
```bash
# 認証を再実行
gcloud auth application-default login

# プロジェクトIDを確認
gcloud config get-value project
```

### 2. "対象特許のembeddingが取得できませんでした"

**原因**: 
- publication_numberが間違っている
- BigQueryに該当データが存在しない
- XMLのパースが失敗している

**解決策**:
```python
# デバッグ用にログレベルを変更
import logging
logging.basicConfig(level=logging.DEBUG)

# publication_numberを直接確認
parser = PatentXMLParser('your_file.xml')
info = parser.parse()
print(f"Parsed number: {info.publication_number}")
```

### 3. "候補特許が見つかりませんでした"

**原因**: 分類コードが取得できていない、またはマッチする特許がない

**解決策**:
```python
# 分類コードを確認
info = parser.parse()
print(f"Classification codes: {info.classification_codes}")
print(f"Theme codes: {info.theme_codes}")

# より広範な検索に変更（先頭1文字にするなど）
```

### 4. メモリ不足エラー

**原因**: 候補特許が多すぎる（10万件以上など）

**解決策**:
```python
# BigQueryクエリに制限を追加
query = f"""
...
ORDER BY filing_date DESC
LIMIT 50000  -- 候補数を制限
"""
```

## 📈 性能最適化

### 1. バッチサイズの調整

大量の候補がある場合、バッチ処理を実装：

```python
def batch_process(candidates_df, batch_size=10000):
    results = []
    for i in range(0, len(candidates_df), batch_size):
        batch = candidates_df.iloc[i:i+batch_size]
        result = calculator.find_top_similar(target_emb, batch, top_k=100)
        results.append(result)
    
    # 最終的なTop-Kを取得
    final = pd.concat(results).nlargest(1000, 'similarity_score')
    return final
```

### 2. キャッシング

同じ特許を何度も検索する場合：

```python
import pickle

# Embeddingをキャッシュ
cache_file = f"cache_{publication_number}.pkl"
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        target_embedding = pickle.load(f)
else:
    target_embedding = searcher.get_target_embedding(...)
    with open(cache_file, 'wb') as f:
        pickle.dump(target_embedding, f)
```

## 🧪 テスト

```bash
# サンプルXMLでテスト実行
python patent_similarity_search.py sample_patent.xml \
  --project-id YOUR_PROJECT_ID \
  --top-k 10

# 期待される出力:
# - 処理ログが表示される
# - similar_patents.csv が生成される
# - Top 10の結果がコンソールに表示される
```

## 📚 参考情報

- [Google Patents Public Data](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data)
- [BigQuery Python Client](https://cloud.google.com/python/docs/reference/bigquery/latest)
- [特許分類（IPC、FI、Fターム）](https://www.jpo.go.jp/system/patent/gaiyo/bunrui/index.html)

## 📄 ライセンス

MIT License

## 🤝 貢献

バグ報告や機能要望は Issue でお願いします。

## 📞 サポート

質問や問題がある場合は、以下を確認してください：
1. このREADMEのトラブルシューティングセクション
2. BigQueryのクォータ状況
3. XMLファイルの構造が正しいか
