# クイックスタートガイド

5分で特許類似検索を始めましょう！

## ステップ1: 環境構築（3分）

```bash
# 1. 依存パッケージのインストール
pip install -r requirements.txt

# 2. Google Cloud認証
gcloud auth application-default login

# 3. プロジェクトIDを設定
export GCP_PROJECT_ID="your-project-id"
```

## ステップ2: テスト実行（1分）

```bash
# システムが正しく動作するか確認
python test_utils.py --test all --project-id $GCP_PROJECT_ID
```

**期待される出力:**
```
✓ xml       : PASS
✓ bigquery  : PASS
✓ embedding : PASS
✓ candidate : PASS
✓ sample    : PASS

総合結果: ✓ すべて成功
```

## ステップ3: 実行（1分）

```bash
# サンプルXMLで類似特許を検索
python patent_similarity_search.py sample_patent.xml \
  --project-id $GCP_PROJECT_ID \
  --top-k 1000 \
  --output results.csv

# 結果を確認
head -20 results.csv
```

## 完成！ 🎉

`results.csv` に類似特許のTop 1000が出力されています。

---

## 実際のファイルで実行する場合

```bash
# 1. XMLファイルを準備
cp /path/to/your/patent.xml ./my_patent.xml

# 2. 検索実行
python patent_similarity_search.py my_patent.xml \
  --project-id $GCP_PROJECT_ID \
  --top-k 1000 \
  --output my_results.csv

# 3. 結果を分析
python -c "
import pandas as pd
df = pd.read_csv('my_results.csv')
print(f'総件数: {len(df)}')
print(f'平均類似度: {df["similarity_score"].mean():.4f}')
print(f'最高類似度: {df["similarity_score"].max():.4f}')
print(f'\nTop 5:')
print(df.head(5)[['publication_number', 'similarity_score', 'title']])
"
```

---

## トラブルシューティング

### ❌ "Permission denied" エラー

```bash
# 再認証してください
gcloud auth application-default login
```

### ❌ "対象特許のembeddingが取得できませんでした"

XMLのpublication_numberが正しいか確認：

```bash
python -c "
from patent_similarity_search import PatentXMLParser
parser = PatentXMLParser('your_file.xml')
info = parser.parse()
print(f'Publication Number: {info.publication_number}')
"
```

### ❌ "候補特許が見つかりませんでした"

分類コードが正しく取得されているか確認：

```bash
python test_utils.py --test xml --xml-path your_file.xml
```

---

## 次のステップ

- [README.md](README.md) - 詳細なドキュメント
- [TECHNICAL.md](TECHNICAL.md) - 技術詳細
- カスタマイズ方法は README.md の「カスタマイズ」セクションを参照

## サポート

質問があれば Issue を作成してください。
