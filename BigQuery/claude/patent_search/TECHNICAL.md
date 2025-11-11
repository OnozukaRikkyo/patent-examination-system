# 技術詳細ドキュメント

## アーキテクチャ概要

### システム構成図

```
┌─────────────────────────────────────────────────────────────┐
│                    Patent XML Input                          │
│                   (JP Patent Document)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PatentXMLParser                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ • publication_number 抽出                             │  │
│  │ • country_code 取得                                   │  │
│  │ • classification_codes 抽出 (IPC, CPC)              │  │
│  │ • theme_codes 抽出 (FI, F-term)                      │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              BigQueryPatentSearcher                          │
│                                                              │
│  ┌─────────────────────┐      ┌────────────────────────┐   │
│  │ get_target_embedding │      │ get_candidate_embeddings│   │
│  │                      │      │                         │   │
│  │ • Query: 1件        │      │ • Query: N件 (filtered) │   │
│  │ • Return: 64D vector│      │ • Filter: 先頭2文字     │   │
│  └─────────────────────┘      └────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              SimilarityCalculator                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ batch_cosine_similarity()                             │  │
│  │                                                       │  │
│  │ Input:  target (64,) + candidates (N, 64)           │  │
│  │ Output: similarities (N,)                            │  │
│  │                                                       │  │
│  │ Algorithm: Vectorized NumPy operations               │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Top-K Selection                            │
│            (similarity_score descending)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    CSV Output                                │
│     (publication_number, title, similarity_score, ...)      │
└─────────────────────────────────────────────────────────────┘
```

---

## データフロー詳細

### 1. XMLパース処理

**入力:** 日本特許XML（WIPO ST.36形式など）

**処理フロー:**
```python
XML Document
    ├─ bibliographic-data
    │   ├─ publication-reference → publication_number
    │   ├─ classification-ipcr   → IPC codes
    │   ├─ patent-classifications → CPC codes
    │   └─ classifications-national → FI codes
    └─ f-terms → F-term codes
```

**出力:** `PatentInfo` オブジェクト
```python
PatentInfo(
    publication_number='JP-2023123456-A',
    country_code='JP',
    classification_codes=['H04L9/00', 'G06K7/14', ...],
    theme_codes=['H04L9/00301', 'G06K7/14D', ...]
)
```

### 2. BigQuery検索処理

#### 2.1 対象特許のEmbedding取得

**クエリ構造:**
```sql
SELECT 
    publication_number,
    embedding_v1  -- 64次元ベクトル
FROM 
    `patents-public-data.google_patents_research.publications`
WHERE 
    publication_number = @pub_number  -- パラメータ化
    AND country_code = @country
    AND ARRAY_LENGTH(embedding_v1) > 0  -- 空ベクトル除外
LIMIT 1
```

**注意点:**
- `embedding_v1` は `ARRAY<FLOAT64>` 型
- 一部の特許にはembeddingが存在しない
- パラメータ化クエリでSQLインジェクション対策

#### 2.2 候補特許のEmbedding取得

**クエリ構造:**
```sql
WITH classified_patents AS (
    SELECT DISTINCT
        p.publication_number,
        p.title_localized[SAFE_OFFSET(0)].text AS title,
        p.filing_date,
        p.country_code,
        p.embedding_v1
    FROM 
        `patents-public-data.google_patents_research.publications` p,
        UNNEST(p.cpc) AS c  -- CPC分類を展開
    WHERE 
        p.country_code = 'JP'
        AND ARRAY_LENGTH(p.embedding_v1) > 0
        AND (
            c.code LIKE 'H0%' OR  -- 先頭2文字マッチング
            c.code LIKE 'G0%' OR
            ...
        )
)
SELECT * FROM classified_patents
ORDER BY filing_date DESC
```

**最適化ポイント:**
- `UNNEST(p.cpc)` で分類コード配列を展開
- `DISTINCT` で重複除去
- `LIKE` 条件によるインデックス活用
- `filing_date` でソートして最新特許を優先

**予想される結果件数:**
| 分類範囲 | 予想件数 | 処理時間 |
|---------|---------|----------|
| 1分類の先頭2文字 | 1,000〜10,000 | 5〜30秒 |
| 2分類の先頭2文字 | 5,000〜50,000 | 10〜60秒 |
| 5分類の先頭2文字 | 20,000〜100,000 | 30〜120秒 |

### 3. 類似度計算処理

#### 3.1 コサイン類似度の計算式

```
           A · B
cos(θ) = ─────────
         ||A|| ||B||

where:
  A: target embedding (64D)
  B: candidate embedding (64D)
  A·B: ドット積
  ||A||: Aのノルム（L2ノルム）
```

#### 3.2 バッチ処理実装

**ナイーブな実装（遅い）:**
```python
similarities = []
for candidate in candidates:
    sim = cosine_similarity(target, candidate)
    similarities.append(sim)
```
**時間計算量:** O(N × D)  
**例:** N=10,000, D=64 → 640,000回の演算

**最適化実装（高速）:**
```python
# 1. 正規化
target_norm = target / np.linalg.norm(target)
candidate_norms = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

# 2. 行列積で一括計算
similarities = np.dot(candidate_norms, target_norm)
```
**時間計算量:** O(N × D) だが、ベクトル化により高速  
**高速化率:** 約50〜100倍（NumPyのSIMD最適化による）

**メモリ使用量:**
```
target_norm:      64 × 8 bytes = 512 bytes
candidate_norms:  N × 64 × 8 bytes = N × 512 bytes
similarities:     N × 8 bytes

例: N = 10,000
→ 約 5.1 MB（候補） + 80 KB（類似度） = 約 5.2 MB
```

#### 3.3 Top-K選択

**実装:**
```python
top_k_indices = np.argpartition(similarities, -k)[-k:]
top_k_sorted = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]
```

**代替実装（Pandasを使用）:**
```python
result_df['similarity_score'] = similarities
top_k_df = result_df.nlargest(k, 'similarity_score')
```

---

## パフォーマンス最適化

### 1. BigQueryクエリ最適化

#### 候補数の制限

```sql
-- 方法1: LIMIT句を追加
SELECT * FROM classified_patents
ORDER BY filing_date DESC
LIMIT 50000  -- 最大候補数を制限

-- 方法2: 日付範囲でフィルタ
WHERE p.filing_date BETWEEN '2020-01-01' AND '2024-12-31'

-- 方法3: より厳密な分類マッチング
WHERE c.code LIKE 'H04L9%'  -- 先頭2文字 → 先頭5文字
```

#### パーティション化の活用

```sql
-- country_code と filing_date でパーティション分割されている場合
WHERE 
    p.country_code = 'JP'  -- パーティションキー
    AND p.filing_date >= '2020-01-01'  -- パーティション範囲
```

### 2. Python処理の最適化

#### メモリ効率化

```python
# チャンク処理で大量データを扱う
def process_large_dataset(candidates_df, chunk_size=10000):
    results = []
    
    for i in range(0, len(candidates_df), chunk_size):
        chunk = candidates_df.iloc[i:i+chunk_size]
        
        # チャンクごとに処理
        chunk_embeddings = np.array([emb for emb in chunk['embedding_v1']])
        similarities = batch_cosine_similarity(target_emb, chunk_embeddings)
        
        # Top-K候補を保持
        chunk_results = chunk.copy()
        chunk_results['similarity_score'] = similarities
        top_chunk = chunk_results.nlargest(1000, 'similarity_score')
        results.append(top_chunk)
    
    # 最終的なTop-Kを選択
    final_results = pd.concat(results).nlargest(1000, 'similarity_score')
    return final_results
```

#### 並列処理

```python
from multiprocessing import Pool
import numpy as np

def compute_similarity_chunk(args):
    target_emb, chunk_embeddings = args
    return batch_cosine_similarity(target_emb, chunk_embeddings)

def parallel_similarity_computation(target_emb, candidates_embeddings, n_workers=4):
    # データを分割
    chunks = np.array_split(candidates_embeddings, n_workers)
    
    # 並列処理
    with Pool(n_workers) as pool:
        chunk_results = pool.map(
            compute_similarity_chunk,
            [(target_emb, chunk) for chunk in chunks]
        )
    
    # 結合
    return np.concatenate(chunk_results)
```

### 3. キャッシング戦略

```python
import hashlib
import pickle
from pathlib import Path

class EmbeddingCache:
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, publication_number):
        return hashlib.md5(publication_number.encode()).hexdigest()
    
    def get(self, publication_number):
        cache_file = self.cache_dir / f"{self.get_cache_key(publication_number)}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, publication_number, embedding):
        cache_file = self.cache_dir / f"{self.get_cache_key(publication_number)}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)

# 使用例
cache = EmbeddingCache()

def get_embedding_with_cache(searcher, publication_number):
    # キャッシュチェック
    cached = cache.get(publication_number)
    if cached is not None:
        return cached
    
    # BigQueryから取得
    embedding = searcher.get_target_embedding(publication_number)
    
    # キャッシュに保存
    if embedding is not None:
        cache.set(publication_number, embedding)
    
    return embedding
```

---

## エラーハンドリング

### 想定されるエラーと対処

#### 1. BigQuery関連

```python
from google.cloud.exceptions import NotFound, Forbidden

try:
    results = client.query(query).to_dataframe()
except NotFound:
    # データセットやテーブルが見つからない
    logger.error("指定されたデータセットが存在しません")
except Forbidden:
    # アクセス権限がない
    logger.error("BigQueryへのアクセス権限がありません")
except Exception as e:
    # その他のエラー
    logger.error(f"予期しないエラー: {e}")
```

#### 2. XML解析関連

```python
import xml.etree.ElementTree as ET

try:
    tree = ET.parse(xml_path)
except ET.ParseError as e:
    logger.error(f"XML形式が不正です: {e}")
except FileNotFoundError:
    logger.error(f"ファイルが見つかりません: {xml_path}")
except PermissionError:
    logger.error(f"ファイルへのアクセス権限がありません: {xml_path}")
```

#### 3. データ検証

```python
def validate_embedding(embedding):
    """Embeddingの妥当性を検証"""
    if embedding is None:
        raise ValueError("Embeddingがnullです")
    
    if not isinstance(embedding, np.ndarray):
        raise TypeError("Embeddingはnumpy配列である必要があります")
    
    if embedding.shape != (64,):
        raise ValueError(f"Embeddingの次元が不正です: {embedding.shape}")
    
    if np.any(np.isnan(embedding)):
        raise ValueError("EmbeddingにNaNが含まれています")
    
    if np.all(embedding == 0):
        raise ValueError("Embeddingがゼロベクトルです")
    
    return True
```

---

## テスト戦略

### 1. ユニットテスト

```python
import unittest
import numpy as np
from patent_similarity_search import SimilarityCalculator

class TestSimilarityCalculator(unittest.TestCase):
    def test_cosine_similarity_identical(self):
        """同一ベクトルの類似度は1.0"""
        vec = np.random.rand(64)
        sim = SimilarityCalculator.cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=5)
    
    def test_cosine_similarity_orthogonal(self):
        """直交ベクトルの類似度は0.0"""
        vec1 = np.zeros(64)
        vec1[0] = 1.0
        vec2 = np.zeros(64)
        vec2[1] = 1.0
        sim = SimilarityCalculator.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(sim, 0.0, places=5)
    
    def test_batch_similarity(self):
        """バッチ計算と個別計算の一致"""
        target = np.random.rand(64)
        candidates = np.random.rand(100, 64)
        
        # バッチ計算
        batch_sims = SimilarityCalculator.batch_cosine_similarity(target, candidates)
        
        # 個別計算
        individual_sims = [
            SimilarityCalculator.cosine_similarity(target, cand)
            for cand in candidates
        ]
        
        np.testing.assert_array_almost_equal(batch_sims, individual_sims, decimal=5)

if __name__ == '__main__':
    unittest.main()
```

### 2. 統合テスト

```python
def test_end_to_end():
    """エンドツーエンドテスト"""
    # 1. テストデータ準備
    test_xml = 'test_data/sample_patent.xml'
    
    # 2. システム初期化
    system = PatentSimilaritySearchSystem(project_id='test-project')
    
    # 3. 検索実行
    results = system.search(test_xml, top_k=100)
    
    # 4. 結果検証
    assert len(results) > 0, "結果が空です"
    assert len(results) <= 100, "Top-K制限が効いていません"
    assert 'similarity_score' in results.columns, "類似度列がありません"
    assert results['similarity_score'].max() <= 1.0, "類似度の範囲が不正です"
    assert results['similarity_score'].min() >= -1.0, "類似度の範囲が不正です"
    
    print("✓ エンドツーエンドテスト成功")
```

---

## ベンチマーク

### 処理時間の目安

**テスト環境:**
- CPU: Intel Core i7 (4コア)
- RAM: 16GB
- Network: 100Mbps

**結果:**

| 候補数 | XMLパース | BQ取得 | 類似度計算 | 合計 |
|-------|----------|--------|-----------|------|
| 1,000 | 0.1秒 | 5秒 | 0.01秒 | 5秒 |
| 10,000 | 0.1秒 | 15秒 | 0.05秒 | 15秒 |
| 50,000 | 0.1秒 | 45秒 | 0.2秒 | 45秒 |
| 100,000 | 0.1秒 | 90秒 | 0.5秒 | 90秒 |

**ボトルネック:** BigQueryからのデータ取得（ネットワークI/O）

---

## 今後の拡張可能性

### 1. 高度なフィルタリング

```python
def get_candidate_embeddings_advanced(
    self,
    classification_codes,
    min_filing_date='2015-01-01',
    max_filing_date='2024-12-31',
    applicant_filter=None,
    citation_filter=None
):
    """より詳細なフィルタリング"""
    # 実装例...
```

### 2. 多段階検索

```python
# Stage 1: 粗い検索（高速）
coarse_results = search_with_2char_prefix(codes, top_k=10000)

# Stage 2: 詳細検索（精密）
fine_results = rerank_with_full_text_similarity(coarse_results, top_k=1000)
```

### 3. ハイブリッド検索

```python
# Embedding検索 + キーワード検索の組み合わせ
embedding_score = 0.7 * cosine_similarity(emb1, emb2)
keyword_score = 0.3 * keyword_match_score(text1, text2)
final_score = embedding_score + keyword_score
```

---

## 参考文献

1. Google Patents Public Data Documentation
2. BigQuery Best Practices
3. NumPy Performance Optimization Guide
4. Patent Classification Systems (IPC, CPC, FI)

---

**更新日:** 2024-11-10  
**バージョン:** 1.0.0
