from google.cloud import bigquery
import json
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv('config.env')

# APIキーの設定（環境変数から取得）
project_id = os.getenv("GCP_PROJECT_ID")

client = bigquery.Client(project=project_id)

query = """
SELECT
    publication_number,
    embedding_v1
FROM
    `patents-public-data.google_patents_research.publications`
WHERE
    country = "Japan"
"""

# query = """
# SELECT
#     publication_number,
#     embedding_v1
# FROM
#     `patents-public-data.google_patents_research.publications`
#     TABLESAMPLE SYSTEM (1 PERCENT)
# WHERE
#     country = "Japan"
# """

print("クエリ実行中...")
query_job = client.query(query)

# 総件数を取得
total_rows = query_job.result().total_rows
print(f"総件数: {total_rows:,} 件")

print("ダウンロード中...")
with open('jp_embeddings_all.jsonl', 'w') as f:
    for i, row in enumerate(tqdm(query_job, total=total_rows), 1):
        data = {
            'publication_number': row.publication_number,
            'embedding_v1': list(row.embedding_v1)
        }
        f.write(json.dumps(data) + '\n')
        
        # 1000件ごとに保存（安全のため）
        if i % 1000 == 0:
            f.flush()

print("完了！")