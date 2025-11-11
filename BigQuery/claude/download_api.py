from google.cloud import bigquery
import pandas as pd
from pathlib import Path

client = bigquery.Client()
output_dir = Path("embeddings_chunks")
output_dir.mkdir(exist_ok=True)

from dotenv import load_dotenv
import os

load_dotenv('config.env')

# APIキーの設定（環境変数から取得）
project_id = os.getenv("GCP_PROJECT_ID")

client = bigquery.Client(project=project_id)

# クエリ結果を一時テーブルに保存（課金1回）
query = """
SELECT 
    publication_number,
    embedding_v1
FROM 
    `patents-public-data.google_patents_research.publications`
WHERE 
    country = "JP"
"""

temp_table_id = ".your_dataset.temp_embeddings"
job_config = bigquery.QueryJobConfig(destination=temp_table_id)
query_job = client.query(query, job_config=job_config)
query_job.result()

print(f"Bytes processed: {query_job.total_bytes_processed / (1024**3):.2f} GB")

# Storage APIでストリーミングダウンロード（追加課金なし）
rows_iter = client.list_rows(temp_table_id, page_size=10000)

chunk_num = 0
chunk_data = []
ROWS_PER_CHUNK = 50000

for i, row in enumerate(rows_iter):
    chunk_data.append({
        'publication_number': row.publication_number,
        'embedding_v1': row.embedding_v1
    })
    
    if len(chunk_data) >= ROWS_PER_CHUNK:
        df = pd.DataFrame(chunk_data)
        output_file = output_dir / f"embeddings_chunk_{chunk_num:04d}.parquet"
        df.to_parquet(output_file, compression='snappy')
        
        print(f"Saved chunk {chunk_num}: {len(df)} records")
        
        chunk_data = []
        chunk_num += 1
        del df

# 残りを保存
if chunk_data:
    df = pd.DataFrame(chunk_data)
    output_file = output_dir / f"embeddings_chunk_{chunk_num:04d}.parquet"
    df.to_parquet(output_file, compression='snappy')
    print(f"Saved final chunk {chunk_num}: {len(df)} records")

# クリーンアップ
client.delete_table(temp_table_id, not_found_ok=True)