from google.cloud import bigquery
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv('config.env')

# APIキーの設定（環境変数から取得）
project_id = os.getenv("GCP_PROJECT_ID")

client = bigquery.Client(project=project_id)

print(f"Using project: {client.project}")

# データセット作成（初回のみ）
dataset_id = f"{client.project}.patent_temp"
dataset = bigquery.Dataset(dataset_id)
dataset.location = "US"
client.create_dataset(dataset, exists_ok=True)
print(f"Dataset ready: {dataset_id}")

# 出力ディレクトリ作成
output_dir = Path("embeddings_chunks")
output_dir.mkdir(exist_ok=True)

# 一時テーブルID設定（自動的にあなたのプロジェクトIDが使われる）
temp_table_id = f"{client.project}.patent_temp.temp_embeddings"
print(f"Temp table: {temp_table_id}")

# クエリ実行（課金1回のみ）
query = """
SELECT 
    publication_number,
    embedding_v1
FROM 
    `patents-public-data.google_patents_research.publications`
WHERE 
    country = "JP"
"""

job_config = bigquery.QueryJobConfig(destination=temp_table_id)
query_job = client.query(query, job_config=job_config)
query_job.result()

print(f"Bytes processed: {query_job.total_bytes_processed / (1024**3):.2f} GB")
print(f"Query completed. Starting download...")

# Storage APIでストリーミングダウンロード（追加課金なし）
rows_iter = client.list_rows(temp_table_id, page_size=10000)

chunk_num = 0
chunk_data = []
ROWS_PER_CHUNK = 50000

print(f"Downloading and saving data in chunks of {ROWS_PER_CHUNK} rows...")

for i, row in enumerate(rows_iter):
    chunk_data.append({
        'publication_number': row.publication_number,
        'embedding_v1': row.embedding_v1
    })
    
    # 進捗表示（10000行ごと）
    if (i + 1) % 10000 == 0:
        print(f"  Processed {i + 1} rows...")
    
    # チャンクサイズに達したら保存
    if len(chunk_data) >= ROWS_PER_CHUNK:
        df = pd.DataFrame(chunk_data)
        output_file = output_dir / f"embeddings_chunk_{chunk_num:04d}.parquet"
        df.to_parquet(output_file, compression='snappy')
        
        print(f"✓ Saved chunk {chunk_num}: {len(df)} records -> {output_file}")
        
        chunk_data = []
        chunk_num += 1
        del df

# 残りのデータを保存
if chunk_data:
    df = pd.DataFrame(chunk_data)
    output_file = output_dir / f"embeddings_chunk_{chunk_num:04d}.parquet"
    df.to_parquet(output_file, compression='snappy')
    print(f"✓ Saved final chunk {chunk_num}: {len(df)} records -> {output_file}")
    del df

print(f"\n=== Download Complete ===")
print(f"Total chunks saved: {chunk_num + 1}")
print(f"Output directory: {output_dir.absolute()}")

# クリーンアップ（一時テーブルを削除してストレージ料金を節約）
print(f"\nCleaning up temporary table...")
client.delete_table(temp_table_id, not_found_ok=True)
print(f"✓ Temporary table deleted: {temp_table_id}")

print("\n=== All Done! ===")