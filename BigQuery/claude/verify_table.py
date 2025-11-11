from google.cloud import bigquery
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv('config.env')
project_id = os.getenv("GCP_PROJECT_ID")
client = bigquery.Client(project=project_id)

print(f"Project: {client.project}\n")
print("=== Checking recent jobs ===\n")

# 過去1時間のジョブを確認
for job in client.list_jobs(max_results=10):
    if job.job_type == "query":
        print(f"Job ID: {job.job_id}")
        print(f"  State: {job.state}")
        print(f"  Created: {job.created}")
        print(f"  Bytes: {job.total_bytes_processed / (1024**3):.2f} GB" if job.total_bytes_processed else "  Bytes: N/A")
        
        # 成功したジョブの destination を確認
        if job.state == "DONE" and job.destination:
            print(f"  ✓ Destination: {job.destination}")
            
            # そのテーブルが存在するか確認
            try:
                table = client.get_table(job.destination)
                print(f"    → Table EXISTS! Rows: {table.num_rows:,}, Size: {table.num_bytes / (1024**3):.2f} GB")
            except Exception as e:
                print(f"    → Table not found (may have been deleted)")
        
        if job.error_result:
            print(f"  ✗ Error: {job.error_result}")
        
        print()
        
# from google.cloud import bigquery
# from dotenv import load_dotenv
# import os

# load_dotenv('config.env')
# project_id = os.getenv("GCP_PROJECT_ID")
# client = bigquery.Client(project=project_id)

# print(f"Project: {client.project}")

# # 既存のテーブルをチェック
# temp_table_id = f"{client.project}.patent_temp.temp_embeddings"

# try:
#     table = client.get_table(temp_table_id)
#     print(f"\n✓ Table EXISTS: {temp_table_id}")
#     print(f"  Rows: {table.num_rows:,}")
#     print(f"  Size: {table.num_bytes / (1024**3):.2f} GB")
#     print(f"  Created: {table.created}")
    
#     if table.num_rows > 0:
#         print("\n✓ Data is available! We can download it without running the query again.")
#     else:
#         print("\n⚠️ Table exists but has 0 rows")
        
# except Exception as e:
#     print(f"\n✗ Table does not exist: {temp_table_id}")
#     print(f"  Error: {e}")