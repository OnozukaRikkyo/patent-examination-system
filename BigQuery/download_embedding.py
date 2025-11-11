from google.cloud import bigquery
import pyarrow.parquet as pq
import os
from dotenv import load_dotenv


load_dotenv('config.env')

# APIキーの設定（環境変数から取得）
project_id = os.getenv("GCP_PROJECT_ID")
if not project_id:
    print("⚠️ config.envファイルにGCP_PROJECT_IDを設定してください")
    exit(1)
    # APIキーの設定（環境変数から取得）

# BigQueryクライアント初期化
client = bigquery.Client(project=project_id)

# 出力ファイルパス
output_file = "jp_patents_embeddings.parquet"

# ファイル書き込み可能かチェック
try:
    # ディレクトリの存在確認
    output_dir = os.path.dirname(output_file) or '.'
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"ディレクトリが存在しません: {output_dir}")

    # 書き込み権限チェック（空ファイル作成で確認）
    with open(output_file, 'w') as f:
        pass
    os.remove(output_file)  # チェック用ファイル削除
    print(f"✓ ファイル書き込み確認OK: {output_file}")
except Exception as e:
    print(f"✗ エラー: ファイルに書き込めません - {e}")
    exit(1)

# クエリ実行
query = """
SELECT
    publication_number,
    embedding_v1
FROM
    `patents-public-data.google_patents_research.publications`
WHERE
    country = "JP"
    AND ARRAY_LENGTH(embedding_v1) > 0
"""

print("クエリ実行中...")
query_job = client.query(query)

# BigQueryの結果をPyArrow Tableに変換してParquet形式で保存
print("データをParquet形式で保存中...")
arrow_table = query_job.to_arrow()
pq.write_table(arrow_table, output_file, compression='snappy')

row_count = len(arrow_table)
print(f"完了: {row_count:,} 件を {output_file} に保存しました")
