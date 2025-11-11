from google.cloud import bigquery
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
# pip install db-dtypes

def search_similar_patents(target_patent_number, output_csv='similar_patents.csv', top_k=1000):

    load_dotenv('config.env')

    # APIキーの設定（環境変数から取得）
    project_id = os.getenv("GCP_PROJECT_ID")

    client = bigquery.Client(project=project_id)

    """
    指定した特許番号に類似する特許を検索し、CSVファイルに保存
    
    Parameters:
    -----------
    target_patent_number : str
        基準とする特許番号（例: 'JP-2023123456-A'）
    output_csv : str
        出力CSVファイル名
    top_k : int
        取得する類似特許の件数（デフォルト: 1000）
    """
        
    # パラメータ化されたクエリ
    query = """
    WITH
    target_patent AS (
      SELECT
        publication_number,
        embedding_v1
      FROM
        `patents-public-data.google_patents_research.publications`
      WHERE
        publication_number = @target_patent_number
    ),
    sample_patents AS (
      SELECT
        publication_number,
        embedding_v1
      FROM
        `patents-public-data.google_patents_research.publications`
      WHERE
        country = "Japan"
        AND MOD(ABS(FARM_FINGERPRINT(publication_number)), 100) = 0
    ),
    distances AS (
      SELECT
        s.publication_number,
        ML.DISTANCE(s.embedding_v1, t.embedding_v1, 'COSINE') AS cosine_distance
      FROM
        sample_patents s,
        target_patent t
      WHERE
        s.publication_number != t.publication_number
    )
    SELECT
      publication_number,
      cosine_distance,
      1 - cosine_distance AS cosine_similarity
    FROM
      distances
    ORDER BY
      cosine_distance ASC
    LIMIT @top_k
    """
    
    # クエリパラメータの設定
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("target_patent_number", "STRING", target_patent_number),
            bigquery.ScalarQueryParameter("top_k", "INT64", top_k)
        ]
    )
    
    print(f"検索開始: {target_patent_number}")
    print(f"取得件数: {top_k}")
    
    # クエリの実行
    query_job = client.query(query, job_config=job_config)
    
    # 結果の取得（DataFrameに変換）
    df = query_job.to_dataframe()
    
    print(f"検索完了: {len(df)}件の類似特許を発見")
    
    # CSVファイルに保存
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"CSVファイルに保存: {output_csv}")
    
    # 統計情報の表示
    if len(df) > 0:
        print("\n--- 統計情報 ---")
        print(f"最大類似度: {df['cosine_similarity'].max():.4f}")
        print(f"最小類似度: {df['cosine_similarity'].min():.4f}")
        print(f"平均類似度: {df['cosine_similarity'].mean():.4f}")
        print("\n--- Top 5 類似特許 ---")
        print(df.head())
    
    return df


# 使用例
if __name__ == "__main__":
    # 基準とする特許番号
    target_patent = 'JP-S4926374-B1'  # ← ここに実際の特許番号を入力
    
    # 実行
    results_df = search_similar_patents(
        target_patent_number=target_patent,
        output_csv=f'similar_patents_{target_patent}.csv',
        top_k=1000
    )