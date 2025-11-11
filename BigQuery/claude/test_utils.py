"""
デバッグ・テスト用ユーティリティスクリプト
"""

import sys
from patent_similarity_search import PatentXMLParser, BigQueryPatentSearcher
from google.cloud import bigquery
import pandas as pd


def test_xml_parser(xml_path: str):
    """XMLパーサーのテスト"""
    print("=" * 80)
    print("XMLパーサーテスト")
    print("=" * 80)
    
    try:
        parser = PatentXMLParser(xml_path)
        info = parser.parse()
        
        print(f"\n✓ パース成功!")
        print(f"\n公開番号: {info.publication_number}")
        print(f"国コード: {info.country_code}")
        print(f"\n分類コード ({len(info.classification_codes)}件):")
        for i, code in enumerate(info.classification_codes[:10], 1):
            print(f"  {i}. {code}")
        if len(info.classification_codes) > 10:
            print(f"  ... 他 {len(info.classification_codes) - 10}件")
        
        print(f"\nテーマコード ({len(info.theme_codes)}件):")
        for i, code in enumerate(info.theme_codes[:10], 1):
            print(f"  {i}. {code}")
        if len(info.theme_codes) > 10:
            print(f"  ... 他 {len(info.theme_codes) - 10}件")
        
        print(f"\n先頭2文字の抽出:")
        prefix_2chars = set()
        for code in info.classification_codes + info.theme_codes:
            if len(code) >= 2:
                prefix_2chars.add(code[:2])
        print(f"  {sorted(prefix_2chars)}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bigquery_connection(project_id: str):
    """BigQuery接続テスト"""
    print("\n" + "=" * 80)
    print("BigQuery接続テスト")
    print("=" * 80)
    
    try:
        client = bigquery.Client(project=project_id)
        
        # 簡単なクエリを実行
        query = """
        SELECT 
            COUNT(*) as total_count
        FROM 
            `patents-public-data.google_patents_research.publications`
        WHERE 
            country_code = 'JP'
        LIMIT 1
        """
        
        print(f"\nプロジェクトID: {project_id}")
        print(f"クエリ実行中...")
        
        result = client.query(query).to_dataframe()
        
        print(f"\n✓ 接続成功!")
        print(f"日本特許データ件数: {result['total_count'].iloc[0]:,}件")
        
        return True
        
    except Exception as e:
        print(f"\n✗ エラー発生: {e}")
        print(f"\n確認事項:")
        print(f"  1. プロジェクトIDが正しいか")
        print(f"  2. GOOGLE_APPLICATION_CREDENTIALSが設定されているか")
        print(f"  3. BigQueryのアクセス権限があるか")
        return False


def test_embedding_retrieval(project_id: str, publication_number: str = "JP-2020123456-A"):
    """Embedding取得テスト"""
    print("\n" + "=" * 80)
    print("Embedding取得テスト")
    print("=" * 80)
    
    try:
        searcher = BigQueryPatentSearcher(project_id)
        
        print(f"\n対象特許: {publication_number}")
        print(f"Embedding取得中...")
        
        embedding = searcher.get_target_embedding(publication_number, 'JP')
        
        if embedding is not None:
            print(f"\n✓ 取得成功!")
            print(f"  Shape: {embedding.shape}")
            print(f"  Dtype: {embedding.dtype}")
            print(f"  Min: {embedding.min():.6f}")
            print(f"  Max: {embedding.max():.6f}")
            print(f"  Mean: {embedding.mean():.6f}")
            print(f"\n  先頭10次元: {embedding[:10]}")
            return True
        else:
            print(f"\n✗ 取得失敗")
            print(f"\n対処方法:")
            print(f"  1. publication_numberを変更してみる")
            print(f"  2. BigQueryで直接確認:")
            print(f"     SELECT publication_number FROM")
            print(f"     `patents-public-data.google_patents_research.publications`")
            print(f"     WHERE country_code = 'JP' LIMIT 10")
            return False
            
    except Exception as e:
        print(f"\n✗ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_candidate_search(project_id: str):
    """候補検索テスト（軽量）"""
    print("\n" + "=" * 80)
    print("候補検索テスト")
    print("=" * 80)
    
    try:
        searcher = BigQueryPatentSearcher(project_id)
        
        # テスト用の分類コード
        test_codes = ['H04L9/00', 'G06K7/14']
        
        print(f"\nテスト分類コード: {test_codes}")
        print(f"候補を検索中（制限: 100件）...")
        
        # 一時的にクエリを修正（テスト用）
        prefix_2chars = set()
        for code in test_codes:
            if len(code) >= 2:
                prefix_2chars.add(code[:2])
        
        like_conditions = " OR ".join([f"c.code LIKE '{prefix}%'" for prefix in prefix_2chars])
        
        query = f"""
        WITH classified_patents AS (
            SELECT DISTINCT
                p.publication_number,
                p.title_localized[SAFE_OFFSET(0)].text AS title,
                p.filing_date,
                p.country_code,
                p.embedding_v1
            FROM 
                `patents-public-data.google_patents_research.publications` p,
                UNNEST(p.cpc) AS c
            WHERE 
                p.country_code = 'JP'
                AND ARRAY_LENGTH(p.embedding_v1) > 0
                AND ({like_conditions})
        )
        
        SELECT 
            publication_number,
            title,
            filing_date
        FROM 
            classified_patents
        ORDER BY 
            filing_date DESC
        LIMIT 100
        """
        
        client = bigquery.Client(project=project_id)
        results = client.query(query).to_dataframe()
        
        if len(results) > 0:
            print(f"\n✓ 検索成功!")
            print(f"  取得件数: {len(results)}件")
            print(f"\n  サンプル (最新5件):")
            for idx, row in results.head(5).iterrows():
                print(f"    {row['publication_number']} | {row['filing_date']} | {row['title'][:50]}...")
            return True
        else:
            print(f"\n✗ 候補が見つかりません")
            return False
            
    except Exception as e:
        print(f"\n✗ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_sample_patent(project_id: str, classification_prefix: str = 'H04L'):
    """サンプル特許を検索"""
    print("\n" + "=" * 80)
    print(f"サンプル特許検索 (分類: {classification_prefix})")
    print("=" * 80)
    
    try:
        client = bigquery.Client(project=project_id)
        
        query = f"""
        SELECT 
            p.publication_number,
            p.title_localized[SAFE_OFFSET(0)].text AS title,
            p.abstract_localized[SAFE_OFFSET(0)].text AS abstract,
            p.filing_date,
            STRING_AGG(DISTINCT c.code, ', ') as cpc_codes
        FROM 
            `patents-public-data.google_patents_research.publications` p,
            UNNEST(p.cpc) AS c
        WHERE 
            p.country_code = 'JP'
            AND ARRAY_LENGTH(p.embedding_v1) > 0
            AND c.code LIKE '{classification_prefix}%'
        GROUP BY 
            p.publication_number, p.title_localized, p.abstract_localized, p.filing_date
        ORDER BY 
            p.filing_date DESC
        LIMIT 5
        """
        
        print(f"\n検索中...")
        results = client.query(query).to_dataframe()
        
        if len(results) > 0:
            print(f"\n✓ {len(results)}件見つかりました:\n")
            for idx, row in results.iterrows():
                print(f"[{idx + 1}] {row['publication_number']}")
                print(f"    出願日: {row['filing_date']}")
                print(f"    タイトル: {row['title']}")
                print(f"    分類: {row['cpc_codes']}")
                print(f"    要約: {row['abstract'][:100]}...")
                print()
            
            return results['publication_number'].tolist()
        else:
            print(f"\n✗ 見つかりませんでした")
            return []
            
    except Exception as e:
        print(f"\n✗ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """メインテスト実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='テスト・デバッグユーティリティ')
    parser.add_argument('--test', choices=['xml', 'bigquery', 'embedding', 'candidate', 'sample', 'all'],
                       default='all', help='実行するテスト')
    parser.add_argument('--xml-path', default='sample_patent.xml', help='テスト用XMLファイル')
    parser.add_argument('--project-id', help='Google Cloud Project ID')
    parser.add_argument('--publication-number', default='JP-2020123456-A', 
                       help='テスト用公開番号')
    parser.add_argument('--classification', default='H04L', help='サンプル検索用分類')
    
    args = parser.parse_args()
    
    print("\n")
    print("*" * 80)
    print("特許類似検索システム - テストユーティリティ")
    print("*" * 80)
    
    results = {}
    
    # XMLパーサーテスト
    if args.test in ['xml', 'all']:
        results['xml'] = test_xml_parser(args.xml_path)
    
    # 以下はproject_idが必要
    if args.test in ['bigquery', 'embedding', 'candidate', 'sample', 'all']:
        if not args.project_id:
            print("\n⚠ --project-id が必要です")
            sys.exit(1)
    
    # BigQuery接続テスト
    if args.test in ['bigquery', 'all']:
        results['bigquery'] = test_bigquery_connection(args.project_id)
    
    # Embedding取得テスト
    if args.test in ['embedding', 'all']:
        results['embedding'] = test_embedding_retrieval(args.project_id, args.publication_number)
    
    # 候補検索テスト
    if args.test in ['candidate', 'all']:
        results['candidate'] = test_candidate_search(args.project_id)
    
    # サンプル特許検索
    if args.test in ['sample', 'all']:
        sample_patents = find_sample_patent(args.project_id, args.classification)
        results['sample'] = len(sample_patents) > 0
    
    # 結果サマリー
    if results:
        print("\n" + "=" * 80)
        print("テスト結果サマリー")
        print("=" * 80)
        for test_name, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  {test_name:15s}: {status}")
        
        all_pass = all(results.values())
        print(f"\n総合結果: {'✓ すべて成功' if all_pass else '✗ 一部失敗'}")
        print("=" * 80)


if __name__ == "__main__":
    main()
