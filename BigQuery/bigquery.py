"""
特許文献XMLから類似特許を検索するシステム
Google Patents Public Data (BigQuery) を使用
"""

import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from google.cloud import bigquery
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import logging
import os
from pathlib import Path

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PatentInfo:
    """特許情報を保持するデータクラス"""
    publication_number: str
    country_code: str
    classification_codes: List[str]  # IPC, CPC等
    theme_codes: List[str]  # FI, Fターム等


class PatentXMLParser:
    """特許XMLパーサー"""
    
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = None
        self.root = None
    
    def parse(self) -> PatentInfo:
        """XMLファイルをパースして特許情報を抽出"""
        logger.info(f"XMLファイルをパース中: {self.xml_path}")
        
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        
        # publication_numberとcountry_codeを取得
        publication_number = self._get_publication_number()
        country_code = self._get_country_code()
        
        # 分類コードを取得
        classification_codes = self._get_classification_codes()
        
        # テーマコード（FI, Fターム）を取得
        theme_codes = self._get_theme_codes()
        
        logger.info(f"パース完了: {publication_number} (分類: {len(classification_codes)}件, テーマ: {len(theme_codes)}件)")
        
        return PatentInfo(
            publication_number=publication_number,
            country_code=country_code,
            classification_codes=classification_codes,
            theme_codes=theme_codes
        )
    
    def _get_publication_number(self) -> str:
        """公開番号を取得"""
        # 日本特許の場合の一般的なXML構造
        # 実際のXML構造に応じて調整が必要
        
        # パターン1: bibliographic-data/publication-reference
        pub_ref = self.root.find('.//publication-reference/document-id/doc-number')
        if pub_ref is not None:
            doc_number = pub_ref.text
            kind = self.root.find('.//publication-reference/document-id/kind').text
            return f"JP-{doc_number}-{kind}"
        
        # パターン2: 直接的な公開番号
        pub_num = self.root.find('.//publication-number')
        if pub_num is not None:
            return pub_num.text
        
        raise ValueError("公開番号が見つかりません")
    
    def _get_country_code(self) -> str:
        """国コードを取得（常に'JP'を返す）"""
        # XMLから取得する場合
        country = self.root.find('.//publication-reference/document-id/country')
        if country is not None and country.text:
            return country.text
        
        # デフォルトでJP
        return 'JP'
    
    def _get_classification_codes(self) -> List[str]:
        """分類コード（IPC, CPC等）を取得"""
        codes = []
        
        # IPC分類
        for ipc in self.root.findall('.//classification-ipc'):
            ipc_text = ipc.find('.//text')
            if ipc_text is not None:
                codes.append(ipc_text.text.strip().replace(' ', ''))
        
        # CPC分類
        for cpc in self.root.findall('.//classification-cpc'):
            section = cpc.find('.//section')
            class_elem = cpc.find('.//class')
            subclass = cpc.find('.//subclass')
            
            if all([section, class_elem, subclass]):
                code = f"{section.text}{class_elem.text}{subclass.text}"
                codes.append(code)
        
        return list(set(codes))  # 重複削除
    
    def _get_theme_codes(self) -> List[str]:
        """テーマコード（FI, Fターム）を取得"""
        codes = []
        
        # FI分類（日本特許特有）
        for fi in self.root.findall('.//classification-national'):
            fi_text = fi.find('.//text')
            if fi_text is not None:
                codes.append(fi_text.text.strip().replace(' ', ''))
        
        # Fターム
        for fterm in self.root.findall('.//f-term'):
            if fterm.text:
                codes.append(fterm.text.strip())
        
        return list(set(codes))


class BigQueryPatentSearcher:
    """BigQueryを使った特許検索"""
    
    def __init__(self, project_id: str):
        self.client = bigquery.Client(project=project_id)
        self.dataset = "patents-public-data.google_patents_research"
    
    def get_target_embedding(self, publication_number: str, country_code: str = 'JP') -> Optional[np.ndarray]:
        """対象特許のembedding_v1を取得"""
        logger.info(f"対象特許のembeddingを取得: {publication_number}")
        
        query = f"""
        SELECT 
            publication_number,
            embedding_v1
        FROM 
            `{self.dataset}.publications`
        WHERE 
            publication_number = @pub_number
            AND country_code = @country
            AND ARRAY_LENGTH(embedding_v1) > 0
        LIMIT 1
        """
        
        """
          SELECT 
      publication_number,
      embedding_v1
  FROM 
    `patents-public-data.patents.publications`

  WHERE 
      publication_number = "A1234"
      AND country_code = "JP"
      AND ARRAY_LENGTH(embedding_v1) > 0
  LIMIT 1

        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("pub_number", "STRING", publication_number),
                bigquery.ScalarQueryParameter("country", "STRING", country_code),
            ]
        )
        
        try:
            results = self.client.query(query, job_config=job_config).to_dataframe()
            
            if len(results) == 0:
                logger.warning(f"特許が見つかりません: {publication_number}")
                return None
            
            embedding = np.array(results['embedding_v1'].iloc[0])
            logger.info(f"Embedding取得成功: shape={embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding取得エラー: {e}")
            return None
    
    def get_candidate_embeddings(
        self, 
        classification_codes: List[str], 
        theme_codes: List[str],
        country_code: str = 'JP'
    ) -> pd.DataFrame:
        """分類コード・テーマコードに基づいて候補特許のembeddingを取得"""
        logger.info(f"候補特許を取得中... (分類: {len(classification_codes)}件, テーマ: {len(theme_codes)}件)")
        
        # 先頭2文字を抽出
        prefix_2chars = set()
        for code in classification_codes + theme_codes:
            if len(code) >= 2:
                prefix_2chars.add(code[:2])
        
        if not prefix_2chars:
            logger.warning("分類コードが不十分です")
            return pd.DataFrame()
        
        # LIKE条件を構築
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
                `{self.dataset}.publications` p,
                UNNEST(p.cpc) AS c
            WHERE 
                p.country_code = @country
                AND ARRAY_LENGTH(p.embedding_v1) > 0
                AND ({like_conditions})
        )
        
        SELECT 
            publication_number,
            title,
            filing_date,
            country_code,
            embedding_v1
        FROM 
            classified_patents
        ORDER BY 
            filing_date DESC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("country", "STRING", country_code),
            ]
        )
        
        try:
            logger.info("BigQueryクエリを実行中...")
            results = self.client.query(query, job_config=job_config).to_dataframe()
            logger.info(f"候補特許取得完了: {len(results)}件")
            return results
            
        except Exception as e:
            logger.error(f"候補特許取得エラー: {e}")
            return pd.DataFrame()


class SimilarityCalculator:
    """ベクトル類似度計算"""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """コサイン類似度を計算"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    @staticmethod
    def batch_cosine_similarity(target_vec: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
        """バッチでコサイン類似度を計算（高速化）"""
        # target_vec: (d,)
        # candidate_vecs: (n, d)
        
        # 正規化
        target_norm = target_vec / np.linalg.norm(target_vec)
        candidate_norms = candidate_vecs / np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
        
        # 内積計算
        similarities = np.dot(candidate_norms, target_norm)
        
        return similarities
    
    @classmethod
    def find_top_similar(
        cls, 
        target_embedding: np.ndarray, 
        candidates_df: pd.DataFrame, 
        top_k: int = 1000
    ) -> pd.DataFrame:
        """Top-K類似特許を検索"""
        logger.info(f"類似度計算中... (候補: {len(candidates_df)}件, Top-K: {top_k})")
        
        if len(candidates_df) == 0:
            logger.warning("候補特許がありません")
            return pd.DataFrame()
        
        # embeddingをnumpy配列に変換
        candidate_embeddings = np.array([
            np.array(emb) for emb in candidates_df['embedding_v1'].values
        ])
        
        # バッチ類似度計算
        similarities = cls.batch_cosine_similarity(target_embedding, candidate_embeddings)
        
        # 結果をデータフレームに追加
        result_df = candidates_df.copy()
        result_df['similarity_score'] = similarities
        
        # Top-Kを取得
        top_k_df = result_df.nlargest(min(top_k, len(result_df)), 'similarity_score')
        
        # embeddingカラムを削除（表示用）
        top_k_df = top_k_df.drop('embedding_v1', axis=1)
        
        logger.info(f"Top-K抽出完了: {len(top_k_df)}件 (最高スコア: {top_k_df['similarity_score'].max():.4f})")
        
        return top_k_df


class PatentSimilaritySearchSystem:
    """特許類似検索システム統合クラス"""
    
    def __init__(self, project_id: str):
        self.searcher = BigQueryPatentSearcher(project_id)
        self.calculator = SimilarityCalculator()
    
    def search(self, xml_path: str, top_k: int = 1000) -> pd.DataFrame:
        """XMLファイルから類似特許を検索"""
        logger.info("=" * 80)
        logger.info("特許類似検索システム開始")
        logger.info("=" * 80)
        
        # 1. XMLをパース
        parser = PatentXMLParser(xml_path)
        patent_info = parser.parse()
        
        logger.info(f"対象特許: {patent_info.publication_number}")
        logger.info(f"分類コード: {patent_info.classification_codes[:5]}..." if len(patent_info.classification_codes) > 5 else f"分類コード: {patent_info.classification_codes}")
        
        # 2. 対象特許のembeddingを取得
        target_embedding = self.searcher.get_target_embedding(
            patent_info.publication_number, 
            patent_info.country_code
        )
        
        if target_embedding is None:
            logger.error("対象特許のembeddingが取得できませんでした")
            return pd.DataFrame()
        
        # 3. 候補特許のembeddingを取得
        candidates_df = self.searcher.get_candidate_embeddings(
            patent_info.classification_codes,
            patent_info.theme_codes,
            patent_info.country_code
        )
        
        if len(candidates_df) == 0:
            logger.error("候補特許が見つかりませんでした")
            return pd.DataFrame()
        
        # 4. 類似度計算してTop-Kを取得
        top_similar = self.calculator.find_top_similar(
            target_embedding, 
            candidates_df, 
            top_k
        )
        
        logger.info("=" * 80)
        logger.info("検索完了")
        logger.info("=" * 80)
        
        return top_similar


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特許類似検索システム')
    # parser.add_argument('xml_path', help='特許XMLファイルのパス')
    # parser.add_argument('--project-id', required=True, help='Google Cloud Project ID')
    # parser.add_argument('--top-k', type=int, default=1000, help='Top-K件数（デフォルト: 1000）')
    # parser.add_argument('--output', default='similar_patents.csv', help='出力CSVファイル名')
    
    load_dotenv('config.env')

    # APIキーの設定（環境変数から取得）
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        print("⚠️ config.envファイルにGCP_PROJECT_IDを設定してください")
        return None
        # APIキーの設定（環境変数から取得）

    xml_path = os.getenv("GCP_XML_PATH")
    if not project_id:
        print("⚠️ config.envファイルにGCP_PROJECT_IDを設定してください")
        return None
    
    # システム初期化
    system = PatentSimilaritySearchSystem(project_id)
    
    # 検索実行
    results = system.search(xml_path, 1000)
    
    # 結果を保存
    if len(results) > 0:
        results.to_csv(args.output, index=False, encoding='utf-8-sig')
        logger.info(f"結果を保存しました: {args.output}")
        
        # 上位10件を表示
        print("\n" + "=" * 80)
        print("Top 10 類似特許:")
        print("=" * 80)
        print(results.head(10).to_string(index=False))
    else:
        logger.error("結果が得られませんでした")


if __name__ == "__main__":
    main()