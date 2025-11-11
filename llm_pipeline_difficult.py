"""
特許審査の段階的進歩性判断システム (RAG統合版)
提案されたワークフローに基づく実装:
- ステップ1: 本願発明の構造化
- ステップ2: 対比用RAG（先行技術1からスニペット抽出）
- ステップ3: LLMによる対比表作成
- ステップ4: 新規性判定と相違点確定
- ステップ5: 動機付け検索（RAG）
- ステップ6: 進歩性の仮判定と自信度評価
- ステップ7: 人間レビュー用データ出力
"""

import google.generativeai as genai
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from dotenv import load_dotenv
import time
from google.api_core import exceptions as google_exceptions
import re


# ==================== データクラス定義 ====================

@dataclass
class PatentDocument:
    """特許文献の構造化データ"""
    problem: str
    solution_principle: str
    claim1_requirements: List[str]


@dataclass
class NoveltyReport:
    """新規性判定レポート"""
    novelty_judgement: str  # "あり" or "なし"
    difference_points: List[str]


@dataclass
class InventiveStepReport:
    """進歩性判定レポート"""
    judgement: str  # "進歩性あり" or "進歩性なし" or "判断困難"
    confidence: str  # "高" or "中" or "低"
    rationale: str
    low_confidence_points: List[str]


# ==================== プロンプトテンプレート ====================

class PromptTemplates:
    """プロンプトテンプレートを管理するクラス"""

    # ステップ1: 本願発明の構造化
    STEP1_STRUCTURE_APPLICATION = """以下の「本願発明」のAbstractおよび全てのClaimを読み、特許判断に必要な要素を以下のJSON形式で抽出・構造化してください。

【本願発明テキスト】
{application_text}

【構造化出力フォーマット】
以下のJSON形式で出力してください：
```json
{{
  "problem": "課題（例：ノズルプレートの機械的頑強性の向上）",
  "solution_principle": "解決原理（例：高熱安定性・特定の物性を持つ疎油性被膜の適用）",
  "claim1_requirements": [
    "要件A: （例：最高300℃で15%未満の重量損失）",
    "要件B: （例：接触角度 約50°超）",
    "要件C: （例：滑走角度 約30°未満）"
  ]
}}
```

JSON形式のみで回答してください。"""

    # ステップ3: 対比表（クレームチャート）の作成
    STEP3_CLAIM_CHART = """あなたは、特許分析を支援する優秀なパラリーガルです。
以下の【本願発明の構成要件】と【先行技術1の関連抜粋】を厳密に比較し、法的判断（新規性・進歩性の有無）は一切行わず、**事実の対比のみ**を行った「対比表（クレームチャート）」をMarkdown形式で作成してください。

* 「先行技術1の対応記載」列には、【先行技術1の関連抜粋】から最も関連する記述を**正確に引用**してください。
* もし【先行技術1の関連抜粋】の中に、該当する記述が見当たらない場合は、明確に「**記載なし**」と記入してください。

-----

**【本願発明の構成要件】:**
```json
{claim1_requirements}
```

**【先行技術1の関連抜粋】:**
```text
{prior_art_snippets}
```

-----

**【出力：対比表】**
以下のMarkdownテーブル形式で出力してください：

| 本願発明の要件 | 先行技術1の対応記載（引用） | 一致/相違 |
| :--- | :--- | :--- |
| ... | ... | ... |"""

    # ステップ4: 新規性の判定と相違点の確定
    STEP4_NOVELTY_JUDGEMENT = """あなたは特許審査官です。
以下の【対比表】を読み、本願発明の「新規性」の有無を機械的に判定し、「相違点」を正確にリストアップしてください。

**判定ルール:**
1. 「一致/相違」列に「相違」が1つでも存在する場合、新規性は「あり」です。
2. すべての要件が「一致」または「実質一致」の場合、新規性は「なし」です。

**【対比表】:**
```markdown
{claim_chart}
```

**【出力フォーマット（JSON）】:**
```json
{{
  "novelty_judgement": "あり" | "なし",
  "difference_points": [
    "（「相違」と判定された要件のテキストをここに記載）"
  ]
}}
```

JSON形式のみで回答してください。"""

    # ステップ6: 進歩性の仮判定と自信度評価
    STEP6_INVENTIVE_STEP = """あなたは、特許審査官（訓練中）です。
以下の【証拠リスト】に基づき、本願発明（クレーム1）の「進歩性」について仮判定を行ってください。

【証拠リスト】
* **証拠1 (本願発明):**
```json
{application_data}
```

* **証拠2 (先行技術1との対比):**
```markdown
{claim_chart}
```

* **証拠3 (相違点):**
```json
{difference_points}
```

* **証拠4 (他の先行技術からの示唆):**
```text
{motivation_snippets}
```

-----

【タスク】
以下のJSON形式で「進歩性仮判定レポート」を作成してください。

```json
{{
  "judgement": "進歩性あり" | "進歩性なし" | "判断困難",
  "confidence": "高" | "中" | "低",
  "rationale": "（なぜそのように判定したかの具体的根拠。証拠4の引用を含む）",
  "low_confidence_points": [
    "（自信度が「中」または「低」の場合、判断を迷わせている要因や、追加調査が必要な点を具体的に記述する。）"
  ]
}}
```

JSON形式のみで回答してください。"""


# ==================== RAGシステム（簡易版）====================

class SimpleRAGSystem:
    """
    簡易的なRAGシステム
    実際の実装では、Chroma/Pinecone/FAISSなどのベクトルDBを使用
    """

    def __init__(self):
        self.documents = {}  # doc_id -> full_text
        self.chunks = {}  # chunk_id -> (doc_id, chunk_text)

    def index_document(self, doc_id: str, full_text: str, chunk_size: int = 500):
        """
        文書をチャンク化してインデックス
        
        Args:
            doc_id: 文書ID
            full_text: 文書全文
            chunk_size: チャンクサイズ（文字数）
        """
        self.documents[doc_id] = full_text
        
        # 簡易的なチャンク化（段落ベース）
        paragraphs = full_text.split('\n\n')
        chunk_id = 0
        for para in paragraphs:
            if para.strip():
                self.chunks[f"{doc_id}_chunk_{chunk_id}"] = (doc_id, para.strip())
                chunk_id += 1

    def search(self, queries: List[str], top_k: int = 5) -> str:
        """
        クエリに関連するスニペットを検索
        
        Args:
            queries: 検索クエリのリスト
            top_k: 各クエリで取得する上位件数
            
        Returns:
            統合されたスニペットテキスト
        """
        # 簡易的なキーワードマッチング（実際はベクトル検索を使用）
        all_snippets = set()
        
        for query in queries:
            keywords = query.lower().split()
            scored_chunks = []
            
            for chunk_id, (doc_id, chunk_text) in self.chunks.items():
                score = sum(1 for keyword in keywords if keyword in chunk_text.lower())
                if score > 0:
                    scored_chunks.append((score, chunk_id, chunk_text))
            
            # スコア順にソート
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            
            # Top-k件を取得
            for _, _, chunk_text in scored_chunks[:top_k]:
                all_snippets.add(chunk_text)
        
        return "\n...\n".join(all_snippets)


# ==================== メインシステムクラス ====================

class PatentExaminationSystemRAG:
    """RAG統合版特許審査システム"""

#    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        """
        Args:
            api_key: Google AI Studio APIキー
            model_name: 使用するGeminiモデル
        """
        if not api_key:
            raise ValueError("APIキーが設定されていません。")

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        # JSON出力用のモデル
        self.json_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )

        # RAGシステム
        self.rag_system = SimpleRAGSystem()
        
        # 処理履歴
        self.processing_history = []

    def _generate_with_retry(self, use_json_model: bool, prompt: str, 
                            max_retries: int = 3) -> str:
        """
        リトライ機能付きでLLMを呼び出す
        
        Args:
            use_json_model: JSON出力モデルを使用するか
            prompt: プロンプトテキスト
            max_retries: 最大リトライ回数
            
        Returns:
            生成されたテキスト
        """
        model = self.json_model if use_json_model else self.model
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except google_exceptions.ResourceExhausted:
                wait_time = (attempt + 1) * 5
                print(f"⚠️ レート制限に達しました。{wait_time}秒待機します...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"⚠️ エラー発生（試行 {attempt + 1}/{max_retries}）: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
        
        raise Exception("最大リトライ回数を超えました")

    def _parse_json_response(self, response_text: str) -> Dict:
        """
        JSONレスポンスを堅牢にパース
        
        Args:
            response_text: レスポンステキスト
            
        Returns:
            パースされたJSON辞書
        """
        try:
            result = json.loads(response_text)
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            return result
        except json.JSONDecodeError:
            # マークダウンのコードブロックを除去して再試行
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                return json.loads(response_text.strip())

    def step1_structure_application(self, application_text: str) -> Dict:
        """
        ステップ1: 本願発明の構造化
        
        Args:
            application_text: 本願発明のテキスト（Abstract + Claims）
            
        Returns:
            構造化されたJSON辞書
        """
        print("\n" + "=" * 80)
        print("📋 ステップ1: 本願発明の構造化")
        print("=" * 80)

        prompt = PromptTemplates.STEP1_STRUCTURE_APPLICATION.format(
            application_text=application_text
        )

        response_text = self._generate_with_retry(use_json_model=True, prompt=prompt)
        result = self._parse_json_response(response_text)

        print("\n✅ 構造化完了:")
        print(f"課題: {result['problem']}")
        print(f"解決原理: {result['solution_principle']}")
        print(f"Claim 1要件: {len(result['claim1_requirements'])}個")

        self.processing_history.append({
            "step": "1",
            "name": "本願発明の構造化",
            "output": result
        })

        return result

    def step2_rag_comparison(self, claim1_requirements: List[str], 
                           prior_art_text: str) -> str:
        """
        ステップ2: 対比用RAG（先行技術1からスニペット抽出）
        
        Args:
            claim1_requirements: 本願のクレーム1の要件リスト
            prior_art_text: 先行技術1の全文
            
        Returns:
            抽出されたスニペットテキスト
        """
        print("\n" + "=" * 80)
        print("🔍 ステップ2: 対比用RAG（スニペット抽出）")
        print("=" * 80)

        # 先行技術1をインデックス化
        self.rag_system.index_document("PriorArt_1", prior_art_text)

        # 各要件を検索クエリとして使用
        print(f"\n検索クエリ数: {len(claim1_requirements)}")
        snippets = self.rag_system.search(claim1_requirements, top_k=5)

        print(f"\n✅ スニペット抽出完了（{len(snippets.split('...'))}件）")

        self.processing_history.append({
            "step": "2",
            "name": "対比用RAG",
            "queries": claim1_requirements,
            "output": snippets
        })

        return snippets

    def step3_claim_chart(self, claim1_requirements: List[str], 
                         prior_art_snippets: str) -> str:
        """
        ステップ3: LLMによる対比表（クレームチャート）の作成
        
        Args:
            claim1_requirements: 本願のクレーム1の要件リスト
            prior_art_snippets: 先行技術1の関連スニペット
            
        Returns:
            対比表（Markdown形式）
        """
        print("\n" + "=" * 80)
        print("📊 ステップ3: 対比表（クレームチャート）の作成")
        print("=" * 80)

        prompt = PromptTemplates.STEP3_CLAIM_CHART.format(
            claim1_requirements=json.dumps(claim1_requirements, ensure_ascii=False, indent=2),
            prior_art_snippets=prior_art_snippets
        )

        claim_chart = self._generate_with_retry(use_json_model=False, prompt=prompt)

        print("\n✅ 対比表作成完了")
        print("\n" + "-" * 80)
        print(claim_chart[:500] + "..." if len(claim_chart) > 500 else claim_chart)
        print("-" * 80)

        self.processing_history.append({
            "step": "3",
            "name": "対比表作成",
            "output": claim_chart
        })

        return claim_chart

    def step4_novelty_judgement(self, claim_chart: str) -> Dict:
        """
        ステップ4: LLMによる新規性の判定と「相違点」の確定
        
        Args:
            claim_chart: ステップ3で作成した対比表
            
        Returns:
            新規性判定レポート（JSON）
        """
        print("\n" + "=" * 80)
        print("⚖️ ステップ4: 新規性の判定と相違点の確定")
        print("=" * 80)

        prompt = PromptTemplates.STEP4_NOVELTY_JUDGEMENT.format(
            claim_chart=claim_chart
        )

        response_text = self._generate_with_retry(use_json_model=True, prompt=prompt)
        result = self._parse_json_response(response_text)

        print(f"\n✅ 新規性判定: {result['novelty_judgement']}")
        if result['difference_points']:
            print(f"相違点数: {len(result['difference_points'])}")
            for i, diff in enumerate(result['difference_points'], 1):
                print(f"  {i}. {diff[:100]}...")

        self.processing_history.append({
            "step": "4",
            "name": "新規性判定",
            "output": result
        })

        return result

    def step5_motivation_search(self, problem: str, difference_points: List[str],
                               all_patents_text: str) -> str:
        """
        ステップ5: 進歩性判断のための「動機付け」検索（RAG）
        
        Args:
            problem: 本願の課題
            difference_points: ステップ4で特定された相違点
            all_patents_text: 先行技術1以外の全特許文献
            
        Returns:
            動機付けスニペット
        """
        print("\n" + "=" * 80)
        print("🔍 ステップ5: 動機付け検索（RAG）")
        print("=" * 80)

        # 全特許DBをインデックス化
        self.rag_system.index_document("All_Patents", all_patents_text)

        # 検索クエリを生成（課題 + 相違点）
        queries = [problem] + difference_points

        print(f"\n検索クエリ数: {len(queries)}")
        motivation_snippets = self.rag_system.search(queries, top_k=3)

        print(f"\n✅ 動機付けスニペット抽出完了")

        self.processing_history.append({
            "step": "5",
            "name": "動機付け検索",
            "queries": queries,
            "output": motivation_snippets
        })

        return motivation_snippets

    def step6_inventive_step_judgement(self, application_data: Dict, 
                                      claim_chart: str,
                                      difference_points: List[str],
                                      motivation_snippets: str) -> Dict:
        """
        ステップ6: LLMによる進歩性の「仮判定」と「自信度」評価
        
        Args:
            application_data: 本願発明の構造化データ
            claim_chart: 対比表
            difference_points: 相違点リスト
            motivation_snippets: 動機付けスニペット
            
        Returns:
            進歩性判定レポート（JSON）
        """
        print("\n" + "=" * 80)
        print("⚖️ ステップ6: 進歩性の仮判定と自信度評価")
        print("=" * 80)

        prompt = PromptTemplates.STEP6_INVENTIVE_STEP.format(
            application_data=json.dumps(application_data, ensure_ascii=False, indent=2),
            claim_chart=claim_chart,
            difference_points=json.dumps(difference_points, ensure_ascii=False, indent=2),
            motivation_snippets=motivation_snippets
        )

        response_text = self._generate_with_retry(use_json_model=True, prompt=prompt)
        result = self._parse_json_response(response_text)

        print(f"\n✅ 進歩性仮判定: {result['judgement']}")
        print(f"自信度: {result['confidence']}")
        if result.get('low_confidence_points'):
            print(f"\n⚠️ 低自信度ポイント数: {len(result['low_confidence_points'])}")
            for i, point in enumerate(result['low_confidence_points'], 1):
                print(f"  {i}. {point[:100]}...")

        self.processing_history.append({
            "step": "6",
            "name": "進歩性仮判定",
            "output": result
        })

        return result

    def step7_prepare_human_review(self) -> Dict:
        """
        ステップ7: 人間（専門家）による最終レビュー用データの準備
        
        Returns:
            人間レビュー用データ
        """
        print("\n" + "=" * 80)
        print("👤 ステップ7: 人間レビュー用データの準備")
        print("=" * 80)

        review_data = {
            "processing_history": self.processing_history,
            "review_instructions": """
【人間レビュー指示】
1. ステップ6の「judgement」と「confidence」を確認してください。
2. 「confidence」が「中」または「低」の場合、「low_confidence_points」を集中的にレビューしてください。
3. 必要に応じて追加調査を実施し、最終判断を下してください。
4. 最終判断を以下の形式で記録してください：
   - 進歩性の有無: [あり/なし]
   - 判断理由: [詳細な理由]
   - 追加調査内容: [実施した追加調査の内容]
"""
        }

        print("\n✅ 人間レビュー用データ準備完了")

        return review_data

    def run_full_examination(self, application_text: str, 
                           prior_art_1_text: str,
                           all_patents_text: str) -> Dict:
        """
        完全な審査プロセスの実行
        
        Args:
            application_text: 本願発明のテキスト
            prior_art_1_text: 先行技術1のテキスト
            all_patents_text: 先行技術1以外の全特許文献
            
        Returns:
            審査結果の辞書
        """
        print("\n" + "🚀" * 40)
        print("特許審査プロセス開始 (RAG統合版)")
        print("🚀" * 40)

        try:
            # ステップ1: 本願発明の構造化
            app_data = self.step1_structure_application(application_text)

            # ステップ2: 対比用RAG
            prior_art_snippets = self.step2_rag_comparison(
                app_data['claim1_requirements'],
                prior_art_1_text
            )

            # ステップ3: 対比表作成
            claim_chart = self.step3_claim_chart(
                app_data['claim1_requirements'],
                prior_art_snippets
            )

            # ステップ4: 新規性判定
            novelty_report = self.step4_novelty_judgement(claim_chart)

            # 新規性がない場合は、進歩性判定をスキップ
            if novelty_report['novelty_judgement'] == 'なし':
                print("\n⚠️ 新規性なし。進歩性判定をスキップします。")
                inventive_step_report = {
                    "judgement": "進歩性なし",
                    "confidence": "高",
                    "rationale": "新規性がないため、進歩性も認められません。",
                    "low_confidence_points": []
                }
            else:
                # ステップ5: 動機付け検索
                motivation_snippets = self.step5_motivation_search(
                    app_data['problem'],
                    novelty_report['difference_points'],
                    all_patents_text
                )

                # ステップ6: 進歩性仮判定
                inventive_step_report = self.step6_inventive_step_judgement(
                    app_data,
                    claim_chart,
                    novelty_report['difference_points'],
                    motivation_snippets
                )

            # ステップ7: 人間レビュー用データ準備
            human_review_data = self.step7_prepare_human_review()

            print("\n" + "✅" * 40)
            print("特許審査プロセス完了")
            print("✅" * 40)

            return {
                "step1_application_structure": app_data,
                "step2_prior_art_snippets": prior_art_snippets,
                "step3_claim_chart": claim_chart,
                "step4_novelty_report": novelty_report,
                "step6_inventive_step_report": inventive_step_report,
                "step7_human_review_data": human_review_data,
                "summary": {
                    "novelty": novelty_report['novelty_judgement'],
                    "inventive_step": inventive_step_report['judgement'],
                    "confidence": inventive_step_report['confidence'],
                    "requires_human_review": inventive_step_report['confidence'] in ['中', '低']
                }
            }

        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "processing_history": self.processing_history,
                "partial_results": "処理が途中で中断されました"
            }

    def save_results(self, results: Dict, output_path: str):
        """
        審査結果をJSONファイルに保存
        
        Args:
            results: 審査結果の辞書
            output_path: 出力ファイルパス
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 結果を保存しました: {output_path}")


# ==================== メイン実行関数 ====================

def entry(application_dict: Dict, prior_art_dict: Dict, 
         all_patents_dict: Optional[Dict] = None):
    """
    特許審査を実行し、結果を返す
    
    Args:
        application_dict: 本願発明のテキスト辞書
            例: {"abstract": "...", "claims": "..."}
        prior_art_dict: 先行技術1のテキスト辞書
            例: {"abstract": "...", "claims": "..."}
        all_patents_dict: その他の全特許文献（オプション）
            例: {"document1": "...", "document2": "..."}
    
    Returns:
        dict: 審査結果の辞書、エラー時はNone
    """
    try:
        # config.envファイルから環境変数を読み込む
        load_dotenv('config.env')

        # APIキーの設定
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ config.envファイルにGOOGLE_API_KEYを設定してください")
            return None

        # システムの初期化
        system = PatentExaminationSystemRAG(api_key)

        # テキストの統合
        application_text = f"""Abstract: {application_dict.get('abstract', '')}

Claims: {application_dict.get('claims', '')}"""

        prior_art_1_text = f"""Abstract: {prior_art_dict.get('abstract', '')}

Claims: {prior_art_dict.get('claims', '')}"""

        # 全特許文献の統合（オプション）
        all_patents_text = ""
        if all_patents_dict:
            all_patents_text = "\n\n".join(all_patents_dict.values())
        else:
            # デフォルトで先行技術1以外の架空の文献を追加
            all_patents_text = """
[文献A] プラズマCVD法を改良し、DLC膜の硬度を最大6Gpaまで高めることに成功した。
[文献B] 高温環境下での使用（350℃）に対応するため、DLC膜にシリコン（Si）をドープすることで耐熱性を350℃まで向上させた。
"""

        # 完全な審査プロセスの実行
        results = system.run_full_examination(
            application_text,
            prior_art_1_text,
            all_patents_text
        )

        return results

    except ValueError as e:
        print(f"❌ 初期化エラー: {e}")
        return None
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # テスト用のサンプルデータ
    sample_application = {
        "abstract": "インクジェットノズルの耐摩耗性と耐熱性を向上させる。",
        "claims": """【請求項１】
炭化ケイ素（SiC）からなる基材と、
前記基材上に形成された、ダイヤモンドライクカーボン（DLC）膜と、
前記DLC膜が、5Gpa以上の硬度と、300℃以上の耐熱性を有することを特徴とする、ノズルヘッド。"""
    }

    sample_prior_art = {
        "abstract": "高温加熱による表面特性の低下を防止し、汚れを低減する。",
        "claims": """【請求項１】
[0025] 本発明のノズルは、基材として炭化ケイ素（SiC）を用いることで、優れた剛性を確保する。
[0038] 耐摩耗性を高めるため、表面にDLC（ダイヤモンドライクカーボン）によるコーティングを施す。
[0042] 実施例１のDLC膜は、ナノインデンテーション試験において4.5Gpaの硬度を示した。
[0056] 熱安定性試験において、本コーティングは250℃までの環境下で安定した特性を維持した。"""
    }

    print("=" * 80)
    print("テスト実行開始")
    print("=" * 80)

    results = entry(sample_application, sample_prior_art)

    if results and "error" not in results:
        print("\n" + "=" * 80)
        print("📊 最終結果サマリー")
        print("=" * 80)
        print(f"新規性: {results['summary']['novelty']}")
        print(f"進歩性: {results['summary']['inventive_step']}")
        print(f"自信度: {results['summary']['confidence']}")
        print(f"人間レビュー必要: {results['summary']['requires_human_review']}")