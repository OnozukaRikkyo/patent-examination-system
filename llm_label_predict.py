"""
特許審査の段階的進歩性判断システム (統合版)
幹（Claim 1）と枝葉（Claim 2以降）を段階的に検証

【統合された特徴】
- データクラスによる型安全性 (llm_pipeline.py)
- チャットセッション方式による文脈保持 (llm_pipline_gemini.py)
- 堅牢なJSONパース処理 (llm_pipeline_chatgpt.py)
- プロンプトテンプレートの外部化 (llm_pipline_gemini.py)
- 詳細な進捗表示と結果保存 (llm_pipeline.py)

【追加された特徴 (2025/11/08)】
- AIモデル2, 3, 4による先行技術調査のための検索クエリ拡張機能
- PatentSearchExpander クラスの追加
"""

import google.generativeai as genai
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from dotenv import load_dotenv
import time
from google.api_core import exceptions as google_exceptions
import re


# ==================== データクラス定義 ====================

@dataclass
class ClaimStructure:
    """クレーム構造を保持するデータクラス"""
    claim_number: int
    requirements: List[str]
    additional_limitations: Optional[List[str]] = None


@dataclass
class PatentDocument:
    """特許文献の構造化データ"""
    problem: str
    solution_principle: str
    claim1_requirements: List[str]
    claim2_limitations: Optional[List[str]] = None
    claim3_limitations: Optional[List[str]] = None
    abstract_hints: Optional[Dict[str, str]] = None


# ==================== プロンプトテンプレート ====================

class PromptTemplates:
    """プロンプトテンプレートを管理するクラス"""

    # --- 元の進歩性判断用プロンプト (そのまま) ---
    STEP_0_1_STRUCTURE_APPLICATION = """（...省略: 元のプロンプト...）"""
    STEP_0_2_STRUCTURE_PRIOR_ART = """（...省略: 元のプロンプト...）"""
    STEP_1_APPLICANT_ARGUMENTS = """（...省略: 元のプロンプト...）"""
    STEP_2_EXAMINER_REVIEW = """（...省略: 元のプロンプト...）"""
    STEP_3_FINAL_DECISION = """（...省略: 元のプロンプト...）"""

    # --- ここからAIモデル2, 3, 4用の新しいプロンプト ---

    MODEL_2_DECOMPOSE = """あなたは特許分析の専門家です。
以下の「本願発明の請求項」を読み、その発明を構成する**独立した技術的構成要素（コンポーネント）**に分解してください。

【本願発明の請求項】
{claims_text}

---
【制約事項】
- 各構成要素は、発明の必須の構成が分かるように簡潔に表現してください。
- 出力は以下のJSON形式のみとしてください。

【構造化出力フォーマット】
{{
  "components": [
    "構成要素A（例：インク滴を吐出するプリントヘッド）",
    "構成要素B（例：プリントヘッドを覆う疎油性被膜）",
    "構成要素C（例：被膜の特定の熱安定性（300℃で15%未満の重量損失））",
    "構成要素D（例：被膜の特定の物性（接触角度50°超、滑走角度30°未満））"
  ]
}}
"""

    MODEL_3_CLASSIFY_ELEMENTS = """あなたは特許分類（IPC/CPC）の専門家です。
以下の「技術的構成要素」のリストについて、**それぞれに**関連する特許分類コード（IPCまたはCPC）を予測してください。

【技術的構成要素リスト】
{components_list}

---
【制約事項】
- 各構成要素に対して、最も関連性が高いと予測される分類コードを3つ挙げてください。
- 出力は以下のJSON形式のみとしてください。

【構造化出力フォーマット】
{{
  "component_classifications": [
    {{
      "component": "構成要素A（例：インク滴を吐出するプリントヘッド）",
      "predicted_codes": ["B41J 2/14", "B41J 2/16", "B41J 2/045"]
    }},
    {{
      "component": "構成要素B（例：プリントヘッドを覆う疎油性被膜）",
      "predicted_codes": ["B41J 2/16", "C09D 127/12", "C23C 14/06"]
    }}
  ]
}}
"""

    MODEL_4_EXPAND_SEARCH = """あなたはベテランの特許調査員（サーチャー）です。
先行技術調査の網羅性を高めるため、以下の「既知の分類コード」のリストに基づき、**統計的または意味的に関連が深く、先行文献が存在しうる**他の分類コードを推薦してください。

【既知の分類コード】
{class_codes_list}

---
【制約事項】
- なぜそのコードを推薦するのか、簡潔な理由（例：「B41J 2/14」の関連技術、「C09D」の下位分類）を付与してください。
- 出力は以下のJSON形式のみとしてください。

【構造化出力フォーマット】
{{
  "recommended_codes": [
    {{
      "code": "G01N 21/00",
      "reason": "（例：被膜の物性（接触角など）を測定する技術に関連）"
    }},
    {{
      "code": "H01L 21/00",
      "reason": "（例：半導体製造プロセスにおける類似の被膜技術）"
    }},
    {{
      "code": "B05D 5/08",
      "reason": "（例：基板への特定の表面特性（撥油性など）の付与技術）"
    }}
  ]
}}
"""


# ==================== 元のメインシステムクラス (変更なし) ====================

class PatentExaminationSystemIntegrated:
    """統合版特許審査システム"""

    # def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
    # def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Args:
            api_key: Google AI Studio APIキー
            model_name: 使用するGeminiモデル
        """
        if not api_key:
            raise ValueError("APIキーが設定されていません。config.envファイルを確認してください。")

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        # JSON出力用のモデル（構造化データ用）
        self.json_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )

        # チャットセッション（文脈保持用）
        self.chat = None
        self.conversation_history = []

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
            # リスト形式で返ってきた場合は最初の要素を取得
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            return result
        except json.JSONDecodeError:
            # マークダウンのコードブロックを除去して再試行
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # ```なしのコードブロックも試す
                json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                # 最後の手段として素のテキストをパース
                return json.loads(response_text.strip())

    def _generate_with_retry(self, use_json_model: bool, prompt: str,
                            max_retries: int = 5, initial_wait: int = 2) -> str:
        """
        リトライロジック付きでコンテンツを生成

        Args:
            use_json_model: JSON出力モデルを使用するか
            prompt: プロンプト
            max_retries: 最大リトライ回数
            initial_wait: 初期待機時間（秒）

        Returns:
            レスポンステキスト
        """
        model = self.json_model if use_json_model else self.model

        for attempt in range(max_retries):
            try:
                if self.chat and not use_json_model:
                    # チャットセッションを使用（文脈保持）
                    response = self.chat.send_message(prompt)
                else:
                    # 単発のリクエスト（JSON構造化用）
                    response = model.generate_content(prompt)
                return response.text
            except google_exceptions.ResourceExhausted as e:
                if attempt < max_retries - 1:
                    wait_time = initial_wait * (4 ** attempt)  # 指数バックオフ
                    print(f"\n⏳ レート制限エラー。{wait_time}秒待機してリトライします... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n❌ 最大リトライ回数に達しました。エラー: {e}")
                    raise
            except Exception as e:
                print(f"\n❌ 予期しないエラー: {e}")
                raise

    # ... (step0_structure_application から save_results までの全メソッドは変更なし) ...
    def step0_structure_application(self, doc_dict: Dict) -> PatentDocument:
        """
        ステップ0.1: 本願発明の構造化

        Args:
            abstract: 本願発明のAbstract
            claims: 本願発明のClaimリスト

        Returns:
            構造化された本願発明データ
        """
        print("=" * 80)
        print("📋 ステップ0.1: 本願発明の構造化")
        print("=" * 80)

        abstract = doc_dict.get("abstract", "")
        claims_text = doc_dict.get("claims", "")

        prompt = PromptTemplates.STEP_0_1_STRUCTURE_APPLICATION.format(
            abstract=abstract,
            claims_text=claims_text
        )

        response_text = self._generate_with_retry(use_json_model=True, prompt=prompt)
        result = self._parse_json_response(response_text)

        print("\n✅ 構造化完了:")
        print(f"課題: {result['problem']}")
        print(f"解決原理: {result['solution_principle']}")
        print(f"Claim 1要件: {len(result['claim1_requirements'])}個")

        self.conversation_history.append({
            "step": doc_dict["step"],
            "role": "構造化",
            "content": result
        })

        return result


    def step1_applicant_arguments(self, app_data: Dict, prior_data: Dict) -> str:
        """
        ステップ1: 代理人の段階的主張

        Args:
            app_data: 本願発明の構造化データ
            prior_data: 先行技術の構造化データ

        Returns:
            代理人の主張テキスト
        """
        print("\n" + "=" * 80)
        print("⚖️ ステップ1: 代理人の段階的主張")
        print("=" * 80)

        prompt = PromptTemplates.STEP_1_APPLICANT_ARGUMENTS.format(
            app_data=json.dumps(app_data, ensure_ascii=False, indent=2),
            prior_data=json.dumps(prior_data, ensure_ascii=False, indent=2)
        )

        arguments = self._generate_with_retry(use_json_model=False, prompt=prompt)

        print("\n✅ 代理人の主張を生成しました")
        print("\n" + "-" * 80)
        print(arguments)
        print("-" * 80)

        self.conversation_history.append({
            "step": "1",
            "role": "代理人",
            "content": arguments
        })

        return arguments

    def step2_examiner_review(self, app_data: Dict, prior_data: Dict, arguments: str) -> str:
        """
        ステップ2: 審査官の段階的批評（7質問による検証）

        Args:
            app_data: 本願発明の構造化データ
            prior_data: 先行技術の構造化データ
            arguments: 代理人の主張

        Returns:
            審査官の検証・反論テキスト
        """
        print("\n" + "=" * 80)
        print("🔍 ステップ2: 審査官の専門的判断")
        print("=" * 80)

        prompt = PromptTemplates.STEP_2_EXAMINER_REVIEW.format(
            app_data=json.dumps(app_data, ensure_ascii=False, indent=2),
            prior_data=json.dumps(prior_data, ensure_ascii=False, indent=2),
            arguments=arguments
        )

        review = self._generate_with_retry(use_json_model=False, prompt=prompt)

        print("\n✅ 審査官の検証を生成しました")
        print("\n" + "-" * 80)
        print(review)
        print("-" * 80)

        self.conversation_history.append({
            "step": "2",
            "role": "審査官",
            "content": review
        })

        return review

    def step3_final_decision(self, arguments: str, review: str) -> str:
        """
        ステップ3: 主任審査官の段階的統合判断

        Args:
            arguments: 代理人の主張
            review: 審査官の検証・反論

        Returns:
            最終判断テキスト
        """
        print("\n" + "=" * 80)
        print("⚖️ ステップ3: 主任審査官の段階的統合判断")
        print("=" * 80)

        prompt = PromptTemplates.STEP_3_FINAL_DECISION.format(
            arguments=arguments,
            review=review
        )

        decision = self._generate_with_retry(use_json_model=False, prompt=prompt)

        print("\n✅ 最終判断を生成しました")
        print("\n" + "=" * 80)
        print(decision)
        print("=" * 80)

        self.conversation_history.append({
            "step": "3",
            "role": "主任審査官",
            "content": decision
        })

        return decision

    def run_full_examination(self,
                            dict_a: Dict,
                            dict_b: Dict) -> Dict:
        """
        完全な審査プロセスの実行

        Args:
            dict_a: 本願発明の構造化データ
            app_claims: 本願発明のClaimリスト
            prior_abstract: 先行技術のAbstract
            prior_claims: 先行技術のClaimリスト

        Returns:
            審査結果の辞書
        """
        print("\n" + "🚀" * 40)
        print("特許審査プロセス開始 (統合版)")
        print("🚀" * 40)

        # チャットセッションを開始（文脈保持用）
        self.chat = self.model.start_chat(history=[])

        try:
            # ステップ0: 構造化
            dict_a["step"] = "0.1 Claim"
            dict_b["step"] = "0.2 Candidate Prior Art"
            app_data = self.step0_structure_application(dict_a)
            prior_data = self.step0_structure_application(dict_b)

            # ステップ1: 代理人の主張
            arguments = self.step1_applicant_arguments(app_data, prior_data)

            # ステップ2: 審査官の検証
            review = self.step2_examiner_review(app_data, prior_data, arguments)

            # ステップ3: 最終判断
            decision = self.step3_final_decision(arguments, review)

            print("\n" + "✅" * 40)
            print("特許審査プロセス完了")
            print(decision)
            print("✅" * 40)

            inventiveness = self.judge_inventiveness(decision)

            return {
                "application_structure": app_data,
                "prior_art_structure": prior_data,
                "applicant_arguments": arguments,
                "examiner_review": review,
                "final_decision": decision,
                "conversation_history": self.conversation_history,
                "inventiveness": inventiveness
            }

        except Exception as e:
            print(f"\n--- エラーが発生しました ---")
            print(f"エラー内容: {e}")
            # エラー発生時でも部分的な結果を返す
            return {
                "error": str(e),
                "conversation_history": self.conversation_history,
                "partial_results": "処理が途中で中断されました"
            }

    def judge_inventiveness(self, final_decision_text: str) -> Dict[str, bool]:
        """
        最終判断テキストから各クレームの進歩性を抽出
        このjsonテキストを抽出して、json形式で返す。
        ```json
{
  "claim1": {
    "inventive": false,
    "reason": "レイトレーシングにおける処理速度向上ニーズは自明であり、パイプラインの分割・並列化は通常の最適化手段であるため。"
  },
  "claim2": {
    "inventive": false,
    "reason": "Claim 1の並列化が容易想到である場合、各ユニットが異なるレイを処理することは並列処理効率最大化のための技術常識であるため。"
  },
  "claim3": {
    "inventive": false,


        Args:
            final_decision_text: 最終判断のテキスト

        Returns:
            各クレームの進歩性を示す辞書

        """
        inventiveness = {}
        # ’’’json形式の部分を抽出
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', final_decision_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
            try:
                json_data = json.loads(json_text)
                # claimは何番まであるか不明なので、動的に処理
                for claim_key in json_data.keys():
                    if claim_key.startswith("claim"):
                        inventiveness[claim_key] = {
                            'inventive': json_data[claim_key]['inventive'],
                            'reason': json_data[claim_key]['reason']
                        }
                return inventiveness
            except json.JSONDecodeError:
                print("❌ 最終判断のJSONパースに失敗しました。")
                print(final_decision_text)
                return {"error": final_decision_text}



        for claim_num in range(1, 4):
            pattern = rf"### {claim_num}\. Claim {claim_num} .*?\n\*\*判断:\*\* \[(容易想到である|容易想到ではない)\]"
            match = re.search(pattern, final_decision_text, re.DOTALL)
            if match:
                inventiveness[claim_num] = (match.group(1) == "容易想到ではない")
            else:
                inventiveness[claim_num] = None  # 判定できなかった場合

        return inventiveness
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


# ==================== ★★★ 新しい実験用クラス ★★★ ====================

class PatentSearchExpander:
    """
    AIモデル2, 3, 4を実行し、特許調査クエリを拡張するためのクラス
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Args:
            api_key: Google AI Studio APIキー
            model_name: 使用するGeminiモデル
        """
        if not api_key:
            raise ValueError("APIキーが設定されていません。config.envファイルを確認してください。")

        genai.configure(api_key=api_key)
        self.model_name = model_name

        # JSON出力用のモデル（全てのステップでJSONを期待するため）
        self.json_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        JSONレスポンスを堅牢にパース (PatentExaminationSystemIntegratedから流用)
        """
        try:
            result = json.loads(response_text)
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

    def _generate_with_retry(self, prompt: str,
                            max_retries: int = 5, initial_wait: int = 2) -> str:
        """
        リトライロジック付きでコンテンツを生成 (JSONモデル専用)
        """
        for attempt in range(max_retries):
            try:
                # 常にJSONモデルを使用
                response = self.json_model.generate_content(prompt)
                return response.text
            except google_exceptions.ResourceExhausted as e:
                if attempt < max_retries - 1:
                    wait_time = initial_wait * (4 ** attempt)  # 指数バックオフ
                    print(f"\n⏳ レート制限エラー。{wait_time}秒待機してリトライします... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n❌ 最大リトライ回数に達しました。エラー: {e}")
                    raise
            except Exception as e:
                print(f"\n❌ 予期しないエラー: {e}")
                raise

    def run_model_2_decompose(self, claims_text: str) -> List[str]:
        """
        AIモデル2（分解）: 請求項から構成要素を抽出する
        """
        print("\n" + "=" * 80)
        print("🤖 AIモデル2: 構成要素の分解")
        print("=" * 80)
        prompt = PromptTemplates.MODEL_2_DECOMPOSE.format(claims_text=claims_text)
        
        response_text = self._generate_with_retry(prompt=prompt)
        parsed_json = self._parse_json_response(response_text)
        
        components = parsed_json.get("components", [])
        print(f"✅ {len(components)}個の構成要素を抽出しました。")
        for i, comp in enumerate(components):
            print(f"  [{i+1}] {comp}")
            
        return components

    def run_model_3_classify_elements(self, components: List[str]) -> Dict[str, List[str]]:
        """
        AIモデル3（要素分類）: 各構成要素に関連する分類コードを予測する
        """
        print("\n" + "=" * 80)
        print("🤖 AIモデル3: 構成要素の分類")
        print("=" * 80)
        
        # コンポーネントリストを文字列に変換
        components_list_str = "\n".join([f"- {c}" for c in components])
        
        prompt = PromptTemplates.MODEL_3_CLASSIFY_ELEMENTS.format(
            components_list=components_list_str
        )
        
        response_text = self._generate_with_retry(prompt=prompt)
        parsed_json = self._parse_json_response(response_text)
        
        classifications = parsed_json.get("component_classifications", [])
        
        # 扱いやすいように Dict[str, List[str]] 形式に変換
        result_dict = {}
        print("✅ 構成要素ごとの分類コードを予測しました。")
        for item in classifications:
            comp = item.get("component")
            codes = item.get("predicted_codes", [])
            if comp:
                result_dict[comp] = codes
                print(f"  ▶ {comp}: {codes}")
                
        return result_dict

    def run_model_4_expand_search(self, all_class_codes: List[str]) -> List[Dict[str, str]]:
        """
        AIモデル4（探索拡張）: 既知の分類コードから関連コードを推薦する
        """
        print("\n" + "=" * 80)
        print("🤖 AIモデル4: 検索クエリの拡張")
        print("=" * 80)
        
        # 重複を除去したコードリストを作成
        unique_codes = sorted(list(set(all_class_codes)))
        class_codes_list_str = "\n".join([f"- {c}" for c in unique_codes])
        
        print(f"入力コード ({len(unique_codes)}件): {unique_codes}")
        
        prompt = PromptTemplates.MODEL_4_EXPAND_SEARCH.format(
            class_codes_list=class_codes_list_str
        )
        
        response_text = self._generate_with_retry(prompt=prompt)
        parsed_json = self._parse_json_response(response_text)
        
        recommended_codes = parsed_json.get("recommended_codes", [])
        print(f"\n✅ {len(recommended_codes)}件の関連コードを推薦しました。")
        for item in recommended_codes:
            print(f"  ▶ {item.get('code')}: {item.get('reason')}")
            
        return recommended_codes

    def run_full_expansion(self, 
                           claims_text: str, 
                           invention_class_codes: List[str] = None) -> Dict[str, Any]:
        """
        モデル2, 3, 4を順番に実行し、検索クエリ拡張の全プロセスを実行する
        
        Args:
            claims_text: 本願発明の請求項テキスト
            invention_class_codes: (オプション) AIモデル1で予測された発明自体の分類コード
        
        Returns:
            実験結果の全プロセスを格納した辞書
        """
        
        if invention_class_codes is None:
            invention_class_codes = []
            
        print("\n" + "🚀" * 40)
        print("特許調査クエリ拡張プロセス開始")
        print("🚀" * 40)
        
        results = {
            "model_1_input_codes": invention_class_codes,
            "model_2_components": [],
            "model_3_classifications": {},
            "model_4_recommendations": []
        }
        
        try:
            # AIモデル2: 分解
            components = self.run_model_2_decompose(claims_text)
            results["model_2_components"] = components
            
            if not components:
                print("⚠️ モデル2で構成要素が抽出されなかったため、以降のステップをスキップします。")
                return results

            # AIモデル3: 要素分類
            classifications = self.run_model_3_classify_elements(components)
            results["model_3_classifications"] = classifications
            
            # AIモデル4への入力コードを準備
            all_codes = list(invention_class_codes) # モデル1のコード
            for codes_list in classifications.values():
                all_codes.extend(codes_list) # モデル3のコード
                
            if not all_codes:
                print("⚠️ モデル1および3で分類コードが一切得られなかったため、モデル4をスキップします。")
                return results

            # AIモデル4: 探索拡張
            recommendations = self.run_model_4_expand_search(all_codes)
            results["model_4_recommendations"] = recommendations
            
            print("\n" + "✅" * 40)
            print("特許調査クエリ拡張プロセス完了")
            print("✅" * 40)
            
            return results

        except Exception as e:
            print(f"\n--- エラーが発生しました ---")
            print(f"エラー内容: {e}")
            results["error"] = str(e)
            return results

# ==================== メイン実行関数 (変更なし) ====================

def entry(doc_dict_a, doc_dict_b):
    """
    2つのクレームファイルから特許審査を実行し、結果を返す
    (元の PatentExaminationSystemIntegrated を呼び出す)
    """
    try:
        load_dotenv('config.env')
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ config.envファイルにGOOGLE_API_KEYを設定してください")
            return None

        # 元の審査システムを初期化
        system = PatentExaminationSystemIntegrated(api_key)
        results = system.run_full_examination(doc_dict_a, doc_dict_b)   
        return results

    except ValueError as e:
        print(f"❌ 初期化エラー: {e}")
        return None
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return None

# ==================== ★★★ 新しい実験用実行関数 ★★★ ====================

def run_search_expansion_experiment():
    """
    新しい PatentSearchExpander クラスを使った実験を実行する
    """
    try:
        load_dotenv('config.env')
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ config.envファイルにGOOGLE_API_KEYを設定してください")
            return None

        # 新しい拡張システムを初期化
        expander = PatentSearchExpander(api_key)

        # --- ダミーデータ (実験用に書き換えてください) ---
        
        # AIモデル1 (分類) の結果（仮）
        model_1_results = ["B41J 2/00", "C09D 11/00"]
        
        # AIモデル2 (分解) の入力となる請求項テキスト（仮）
        claims_text_input = """
【請求項１】
インク滴を吐出するための複数のノズルが形成されたノズルプレートと、
前記ノズルプレートの表面に形成された被膜と、を備え、
前記被膜は、３００℃の温度で１５％未満の重量損失を示し、
約５０°を超える水接触角度と、約３０°未満の滑走角度を有し、
２９０℃かつ３５０ｐｓｉの環境に曝露された後も前記特性を維持する、
インクジェットプリントヘッド。

【請求項２】
前記被膜がフッ素系ポリマーを含む、ことを特徴とする請求項１に記載のインクジェットプリントヘッド。
"""
        # --- ここまでダミーデータ ---

        # 実験実行
        results = expander.run_full_expansion(
            claims_text=claims_text_input,
            invention_class_codes=model_1_results
        )
        
        print("\n--- 最終実験結果 (JSON) ---")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        
        # 結果をファイルにも保存
        output_path = "search_expansion_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 実験結果を保存しました: {output_path}")

        return results

    except ValueError as e:
        print(f"❌ 初期化エラー: {e}")
        return None
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return None

    
if __name__ == "__main__":
    # --- ★★★ こちらの実験を実行します ★★★ ---
    run_search_expansion_experiment()
    
    # --- 元の進歩性判断を実行したい場合は、以下をコメント解除 ---
    # print("（進歩性判断は実行されません）")
    # pass