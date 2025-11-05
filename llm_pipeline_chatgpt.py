"""
特許審査の段階的進歩性判断システム (Gemini版 - 旧ChatGPT版)
幹（Claim 1）と枝葉（Claim 2以降）を段階的に検証
"""

import google.generativeai as genai
import os
from typing import Dict, List
import json
from dotenv import load_dotenv
import time
from google.api_core import exceptions as google_exceptions

load_dotenv('config.env')


class PatentExaminationSystemChatGPT:
    """特許審査システム (Gemini版)"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Args:
            api_key: Google API キー
            model_name: 使用するGeminiモデル
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        # JSON出力用のモデル（構造化データ用）
        self.json_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        self.model_name = model_name
        self.conversation_history = []

    def _call_api_with_retry(self, model, prompt: str, max_retries: int = 3) -> str:
        """APIをリトライロジック付きで呼び出す"""
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except google_exceptions.ResourceExhausted as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * (2 ** attempt)
                    print(f"\n⏳ レート制限エラー。{wait_time}秒待機... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n❌ 最大リトライ回数到達: {e}")
                    raise
            except Exception as e:
                print(f"\n❌ エラー: {e}")
                raise

    def run_examination(self, app_abstract: str, app_claims: List[str],
                       prior_abstract: str, prior_claims: List[str]) -> Dict:
        """完全な審査プロセスを実行"""
        print("\n" + "🚀" * 40)
        print("特許審査プロセス開始 (Gemini版)")
        print("🚀" * 40)

        # ステップ0.1: 本願発明の構造化
        print("\n" + "=" * 80)
        print("📋 ステップ0.1: 本願発明の構造化")
        print("=" * 80)

        claims_text = "\n".join([f"Claim {i+1}: {c}" for i, c in enumerate(app_claims)])
        prompt = f"""以下の本願発明を構造化してJSON形式で出力:

【本願発明】
Abstract: {app_abstract}
{claims_text}

JSON形式:
{{
  "problem": "課題",
  "solution_principle": "解決原理",
  "claim1_requirements": ["要件A", "要件B"],
  "claim2_limitations": ["追加限定"],
  "claim3_limitations": ["追加限定"]
}}

JSON形式のみで回答してください。"""

        app_data_text = self._call_api_with_retry(self.json_model, prompt)
        try:
            app_data = json.loads(app_data_text)
            # リスト形式で返ってきた場合は最初の要素を取得
            if isinstance(app_data, list) and len(app_data) > 0:
                app_data = app_data[0]
        except json.JSONDecodeError:
            # JSONパースに失敗した場合、マークダウンのコードブロックを除去して再試行
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', app_data_text, re.DOTALL)
            if json_match:
                app_data = json.loads(json_match.group(1))
            else:
                app_data = json.loads(app_data_text.strip())
        print(f"✅ 課題: {app_data['problem']}")

        # ステップ0.2: 先行技術の構造化
        print("\n" + "=" * 80)
        print("📋 ステップ0.2: 先行技術の構造化")
        print("=" * 80)

        prior_claims_text = "\n".join([f"Claim {i+1}: {c}" for i, c in enumerate(prior_claims)])
        prompt = f"""以下の先行技術を構造化してJSON形式で出力（Abstractの示唆を含む）:

【先行技術】
Abstract: {prior_abstract}
{prior_claims_text}

JSON形式:
{{
  "problem": "課題",
  "solution_principle": "解決原理",
  "claim1_requirements": ["要件X", "要件Y"],
  "abstract_hints": {{"temperature_range": "範囲"}}
}}

JSON形式のみで回答してください。"""

        prior_data_text = self._call_api_with_retry(self.json_model, prompt)
        try:
            prior_data = json.loads(prior_data_text)
            # リスト形式で返ってきた場合は最初の要素を取得
            if isinstance(prior_data, list) and len(prior_data) > 0:
                prior_data = prior_data[0]
        except json.JSONDecodeError:
            # JSONパースに失敗した場合、マークダウンのコードブロックを除去して再試行
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', prior_data_text, re.DOTALL)
            if json_match:
                prior_data = json.loads(json_match.group(1))
            else:
                prior_data = json.loads(prior_data_text.strip())
        print(f"✅ 課題: {prior_data['problem']}")

        # ステップ1: 代理人の主張
        print("\n" + "=" * 80)
        print("⚖️ ステップ1: 代理人の主張")
        print("=" * 80)

        prompt = f"""あなたは本願発明の代理人です。進歩性を主張してください。

【本願発明】
{json.dumps(app_data, ensure_ascii=False, indent=2)}

【先行技術】
{json.dumps(prior_data, ensure_ascii=False, indent=2)}

Claim 1の進歩性とClaim 2以降の予備的主張を展開してください。"""

        arguments = self._call_api_with_retry(self.model, prompt)
        print("✅ 主張を生成しました")

        # ステップ2: 審査官の検証
        print("\n" + "=" * 80)
        print("🔍 ステップ2: 審査官の検証")
        print("=" * 80)

        prompt = f"""あなたは審査官です。代理人の主張を検証してください。

【代理人の主張】
{arguments}

7つの質問（課題共通性、解決原理、最適化、動機付け、阻害要因、予期せぬ効果、結論）に答え、
Claim 1-3の容易想到性を検証してください。"""

        review = self._call_api_with_retry(self.model, prompt)
        print("✅ 検証を生成しました")

        # ステップ3: 最終判断
        print("\n" + "=" * 80)
        print("⚖️ ステップ3: 最終判断")
        print("=" * 80)

        prompt = f"""あなたは主任審査官です。最終判断を下してください。

【代理人の主張】
{arguments}

【審査官の検証】
{review}

Claim 1-3それぞれの進歩性判断と総合結論を述べてください。"""

        decision = self._call_api_with_retry(self.model, prompt)
        print("✅ 最終判断を生成しました")
        print("\n" + decision)

        print("\n" + "✅" * 40)
        print("審査プロセス完了")
        print("✅" * 40)

        return {
            "application_structure": app_data,
            "prior_art_structure": prior_data,
            "applicant_arguments": arguments,
            "examiner_review": review,
            "final_decision": decision
        }


def main():
    """メイン実行関数"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ config.envにGOOGLE_API_KEYを設定してください")
        return

    system = PatentExaminationSystemChatGPT(api_key)

    app_abstract = """
    本発明は、インクジェットプリントヘッドのノズルプレートに関し、
    特に高温・高圧環境下での耐久性を向上させた疎油性被膜を提供する。
    この被膜は、300℃で15%未満の重量損失、50°超の接触角度、
    30°未満の滑走角度を有し、290℃かつ350psiに曝露後も性能を維持する。
    """

    app_claims = [
        "最高300℃で15%未満の重量損失を有し、接触角度が約50°超であり、滑走角度が約30°未満であり、290℃かつ350psiに曝露後も性能を維持する疎油性被膜を備えたノズルプレート。",
        "前記被膜がフッ素系ポリマーを含む、請求項1に記載のノズルプレート。",
        "前記被膜の膜厚が1μm～5μmである、請求項1または2に記載のノズルプレート。"
    ]

    prior_abstract = """
    高温加熱による表面特性の低下を防止し、汚れを低減するための
    熱に安定な撥油性低接着性コーティングを提供する。
    このコーティングは、滑走角度が約30°未満であり、
    200℃に30分曝露後も性能を維持する。
    好ましくは、接触角度は45°よりも大きく、
    180℃〜320℃の温度範囲および100psi〜400psiの圧力範囲で使用可能である。
    """

    prior_claims = [
        "滑走角度が約30°未満であり、200℃に30分曝露後も性能を維持する撥油性コーティング。"
    ]

    results = system.run_examination(app_abstract, app_claims, prior_abstract, prior_claims)

    output_path = "patent_examination_results_chatgpt.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 結果を保存: {output_path}")


if __name__ == "__main__":
    main()
