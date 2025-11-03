"""
特許審査の段階的進歩性判断システム (統合版)
幹（Claim 1）と枝葉（Claim 2以降）を段階的に検証

【統合された特徴】
- データクラスによる型安全性 (llm_pipeline.py)
- チャットセッション方式による文脈保持 (llm_pipline_gemini.py)
- 堅牢なJSONパース処理 (llm_pipeline_chatgpt.py)
- プロンプトテンプレートの外部化 (llm_pipline_gemini.py)
- 詳細な進捗表示と結果保存 (llm_pipeline.py)
"""

import google.generativeai as genai
import os
from typing import Dict, List, Optional
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

    STEP_0_1_STRUCTURE_APPLICATION = """以下の「本願発明」のAbstractおよび全てのClaimを読み、特許判断に必要な要素を以下の形式で抽出・構造化してください。

【本願発明】
Abstract: {abstract}

{claims_text}

---
【構造化出力フォーマット】
以下のJSON形式で出力してください：

{{
  "problem": "課題（例：ノズルプレートの機械的頑強性の向上）",
  "solution_principle": "解決原理（例：高熱安定性・特定の物性を持つ疎油性被膜の適用）",
  "claim1_requirements": [
    "要件A: （例：最高300℃で15%未満の重量損失）",
    "要件B: （例：接触角度 約50°超）",
    "要件C: （例：滑走角度 約30°未満）",
    "要件D: （例：290℃ かつ 350psiに曝露後も性能維持）"
  ],
  "claim2_limitations": [
    "（例：前記被膜がフッ素系ポリマーを含む、こと。）"
  ],
  "claim3_limitations": [
    "（例：前記被膜の膜厚が1μm～5μmである、こと。）"
  ]
}}

JSON形式のみで回答してください。"""

    STEP_0_2_STRUCTURE_PRIOR_ART = """同様に、以下の「先行技術」のAbstractおよび全てのClaimを読み、同じ形式で構造化してください。**特にAbstractの「示唆（ヒント）」**を重要視してください。

【先行技術】
Abstract: {abstract}

{claims_text}

---
【構造化出力フォーマット】
以下のJSON形式で出力してください：

{{
  "problem": "課題（例：高温加熱による表面特性の低下防止、汚れ低減）",
  "solution_principle": "解決原理（例：熱に安定な撥油性低接着性コーティングの適用）",
  "claim1_requirements": [
    "要件X: （例：滑走角度 約30°未満）",
    "要件Y: （例：200℃に30分曝露後も性能維持）"
  ],
  "abstract_hints": {{
    "contact_angle": "（例：45°よりも大きな）",
    "temperature_range": "（例：180℃〜320℃の範囲）",
    "pressure_range": "（例：100psi〜400psiの範囲）"
  }}
}}

JSON形式のみで回答してください。"""

    STEP_1_APPLICANT_ARGUMENTS = """あなたは「本願発明」の代理人です。
先ほど構造化した2つの文献データに基づき、以下の2段階で「進歩性がある（容易に考えつけない）」という論理的な主張を構築してください。

【本願発明の構造化データ】
{app_data}

【先行技術の構造化データ】
{prior_data}

---

1. **第一の主張 (幹):**
   まず、**本願発明のClaim 1 (幹)**が、先行技術と比較して進歩性を有することを主張してください。
   *（ヒント：先行技術のClaim 1にはない要件の存在や、共通する要件の決定的な差異を強調する。）*

2. **予備的主張 (枝葉):**
   **仮に、Claim 1の進歩性が否定されたとしても**、**Claim 2の追加限定 (枝1)**や**Claim 3の追加限定 (枝2)**を先行技術に適用することは、先行技術からは動機付けがなく、容易想到ではないと主張してください。

---

以下の構造で主張を展開してください：

## 第一の主張：Claim 1の進歩性

### 1. 課題・解決原理の相違点
[本願発明と先行技術の課題・解決原理の違いを説明]

### 2. 構成要件の相違点
[Claim 1の要件と先行技術の要件の具体的な違いを列挙]

### 3. 進歩性の根拠
[なぜこの相違点が単なる最適化ではなく、進歩性を有するのかを論理的に説明]

## 予備的主張：Claim 2以降の進歩性

### Claim 2の追加限定について
[Claim 2の追加限定が先行技術から容易想到でない理由]

### Claim 3の追加限定について
[Claim 3の追加限定が先行技術から容易想到でない理由]
"""

    STEP_2_EXAMINER_REVIEW = """役割を変更します。あなたは特許庁の「審査官」です。
ステップ1の「代理人の主張」を論破するため、以下の2段階で検証と反論（＝進歩性なしのロジック）を構築してください。

【本願発明の構造化データ】
{app_data}

【先行技術の構造化データ】
{prior_data}

【代理人の主張】
{arguments}

---

## 第1段階：Claim 1 (幹) の検証

ステップ0の構造化データ（特に先行技術の**Abstractの示唆**）を参照し、以下の7つの質問に答える形式で、**本願発明のClaim 1 (幹)**が**進歩性を欠く（容易想到である）**という結論を導いてください。

### 質問1: 課題は共通か？
両者の「課題」は実質的に同一（例：耐久性向上）か？

### 質問2: 解決原理は同一か？
両者の「解決原理」は実質的に同一（例：耐熱性低接着被膜）か？

### 質問3: 差分は最適化か？
Claim 1の要件は、先行技術の原理に対する**単なる最適化や設計変更**の範囲内ではないか？

### 質問4: 動機付けはあるか？
先行技術のAbstractの**示唆（温度範囲、圧力範囲等）**は、当業者がClaim 1の数値を試みる十分な**動機付け**にならないか？

### 質問5: 阻害要因はないか？
先行技術に、本願発明の方向性を**妨げる記載**はあるか？ なければ阻害要因なし。

### 質問6: 予期せぬ効果はあるか？
Claim 1の数値にしたことで、先行技術からは**予測できない異質な効果**が生じているか？ 単なる「耐久性が向上した」という**程度の差**ではないか？

### 質問7: 結論（容易想到か）？
上記1〜6より、当業者が先行技術に基づき、通常の実験（最適化）でClaim 1に到達することは**容易**ではないか？

---

## 第2段階：Claim 2以降 (枝葉) の検証

第1段階の結論に基づき、**「Claim 1は進歩性なし」と仮定**します。

### Claim 2の追加限定の検証
この技術分野において、Claim 2の追加限定を適用することは**周知の選択肢**または**技術常識**ではありませんか？ 先行技術に適用することに、何か**困難や阻害要因**がありますか？ なければ、この追加限定も**容易**ではないですか？

### Claim 3の追加限定の検証
同様に、Claim 3の追加限定についても検証してください。
"""

    STEP_3_FINAL_DECISION = """あなたは「主任審査官」です。
ステップ1の「代理人の段階的主張」とステップ2の「審査官の段階的検証（反論）」を比較検討してください。

【代理人の主張】
{arguments}

【審査官の検証・反論】
{review}

---

以下の項目について、最終的な進歩性の判断（容易想到である / 容易想到ではない）とその理由を簡潔に述べてください。

## 判断項目

### 1. Claim 1 (幹) の進歩性
**判断:** [容易想到である / 容易想到ではない]
**理由:** [簡潔に説明]

### 2. Claim 2 (枝1) の進歩性
**判断:** [容易想到である / 容易想到ではない]
**理由:** [簡潔に説明]

### 3. Claim 3 (枝2) の進歩性
**判断:** [容易想到である / 容易想到ではない]
**理由:** [簡潔に説明]

### 4. 総合結論
[例: Claim 1は先行技術の示唆に基づく単なる最適化であり進歩性なし。しかし、Claim 2の追加限定は周知技術とは言えず進歩性あり。よって、Claim 2以降のクレームは特許可能と判断する。]

---

以下のJSON形式でも出力してください：

{{
  "claim1": {{
    "inventive": true/false,
    "reason": "理由"
  }},
  "claim2": {{
    "inventive": true/false,
    "reason": "理由"
  }},
  "claim3": {{
    "inventive": true/false,
    "reason": "理由"
  }},
  "conclusion": "総合結論"
}}
"""


# ==================== メインシステムクラス ====================

class PatentExaminationSystemIntegrated:
    """統合版特許審査システム"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
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
                            max_retries: int = 3, initial_wait: int = 2) -> str:
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
                    wait_time = initial_wait * (2 ** attempt)  # 指数バックオフ
                    print(f"\n⏳ レート制限エラー。{wait_time}秒待機してリトライします... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n❌ 最大リトライ回数に達しました。エラー: {e}")
                    raise
            except Exception as e:
                print(f"\n❌ 予期しないエラー: {e}")
                raise

    def step0_1_structure_application(self, abstract: str, claims: List[str]) -> PatentDocument:
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

        claims_text = "\n".join([f"Claim {i+1}: {claim}" for i, claim in enumerate(claims)])
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
            "step": "0.1",
            "role": "構造化",
            "content": result
        })

        return result

    def step0_2_structure_prior_art(self, abstract: str, claims: List[str]) -> PatentDocument:
        """
        ステップ0.2: 先行技術の構造化

        Args:
            abstract: 先行技術のAbstract
            claims: 先行技術のClaimリスト

        Returns:
            構造化された先行技術データ
        """
        print("\n" + "=" * 80)
        print("📋 ステップ0.2: 先行技術の構造化")
        print("=" * 80)

        claims_text = "\n".join([f"Claim {i+1}: {claim}" for i, claim in enumerate(claims)])
        prompt = PromptTemplates.STEP_0_2_STRUCTURE_PRIOR_ART.format(
            abstract=abstract,
            claims_text=claims_text
        )

        response_text = self._generate_with_retry(use_json_model=True, prompt=prompt)
        result = self._parse_json_response(response_text)

        print("\n✅ 構造化完了:")
        print(f"課題: {result['problem']}")
        print(f"解決原理: {result['solution_principle']}")
        print(f"Abstractの示唆: {result.get('abstract_hints', {})}")

        self.conversation_history.append({
            "step": "0.2",
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
        print("🔍 ステップ2: 審査官の段階的批評（7質問）")
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
                            app_abstract: str,
                            app_claims: List[str],
                            prior_abstract: str,
                            prior_claims: List[str]) -> Dict:
        """
        完全な審査プロセスの実行

        Args:
            app_abstract: 本願発明のAbstract
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
            app_data = self.step0_1_structure_application(app_abstract, app_claims)
            prior_data = self.step0_2_structure_prior_art(prior_abstract, prior_claims)

            # ステップ1: 代理人の主張
            arguments = self.step1_applicant_arguments(app_data, prior_data)

            # ステップ2: 審査官の検証
            review = self.step2_examiner_review(app_data, prior_data, arguments)

            # ステップ3: 最終判断
            decision = self.step3_final_decision(arguments, review)

            print("\n" + "✅" * 40)
            print("特許審査プロセス完了")
            print("✅" * 40)

            return {
                "application_structure": app_data,
                "prior_art_structure": prior_data,
                "applicant_arguments": arguments,
                "examiner_review": review,
                "final_decision": decision,
                "conversation_history": self.conversation_history
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

def main():
    """メイン実行関数（サンプル）"""

    # config.envファイルから環境変数を読み込む
    load_dotenv('config.env')

    # APIキーの設定（環境変数から取得）
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ config.envファイルにGOOGLE_API_KEYを設定してください")
        return

    # システムの初期化
    try:
        system = PatentExaminationSystemIntegrated(api_key)
        print(f"✅ システム初期化完了 (モデル: {system.model_name})")
    except ValueError as e:
        print(f"❌ 初期化エラー: {e}")
        return

    # サンプルデータ（実際のデータに置き換えてください）
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

    # 完全な審査プロセスの実行
    results = system.run_full_examination(
        app_abstract, app_claims,
        prior_abstract, prior_claims
    )

    # 結果の保存
    output_path = "patent_examination_results_integrated.json"
    system.save_results(results, output_path)

    print("\n" + "=" * 80)
    print("📊 審査プロセスが完了しました")
    print("=" * 80)


if __name__ == "__main__":
    main()