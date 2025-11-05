import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import time
from google.api_core import exceptions as google_exceptions

# --- 1. APIキーとモデルの設定 ---
# config.envファイルから環境変数を読み込む
load_dotenv('config.env')

try:
    # 環境変数からAPIキーを読み込み
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if GOOGLE_API_KEY is None:
        raise ValueError("GOOGLE_API_KEYが設定されていません。config.envファイルを確認してください。")
    genai.configure(api_key=GOOGLE_API_KEY)
except ValueError as e:
    print(e)
    print("config.envファイルにGOOGLE_API_KEYを設定してください。")
    exit()

# 使用するモデル（推論能力の高いモデルを推奨）
MODEL_NAME = "gemini-2.0-flash-exp"  # または gemini-1.5-flash など

# --- 2. プロンプトテンプレートの定義 (No.18で定義したもの) ---

# ステップ0.1: 本願発明の構造化
PROMPT_STEP_0_1 = """
以下の「本願発明」のAbstractおよび全てのClaim（Claim 1, 2, 3...）を読み、特許判断に必要な要素を以下の形式で抽出・構造化してください。

【構造化出力フォーマット】
1.  **課題:** (例：...)
2.  **解決原理:** (例：...)
3.  **Claim 1の構成要件 (幹):**
    * 要件A: (例：...)
    * 要件B: (例：...)
    * (以下続く)
4.  **Claim 2の追加限定 (枝1):**
    * (例：...)
5.  **Claim 3の追加限定 (枝2):**
    * (例：...)
---
[本願発明]
{document_text}
"""

# ステップ0.2: 先行技術の構造化
PROMPT_STEP_0_2 = """
同様に、以下の「先行技術」のAbstractおよび全てのClaimを読み、同じ形式で構造化してください。**特にAbstractの「示唆（ヒント）」**を重要視してください。

【構造化出力フォーマット】
1.  **課題:** (例：...)
2.  **解決原理:** (例：...)
3.  **Claim 1の構成要件:**
    * 要件X: (例：...)
    * (以下続く)
4.  **Abstractの示唆（ヒント）:**
    * (例：...)
---
[先行技術]
{document_text}
"""

# ステップ1: 主張
PROMPT_STEP_1 = """
あなたは「本願発明」の代理人です。
先ほど構造化した2つの文献データ（ステップ0）に基づき、以下の2段階で「進歩性がある（容易に考えつけない）」という論理的な主張を構築してください。

1.  **第一の主張 (幹):**
    まず、本願発明のClaim 1 (幹)が、先行技術と比較して進歩性を有することを主張してください。
    （ヒント：先行技術のClaim 1にはない構成要件の存在や、共通する要件の決定的な差異を強調する。）

2.  **予備的主張 (枝葉):**
    仮に、Claim 1の進歩性が否定されたとしても、Claim 2以降の追加限定（枝）を先行技術に適用することは、先行技術からは動機付けがなく、容易想到ではないと主張してください。
"""

# ステップ2: 批評（自己検証）
PROMPT_STEP_2 = """
役割を変更します。あなたは特許庁の「審査官」です。
ステップ1の「代理人の主張」を論破するため、以下の2段階で検証と反論（＝進P性なしのロジック）を構築してください。

【第1段階：Claim 1 (幹) の検証】
ステップ0の構造化データ（特に先行技術のAbstractの示唆）を参照し、以下の7つの質問に答える形式で、本願発明のClaim 1 (幹)が**進歩性を欠く（容易想到である）**という結論を導いてください。

1.  **課題は共通か？**
2.  **解決原理は同一か？**
3.  **差分は最適化か？**
4.  **動機付けはあるか？**（先行技術Abstractの示唆は動機付けにならないか？）
5.  **阻害要因はないか？**
6.  **予期せぬ効果はあるか？**（単なる程度の差ではないか？）
7.  **結論（容易想到か）？**

【第2段階：Claim 2以降 (枝葉) の検証】
第1段階の結論に基づき、「Claim 1は進歩性なし」と仮定します。
次に、**Claim 2の追加限定 (枝1)**について検討します。これは技術常識や周知の選択肢ではありませんか？ 適用は容易ではありませんか？
（同様に、Claim 3 (枝2) ... についても検証してください）
"""

# ステップ3: 統合
PROMPT_STEP_3 = """
あなたは「主任審査官」です。
ステップ1の「代理人の段階的主張」とステップ2の「審査官の段階的検証（反論）」を比較検討してください。

以下の項目について、最終的な進歩性の判断（容易想到である / 容易想到ではない）とその理由を簡潔に述べてください。

1.  **Claim 1 (幹) の進歩性:**
2.  **Claim 2 (枝1) の進歩性:**
3.  **Claim 3 (枝2) の進歩性:**
4.  **総合結論:** (例: Claim 1は...進歩性なし。しかし、Claim 2は...進歩性あり。...)
"""

# --- 3. ヘルパー関数：リトライロジック ---

def generate_with_retry(chat, prompt, max_retries=3, initial_wait=2):
    """
    リトライロジック付きでコンテンツを生成

    Args:
        chat: チャットセッション
        prompt: プロンプト
        max_retries: 最大リトライ回数
        initial_wait: 初期待機時間（秒）

    Returns:
        レスポンス
    """
    for attempt in range(max_retries):
        try:
            return chat.send_message(prompt)
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

# --- 4. メイン実行関数 ---

def run_patent_analysis(doc_A_text, doc_B_text):
    """
    ステップ0〜3の段階的検証プロンプトチェーンを実行する
    """

    print(f"モデル {MODEL_NAME} を使用して分析を開始します...")

    model = genai.GenerativeModel(MODEL_NAME)
    # チャットセッションを開始（文脈を記憶させるため）
    chat = model.start_chat(history=[])

    analysis_report = {}

    try:
        # --- ステップ0.1: 本願発明の構造化 ---
        print("[ステップ0.1] 本願発明を構造化中...")
        prompt_0_1 = PROMPT_STEP_0_1.format(document_text=doc_A_text)
        response_0_1 = generate_with_retry(chat, prompt_0_1)
        analysis_report["step_0_1_structured_A"] = response_0_1.text
        print("...完了")

        # --- ステップ0.2: 先行技術の構造化 ---
        print("[ステップ0.2] 先行技術を構造化中...")
        prompt_0_2 = PROMPT_STEP_0_2.format(document_text=doc_B_text)
        response_0_2 = generate_with_retry(chat, prompt_0_2)
        analysis_report["step_0_2_structured_B"] = response_0_2.text
        print("...完了")

        # --- ステップ1: 主張 ---
        print("[ステップ1] 代理人の主張を生成中...")
        response_1 = generate_with_retry(chat, PROMPT_STEP_1)
        analysis_report["step_1_claimant_argument"] = response_1.text
        print("...完了")

        # --- ステップ2: 批評（自己検証） ---
        print("[ステップ2] 審査官の批評（7質問）を生成中...")
        response_2 = generate_with_retry(chat, PROMPT_STEP_2)
        analysis_report["step_2_examiner_critique"] = response_2.text
        print("...完了")

        # --- ステップ3: 統合 ---
        print("[ステップ3] 主任審査官の最終判断を生成中...")
        response_3 = generate_with_retry(chat, PROMPT_STEP_3)
        analysis_report["step_3_final_judgement"] = response_3.text
        print("...完了")

        print("\n--- 分析が正常に完了しました ---")
        return analysis_report

    except Exception as e:
        print(f"\n--- エラーが発生しました ---")
        print(e)
        # エラー発生時の部分的なレポートを返す
        return analysis_report

# --- 5. サンプルデータの定義と実行 ---
if __name__ == "__main__":
    
    # サンプルデータ（このスレッドで使用した例）
    # 実際にはファイルから読み込むか、DBから取得します
    DOCUMENT_A = """
    Abstract:
    "【課題】インクジェットプリントヘッドのノズルプレートにおける疎油性の低接着性表面被膜について、所望の機械的な頑強さを得る方法を提供する。【解決手段】インクジェットプリントヘッド前面のための被膜であって、この中で、該被膜は、最高３００℃まで加熱した場合に約１５％未満の重量損失によって示されるような高い熱安定性を有する疎油性低接着性被膜を含み、かつこの中で、１滴の紫外（ＵＶ）ゲルインクおよび１滴の固体インクは、該被膜の表面との約５０°超の接触角度および約３０°未満の滑走角度を呈し、この中で、該被膜は、少なくとも２６０℃の温度に少なくとも３０分間曝露された後に該接触角度および滑走角度を維持する。"
    Claim 1:
    "インクジェットプリントヘッド前面のための被膜であって、この中で、該被膜は、最高３００℃まで加熱した場合に約１５％未満の重量損失によって示される高熱安定性を有する疎油性低接着性被膜を含み、かつこの中で、１滴の紫外（ＵＶ）ゲルインクまたは１滴の固体インクは、該被膜の表面との約５０°超の接触角度および約３０°未満の滑走角度を呈し、この中で、該被膜は、少なくとも２９０℃の温度および少なくとも３５０ｐｓｉの圧力に曝露された後で該接触角度および滑走角度を維持する、該被膜。"
    Claim 2:
    "前記被膜がフッ素系ポリマーを含む、請求項１に記載の被膜。"
    """
    
    DOCUMENT_B = """
    Abstract:
    "【課題】既知の撥油性低接着性プリントヘッド前面コーティングに特徴的な表面特性は、プリントヘッド加工プロセス中に加えられる典型的な温度まで加熱すると、インクジェットプリントヘッド前面の汚れを可能な限り減らすのには足りない状態まで低下する。【解決手段】撥油性低接着性コーティングが、インクジェットプリントヘッド前面の表面に配置される場合、紫外線ゲルインクの液滴、または固体インクの液滴は、４５°よりも大きな接触角と、約３０°よりも小さな低い滑り角を示す。前記撥油性低接着性コーティングは、熱に安定であり、表面の接触角および滑り角は、プリントヘッドの加工および製造中に、コーティングを１８０℃〜３２０℃の範囲または略この範囲の高温、および１００ｐｓｉ〜４００ｐｓｉの範囲または略この範囲の高圧にさらした後、ほとんど分解していないことを示す。"
    Claim 1:
    "インクジェットプリントヘッド前面のためのコーティングであって、このコーティングが、撥油性低接着性コーティングを含み、紫外線（ＵＶ）ゲルインクの液滴、または固体インクの液滴が、このコーティングを少なくとも２００℃の温度に少なくとも３０分間さらした後に、コーティング表面に対し、約３０°未満の滑り角を示する、コーティング。"
    """

    # 推論実行
    report = run_patent_analysis(DOCUMENT_A, DOCUMENT_B)
    
    # 最終結果（ステップ3）の表示
    print("\n--- [最終判断] ---")
    print(report.get("step_3_final_judgement", "（判断の生成に失敗しました）"))

    # すべての推論過程をJSONファイルに保存
    with open("patent_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\nすべての推論過程を 'patent_analysis_report.json' に保存しました。")