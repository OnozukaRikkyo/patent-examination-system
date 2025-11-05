"""
膨大なファイルからループで発明のテキストと新規性、進歩性を否定するテキストを読み込み、
類似度計算を実行するバッチ処理プログラム
"""

from logging import config
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(dotenv_path="config.env")

try:
    from llm_pipeline_integrated import entry
except ImportError:
    print("Warning: llm_pipeline_integrated.py not found.")
    entry = None

class BatchConfig:
    """バッチ処理の設定クラス"""

    def __init__(self):
        self.OUTPUT_ROOT_DIR = os.environ.get("OUTPUT_ROOT", "/mnt/eightthdd")
        self.PROJECT_ROOT = "graph/csv1/result_rank"

        self.csv_path = f"{self.OUTPUT_ROOT_DIR}/llm/csv1/CSV1.csv"
        self.output_csv_path = f"{self.OUTPUT_ROOT_DIR}/graph/csv1/ref/class_cos_sim1.csv"

        # デバッグ
        self.save_interval = True  # 保存間隔（行数）
        self.verbose = True  # 詳細ログ出力
        
        self.abstract_key = "abstract"
        self.claims_key = "claims"
        self.claim1_key = "claim"
        self.description_key = "description"

        self.doc_key_dict = {
            self.abstract_key: str,
            self.claims_key: list,
            self.claim1_key: str,
            self.description_key: str
        }

        self.first_level_keys = [self.abstract_key, self.claims_key, self.description_key]
        self.claim1_list_dict_key = "text"

        self.abstract_dir = Path(self.OUTPUT_ROOT_DIR) / self.PROJECT_ROOT / self.abstract_key
        self.claims_dir = Path(self.OUTPUT_ROOT_DIR) / self.PROJECT_ROOT / self.claims_key
        self.claim1_dir = Path(self.OUTPUT_ROOT_DIR) / self.PROJECT_ROOT / self.claim1_key
        self.description_dir = Path(self.OUTPUT_ROOT_DIR) / self.PROJECT_ROOT / self.description_key

        self.error_log_path = Path(self.OUTPUT_ROOT_DIR) / self.PROJECT_ROOT / "error" / "error_log.txt"



def extract_path_from_matches(matches_paths):
    """self.claim1_list_dict_key = "text"
    matches/matches_ref列からパスを抽出

    Args:
        matches_paths: CSV列の値（文字列リストまたはリスト）

    Returns:
        str: 抽出されたパス、失敗時は空文字列
    """
    if isinstance(matches_paths, str):
        try:
            matches_paths = eval(matches_paths)
            if isinstance(matches_paths, list) and len(matches_paths) > 0:
                return matches_paths[0]
        except:
            pass
        return ""

    if isinstance(matches_paths, list) and len(matches_paths) > 0:
        return matches_paths[0]

    return ""


def convert_similarity_to_jsonl_path(similarity_path):
    """
    similarityパスをjsonl_dataパスに変換
    例: /mnt/eightthdd/similarity/result_4/28/JP2013224028A
    -> /mnt/eightthdd/jsonl_data/result_4/28/JP2013224028A
    """
    return Path(similarity_path.replace('/similarity/', '/jsonl_data/'))


def find_json_file(directory_path):
    """
    ディレクトリ内でJSONファイルを探す
    優先順位: json.txt > text.jsonl > json.json
    """
    for filename in ['text.jsonl']:
        candidate = directory_path / filename
        if candidate.exists():
            return candidate
    return None


def load_document_json(json_file_path, config=None):
    """JSONファイルを読み込んでabstract、claims、claim1、descriptionを取得"""
    if config is None:
        config = BatchConfig()

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            field = data["fields"]
            claims = field.get(config.claims_key, [])
            claim = claims[0] if claims else {}

        return {
            config.abstract_key: field.get(config.abstract_key, ''),
            config.claims_key: claims,
            config.claim1_key: claim.get(config.claim1_key, ''),
            config.description_key: field.get(config.description_key, '')
        }
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {json_file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {json_file_path}")
        return None


def get_doc_json(similarity_path):
    """
    similarityパスからJSONドキュメントを読み込む

    処理フロー:
    1. /similarity/ を /jsonl_data/ に変換
    2. json.txt, text.jsonl, json.json を順に探す
    3. 見つかったファイルからabstractとclaimを読み込む
    """
    if not similarity_path:
        return None

    try:
        jsonl_dir = convert_similarity_to_jsonl_path(similarity_path)
        json_file = find_json_file(jsonl_dir)

        if json_file is None:
            return None

        return load_document_json(json_file)

    except Exception as e:
        print(f"Error in get_doc_json: {similarity_path}")
        return None


def create_temp_text_file(text, temp_dir, filename):
    config = BatchConfig()
    """一時テキストファイルを作成"""
    temp_dir = Path(config.OUTPUT_ROOT_DIR) / Path(config.PROJECT_ROOT) / temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_path = temp_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

    return file_path

import copy
def process_single_row(df, row, index, verbose=False):
    """
    CSVの1行を処理して類似度を計算

    Args:config.
        row: DataFrameの行
        index: 行のインデックス
        temp_dir: 一時ファイル用ディレクトリ
        verbose: 詳細ログ出力
    """
    try:
        config = BatchConfig()
        
        claim_id = row['syutugan']

        extracted_claim_id_df = df[df['syutugan'] == claim_id]
        associated_ref_id_list = extracted_claim_id_df['himotuki'].tolist()
        associated_ref_id_list.append(claim_id)  # 自身も含める 

        # get all files in the project root
        project_root_path = Path(config.OUTPUT_ROOT_DIR) / config.PROJECT_ROOT
        all_files = list(project_root_path.rglob('*'))
        # find filename matching claim_id
        target_file_path = None
        for file in all_files:
            if file.is_file() and file.stem == str(claim_id):
                target_file_path = file
                break
        if not target_file_path:
            raise FileNotFoundError(f"File for claim_id {claim_id} not found.")
        # read the csv file as pandas dataframe
        target_df = pd.read_csv(target_file_path, encoding='utf-8')

        # for loop each row of the target_df
        ranked_target_doc_id_list = []
        for _, target_row in target_df.iterrows():
            doc_id = target_row['doc_id']

            if doc_id in associated_ref_id_list:
                continue
            ranked_target_doc_id_list.append(doc_id)
        # if claim_id in ranked_target_doc_id_list then remove it
        if claim_id in ranked_target_doc_id_list:
            ranked_target_doc_id_list.remove(claim_id)

        OUTPUT_ROOT_DIR = os.environ.get("OUTPUT_ROOT", "/mnt/eightthdd")
        PROJECT_ROOT = "llm/csv1"
        csv_path = f"{OUTPUT_ROOT_DIR}/graph/csv1/ref/class_cos_sim1.csv"
        output_csv_path = f"{OUTPUT_ROOT_DIR}/graph/csv1/ref/class_cos_sim1.csv"
        # read output csv as pandas dataframe
        output_df = pd.read_csv(output_csv_path, encoding='utf-8')
        
        doc_b_path_list = []
        for doc_b_id in ranked_target_doc_id_list:
            # find doc_b_id in output_df
            output_row = output_df[output_df['syutugan'] == doc_b_id]
            if len(output_row) > 0:
                # get the first row's matches column
                first_row = output_row.iloc[0]
                path_str = eval(first_row['matches'])[0]

                doc_b_path_list.append(path_str)
            else:
                output_row = output_df[output_df['himotuki'] == doc_b_id]
                if len(output_row) > 0:
                    first_row = output_row.iloc[0]
                    path_str = eval(first_row['matches'])[0]
                    doc_b_path_list.append(path_str)

            if len(output_row) == 0:
                continue
 
        doc_a_path = eval(output_df[output_df['syutugan'] == claim_id].iloc[0]['matches'])[0]
        for doc_b_path in doc_b_path_list:
            key_dict_a = copy.deepcopy(config.doc_key_dict)
            key_dict_b = copy.deepcopy(config.doc_key_dict)

            # path_a = extract_path_from_matches(doc_a_path)
            # path_b = extract_path_from_matches(doc_b_path)

            # if not path_a or not path_b:
            #     return False, "Empty path from CSV"

            doc_a = get_doc_json(doc_a_path)
            doc_b = get_doc_json(doc_b_path)

            if not doc_a or not doc_b:
                return False, None, "Failed to load JSON documents"

            for key_name in config.first_level_keys:
                if key_name not in doc_a or key_name not in doc_b:

                    continue
                doc_text_a = str(doc_a[key_name])
                doc_text_b = str(doc_b[key_name])

                if not doc_text_a or not doc_text_b:
                    continue

                if type(doc_text_a) is not str or type(doc_text_b) is not str:
                    key_dict_a[key_name] = json.dumps(doc_text_a, ensure_ascii=False, indent=2)
                    key_dict_b[key_name] = json.dumps(doc_text_b, ensure_ascii=False, indent=2)
                else:
                    key_dict_a[key_name] = doc_text_a
                    key_dict_b[key_name] = doc_text_b

            result = entry(key_dict_a, key_dict_b)

        return True, result

    except Exception as e:
        if verbose:
            print(f"\nRow {index} exception: {e}")
        return False, str(e)

import copy
def process_single_row_exec(row, index, verbose=False):
    """
    CSVの1行を処理して類似度を計算

    Args:config.
        row: DataFrameの行
        index: 行のインデックス
        temp_dir: 一時ファイル用ディレクトリ
        verbose: 詳細ログ出力
    """
    try:
        config = BatchConfig()
        
        key_dict_a = copy.deepcopy(config.doc_key_dict)
        key_dict_b = copy.deepcopy(config.doc_key_dict)

        path_a = extract_path_from_matches(row['matches'])
        path_b = extract_path_from_matches(row['matches_ref'])

        if not path_a or not path_b:
            return False, "Empty path from CSV"

        doc_a = get_doc_json(path_a)
        doc_b = get_doc_json(path_b)

        if not doc_a or not doc_b:
            return False, None, "Failed to load JSON documents"

        for key_name in config.first_level_keys:
            if key_name not in doc_a or key_name not in doc_b:

                continue
            doc_text_a = str(doc_a[key_name])
            doc_text_b = str(doc_b[key_name])

            if not doc_text_a or not doc_text_b:
                continue

            if type(doc_text_a) is not str or type(doc_text_b) is not str:
                key_dict_a[key_name] = json.dumps(doc_text_a, ensure_ascii=False, indent=2)
                key_dict_b[key_name] = json.dumps(doc_text_b, ensure_ascii=False, indent=2)
            else:
                key_dict_a[key_name] = doc_text_a
                key_dict_b[key_name] = doc_text_b

            text_file_a = create_temp_text_file(
                doc_text_a, key_name, f"claim_{index}_{row['syutugan']}.txt"
            )
            text_file_b = create_temp_text_file(
                doc_text_b, key_name, f"ref_{index}_{row['himotuki']}.txt"
            )   

        claim1_text_a = eval(key_dict_a[config.claims_key])[0][config.claim1_list_dict_key]
        claim1_text_b = eval(key_dict_b[config.claims_key])[0][config.claim1_list_dict_key]

        if claim1_text_a and claim1_text_b:
            text_file_a = create_temp_text_file(
                claim1_text_a, config.claim1_key, f"claim_{index}_{row['syutugan']}.txt"
            )
            text_file_b = create_temp_text_file(
                claim1_text_b, config.claim1_key, f"ref_{index}_{row['himotuki']}.txt"
            )           

        result = entry(key_dict_a, key_dict_b)

        # text_file_a.unlink(missing_ok=True)
        # text_file_b.unlink(missing_ok=True)

        return True, result

    except Exception as e:
        if verbose:
            print(f"\nRow {index} exception: {e}")
        return False, str(e)


def process_batch(config, start_index=None, end_index=None, overwrite=False):
    """
    バッチ処理のメイン関数

    Args:
        config: BatchConfigインスタンス
        start_index: 処理開始インデックス
        end_index: 処理終了インデックス
        overwrite: 既存のcos_sim値も再計算
    """
    print(f"Loading CSV from: {config.csv_path}")
    df = pd.read_csv(config.csv_path)
    df_copy = df.copy()

    # if "cos_sim" not in df_copy.columns:
    #     df_copy["cos_sim"] = None

    total_files = len(df)
    start = start_index if start_index is not None else 0
    end = end_index if end_index is not None else total_files

    print(f"Processing rows {start} to {end} (Total: {total_files})")
    if config.verbose:
        print("Verbose mode: ON")

    pbar = tqdm(total=end - start, dynamic_ncols=True, desc="Processing", unit="row", smoothing=0)

    processed_count = 0
    error_count = 0
    skipped_count = 0

    for i in range(start, end):
        pbar.update(1)

        try:
            row = df.iloc[i]
            success, msg = process_single_row(df, row, i)
            if success:
                processed_count += 1
            else:
                error_count += 1
                if config.verbose:
                    print(f"\nRow {i} error: {msg}")

                    # save text of row to error log
                    with open(config.error_log_path, "a", encoding="utf-8") as f:
                        f.write(f"Row {i} error: {msg}\n")

            if config.verbose:
                json_msg = json.dumps(msg, ensure_ascii=False, indent=2)

                inventiveness = msg.get("inventiveness", "N/A")
                inventive = inventiveness.get("claim1", "N/A")
                judge = inventive.get("inventive", "N/A")

                if judge:
                    save_negative_path = Path(config.OUTPUT_ROOT_DIR) / config.PROJECT_ROOT / "inventive"
                    save_negative_path.mkdir(parents=True, exist_ok=True)
                    with open(save_negative_path / f"{i}.txt", "w", encoding="utf-8") as f:
                        f.write(json_msg)
                else:
                    save_positive_path = Path(config.OUTPUT_ROOT_DIR) / config.PROJECT_ROOT / "not_inventive"
                    save_positive_path.mkdir(parents=True, exist_ok=True)
                    with open(save_positive_path / f"{i}.txt", "w", encoding="utf-8") as f:
                        f.write(json_msg)
                
                pbar.set_description(f"Processing (saved at row {i+1})")

            pbar.set_postfix({
                'processed': processed_count,
                'errors': error_count,
                'skipped': skipped_count
            })

        except Exception as e:
            print(f"\nUnexpected error at row {i}: {e}")
            error_count += 1
            pbar.set_postfix({
                'processed': processed_count,
                'errors': error_count,
                'skipped': skipped_count
            })
            continue

    pbar.close()

    df_copy.to_csv(config.output_csv_path, index=False, encoding="utf-8")

    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Results saved to: {config.output_csv_path}")


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch similarity processing for patent documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_similarity_processor.py
  python batch_similarity_processor.py --start 0 --end 100
  python batch_similarity_processor.py --overwrite --verbose
  python batch_similarity_processor.py --start 0 --end 1000 --save-interval 50
        """
    )
    parser.add_argument('--start', type=int, default=None,
                        help='Start index (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='End index (default: all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing cos_sim values')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save results every N rows (default: 100)')

    args = parser.parse_args()

    config = BatchConfig()
    # config.verbose = args.verbose
    # config.save_interval = args.save_interval

    if not Path(config.csv_path).exists():
        print(f"Error: CSV file not found at {config.csv_path}")
        return 1

    if entry is None:
        print("Error: llm_pipeline_integrated.py could not be imported.")
        return 1

    process_batch(
        config=config,
        start_index=args.start,
        end_index=args.end,
        overwrite=args.overwrite
    )

    return 0


if __name__ == "__main__":
    exit(main())