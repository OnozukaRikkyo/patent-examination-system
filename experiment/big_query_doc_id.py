import pandas as pd
# input
# /mnt/eightthdd/pipeline/data/CSV1_tokugan.csv
# read iteratively the csv row by syutugan, himotuki and category in python dataframe

json_df = pd.read_json('/mnt/eightthdd/pipeline/json/jp_embeddings.jsonl', lines=True)
# embedding_vectors = []

# for index, row in json_df.iterrows():
#     publication_number = row['publication_number']
#     if "WO" in publication_number:
#         # cast embedding_v1 to list of float
#         embedding_v1 = list(row['embedding_v1'])
#         embedding_vectors.append(embedding_v1)
#         print(embedding_v1)

# # calc cosine similarity between two vectors
# from numpy import dot
# from numpy.linalg import norm
# import numpy as np
# base_vector = np.array(embedding_vectors[0])
# for vec in embedding_vectors[1:]:
#     vec = np.array(vec)
#     cos_sim = dot(base_vector, vec) / (norm(base_vector) * norm(vec))
#     if cos_sim < 0.8:
#         base_vector = vec
#     print(f'Cosine Similarity: {cos_sim}')
# print()
    


def read_csv_iteratively():
    df = pd.read_csv('/mnt/eightthdd/pipeline/data/CSV1_tokugan.csv', usecols=['syutugan', 'himotuki', 'category']) 
    for index, row in df.iterrows():
        syutugan = row['syutugan']
        himotuki = row['himotuki']
        category = row['category']
        claim_json = find_document_id(syutugan)
        if not claim_json:
            continue
        ref_json = find_document_id(himotuki)
        if not ref_json:
            continue

        print(f'Syutugan: {syutugan}, Himotuki: {himotuki}, Category: {category}')

# find document ids from
# /mnt/eightthdd/pipeline/jsonl/jp_embeddings.jsonl
# that match the syutugan values from the csv
import json 
def find_document_id(doc_id):
    global json_df
    # separate JP and rest of the string, 'JP2013224028A' -> '2013224028A'
    doc_id = doc_id[2:]
    # if the end of doc_id is 'A' or 'B', store it in a variable
    if doc_id[-1] in ['A', 'B']:
        suffix = doc_id[-1]
        doc_id = doc_id[:-1]
    # use dataframe to find the docment in one lins
    result = json_df[json_df['publication_number'].str.contains(doc_id)]
    # for to handle multiple results, filter by suffix if exists
    for i, row in result.iterrows():
        # publication number after JP- has WO is not considered
        publication_number = row['publication_number'].to_string()
        separated = publication_number.split('-')[1]  # after JP-
        if separated.startswith('WO'):
            result = result.drop(i)

        if suffix:
            result = result[result['publication_number'].str.endswith(suffix)]
    if not result.empty:
        # if result[publication_number]
        return result.iloc[0].to_dict()
    return None

if __name__ == "__main__":
    read_csv_iteratively()
