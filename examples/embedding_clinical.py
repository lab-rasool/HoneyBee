import logging

import minds
import numpy as np
import pandas as pd
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(filename="./logs/clinical.log", level=logging.INFO)


class ClinicalEmbeddings:
    def __init__(self, model_name, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)

    def generate_embeddings(self, sentences):
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state
        return embeddings.cpu().numpy()


def process_group(group):
    common_fields = {}
    nested_objects = []
    for col in group.columns:
        unique_values = group[col].dropna().unique()
        if len(unique_values) == 1:
            # If only one unique value exists, it's a common field
            common_fields[col] = unique_values[0]

    # Create nested objects for fields that are not common
    for idx, row in group.iterrows():
        nested_object = {
            col: row[col]
            for col in group.columns
            if col not in common_fields and pd.notna(row[col])
        }
        if nested_object:  # Only add if the nested object is not empty
            nested_objects.append(nested_object)

    return common_fields, nested_objects


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def generate_summary_from_json(patient_data):
    # Initialize an empty list to store sentences
    summary_sentences = []

    # Iterate through each key-value pair in the JSON object
    for key, value in patient_data.items():
        # if the key is "case_id" then skip it
        if key == "case_id" or key == "pathology_report_uuid":
            continue

        # remove all _ from the key
        key = key.replace("_", " ")
        sentence = f"{key}: {value};"

        # if the value is a list, then skip it
        if isinstance(value, list):
            continue

        summary_sentences.append(sentence)

    # Compile all sentences into a single summary string
    summary = " ".join(summary_sentences)

    return summary


def get_chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_clinical_json_from_minds():
    tables = minds.get_tables()
    json_objects = {}
    for table in tqdm(tables, desc="Getting data from tables"):
        query = f"SELECT * FROM nihnci.{table} WHERE project_id='TCGA-LUAD'"
        df = minds.query(query)
        for case_id, group in tqdm(df.groupby("case_submitter_id"), leave=False):
            if case_id not in json_objects:
                json_objects[case_id] = {}
            common_fields, nested_objects = process_group(group)
            json_objects[case_id].update(common_fields)
            json_objects[case_id][table] = nested_objects

    return json_objects


def main():
    embedding_model = ClinicalEmbeddings(model_name="UFNLP/gatortron-base")

    json_objects = get_clinical_json_from_minds()
    clinical_df = []
    for case_id, patient_data in tqdm(json_objects.items()):
        summary = generate_summary_from_json(patient_data)

        if len(summary) > 0:
            summary_chunks = get_chunk_text(summary)
            chunk_embeddings = []
            for chunk in summary_chunks:
                chunk_embedding = embedding_model.generate_embeddings([chunk])
                chunk_embeddings.append(chunk_embedding)
            clinical_embedding = np.vstack(chunk_embeddings)
            clinical_embedding = np.array(
                pd.Series(list(clinical_embedding), dtype=object)
            )
        else:
            clinical_embedding = None

        patient_data["summary"] = summary
        patient_data["clinical_embedding"] = clinical_embedding

        # Create a new dictionary for DataFrame conversion, excluding lists
        patient_data_for_df = {
            key: value
            for key, value in patient_data.items()
            if not isinstance(value, list)
        }
        clinical_df.append(patient_data_for_df)

    clinical_df = pd.DataFrame(clinical_df)
    clinical_df.to_parquet("./data/parquet/clinical.parquet", index=False)


if __name__ == "__main__":
    main()
