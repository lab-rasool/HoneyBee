import gc
import json
import os
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from honeybee.loaders import Scan
from honeybee.models import REMEDIS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
load_dotenv()


def manifest_to_df(manifest_path, modality):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    modality_df = pd.DataFrame()

    for patient in manifest:
        patient_id = patient["PatientID"]
        gdc_case_id = patient["gdc_case_id"]

        if modality in patient:
            df = pd.DataFrame(patient[modality])
            df["PatientID"] = patient_id
            df["gdc_case_id"] = gdc_case_id

            modality_df = pd.concat([modality_df, df], ignore_index=True)

    return modality_df if not modality_df.empty else None


def main():
    PROJECTS = [
        "TCGA-ACC",
        "TCGA-COAD",
        "TCGA-KICH",
        "TCGA-LIHC",
        "TCGA-PAAD",
        "TCGA-SKCM",
        "TCGA-UCEC",
        "TCGA-BLCA",
        "TCGA-DLBC",
        "TCGA-KIRC",
        "TCGA-LUAD",
        "TCGA-PCPG",
        "TCGA-STAD",
        "TCGA-UCS",
        "TCGA-BRCA",
        "TCGA-ESCA",
        "TCGA-KIRP",
        "TCGA-LUSC",
        "TCGA-PRAD",
        "TCGA-TGCT",
        "TCGA-UVM",
        "TCGA-CESC",
        "TCGA-GBM",
        "TCGA-LAML",
        "TCGA-MESO",
        "TCGA-READ",
        "TCGA-THCA",
        "TCGA-CHOL",
        "TCGA-HNSC",
        "TCGA-LGG",
        "TCGA-OV",
        "TCGA-SARC",
        "TCGA-THYM",
    ]
    RADIOLOGY_MODALITIES = [
        "MG",
        "CT",
        "MR",
        "SEG",
        "RTSTRUCT",
        "CR",
        "SR",
        "US",
        "PT",
        "DX",
        "RTDOSE",
        "RTPLAN",
        "PR",
        "REG",
        "RWV",
        "NM",
        "KO",
        "FUSION",
        "XA",
        "OT",
        "SC",
        "RF",
    ]

    embedding_model_path = "/mnt/d/Models/REMEDIS/onnx/cxr-50x1-remedis-s.onnx"

    for PROJECT in PROJECTS:
        DATA_DIR = f"/mnt/d/TCGA/raw/{PROJECT}"
        MANIFEST_PATH = f"{DATA_DIR}/manifest.json"

        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)

        radiology_modalities = {
            modality
            for patient in manifest
            for modality in RADIOLOGY_MODALITIES
            if modality in patient
        }

        for modality in radiology_modalities:
            PARQUET = (
                f"/mnt/d/TCGA/parquet/{PROJECT} {modality} Radiology (REMEDIS).parquet"
            )
            # Ensure the directory for the Parquet file exists
            os.makedirs(os.path.dirname(PARQUET), exist_ok=True)

            df = manifest_to_df(MANIFEST_PATH, modality)
            if df is None:
                continue

            schema = pa.schema(
                [
                    ("StudyInstanceUID", pa.string()),
                    ("SeriesInstanceUID", pa.string()),
                    ("SeriesDate", pa.string()),
                    ("BodyPartExamined", pa.string()),
                    ("SeriesNumber", pa.string()),
                    ("Collection", pa.string()),
                    ("Manufacturer", pa.string()),
                    ("ManufacturerModelName", pa.string()),
                    ("SoftwareVersions", pa.string()),
                    ("Visibility", pa.string()),
                    ("ImageCount", pa.int64()),
                    ("PatientID", pa.string()),
                    ("gdc_case_id", pa.string()),
                    ("ProtocolName", pa.string()),
                    ("SeriesDescription", pa.string()),
                    ("embedding", pa.binary()),
                    ("embedding_shape", pa.list_(pa.int64())),
                    ("__index_level_0__", pa.int64()),
                ]
            )

            # Ensure all required columns are present in the DataFrame
            for col in schema.names:
                if col not in df.columns:
                    df[col] = None

            df["embedding"] = None
            df["embedding_shape"] = None

            if os.path.exists(PARQUET):
                # If the Parquet file exists, read it into a DataFrame
                existing_df = pq.read_table(PARQUET).to_pandas()
                # Concatenate the new data with the existing data
                df = pd.concat([existing_df, df], ignore_index=True)

            writer = pq.ParquetWriter(PARQUET, schema)

            for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
                embedding = None  # Initialize embedding before the try block
                patches = None  # Initialize patches before the try block
                try:
                    file_path = f"{DATA_DIR}/raw/{row['PatientID']}/{modality}/{row['SeriesInstanceUID']}/{row['SeriesInstanceUID']}"
                    scanner = Scan(file_path)
                    patches = scanner.load_patches(target_patch_size=224)
                    embedding = REMEDIS.load_model_and_predict(
                        embedding_model_path, patches
                    )
                    df.at[index, "embedding_shape"] = embedding.shape
                    df.at[index, "embedding"] = embedding.tobytes()
                except Exception as e:
                    print(f"Error: {e}")
                    df.at[index, "embedding"] = None

                table = pa.Table.from_pandas(df.iloc[[index]], schema=schema)
                writer.write_table(table)

                del scanner, patches, embedding, table
                gc.collect()
                torch.cuda.empty_cache()

            if writer is not None:
                writer.close()


def combine_parquet_files(output_file):
    all_dfs = []
    parquet_dir = "/mnt/d/TCGA/parquet"

    for root, _, files in os.walk(parquet_dir):
        for file in tqdm(
            files, desc="Reading files", total=len(files), unit="files", leave=False
        ):
            if file.endswith("Radiology (REMEDIS).parquet"):
                file_path = os.path.join(root, file)
                df = pq.read_table(file_path).to_pandas()
                all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_table = pa.Table.from_pandas(combined_df)
    pq.write_table(combined_table, output_file)


if __name__ == "__main__":
    # main()
    combine_parquet_files("/mnt/d/TCGA/parquet/Radiology (REMEDIS).parquet")
