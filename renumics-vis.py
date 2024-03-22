import pandas as pd
from datasets import load_dataset
from renumics import spotlight
import numpy as np

# dataset = load_dataset("Aakash-Tripathi/TCGA-LUAD-minds", "clinical")
# print(dataset)
# df = pd.DataFrame(dataset["train"])

df = pd.read_parquet("./data/parquet/clinical.parquet")

embeddings_series = df["clinical_embedding"]
for i in range(len(embeddings_series)):
    df.at[i, "clinical_embedding"] = np.array(
        embeddings_series[i], dtype=object
    ).reshape(-1)

spotlight.show(df, dtype={"clinical_embedding": spotlight.Embedding})
