import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from renumics import spotlight


def average_pool_embeddings(embedding, shape):
    """Average pool the embeddings across vectors to reduce each
    to a single 1024-dimensional vector."""
    embedding = embedding.reshape(shape)
    return np.mean(embedding, axis=0)


# Load the parquet file
parquet_file = pq.ParquetFile("/mnt/d/TCGA-LUAD/parquet/uni_Slide Image.parquet")

# Container for processed DataFrames
processed_batches = []

# Process each batch
for batch in parquet_file.iter_batches(
    batch_size=100
):  # Adjust batch_size based on memory capacity
    batch_df = batch.to_pandas()
    # Reshape and pool embeddings
    batch_df["embedding"] = batch_df.apply(
        lambda row: average_pool_embeddings(
            np.frombuffer(row["embedding"], dtype=np.float32), row["embedding_shape"]
        ),
        axis=1,
    )
    # Append processed batch to the list
    processed_batches.append(batch_df)

# Concatenate all processed batches into a single DataFrame
df = pd.concat(processed_batches, ignore_index=True)

# Use Spotlight to show the DataFrame
spotlight.show(df, dtype={"embedding": spotlight.Embedding})
