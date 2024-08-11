import datasets
import numpy as np
import pandas as pd
import torch
from cuml.manifold.umap import UMAP
from datasets import load_dataset
from renumics import spotlight
from transformers import AutoModel, ViTImageProcessor


def extract_embeddings(model, feature_extractor, image_name="image"):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        images = batch["image"]
        inputs = feature_extractor(images=images, return_tensors="pt").to(device)
        embeddings = model(**inputs).last_hidden_state[:, 0].cpu()

        return {"embedding": embeddings}

    return pp


def huggingface_embedding(
    df,
    image_name="image",
    inplace=False,
    modelname="google/vit-base-patch16-224",
    batched=True,
    batch_size=24,
):
    feature_extractor = ViTImageProcessor.from_pretrained(modelname, do_normalize=True)

    model = AutoModel.from_pretrained(modelname, output_hidden_states=True)

    # create huggingface dataset from df
    dataset = datasets.Dataset.from_pandas(df).cast_column(image_name, datasets.Image())

    # compute embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_fn = extract_embeddings(model.to(device), feature_extractor, image_name)
    updated_dataset = dataset.map(extract_fn, batched=batched, batch_size=batch_size)

    df_temp = updated_dataset.to_pandas()

    if inplace:
        df["embedding"] = df_temp["embedding"]
        return

    df_emb = pd.DataFrame()
    df_emb["embedding"] = df_temp["embedding"]

    return df_emb


dataset = load_dataset("marmal88/skin_cancer", split="train")
df = dataset.to_pandas()
df_emb = huggingface_embedding(df, modelname="google/vit-base-patch16-224")
df = pd.concat([df, df_emb], axis=1)

embeddings = np.stack(df["embedding"].to_numpy())

reducer = UMAP()
reduced_embedding = reducer.fit_transform(embeddings)

df["embedding_reduced"] = np.array(reduced_embedding).tolist()

spotlight.show(
    df,
    dtype={"image": spotlight.Image, "embedding_reduced": spotlight.Embedding},
)
