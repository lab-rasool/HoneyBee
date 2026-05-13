"""Combine per-modality preprocessed outputs into SeNMo's input vector.

Faithful port of ``DataCombiner`` in
``lab-rasool/SeNMo/package_classes/combine_features.py``. Concatenates
preprocessed clinical, DNA mutation, protein, gene, methylation, and
miRNA features into a single 80,697-dim float vector. Missing
modalities are zero-padded for their slice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Union

import numpy as np
import pandas as pd

# Expected feature count after preprocessing for each modality. Order
# of this dict determines concatenation order — matches upstream
# combine_features.py:18-25. Reordering breaks compatibility with the
# trained SeNMo checkpoints.
_MODALITY_DIMS: Dict[str, int] = {
    "clinical": 4,
    "dna_mutation": 17301,
    "protein": 472,
    "gene_expression": 8794,
    "dna_methylation": 52396,
    "mirna": 1730,
}

# Modalities whose preprocessed output carries a leading ``sample`` ID
# column (the first column is sample id, the rest are features). The
# combiner skips that column. Clinical and DNA mutation outputs are
# already feature-only.
_MODALITIES_WITH_SAMPLE_COL = {
    "protein",
    "gene_expression",
    "dna_methylation",
    "mirna",
}

# Total SeNMo input dim — sum of the above. Hardcoded as a safety check.
SENMO_INPUT_DIM: int = 80697

Source = Union[str, Path, pd.DataFrame, np.ndarray, None]


def combine_modalities(sources: Mapping[str, Source]) -> np.ndarray:
    """Combine preprocessed per-modality outputs into a SeNMo feature vector.

    Args:
        sources: Map from modality name to one of:

            * A path to the preprocessed CSV (e.g. produced by
              ``preprocess_*().to_csv(path, index=False)``).
            * A DataFrame from any of the ``preprocess_*`` functions.
            * A 1D numpy array of exactly the modality's expected dim.
            * ``None`` (or key omitted) to zero-pad that modality's
              slice. Useful for patients with missing data.

            Valid keys: ``clinical``, ``dna_mutation``, ``protein``,
            ``gene_expression``, ``dna_methylation``, ``mirna``.

    Returns:
        A 1D ``np.float32`` array of shape ``(80697,)`` — the input
        vector for :class:`~honeybee.models.SeNMo.SeNMoInference`.
    """
    parts = []
    for modality, expected_dim in _MODALITY_DIMS.items():
        source = sources.get(modality)
        if source is None:
            parts.append(np.zeros(expected_dim, dtype=np.float32))
            continue
        parts.append(_extract_features(source, modality, expected_dim))

    combined = np.concatenate(parts).astype(np.float32)
    if combined.shape != (SENMO_INPUT_DIM,):
        raise ValueError(
            f"Combined vector has shape {combined.shape}, "
            f"expected ({SENMO_INPUT_DIM},)"
        )
    return combined


def _extract_features(
    source: Source,
    modality: str,
    expected_dim: int,
) -> np.ndarray:
    """Convert one modality's source to a 1D ndarray of expected_dim."""
    skip_sample_col = modality in _MODALITIES_WITH_SAMPLE_COL

    if isinstance(source, (str, Path)):
        if not Path(source).is_file():
            return np.zeros(expected_dim, dtype=np.float32)
        # Read with header=None to handle the upstream CSV format
        # (where row 0 is the column header text and row 1 is data).
        df = pd.read_csv(source, header=None, low_memory=False)
        if df.shape[0] < 2:
            raise ValueError(
                f"{modality} CSV at {source} has fewer than 2 rows; "
                f"expected header + data."
            )
        row = df.iloc[1, :]
        if skip_sample_col:
            row = row.iloc[1:]
        vec = pd.to_numeric(row, errors="coerce").to_numpy()

    elif isinstance(source, pd.DataFrame):
        if source.shape[0] < 1:
            raise ValueError(f"{modality} DataFrame is empty")
        row = source.iloc[0, :]
        if skip_sample_col:
            row = row.iloc[1:]
        vec = pd.to_numeric(row, errors="coerce").to_numpy()

    elif isinstance(source, np.ndarray):
        vec = source.flatten().astype(np.float32)

    else:
        raise TypeError(
            f"Unsupported source type for {modality}: {type(source).__name__}"
        )

    # Defensive: if the source is off-by-one (sample col not stripped
    # when expected, or vice versa), reconcile to expected_dim.
    if vec.shape[0] == expected_dim + 1:
        vec = vec[1:]
    elif vec.shape[0] != expected_dim:
        raise ValueError(
            f"{modality}: feature vector has length {vec.shape[0]}, "
            f"expected {expected_dim}."
        )

    return vec.astype(np.float32)