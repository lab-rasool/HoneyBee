"""Protein expression (RPPA) preprocessing.

Faithful port of ``ProteinExpressionPreprocessor`` in
``lab-rasool/SeNMo/package_classes/ProteinExpresn_preprocess.py``.
Reduces a long-format RPPA TSV (one row per protein, columns
``AGID`` + ``protein_expression``) to a single-row DataFrame with
473 columns (``sample`` + 472 protein features) by dropping NaN rows
to hit target and mean-imputing any remainder.

Note: unlike the other modalities, the resulting ``sample`` column
literally holds the string ``"protein_expression"`` (the value of
the AGID label in the transposed data). The combine step ignores the
sample column, so this is a cosmetic upstream quirk we preserve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

# Number of protein features after preprocessing. Matches
# ProteinExpresn_preprocess.py:39.
_PROTEIN_TARGET_ROWS: int = 472


def preprocess_protein(
    source: Union[str, Path, pd.DataFrame],
) -> pd.DataFrame:
    """Preprocess RPPA protein expression data for SeNMo.

    Args:
        source: Path to a TSV file with at least ``AGID`` and
            ``protein_expression`` columns (e.g. TCGA-GDC RPPA data),
            or an already-loaded DataFrame with those columns.

    Returns:
        Single-row DataFrame. Column 0 is ``sample``; columns 1..472
        are protein expression values, one per AGID.
    """
    if isinstance(source, (str, Path)):
        data = pd.read_table(source)
    else:
        data = source.copy()

    if "AGID" not in data.columns or "protein_expression" not in data.columns:
        raise ValueError(
            "Protein expression input must contain 'AGID' and "
            "'protein_expression' columns."
        )

    filtered = data[["AGID", "protein_expression"]].copy()
    # Cast for the mean computation; cast back to float at the end.
    mean_value = filtered["protein_expression"].astype(float).mean()

    # Drop NaN-valued rows in order until we hit target. ``nan_indices``
    # preserves the original row order, so this is deterministic.
    nan_indices = filtered.index[filtered["protein_expression"].isnull()]
    excess = filtered.shape[0] - _PROTEIN_TARGET_ROWS
    if excess > 0 and len(nan_indices) > 0:
        filtered = filtered.drop(index=nan_indices[:excess])

    filtered["protein_expression"] = filtered["protein_expression"].fillna(mean_value)

    if filtered["protein_expression"].isnull().any():
        raise ValueError("Protein expression still contains NaN after imputation.")
    if filtered.shape[0] > _PROTEIN_TARGET_ROWS:
        raise ValueError(
            f"Protein expression has {filtered.shape[0]} rows after dropping "
            f"NaN rows; expected <= {_PROTEIN_TARGET_ROWS}. Insufficient "
            f"NaN rows to drop."
        )

    transposed = filtered.transpose()
    transposed.reset_index(inplace=True)
    transposed.columns = transposed.iloc[0]
    transposed = transposed[1:]
    transposed.rename(columns={transposed.columns[0]: "sample"}, inplace=True)
    return transposed