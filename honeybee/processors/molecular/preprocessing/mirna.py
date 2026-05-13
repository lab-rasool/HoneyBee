"""miRNA expression preprocessing.

Faithful port of ``MiRNAPreprocessor`` in
``lab-rasool/SeNMo/package_classes/miRNA_preprocess.py``. Reduces a
wide miRNA TSV (features-as-rows, samples-as-columns) to a DataFrame
with one row per sample and 1731 columns (``sample`` id + 1730 miRNA
features) by transposing, dropping NaN rows, and trimming zero-valued
features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

# 1 sample-id column + 1730 miRNA features. Matches the
# ``target_size=1731`` constant in upstream miRNA_preprocess.py:37.
_MIRNA_TARGET_COLS: int = 1731


def preprocess_mirna(
    source: Union[str, Path, pd.DataFrame],
) -> pd.DataFrame:
    """Preprocess miRNA expression data for SeNMo.

    Args:
        source: Either a path to a TSV file (features-as-rows,
            samples-as-columns, as TCGA-GDC miRNA expression matrices
            are distributed) or an already-loaded DataFrame in the
            same orientation.

    Returns:
        DataFrame with one row per sample. Column 0 is ``sample`` (the
        original column header from the TSV); columns 1..1730 are
        miRNA features. May have fewer columns if the input had fewer
        than 1731 features to begin with.
    """
    if isinstance(source, (str, Path)):
        data = pd.read_table(source)
    else:
        data = source

    transposed = data.transpose()
    transposed.reset_index(inplace=True)
    transposed.columns = transposed.iloc[0]
    transposed = transposed[1:]
    transposed.rename(columns={transposed.columns[0]: "sample"}, inplace=True)

    transposed = transposed.dropna()

    if transposed.shape[1] > _MIRNA_TARGET_COLS:
        zero_columns = transposed.columns[(transposed == 0).all()]
        transposed = transposed.drop(
            columns=zero_columns[: transposed.shape[1] - _MIRNA_TARGET_COLS]
        )

    return transposed