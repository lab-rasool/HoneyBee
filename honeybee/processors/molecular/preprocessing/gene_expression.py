"""Gene expression (RNA-seq) preprocessing.

Faithful port of ``GeneExpressionPreprocessor`` in
``lab-rasool/SeNMo/package_classes/GeneExpresn_preprocess.py``.
Transposes a wide gene expression TSV, drops all-NaN columns, and
trims to 8794 expression features via a two-step filter:

1. Drop columns that are zero across all samples.
2. If still over target, drop columns whose values are all <7
   (corresponding to <127 FPKM after log+1 transform — the
   low-expression cutoff cited in the paper).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

# Number of expression features after preprocessing. Excludes the
# leading ``sample`` ID column, so the returned DataFrame has up to
# 8795 columns total. Matches upstream GeneExpresn_preprocess.py:40.
_GENE_TARGET_FEATURES: int = 8794

# Log+1 FPKM cutoff for the low-expression filter. SeNMo paper Sec 4.3:
# values above 7 correspond to ~127 FPKM, retained for biological
# relevance. See GeneExpresn_preprocess.py:60.
_LOW_EXPR_CUTOFF: float = 7.0


def preprocess_gene_expression(
    source: Union[str, Path, pd.DataFrame],
) -> pd.DataFrame:
    """Preprocess gene expression (RNA-seq HTSeq-FPKM) data for SeNMo.

    Args:
        source: Path to a TSV file (features-as-rows, samples-as-
            columns, e.g. TCGA-GDC gene-expr-RNAhtseq_fpkm.tsv) or an
            already-loaded DataFrame in the same orientation. Values
            are expected to be already log+1 transformed.

    Returns:
        DataFrame with one row per sample. Column 0 is ``sample``;
        columns 1..N are gene expression features (up to 8794).
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

    transposed = transposed.dropna(axis=1, how="all")

    numeric = transposed.iloc[:, 1:].astype(float)

    if numeric.shape[1] > _GENE_TARGET_FEATURES:
        # Step 1: drop zero columns first.
        zero_columns = numeric.columns[(numeric == 0).all()]
        to_remove = len(numeric.columns) - _GENE_TARGET_FEATURES
        numeric = numeric.drop(columns=zero_columns[:to_remove])

    if numeric.shape[1] > _GENE_TARGET_FEATURES:
        # Step 2: drop low-expression columns.
        low_expr_columns = numeric.columns[(numeric < _LOW_EXPR_CUTOFF).all()]
        to_remove = len(numeric.columns) - _GENE_TARGET_FEATURES
        numeric = numeric.drop(columns=low_expr_columns[:to_remove])

    if numeric.shape[1] > _GENE_TARGET_FEATURES:
        raise ValueError(
            f"Gene expression still has {numeric.shape[1]} features after "
            f"both filter passes; expected <= {_GENE_TARGET_FEATURES}. "
            f"Input may have too many high-expression non-zero features."
        )

    numeric.insert(0, "sample", transposed.iloc[:, 0])
    return numeric