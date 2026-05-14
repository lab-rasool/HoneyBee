"""DNA methylation preprocessing.

Faithful port of ``DNAMethylationPreprocessor`` in
``lab-rasool/SeNMo/package_classes/DNAmethyl_preprocess.py``.
Transposes a wide methylation TSV, drops all-NaN columns, and trims
to 52,396 features by iterating through stepwise value-range filters
that drop low-variance CpG sites bunched in a narrow methylation
range.

Two upstream quirks preserved as-is:

* The first filter interval ``(0, 0)`` is a no-op (no value satisfies
  ``x > 0 AND x <= 0``).
* All-zero columns are not dropped (the filter checks ``(data > 0)
  .all()``, which is False for any column containing zeros).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

# Number of methylation features after preprocessing. Excludes the
# leading ``sample`` ID column. Matches DNAmethyl_preprocess.py:40.
_METHYLATION_TARGET_FEATURES: int = 52396

# Stepwise (lower, upper] intervals scanned in order. A column whose
# values ALL fall inside ``(lower, upper]`` is a candidate for removal.
# First (0, 0) is intentionally a no-op (matches upstream).
_METHYLATION_FILTER_INTERVALS: List[Tuple[float, float]] = [
    (0.0, 0.0),
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
]


def preprocess_dna_methylation(
    source: Union[str, Path, pd.DataFrame],
) -> pd.DataFrame:
    """Preprocess DNA methylation (Illumina 450K beta values) for SeNMo.

    Args:
        source: Path to a TSV file (CpG sites as rows, samples as
            columns; e.g. TCGA-GDC methylation450.tsv) or an
            already-loaded DataFrame in the same orientation.
            Values must be beta values in ``[0, 1]``.

    Returns:
        DataFrame with one row per sample. Column 0 is ``sample``;
        columns 1..N are CpG methylation features (up to 52,396).
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

    for lower, upper in _METHYLATION_FILTER_INTERVALS:
        if numeric.shape[1] <= _METHYLATION_TARGET_FEATURES:
            break
        condition = (numeric > lower).all() & (numeric <= upper).all()
        candidates = numeric.columns[condition]
        to_remove = len(numeric.columns) - _METHYLATION_TARGET_FEATURES
        numeric = numeric.drop(columns=candidates[:to_remove])

    if numeric.shape[1] > _METHYLATION_TARGET_FEATURES:
        raise ValueError(
            f"DNA methylation still has {numeric.shape[1]} features after "
            f"all filter intervals; expected <= {_METHYLATION_TARGET_FEATURES}."
        )

    numeric.insert(0, "sample", transposed.iloc[:, 0])
    return numeric