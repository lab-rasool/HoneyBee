"""DNA mutation preprocessing.

Faithful port of ``DNAMutationPreprocessor`` in
``lab-rasool/SeNMo/package_classes/DNAMut_preprocess.py``. Reads a
TCGA MAF (mutation annotation format) file, binarizes per-Hugo-symbol
HIGH-impact mutations, and trims to 17,301 features by randomly
dropping a deterministic subset of zero-valued rows.

Important: upstream's ``random.sample(...)`` is called against the
unseeded global random state, so identical inputs produce different
outputs run-to-run. This port introduces a ``seed`` parameter
(default 42) so results are reproducible. To reproduce upstream's
output exactly in a given Python process, set ``random.seed(seed)``
before calling upstream's class with the same seed value.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Union

import pandas as pd

# Number of mutation features after preprocessing. Matches
# DNAMut_preprocess.py:55 and combine_features.py:20 (both 17301).
# Paper Table 4 says 17,253 — that's a transcription error; the
# code is internally consistent at 17,301.
_DNAMUT_TARGET_ROWS: int = 17301


def _bundled_hugo_symbols_path() -> Path:
    """Return the package-bundled Hugo_symbols.tsv path."""
    return Path(__file__).resolve().parent.parent / "data" / "Hugo_symbols.tsv"


def preprocess_dna_mutation(
    source: Union[str, Path, pd.DataFrame],
    hugo_symbols: Union[str, Path, None] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Preprocess DNA mutation (MAF) data for SeNMo.

    Args:
        source: Path to a TCGA MAF file (tab-separated, ``#``-comments
            allowed) with ``Hugo_Symbol`` and ``IMPACT`` columns, or
            an already-loaded DataFrame in the same shape.
        hugo_symbols: Path to a one-symbol-per-line TSV used as the
            feature vocabulary. Defaults to the file bundled with
            HoneyBee at
            ``honeybee/processors/molecular/data/Hugo_symbols.tsv``.
        seed: Seed for the random row-drop step. Same seed produces
            same output across runs; upstream omits this and is
            non-deterministic.

    Returns:
        Single-row DataFrame. Column 0 is ``DNA_Mut_values`` (the
        sample identifier column post-transpose, awkwardly named);
        columns 1..N are 0/1 binary mutation features per Hugo
        symbol, totalling 17,301 features.
    """
    if isinstance(source, (str, Path)):
        data = pd.read_csv(source, sep="\t", comment="#", low_memory=False)
    else:
        data = source

    if "Hugo_Symbol" not in data.columns or "IMPACT" not in data.columns:
        raise ValueError(
            "DNA mutation input must contain 'Hugo_Symbol' and 'IMPACT' columns."
        )

    labels_path = Path(hugo_symbols) if hugo_symbols is not None else _bundled_hugo_symbols_path()
    if not labels_path.is_file():
        raise FileNotFoundError(f"Hugo symbols file not found: {labels_path}")
    labels = pd.read_csv(labels_path, sep="\t", header=None)
    labels_list = labels[0].tolist()

    binary_dict = {label: 0 for label in labels_list}
    high_impact_symbols = data.loc[data["IMPACT"] == "HIGH", "Hugo_Symbol"]
    for symbol in high_impact_symbols:
        if symbol in binary_dict:
            binary_dict[symbol] = 1

    binary_df = pd.DataFrame(
        list(binary_dict.items()), columns=["Hugo_Symbol", "DNA_Mut_values"]
    )

    excess = binary_df.shape[0] - _DNAMUT_TARGET_ROWS
    if excess > 0:
        zero_value_indices = binary_df.index[
            binary_df["DNA_Mut_values"] == 0
        ].tolist()
        if len(zero_value_indices) < excess:
            raise ValueError(
                f"Not enough zero rows ({len(zero_value_indices)}) to drop "
                f"{excess} rows. Too many HIGH-impact mutations."
            )
        rng = random.Random(seed)
        drop_indices = rng.sample(zero_value_indices, excess)
        binary_df = binary_df.drop(index=drop_indices)

    transposed = binary_df.set_index("Hugo_Symbol").transpose()
    return transposed