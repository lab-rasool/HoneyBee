"""Clinical covariates preprocessing.

Faithful port of ``ClinicalDataPreprocessor`` in
``lab-rasool/SeNMo/package_classes/Clinical_preprocess.py``. Pulls
the four covariates SeNMo expects (age, gender, race, stage) from a
TCGA-GDC phenotype TSV and maps the categoricals to numeric codes.

Caveat: this preprocessor is hardcoded for TCGA-GDC column names
(``age_at_index.demographic``, ``gender.demographic``, etc.). External
cohorts using different schemas (CPTAC, Moffitt) need to be remapped
to these column names by the caller before invoking this function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import pandas as pd

# TCGA-GDC source column -> SeNMo-expected short name.
_REQUIRED_COLUMNS: Dict[str, str] = {
    "age_at_index.demographic": "age",
    "gender.demographic": "gender",
    "race.demographic": "race",
    "tumor_stage.diagnoses": "stage",
}

_GENDER_MAP: Dict[str, int] = {"male": 1, "female": 2}

_RACE_MAP: Dict[str, int] = {
    "white": 1,
    "asian": 2,
    "black or african american": 3,
    "not reported": 4,
    "american indian or alaska native": 5,
}

_STAGE_MAP: Dict[str, int] = {
    "stage 0": 1,
    "is": 10,
    "stage i": 10,
    "stage ia": 11,
    "stage ib": 12,
    "stage ic": 13,
    "stage ii": 20,
    "i/ii nos": 20,
    "stage iia": 21,
    "stage iib": 22,
    "stage iic": 23,
    "stage iii": 30,
    "stage iiia": 31,
    "stage iiib": 32,
    "stage iiic": 33,
    "stage iv": 40,
    "stage iva": 41,
    "stage ivb": 42,
    "stage ivc": 43,
    "not reported": 50,
    "stage x": 50,
}


def preprocess_clinical_covariates(
    source: Union[str, Path, pd.DataFrame],
) -> pd.DataFrame:
    """Extract and numerically encode SeNMo's clinical covariates.

    Args:
        source: Path to a TCGA-GDC phenotype TSV (tab-separated) or
            an already-loaded DataFrame. Must contain the four
            ``*.demographic`` / ``*.diagnoses`` columns listed in
            ``_REQUIRED_COLUMNS``.

    Returns:
        DataFrame with one row per sample and four numeric columns:
        ``age``, ``gender``, ``race``, ``stage``. Unmapped categorical
        values become NaN (matches upstream ``.map`` semantics).
    """
    if isinstance(source, (str, Path)):
        data = pd.read_csv(source, sep="\t", low_memory=False)
    else:
        data = source

    missing = [c for c in _REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            f"Clinical input missing required TCGA-GDC columns: {missing}"
        )

    selected = data[list(_REQUIRED_COLUMNS.keys())].copy()
    selected = selected.rename(columns=_REQUIRED_COLUMNS)

    selected["age"] = selected["age"].astype(float)
    selected["gender"] = selected["gender"].map(_GENDER_MAP)
    selected["race"] = selected["race"].map(_RACE_MAP)
    selected["stage"] = selected["stage"].map(_STAGE_MAP)

    return selected
