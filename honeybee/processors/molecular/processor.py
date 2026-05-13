"""MolecularProcessor — multi-omics analysis using SeNMo.

Public API for the molecular pillar. Accepts input in three modes:

* Mode A — ``features_pkl=``: a pkl file in
  ``lab-rasool/SeNMo/combine_features.py``'s format.
* Mode B — ``raw=``: dict of raw per-modality TSV paths or DataFrames.
  Preprocess + combine happens internally.
* Mode C — ``features=``: a preprocessed 80,697-dim vector.

All three modes converge on the same underlying SeNMo ensemble inference.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

import numpy as np
import torch

from honeybee.models.SeNMo import SeNMoInference

from .preprocessing import (
    combine_modalities,
    preprocess_clinical_covariates,
    preprocess_dna_methylation,
    preprocess_dna_mutation,
    preprocess_gene_expression,
    preprocess_mirna,
    preprocess_protein,
)
from .result import MolecularResult

logger = logging.getLogger(__name__)

# Modality name in ``raw={}`` -> preprocessing function. Keys must
# match those accepted by :func:`combine_modalities`.
_RAW_PREPROCESSORS: Dict[str, Callable[..., Any]] = {
    "clinical": preprocess_clinical_covariates,
    "dna_methylation": preprocess_dna_methylation,
    "dna_mutation": preprocess_dna_mutation,
    "gene_expression": preprocess_gene_expression,
    "mirna": preprocess_mirna,
    "protein": preprocess_protein,
}


class MolecularProcessor:
    """Pan-cancer multi-omics processor backed by SeNMo.

    Wraps the pretrained 10-checkpoint SeNMo ensemble published by
    Waqas et al. 2025 as a HoneyBee pillar. The underlying
    :class:`SeNMoInference` is lazy-loaded on first call to
    :meth:`process` so simply importing or constructing the processor
    does not download checkpoints.

    Args:
        checkpoint_dir: Local directory of SeNMo ``*.pt`` files. If
            ``None``, downloads them from HuggingFace Hub
            (``Lab-Rasool/SeNMo``) on first use.
        device: Torch device string. Auto-detects ``cuda``/``cpu``.

    Example::

        proc = MolecularProcessor()
        # Mode C: pre-combined feature vector
        result = proc.process(features=my_80697_dim_vector)
        # Mode A: upstream-format pkl
        result = proc.process(features_pkl="multiomic_features.pkl")
        # Mode B: raw per-modality TSVs
        result = proc.process(raw={
            "gene_expression": "gene.tsv",
            "dna_methylation": "methyl.tsv",
            # any modality can be omitted -> zero-padded
        })

        result.embedding       # (48,)
        result.hazard_score    # float
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self._inference: Optional[SeNMoInference] = None

    @property
    def inference(self) -> SeNMoInference:
        """Lazy-initialize the underlying SeNMo ensemble on first access."""
        if self._inference is None:
            logger.info("Initializing SeNMo ensemble (may trigger HF Hub download)")
            self._inference = SeNMoInference(
                checkpoint_dir=self.checkpoint_dir,
                device=self.device,
            )
        return self._inference

    def process(
        self,
        features: Optional[Union[np.ndarray, torch.Tensor]] = None,
        features_pkl: Optional[Union[str, Path]] = None,
        raw: Optional[Mapping[str, Any]] = None,
        seed: int = 42,
    ) -> MolecularResult:
        """Run pan-cancer multi-omics analysis.

        Exactly one of ``features``, ``features_pkl``, or ``raw`` must
        be provided.

        Args:
            features: Preprocessed multi-omics feature vector, shape
                ``(80697,)`` or ``(N, 80697)``. Pass-through to SeNMo.
            features_pkl: Path to a pkl in upstream
                ``combine_features.py``'s format
                (``data['cv_splits'][1]['test']['x_omic'][0]``).
            raw: Mapping of modality name to raw per-modality data
                (TSV/MAF path or already-loaded DataFrame in the
                orientation upstream's preprocessing expects). Keys:
                ``clinical``, ``dna_mutation``, ``protein``,
                ``gene_expression``, ``dna_methylation``, ``mirna``.
                Omitted modalities are zero-padded.
            seed: Seed for the DNA mutation preprocessor's random
                row-drop. Ignored unless ``'dna_mutation'`` appears
                in ``raw``.

        Returns:
            :class:`MolecularResult` with the 48-dim embedding,
            hazard score, and the resolved 80,697-dim input vector.
        """
        provided = [x is not None for x in (features, features_pkl, raw)]
        if sum(provided) != 1:
            raise ValueError(
                "Provide exactly one of features, features_pkl, or raw "
                f"(got {sum(provided)})."
            )

        if features_pkl is not None:
            resolved = _load_features_pkl(features_pkl)
        elif raw is not None:
            resolved = _preprocess_and_combine(raw, seed=seed)
        else:
            resolved = (
                features.detach().cpu().numpy()
                if torch.is_tensor(features)
                else np.asarray(features)
            )

        embedding, hazard_score = self.inference.predict(resolved)
        return MolecularResult(
            embedding=embedding,
            hazard_score=hazard_score,
            input_features=resolved,
        )


def _load_features_pkl(pkl_path: Union[str, Path]) -> np.ndarray:
    """Load a SeNMo combine_features.py-format pkl and return the
    single-patient 80,697-dim vector."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    try:
        x_omic = data["cv_splits"][1]["test"]["x_omic"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(
            f"pkl at {pkl_path} does not match the upstream "
            f"combine_features.py format "
            f"(expected data['cv_splits'][1]['test']['x_omic'][0]): {e}"
        ) from e
    return np.asarray(x_omic, dtype=np.float32)


def _preprocess_and_combine(
    raw: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    """Preprocess each provided raw modality and combine into one vector."""
    unknown = set(raw.keys()) - set(_RAW_PREPROCESSORS.keys())
    if unknown:
        raise ValueError(
            f"Unknown modality keys in raw: {sorted(unknown)}. "
            f"Allowed: {sorted(_RAW_PREPROCESSORS.keys())}"
        )

    preprocessed: Dict[str, Any] = {}
    for modality, source in raw.items():
        fn = _RAW_PREPROCESSORS[modality]
        # DNA mutation accepts a seed for its random row-drop; others
        # take a single source argument.
        if modality == "dna_mutation":
            preprocessed[modality] = fn(source, seed=seed)
        else:
            preprocessed[modality] = fn(source)

    return combine_modalities(preprocessed)