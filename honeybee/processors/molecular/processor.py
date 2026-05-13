"""MolecularProcessor — multi-omics analysis using SeNMo.

PR 1 scope: accepts a preprocessed 80,697-dim multi-omics feature
vector (Mode C in the implementation plan) and returns survival
embedding + hazard score. Modes A and B (raw per-modality files and
combined pkl) are added in PR 2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from honeybee.models.SeNMo import SeNMoInference

from .result import MolecularResult

logger = logging.getLogger(__name__)


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
        result = proc.process(features=my_80697_dim_vector)
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
        features: Union[np.ndarray, torch.Tensor],
    ) -> MolecularResult:
        """Run pan-cancer multi-omics analysis on a feature vector.

        Args:
            features: Preprocessed multi-omics feature vector. Shape
                ``(80697,)`` for one patient or ``(N, 80697)`` for a
                batch. Use the helpers in
                ``honeybee.processors.molecular.preprocessing`` to
                build this vector from raw per-modality files (added
                in PR 2).

        Returns:
            A :class:`MolecularResult` with the 48-dim embedding,
            hazard score, and original input.
        """
        input_features = (
            features.detach().cpu().numpy() if torch.is_tensor(features) else np.asarray(features)
        )
        embedding, hazard_score = self.inference.predict(features)
        return MolecularResult(
            embedding=embedding,
            hazard_score=hazard_score,
            input_features=input_features,
        )
