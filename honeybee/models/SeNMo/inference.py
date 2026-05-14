"""SeNMo ensemble inference.

Loads (or downloads) the published 10-checkpoint pan-cancer ensemble
from HuggingFace Hub and averages predictions across all 10 models.
Reproduces the upstream behavior of
``lab-rasool/SeNMo/generate_embeddings.py``, packaged as a class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download

from .model import SeNMo

logger = logging.getLogger(__name__)

# Pan-cancer ensemble checkpoints from Waqas et al. 2025.
_HF_REPO_ID = "Lab-Rasool/SeNMo"


class SeNMoInference:
    """10-checkpoint ensemble inference for SeNMo.

    On construction, locates the ensemble checkpoints (downloading from
    HuggingFace Hub the first time if a local path is not supplied),
    instantiates 10 :class:`SeNMo` copies, and sets them to ``eval()``.
    Calling :meth:`predict` runs all 10 models and averages the outputs
    to reproduce the published ensemble inference.

    Args:
        checkpoint_dir: Directory holding the 10 ``*.pt`` checkpoint
            files. If ``None``, downloads from ``Lab-Rasool/SeNMo`` on
            HF Hub (cached via the standard ``HF_HOME`` mechanism).
        device: Torch device string. Defaults to ``cuda`` if available,
            else ``cpu``.

    Example::

        inf = SeNMoInference()             # first call downloads ~3 GB
        emb, hz = inf.predict(np.random.randn(80697))
        emb.shape   # (48,)
        type(hz)    # float
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_dir = self._resolve_checkpoint_dir(checkpoint_dir)
        self.models: List[SeNMo] = self._load_ensemble(self.checkpoint_dir)
        # Read the expected input dim from the first encoder Linear layer
        # so the shape check stays in sync if the architecture changes.
        self.input_dim: int = self.models[0].encoder[0].in_features
        logger.info(
            "SeNMoInference ready: %d checkpoints on %s, input_dim=%d",
            len(self.models),
            self.device,
            self.input_dim,
        )

    @staticmethod
    def _resolve_checkpoint_dir(checkpoint_dir: Optional[Union[str, Path]]) -> Path:
        if checkpoint_dir is not None:
            path = Path(checkpoint_dir)
            if not path.is_dir():
                raise FileNotFoundError(f"checkpoint_dir not found: {path}")
            return path
        local_path = snapshot_download(
            repo_id=_HF_REPO_ID,
            allow_patterns=["*.pt"],
        )
        return Path(local_path)

    def _load_ensemble(self, checkpoint_dir: Path) -> List[SeNMo]:
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No .pt checkpoints found in {checkpoint_dir}. "
                f"Expected pretrained ensemble from HF repo `{_HF_REPO_ID}`."
            )

        models: List[SeNMo] = []
        for ckpt_path in checkpoints:
            model = SeNMo()
            # weights_only=False: the published checkpoints bundle
            # optimizer state and other pickled objects (protocol 4)
            # that PyTorch 2.6+'s safe unpickler rejects. We trust the
            # source (verified HF repo Lab-Rasool/SeNMo).
            state = torch.load(
                ckpt_path, map_location=self.device, weights_only=False
            )
            state_dict = (
                state["model_state_dict"]
                if isinstance(state, dict) and "model_state_dict" in state
                else state
            )
            # Strip DataParallel ``module.`` prefix if checkpoint was
            # saved from a DataParallel-wrapped model.
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("module.", "", 1): v for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            models.append(model)
        return models

    @torch.inference_mode()
    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """Run ensemble inference on a multi-omics feature vector.

        Args:
            features: Multi-omics features. Shape ``(80697,)`` for a
                single patient or ``(N, 80697)`` for a batch.

        Returns:
            ``(embedding, hazard_score)`` averaged across the 10
            checkpoints. For 1D input: embedding is ``(48,)`` and
            ``hazard_score`` is a Python ``float``. For batched input:
            embedding is ``(N, 48)`` and ``hazard_score`` is ``(N,)``.
        """
        was_1d, x = self._coerce_input(features)

        all_features: List[torch.Tensor] = []
        all_hazards: List[torch.Tensor] = []
        for model in self.models:
            feat, hz = model(x_omic=x)
            all_features.append(feat)
            all_hazards.append(hz)

        mean_features = torch.stack(all_features).mean(dim=0)
        mean_hazards = torch.stack(all_hazards).mean(dim=0).squeeze(-1)

        embeddings = mean_features.cpu().numpy()
        hazards = mean_hazards.cpu().numpy()

        if was_1d:
            return embeddings[0], float(hazards.item())
        return embeddings, hazards

    def _coerce_input(
        self, features: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[bool, torch.Tensor]:
        if isinstance(features, np.ndarray):
            x = torch.from_numpy(features).float()
        elif torch.is_tensor(features):
            x = features.float()
        else:
            raise TypeError(
                f"features must be ndarray or Tensor, got {type(features).__name__}"
            )
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected features of shape ({self.input_dim},) or "
                f"(N, {self.input_dim}), got {tuple(x.shape)}"
            )
        return was_1d, x.to(self.device)