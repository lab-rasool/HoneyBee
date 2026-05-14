"""SeNMo multi-omics survival model.

Ported from Waqas et al. 2025 (Int. J. Mol. Sci. 26:7358). Architecture
mirrors the ``SeNMo`` class in the original repo at
``lab-rasool/SeNMo/SeNMo_Ensemble.py:171``. Module names, the
``nn.Sequential(nn.Linear(...))`` classifier wrap, the keyword-only
``forward(**kwargs)`` signature, and the ``output_range``/``output_shift``
buffers are preserved verbatim so the published checkpoints on
HuggingFace (``Lab-Rasool/SeNMo``) load via ``load_state_dict`` without
any key remapping.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# Pan-cancer architecture from Fig 1C of the paper (and matches the
# ``Lab-Rasool/SeNMo`` checkpoints). The class default in upstream
# SeNMo_Ensemble.py:172 is a different 5-layer shape because runtime
# layers come from CLI args there; here we hardcode the published one.
_DEFAULT_HIDDEN_LAYERS: List[int] = [1024, 512, 256, 128, 48, 48, 48]


class SeNMo(nn.Module):
    """Self-Normalizing Multi-Omics Neural Network.

    Seven-layer MLP that maps an 80,697-dim multi-omics feature vector
    (gene expression + DNA methylation + miRNA + protein + mutation +
    clinical, concatenated) to a 48-dim patient embedding plus a 1-dim
    hazard score.

    Note: the paper (Sec 4.6) describes the activation as SELU, but the
    upstream training code uses ``nn.ELU`` (see
    ``SeNMo_Ensemble.py:182``). The published checkpoints were trained
    with ELU, so this port uses ELU. A parity test against the upstream
    ``generate_embeddings.py`` is the definitive check.

    Args:
        input_dim: Multi-omics feature dim. Default 80697 matches the
            pan-cancer pretrained checkpoint.
        hidden_layers: Hidden layer widths.
        dropout_rate: AlphaDropout probability. Has no effect at
            inference (``model.eval()`` makes it a pass-through).
        act: Optional activation applied to the output head. ``None``
            for raw regression (hazard score); ``nn.Sigmoid`` clamps the
            output to ``[-3, 3]`` via ``output_range`` and
            ``output_shift``; ``nn.LogSoftmax`` for classification.
        label_dim: Output dim. 1 for survival regression, ignored when
            ``regression=False`` (33 classes hardcoded).
        regression: Survival regression head when True; 33-way cancer
            type classification head when False.
    """

    def __init__(
        self,
        input_dim: int = 80697,
        hidden_layers: Optional[List[int]] = None,
        dropout_rate: float = 0.25,
        act: Optional[nn.Module] = None,
        label_dim: int = 1,
        regression: bool = True,
    ) -> None:
        super().__init__()
        self.act = act
        self.regression = regression

        if hidden_layers is None:
            hidden_layers = list(_DEFAULT_HIDDEN_LAYERS)

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.AlphaDropout(p=dropout_rate, inplace=False))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Wrapped in Sequential (rather than bare Linear) so published
        # checkpoints' ``classifier.0.weight`` / ``classifier.0.bias``
        # keys load without remapping.
        if regression:
            self.classifier = nn.Sequential(nn.Linear(prev_dim, label_dim))
        else:
            self.classifier = nn.Sequential(nn.Linear(prev_dim, 33))

        # Non-learnable but stored as Parameters (matching the checkpoint
        # state dict structure). Only used when ``act`` is ``nn.Sigmoid``
        # to rescale output from [0, 1] to [-3, 3].
        if regression:
            self.output_range = nn.Parameter(
                torch.FloatTensor([6]), requires_grad=False
            )
            self.output_shift = nn.Parameter(
                torch.FloatTensor([-3]), requires_grad=False
            )
        else:
            self.output_range = None
            self.output_shift = None

    def forward(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keyword-only ``x_omic`` signature mirrors upstream so the same
        # inference loops work over this class without modification.
        x = kwargs["x_omic"]
        features = self.encoder(x)
        out = self.classifier(features)
        if self.regression:
            if self.act is not None:
                out = self.act(out)
                if isinstance(self.act, nn.Sigmoid):
                    out = out * self.output_range + self.output_shift
        else:
            if self.act is not None:
                out = self.act(out)
        return features, out