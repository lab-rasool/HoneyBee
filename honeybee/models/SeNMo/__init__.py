"""SeNMo: Self-Normalizing Multi-Omics Neural Network.

The pretrained pan-cancer model from Waqas et al. 2025 wrapped for use
within HoneyBee. :class:`SeNMo` is the model architecture;
:class:`SeNMoInference` handles the 10-checkpoint ensemble download
and inference.
"""

from .inference import SeNMoInference
from .model import SeNMo

__all__ = ["SeNMo", "SeNMoInference"]