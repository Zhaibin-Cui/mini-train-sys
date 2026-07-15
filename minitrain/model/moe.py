"""Compatibility imports for the unified configurable Transformer model."""

from minitrain.model.blocks import MoEFeedForward
from minitrain.model.config import MoEModelConfig
from minitrain.model.transformer import MiniMoETransformer

__all__ = ["MiniMoETransformer", "MoEFeedForward", "MoEModelConfig"]
