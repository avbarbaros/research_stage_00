"""Model adapters.

Deliberately dependency-light: at stage 0.2 both VLMEvalKit and lmms-eval will
import this package from their own venvs, and those two pin conflicting
dependencies. Nothing here may import anything heavier than config, torch, and
transformers.
"""

from .base import ADAPTERS, GenOutput, VLMAdapter, get_adapter_class, load_adapter

__all__ = ["ADAPTERS", "GenOutput", "VLMAdapter", "get_adapter_class", "load_adapter"]
