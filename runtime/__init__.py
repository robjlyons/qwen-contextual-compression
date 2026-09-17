"""Single-layer sparse FFN runtime prototypes (not full-model inference)."""

from runtime.ffn import DenseFFN, StaticPackedFFN, TorchDynamicSparseFFN
from runtime.selector import RuntimeSelector

__all__ = ["DenseFFN", "StaticPackedFFN", "TorchDynamicSparseFFN", "RuntimeSelector"]
