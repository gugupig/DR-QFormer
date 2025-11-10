"""
DR-QFormer: Parameter-Efficient Middleware for RAG.

BLIP-2-style Q-Former architecture where:
- Retriever & LLM are frozen
- Only Q-Former + task-specific heads are trainable
"""

__version__ = "0.1.0"

try:
    from .models.qformer import DRQFormer
    from .models.heads import EntailmentHead, SortingHead, CondenseHead
    from .adapters.retriever import Retriever
    from .adapters.llm import FrozenLLM
except ImportError:
    # Graceful fallback if dependencies are missing
    pass

__all__ = [
    "DRQFormer",
    "EntailmentHead",
    "SortingHead",
    "CondenseHead",
    "Retriever",
    "FrozenLLM",
]
