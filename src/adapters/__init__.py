"""Adapters for external frozen components."""

try:
    from .retriever import Retriever
    from .llm import FrozenLLM
except ImportError:
    pass

__all__ = ["Retriever", "FrozenLLM"]
