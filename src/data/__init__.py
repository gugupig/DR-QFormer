"""Data interfaces and dataset utilities."""

try:
    from .interfaces import Fragment, Example
except ImportError:
    pass

__all__ = ["Fragment", "Example"]
