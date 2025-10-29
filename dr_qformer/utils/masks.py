"""Utility functions for attention masks."""

from typing import Optional

try:
    import torch
    from torch import Tensor
except ImportError:
    Tensor = None


def build_sa_mask(n_queries: int, seq_len: int = 0) -> Optional[object]:
    """
    Build self-attention mask for Q-Former.
    
    Self-attention operates over [LQs, query_tokens, answer_tokens].
    
    Args:
        n_queries (int): Number of learnable query tokens (N)
        seq_len (int): Length of text sequence (query + answer)
    
    Returns:
        mask: Attention mask [N+seq_len, N+seq_len]
              - LQs can attend to all tokens (bidirectional)
              - Text tokens can attend to LQs and themselves
    
    TODO:
    - Implement bidirectional attention mask
    - Handle causal masking if needed for text
    - Support batch dimension
    - Return boolean or additive mask format
    
    Mask format:
    - False/0: Can attend
    - True/-inf: Cannot attend
    """
    total_len = n_queries + seq_len
    
    # TODO: Implement mask creation
    # mask = torch.zeros(total_len, total_len, dtype=torch.bool)
    pass
    
    return None


def build_ca_mask(n_queries: int, k_fragments: int) -> Optional[object]:
    """
    Build cross-attention mask for Q-Former.
    
    Cross-attention: LQs attend to retrieved fragment embeddings.
    
    Args:
        n_queries (int): Number of learnable query tokens (N)
        k_fragments (int): Number of retrieved fragments (k)
    
    Returns:
        mask: Attention mask [N, k]
              - All LQs can attend to all fragment embeddings
              - Used to mask out padding fragments
    
    TODO:
    - Create mask for cross-attention
    - Handle variable number of valid fragments (< k)
    - Support batch dimension
    """
    # TODO: Implement mask creation
    # mask = torch.zeros(n_queries, k_fragments, dtype=torch.bool)
    pass
    
    return None


def build_padding_mask(
    lengths: list,
    max_length: Optional[int] = None,
) -> Optional[object]:
    """
    Build padding mask for variable-length sequences.
    
    Args:
        lengths: List of sequence lengths in batch
        max_length: Maximum sequence length (uses max(lengths) if None)
    
    Returns:
        mask: Padding mask [batch, max_length]
              - False: Valid position
              - True: Padding position
    
    TODO:
    - Create padding mask from lengths
    - Support both 2D and 4D mask formats
    """
    if max_length is None:
        max_length = max(lengths) if lengths else 0
    
    # TODO: Implement padding mask
    # batch_size = len(lengths)
    # mask = torch.ones(batch_size, max_length, dtype=torch.bool)
    # for i, length in enumerate(lengths):
    #     mask[i, :length] = False
    pass
    
    return None


def build_causal_mask(seq_len: int) -> Optional[object]:
    """
    Build causal (lower triangular) attention mask.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        mask: Causal mask [seq_len, seq_len]
              - Position i can only attend to positions <= i
    
    TODO:
    - Create lower triangular mask
    - Support batch dimension
    """
    # TODO: Implement causal mask
    # mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    pass
    
    return None


def combine_masks(*masks) -> Optional[object]:
    """
    Combine multiple attention masks (logical OR).
    
    Args:
        *masks: Variable number of masks to combine
    
    Returns:
        combined_mask: Combined mask (True if any input is True)
    
    TODO:
    - Handle broadcasting of different mask shapes
    - Support both boolean and additive masks
    """
    # TODO: Implement mask combination
    pass
    return None
