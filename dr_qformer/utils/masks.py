"""
Utility functions for attention masks (Cross-Attention Architecture).

DR-QFormer uses PERMISSIVE masking by default:
- SA mask: (N+1) x (N+1) all-ones → fully bidirectional
- CA mask: N x k all-ones → each LQ attends to all fragments
- Learning: Tasks E, S, C train attention to filter/rank effectively
"""

from typing import Optional

try:
    import torch
    from torch import Tensor
except ImportError:
    Tensor = None


def build_sa_mask(n_queries: int, seq_len: int = 1) -> Optional[object]:
    """
    Build self-attention mask for Q-Former SA stage.
    
    Cross-Attention Architecture Design:
    ====================================
    SA operates over: Concat([LQs, q_embed]) or Concat([LQs, a_embed])
    - Sequence length: N (LQs) + 1 (single query or answer embedding)
    - NOT concatenating full query/answer token sequences
    
    Mask Shape: (N+1) x (N+1)
    ==========================
    Default: ALL-ONES matrix (or no mask) - fully bidirectional attention
    
    Layout:
             LQ1 LQ2 ... LQN q/a_embed
        LQ1   1   1  ...  1      1
        LQ2   1   1  ...  1      1
        ...  ..  ..  ...  ..    ..
        LQN   1   1  ...  1      1
    q/a_embed 1   1  ...  1      1
    
    Behavior:
    - LQs can attend to: all other LQs + query/answer embedding
    - Query/answer embedding can attend to: all LQs + itself
    - Fully bidirectional - no causal restrictions
    
    Args:
        n_queries (int): Number of learnable query tokens (N). Default: 32
        seq_len (int): Length of q/a sequence. Should be 1 for single embedding.
                      Can be > 1 if using multiple query/answer embeddings.
    
    Returns:
        mask: Attention mask [(N+seq_len), (N+seq_len)]
              - False/0: Can attend (default for all positions)
              - True/-inf: Cannot attend (none by default)
              - OR None (no mask applied, equivalent to all-ones)
    
    Implementation Options:
    =======================
    Option 1 (Recommended): Return None
        - Most efficient, no mask overhead
        - PyTorch attention treats None as "attend to everything"
    
    Option 2: Return all-zeros boolean mask
        - Explicit representation: False = can attend
        - mask = torch.zeros(N+seq_len, N+seq_len, dtype=torch.bool)
    
    Option 3: Return all-ones additive mask (for older APIs)
        - Additive format: 0.0 = can attend, -inf = cannot
        - mask = torch.zeros(N+seq_len, N+seq_len)
    
    TODO:
    - Decide on mask format (None vs explicit)
    - Support batch dimension if needed: [batch, N+seq_len, N+seq_len]
    - Handle edge case: seq_len = 0 (LQs only)
    """
    total_len = n_queries + seq_len
    
    # TODO: Implement mask creation
    # Option 1: No mask (fully permissive)
    # return None
    #
    # Option 2: Explicit all-zeros boolean mask
    # mask = torch.zeros(total_len, total_len, dtype=torch.bool)
    # return mask
    #
    # Option 3: Additive mask (all zeros, no blocking)
    # mask = torch.zeros(total_len, total_len)
    # return mask
    pass
    
    return None


def build_ca_mask(n_queries: int, k_fragments: int) -> Optional[object]:
    """
    Build cross-attention mask for Q-Former CA stage.
    
    Cross-Attention Architecture Design:
    ====================================
    CA stage: LQs_aware (Query) attend to P_embeds (Key/Value)
    - Query: LQs_aware [batch, N, d] from SA stage
    - Key/Value: P_embeds [batch, k, d] from frozen retriever
    
    Mask Shape: N x k
    =================
    Default: ALL-ONES matrix (or no mask) - each LQ attends to all k fragments
    
    Layout:
             Frag1 Frag2 ... Fragk
        LQ1    1     1   ...   1
        LQ2    1     1   ...   1
        ...   ..    ..   ...  ..
        LQN    1     1   ...   1
    
    Behavior:
    - Each LQ can attend to: ALL k fragment embeddings
    - No restrictions by default
    - Learning: Tasks E and S train attention weights to focus on relevant fragments
    
    Args:
        n_queries (int): Number of learnable query tokens (N). Default: 32
        k_fragments (int): Number of retrieved fragments (k). Default: 10
    
    Returns:
        mask: Attention mask [N, k]
              - False/0: Can attend (default for all positions)
              - True/-inf: Cannot attend (used for padding only)
              - OR None (no mask, equivalent to all-ones)
    
    Padding Handling:
    =================
    When actual number of fragments < k (due to padding):
    - Valid fragments: mask = False (can attend)
    - Padding fragments: mask = True (cannot attend)
    
    Example with k=5, but only 3 valid fragments:
        mask = [[F, F, F, T, T],   # LQ1: attend to 3 valid, ignore 2 padding
                [F, F, F, T, T],   # LQ2: same
                ...
                [F, F, F, T, T]]   # LQN: same
    
    Implementation Notes:
    =====================
    This function builds the BASE mask (all-ones / no restrictions).
    Use build_padding_mask() to handle variable-length fragments.
    Combine masks using combine_masks() if needed.
    
    TODO:
    - Create base CA mask (all-ones or None)
    - Support batch dimension: [batch, N, k] if needed
    - Combine with padding mask for variable-length fragments
    """
    # TODO: Implement mask creation
    # Option 1: No mask (fully permissive)
    # return None
    #
    # Option 2: Explicit all-zeros boolean mask
    # mask = torch.zeros(n_queries, k_fragments, dtype=torch.bool)
    # return mask
    pass
    
    return None


def build_padding_mask(
    lengths: list,
    max_length: Optional[int] = None,
) -> Optional[object]:
    """
    Build padding mask for variable-length P_embeds (REQUIRED for batching).
    
    Purpose: Mask out invalid padding positions in fragment embeddings.
    
    Use Cases:
    ==========
    1. Variable number of fragments per query:
       - Query 1 retrieved k=8 fragments
       - Query 2 retrieved k=5 fragments
       - Batch k=10 (padded with zeros)
       - Mask ensures attention ignores padding
    
    2. CA stage masking:
       - Combine with base CA mask using combine_masks()
       - Prevents LQs from attending to padding fragments
    
    Args:
        lengths: List of valid fragment counts per batch [batch_size]
                Example: [8, 5, 10, 7] for batch_size=4, max_k=10
        max_length: Maximum sequence length (k_fragments)
                   If None, uses max(lengths)
    
    Returns:
        mask: Padding mask [batch, max_length]
              - False: Valid fragment position (can attend)
              - True: Padding position (cannot attend)
    
    Example:
    ========
    lengths = [3, 2, 4], max_length = 5
    mask = [[F, F, F, T, T],   # 3 valid, 2 padding
            [F, F, T, T, T],   # 2 valid, 3 padding
            [F, F, F, F, T]]   # 4 valid, 1 padding
    
    Integration with CA:
    ====================
    1. Build base CA mask: [N, k] all-ones
    2. Build padding mask: [batch, k]
    3. Expand padding mask: [batch, 1, k] → broadcast to [batch, N, k]
    4. Combine: final_mask = base_mask | padding_mask
    5. Use in CA attention: prevents attending to padding
    
    Why REQUIRED:
    =============
    Without padding mask:
    - LQs attend to padding (zeros or garbage)
    - Attention weights diluted across invalid positions
    - Softmax includes padding in normalization
    - Gradients affected by meaningless tokens
    
    Transformer libraries (PyTorch, Hugging Face) typically handle this automatically
    if you provide key_padding_mask parameter.
    
    TODO:
    - Create padding mask from lengths
    - Set True for padding positions, False for valid
    - Support 2D format [batch, k] for CA
    - Ensure dtype is torch.bool for PyTorch attention APIs
    """
    if max_length is None:
        max_length = max(lengths) if lengths else 0
    
    # TODO: Implement padding mask
    # batch_size = len(lengths)
    # mask = torch.ones(batch_size, max_length, dtype=torch.bool)
    # for i, length in enumerate(lengths):
    #     mask[i, :length] = False  # Valid positions = False
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
