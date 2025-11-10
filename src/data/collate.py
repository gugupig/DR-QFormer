"""
Data collation functions for DR-QFormer tasks.

Purpose: 数据契约前移 - 在 collate_fn 中统一完成 padding mask 构造与对齐
"""

from typing import List, Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None


def collate_task_e(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for Task E (Entailment Tagging).
    
    Purpose: 统一构造 pool_padding_mask，确保批内数据格式一致
    
    Args:
        batch: List of samples, each sample is a dict with:
            - query: str (single query)
            - fragments: List[str] (retrieved fragments, variable length)
            - gt_labels: List[int] (binary labels, same length as fragments)
            - answer: Optional[str] (for Dual mode)
            - is_longtail: Optional[List[int]] (longtail indicators)
            - mode: str ("primal" or "dual")
    
    Returns:
        Collated batch dict with:
            - queries: List[str] [batch_size]
            - fragments: List[List[str]] [batch_size, K_max] (padded)
            - gt_labels: Tensor [batch_size, K_max]
            - pool_padding_mask: Tensor [batch_size, K_max] (True=valid, False=padding)
            - importance_weights: Optional[Tensor] [batch_size, K_max]
            - answers: Optional[List[str]] [batch_size] (for Dual mode)
            - is_longtail: Optional[Tensor] [batch_size, K_max]
            - mode: str
    
    Key Features:
    =============
    1. 动态 K_pool: 批内最大 K，自动 padding
    2. pool_padding_mask: 标记有效片段 (True) vs padding (False)
    3. importance_weights: 根据 gt_labels + is_longtail 自动构造
    4. 支持 Primal (QA) 和 Dual (QG) 模式
    
    Example:
    ========
    >>> sample1 = {
    ...     "query": "What is AI?",
    ...     "fragments": ["AI is artificial intelligence", "Machine learning"],
    ...     "gt_labels": [1, 0],
    ...     "is_longtail": [0, 0],
    ... }
    >>> sample2 = {
    ...     "query": "What is ML?",
    ...     "fragments": ["ML is machine learning", "Deep learning", "Neural networks"],
    ...     "gt_labels": [1, 1, 0],
    ...     "is_longtail": [0, 1, 0],
    ... }
    >>> batch = collate_task_e([sample1, sample2])
    >>> print(batch["fragments"])  # [[frag1, frag2, ""], [frag1, frag2, frag3]]
    >>> print(batch["pool_padding_mask"])  # [[True, True, False], [True, True, True]]
    """
    if torch is None:
        raise ImportError("torch is required for collate functions")
    
    batch_size = len(batch)
    
    # Extract data
    queries = [sample["query"] for sample in batch]
    fragments_list = [sample["fragments"] for sample in batch]
    gt_labels_list = [sample["gt_labels"] for sample in batch]
    
    # 动态 K_pool: 批内最大片段数
    K_max = max(len(frags) for frags in fragments_list)
    
    # Pad fragments to K_max
    fragments_padded = []
    gt_labels_padded = []
    pool_padding_mask = []
    
    for frags, labels in zip(fragments_list, gt_labels_list):
        K_sample = len(frags)
        
        # Pad fragments with empty strings
        frags_padded = frags + [""] * (K_max - K_sample)
        fragments_padded.append(frags_padded)
        
        # Pad labels with 0
        labels_padded = labels + [0] * (K_max - K_sample)
        gt_labels_padded.append(labels_padded)
        
        # Create padding mask: True for valid, False for padding
        mask = [True] * K_sample + [False] * (K_max - K_sample)
        pool_padding_mask.append(mask)
    
    # Convert to tensors
    gt_labels_tensor = torch.tensor(gt_labels_padded, dtype=torch.float32)  # [batch_size, K_max]
    pool_padding_mask_tensor = torch.tensor(pool_padding_mask, dtype=torch.bool)  # [batch_size, K_max]
    
    # Construct importance_weights from gt_labels + is_longtail
    importance_weights = None
    if all("is_longtail" in sample for sample in batch):
        is_longtail_list = [sample["is_longtail"] for sample in batch]
        is_longtail_padded = []
        
        for longtail_flags in is_longtail_list:
            K_sample = len(longtail_flags)
            longtail_padded = longtail_flags + [0] * (K_max - K_sample)
            is_longtail_padded.append(longtail_padded)
        
        is_longtail_tensor = torch.tensor(is_longtail_padded, dtype=torch.int64)  # [batch_size, K_max]
        
        # Build importance weights:
        # - Negative class (gt=0): 1.0
        # - Positive class (gt=1): 10.0
        # - Longtail positive (gt=1 & longtail=1): 50.0
        importance_weights = torch.ones_like(gt_labels_tensor)  # Default: 1.0
        importance_weights = torch.where(
            gt_labels_tensor == 1,
            10.0 * torch.ones_like(gt_labels_tensor),
            importance_weights
        )
        importance_weights = torch.where(
            (gt_labels_tensor == 1) & (is_longtail_tensor == 1),
            50.0 * torch.ones_like(gt_labels_tensor),
            importance_weights
        )
    
    # Collect answers for Dual mode
    answers = None
    if all("answer" in sample for sample in batch):
        answers = [sample["answer"] for sample in batch]
    
    # Mode (Primal or Dual)
    mode = batch[0].get("mode", "primal")
    
    # Return collated batch
    collated = {
        "queries": queries,  # List[str]
        "fragments": fragments_padded,  # List[List[str]] [batch_size, K_max]
        "gt_labels": gt_labels_tensor,  # Tensor [batch_size, K_max]
        "pool_padding_mask": pool_padding_mask_tensor,  # Tensor [batch_size, K_max]
        "importance_weights": importance_weights,  # Optional[Tensor] [batch_size, K_max]
        "answers": answers,  # Optional[List[str]]
        "mode": mode,  # str
    }
    
    return collated


def collate_task_s(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for Task S (Sorting).
    
    Args:
        batch: List of samples with:
            - query: str
            - fragments: List[str]
            - gt_soft_weights: List[float] (target probability distribution)
            - mode: str ("primal" or "dual")
    
    Returns:
        Collated batch with padded fragments and soft weights
    """
    if torch is None:
        raise ImportError("torch is required for collate functions")
    
    batch_size = len(batch)
    queries = [sample["query"] for sample in batch]
    fragments_list = [sample["fragments"] for sample in batch]
    gt_soft_weights_list = [sample["gt_soft_weights"] for sample in batch]
    
    # Dynamic K_pool
    K_max = max(len(frags) for frags in fragments_list)
    
    # Pad
    fragments_padded = []
    gt_soft_weights_padded = []
    pool_padding_mask = []
    
    for frags, weights in zip(fragments_list, gt_soft_weights_list):
        K_sample = len(frags)
        
        # Pad fragments
        frags_padded = frags + [""] * (K_max - K_sample)
        fragments_padded.append(frags_padded)
        
        # Pad soft weights with 0.0
        weights_padded = weights + [0.0] * (K_max - K_sample)
        gt_soft_weights_padded.append(weights_padded)
        
        # Mask
        mask = [True] * K_sample + [False] * (K_max - K_sample)
        pool_padding_mask.append(mask)
    
    # Convert to tensors
    gt_soft_weights_tensor = torch.tensor(gt_soft_weights_padded, dtype=torch.float32)
    pool_padding_mask_tensor = torch.tensor(pool_padding_mask, dtype=torch.bool)
    
    # Mode
    mode = batch[0].get("mode", "primal")
    
    return {
        "queries": queries,
        "fragments": fragments_padded,
        "gt_soft_weights": gt_soft_weights_tensor,
        "pool_padding_mask": pool_padding_mask_tensor,
        "mode": mode,
    }


def collate_task_c(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for Task C (Condensing).
    
    Args:
        batch: List of samples with:
            - query: str
            - fragments: List[str]
            - gold_answer: str (for Primal QA)
            - gold_query: Optional[str] (for Dual QG)
            - mode: str ("primal" or "dual")
    
    Returns:
        Collated batch with padded fragments
    """
    if torch is None:
        raise ImportError("torch is required for collate functions")
    
    batch_size = len(batch)
    queries = [sample["query"] for sample in batch]
    fragments_list = [sample["fragments"] for sample in batch]
    gold_answers = [sample["gold_answer"] for sample in batch]
    
    # Dynamic K_pool
    K_max = max(len(frags) for frags in fragments_list)
    
    # Pad
    fragments_padded = []
    pool_padding_mask = []
    
    for frags in fragments_list:
        K_sample = len(frags)
        frags_padded = frags + [""] * (K_max - K_sample)
        fragments_padded.append(frags_padded)
        
        mask = [True] * K_sample + [False] * (K_max - K_sample)
        pool_padding_mask.append(mask)
    
    pool_padding_mask_tensor = torch.tensor(pool_padding_mask, dtype=torch.bool)
    
    # Optional gold queries for Dual mode
    gold_queries = None
    if all("gold_query" in sample for sample in batch):
        gold_queries = [sample["gold_query"] for sample in batch]
    
    mode = batch[0].get("mode", "primal")
    
    return {
        "queries": queries,
        "fragments": fragments_padded,
        "gold_answers": gold_answers,
        "gold_queries": gold_queries,
        "pool_padding_mask": pool_padding_mask_tensor,
        "mode": mode,
    }
