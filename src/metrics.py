"""Evaluation metrics for DR-QFormer tasks."""

from typing import List, Optional


def exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Exact match accuracy.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
    
    Returns:
        em_score: Fraction of exact matches
    
    TODO:
    - Implement case-insensitive exact match
    - Handle normalization (whitespace, punctuation)
    """
    # Placeholder
    return 0.0


def f1_score(predictions: List[str], references: List[str]) -> float:
    """
    Token-level F1 score.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
    
    Returns:
        f1: Average F1 score
    
    TODO:
    - Tokenize predictions and references
    - Compute precision and recall
    - Average F1 over examples
    """
    # Placeholder
    return 0.0


def rouge_score(
    predictions: List[str],
    references: List[str],
    rouge_type: str = "rougeL",
) -> float:
    """
    ROUGE score for summarization/generation.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        rouge_type: ROUGE variant (rouge1, rouge2, rougeL)
    
    Returns:
        rouge: Average ROUGE score
    
    TODO:
    - Install rouge-score package
    - Compute ROUGE metrics
    - Support different ROUGE types
    """
    # TODO: Implement with rouge-score library
    # from rouge_score import rouge_scorer
    # scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    pass
    return 0.0


def bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    BLEU score for generation.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
    
    Returns:
        bleu: Average BLEU score
    
    TODO:
    - Install sacrebleu package
    - Compute BLEU metrics
    - Handle multiple references per prediction
    """
    # TODO: Implement with sacrebleu library
    # from sacrebleu import corpus_bleu
    pass
    return 0.0


def compute_ranking_metrics(
    ranking_logits,  # Tensor [batch, K] or numpy array
    gt_scores,  # Tensor [batch, K] or numpy array
    pool_padding_mask=None,  # Tensor [batch, K] or numpy array
    k_list: list = [5, 10, 20],
) -> dict:
    """
    Ranking evaluation metrics for Task S.
    
    Args:
        ranking_logits: Predicted ranking scores [batch, K]
        gt_scores: Ground truth relevance scores [batch, K]
        pool_padding_mask: Valid fragment mask [batch, K] (True=valid, False=padding)
        k_list: List of k values for NDCG@k (default: [5, 10, 20])
    
    Returns:
        metrics: Dictionary with:
            - ndcg@k: Normalized Discounted Cumulative Gain at k
            - mrr: Mean Reciprocal Rank
            - map: Mean Average Precision
            - spearman: Spearman's rank correlation
    
    Example:
        >>> ranking_logits = torch.randn(2, 50)
        >>> gt_scores = torch.randn(2, 50).softmax(dim=-1)
        >>> mask = torch.ones(2, 50, dtype=torch.bool)
        >>> metrics = compute_ranking_metrics(ranking_logits, gt_scores, mask)
    """
    try:
        import torch
        import numpy as np
        from scipy.stats import spearmanr
        
        # Convert to numpy
        if isinstance(ranking_logits, torch.Tensor):
            ranking_logits = ranking_logits.detach().cpu().numpy()
        if isinstance(gt_scores, torch.Tensor):
            gt_scores = gt_scores.detach().cpu().numpy()
        if pool_padding_mask is not None and isinstance(pool_padding_mask, torch.Tensor):
            pool_padding_mask = pool_padding_mask.detach().cpu().numpy()
        else:
            pool_padding_mask = np.ones_like(ranking_logits, dtype=bool)
        
        batch_size, K = ranking_logits.shape
        
        # Initialize metrics
        ndcg_scores = {f"ndcg@{k}": [] for k in k_list}
        mrr_scores = []
        map_scores = []
        spearman_scores = []
        
        for b in range(batch_size):
            # Get valid positions
            valid_mask = pool_padding_mask[b]
            if not valid_mask.any():
                continue
            
            pred_scores_b = ranking_logits[b][valid_mask]
            gt_scores_b = gt_scores[b][valid_mask]
            K_valid = len(pred_scores_b)
            
            # Sort by predicted scores (descending)
            pred_ranking = np.argsort(-pred_scores_b)
            
            # Get ground truth relevance for predicted ranking
            gt_at_pred_rank = gt_scores_b[pred_ranking]
            
            # === NDCG@k ===
            for k in k_list:
                if K_valid < k:
                    k_actual = K_valid
                else:
                    k_actual = k
                
                # DCG@k
                dcg = np.sum(gt_at_pred_rank[:k_actual] / np.log2(np.arange(2, k_actual + 2)))
                
                # IDCG@k (ideal DCG - sort by ground truth)
                gt_sorted = np.sort(gt_scores_b)[::-1]  # Descending
                idcg = np.sum(gt_sorted[:k_actual] / np.log2(np.arange(2, k_actual + 2)))
                
                # NDCG
                ndcg = dcg / (idcg + 1e-8)
                ndcg_scores[f"ndcg@{k}"].append(ndcg)
            
            # === MRR (Mean Reciprocal Rank) ===
            # Find first relevant item (assume top gt_score as relevant)
            threshold = np.percentile(gt_scores_b, 90)  # Top 10% as relevant
            relevant_mask = gt_scores_b >= threshold
            
            # Find rank of first relevant item in predicted ranking
            rr = 0.0
            for rank, idx in enumerate(pred_ranking, start=1):
                if relevant_mask[idx]:
                    rr = 1.0 / rank
                    break
            mrr_scores.append(rr)
            
            # === MAP (Mean Average Precision) ===
            # Count relevant items
            num_relevant = relevant_mask.sum()
            if num_relevant > 0:
                # Calculate precision at each relevant position
                precisions = []
                num_relevant_so_far = 0
                for rank, idx in enumerate(pred_ranking, start=1):
                    if relevant_mask[idx]:
                        num_relevant_so_far += 1
                        precision_at_k = num_relevant_so_far / rank
                        precisions.append(precision_at_k)
                
                ap = np.mean(precisions) if precisions else 0.0
                map_scores.append(ap)
            
            # === Spearman's Rank Correlation ===
            if K_valid > 1:
                try:
                    corr, _ = spearmanr(pred_scores_b, gt_scores_b)
                    if not np.isnan(corr):
                        spearman_scores.append(corr)
                except:
                    pass
        
        # Aggregate metrics
        metrics = {}
        for k in k_list:
            metrics[f"ndcg@{k}"] = np.mean(ndcg_scores[f"ndcg@{k}"]) if ndcg_scores[f"ndcg@{k}"] else 0.0
        
        metrics["mrr"] = np.mean(mrr_scores) if mrr_scores else 0.0
        metrics["map"] = np.mean(map_scores) if map_scores else 0.0
        metrics["spearman"] = np.mean(spearman_scores) if spearman_scores else 0.0
        
        return metrics
    
    except ImportError:
        # Fallback if scipy not available
        return {
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "ndcg@20": 0.0,
            "mrr": 0.0,
            "map": 0.0,
            "spearman": 0.0,
        }


def ranking_metrics(
    predicted_ranks: List[List[int]],
    true_ranks: List[List[int]],
) -> dict:
    """
    DEPRECATED: Use compute_ranking_metrics() instead.
    
    Ranking evaluation metrics.
    
    Args:
        predicted_ranks: Predicted rankings [batch x k]
        true_ranks: Ground-truth rankings [batch x k]
    
    Returns:
        metrics: Dictionary with:
            - ndcg: Normalized discounted cumulative gain
            - map: Mean average precision
            - kendall_tau: Rank correlation
    """
    metrics = {
        "ndcg": 0.0,
        "map": 0.0,
        "kendall_tau": 0.0,
    }
    return metrics


def compute_entailment_metrics(
    logits,  # Tensor [batch, k] or numpy array
    gt_labels,  # Tensor [batch, k] or numpy array
    pool_padding_mask=None,  # Tensor [batch, k] or numpy array, True=valid
    threshold: float = 0.5,
) -> dict:
    """
    Binary classification metrics for entailment tagging (Task E).
    
    Args:
        logits: Predicted logits [batch, k] (raw scores before sigmoid)
        gt_labels: Ground-truth binary labels [batch, k] (0/1)
        pool_padding_mask: Valid fragment mask [batch, k] (True=valid, False=padding)
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        metrics: Dictionary with:
            - accuracy: Overall accuracy
            - precision: Precision for positive class
            - recall: Recall for positive class
            - f1: F1 score for positive class
            - true_positives: Number of TP
            - false_positives: Number of FP
            - false_negatives: Number of FN
            - true_negatives: Number of TN
    
    Example:
        >>> import torch
        >>> logits = torch.randn(2, 5)
        >>> labels = torch.tensor([[1, 0, 0, 1, 0], [0, 1, 0, 0, 0]])
        >>> mask = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
        >>> metrics = compute_entailment_metrics(logits, labels, mask)
        >>> print(f"F1: {metrics['f1']:.4f}")
    """
    try:
        import torch
        import numpy as np
        
        # Convert to numpy if needed
        if isinstance(logits, torch.Tensor):
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        else:
            probs = 1.0 / (1.0 + np.exp(-logits))  # Manual sigmoid
        
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.detach().cpu().numpy()
        
        if pool_padding_mask is not None:
            if isinstance(pool_padding_mask, torch.Tensor):
                pool_padding_mask = pool_padding_mask.detach().cpu().numpy()
        else:
            pool_padding_mask = np.ones_like(gt_labels, dtype=bool)
        
        # Apply threshold to get binary predictions
        preds = (probs >= threshold).astype(int)
        
        # Flatten and filter by mask
        preds_flat = preds[pool_padding_mask]
        labels_flat = gt_labels[pool_padding_mask]
        
        # Compute confusion matrix components
        tp = ((preds_flat == 1) & (labels_flat == 1)).sum()
        fp = ((preds_flat == 1) & (labels_flat == 0)).sum()
        fn = ((preds_flat == 0) & (labels_flat == 1)).sum()
        tn = ((preds_flat == 0) & (labels_flat == 0)).sum()
        
        # Compute metrics
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }
    
    except ImportError:
        # Fallback if torch/numpy not available
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
        }


def entailment_metrics(
    predictions: List[List[int]],
    labels: List[List[int]],
) -> dict:
    """
    Binary classification metrics for entailment (legacy interface).
    
    DEPRECATED: Use compute_entailment_metrics() instead for tensor inputs.
    
    Args:
        predictions: Predicted labels [batch x k] (0/1)
        labels: Ground-truth labels [batch x k] (0/1)
    
    Returns:
        metrics: Dictionary with accuracy, precision, recall, f1
    """
    try:
        import numpy as np
        
        # Flatten lists
        preds_flat = [p for batch in predictions for p in batch]
        labels_flat = [l for batch in labels for l in batch]
        
        preds_arr = np.array(preds_flat)
        labels_arr = np.array(labels_flat)
        
        # Compute confusion matrix
        tp = ((preds_arr == 1) & (labels_arr == 1)).sum()
        fp = ((preds_arr == 1) & (labels_arr == 0)).sum()
        fn = ((preds_arr == 0) & (labels_arr == 1)).sum()
        tn = ((preds_arr == 0) & (labels_arr == 0)).sum()
        
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    except ImportError:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    return metrics


def compute_all_metrics(
    predictions: dict,
    references: dict,
    task_type: str = "qa",
) -> dict:
    """
    Compute all relevant metrics for a task.
    
    Args:
        predictions: Dictionary of predictions
        references: Dictionary of ground-truth references
        task_type: Task type (qa, sorting, entailment, condense)
    
    Returns:
        metrics: Dictionary of all computed metrics
    
    TODO:
    - Route to appropriate metric functions based on task_type
    - Aggregate metrics across examples
    - Format output consistently
    """
    metrics = {}
    
    if task_type == "qa":
        # TODO: Add QA-specific metrics (EM, F1, etc.)
        pass
    elif task_type == "entailment":
        # TODO: Add entailment metrics
        pass
    elif task_type == "sorting":
        # TODO: Add ranking metrics
        pass
    elif task_type == "condense":
        # TODO: Add generation metrics (ROUGE, BLEU)
        pass
    
    return metrics
