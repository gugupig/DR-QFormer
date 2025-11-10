"""Loss functions for DR-QFormer tasks."""

from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
except ImportError:
    Tensor = None


def compute_focal_loss(
    logits: Tensor,
    gt_labels: Tensor,
    importance_weights: Optional[Tensor] = None,
    pool_padding_mask: Optional[Tensor] = None,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> Tensor:
    """
    Compute Focal Loss with importance weighting for entailment tagging (Task E).
    
    Purpose: Train EntailmentHead to predict which fragments entail query/answer.
             Focal loss reduces weight of easy examples, focusing on hard negatives.
    
    Args:
        logits: [batch, k] predicted logits from EntailmentHead (raw, no sigmoid)
        gt_labels: [batch, k] ground truth binary labels {0, 1}
                   - 0: Fragment does not entail / irrelevant
                   - 1: Fragment entails / is golden evidence
        importance_weights: [batch, k] fragment-level weights (default: all 1.0)
                           Typical values:
                           - Negative class: 1.0
                           - Positive class: 10.0
                           - Longtail positive: 50.0
        pool_padding_mask: [batch, k] valid fragment mask (True=valid, False=padding)
        focal_gamma: Focal loss gamma parameter (default: 2.0)
                     Higher gamma = more focus on hard examples
        focal_alpha: Focal loss alpha parameter (default: 0.25)
                     Balances positive/negative classes (0.25 means 25% weight on positives)
    
    Returns:
        loss: Scalar focal loss
    
    Formula:
        p_t = sigmoid(logits)
        FL_pos = -alpha * (1-p_t)^gamma * log(p_t)           [for positive labels]
        FL_neg = -(1-alpha) * p_t^gamma * log(1-p_t)         [for negative labels]
        FL = gt_labels * FL_pos + (1-gt_labels) * FL_neg
        Loss = sum(importance_weights * FL * mask) / sum(mask)
    
    Key Features:
    =============
    1. Focal Loss: Reduces weight of easy examples (well-classified)
       - Easy negative (p ≈ 0): weight ≈ 0
       - Hard negative (p ≈ 1): weight ≈ 1
       - Easy positive (p ≈ 1): weight ≈ 0
       - Hard positive (p ≈ 0): weight ≈ 1
    
    2. Importance Weighting: Fragment-level sample importance
       - Negative class: 1.0 (baseline)
       - Positive class: 10.0 (emphasize positive examples)
       - Longtail positive: 50.0 (heavily emphasize rare examples)
    
    3. Padding Mask: Ignore padded fragments in batch
       - Valid fragments: mask=True, contribute to loss
       - Padded fragments: mask=False, excluded from loss
    
    Training Modes:
    ===============
    - Primal (QA): Labels mark fragments that contain answer evidence
    - Dual (QG): Labels mark fragments that contain query evidence
    
    Example:
    ========
    >>> logits = torch.randn(2, 5)  # 2 samples, 5 fragments
    >>> gt_labels = torch.tensor([[1, 0, 0, 1, 0], [0, 1, 0, 0, 0]])
    >>> importance_weights = torch.tensor([[10, 1, 1, 10, 1], [1, 10, 1, 1, 1]])
    >>> mask = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    >>> loss = compute_focal_loss(logits, gt_labels, importance_weights, mask)
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)
    
    # Default importance weights (all 1.0)
    if importance_weights is None:
        importance_weights = torch.ones_like(logits)
    
    # Default padding mask (all valid)
    if pool_padding_mask is None:
        pool_padding_mask = torch.ones_like(logits, dtype=torch.bool)
    
    # Focal loss components
    # For positive samples: -alpha * (1-p)^gamma * log(p)
    pos_loss = -focal_alpha * torch.pow(1.0 - probs, focal_gamma) * torch.log(probs + 1e-8)
    
    # For negative samples: -(1-alpha) * p^gamma * log(1-p)
    neg_loss = -(1.0 - focal_alpha) * torch.pow(probs, focal_gamma) * torch.log(1.0 - probs + 1e-8)
    
    # Combine based on gt_labels
    focal_loss = gt_labels * pos_loss + (1.0 - gt_labels) * neg_loss
    
    # Apply importance weights and padding mask
    weighted_loss = focal_loss * importance_weights * pool_padding_mask.float()
    
    # Average over valid fragments
    loss = weighted_loss.sum() / (pool_padding_mask.sum() + 1e-8)
    
    return loss


def entailment_loss(
    logits: Optional[object],
    labels: Optional[object],
    pos_weight: Optional[object] = None,
) -> Optional[object]:
    """
    Binary cross-entropy loss for entailment tagging (Task E).
    
    DEPRECATED: Use compute_focal_loss() instead for better handling of class imbalance.
    
    Purpose: Train EntailmentHead to predict which fragments entail query/answer.
    
    Args:
        logits: Entailment logits [batch, k] from EntailmentHead (raw, no sigmoid)
        labels: Ground-truth binary labels [batch, k] (gt_k)
        pos_weight: Optional positive class weight [k] for handling class imbalance
    
    Returns:
        loss: Scalar BCE loss
    """
    # Simple BCE implementation (use compute_focal_loss for production)
    if logits is None or labels is None:
        return None
    
    loss = F.binary_cross_entropy_with_logits(
        logits, 
        labels.float(),
        pos_weight=pos_weight,
        reduction='mean'
    )
    return loss


def get_curriculum_weights(
    current_step: int,
    total_steps: int,
    lambda_teach_start: float = 1.0,
    lambda_teach_end: float = 0.2,
    lambda_post_start: float = 0.0,
    lambda_post_end: float = 0.8,
) -> dict:
    """
    Compute curriculum learning weights for Task S.
    
    Implements dynamic transition from teacher supervision to posterior alignment:
    - Early training: High λ_teach, Low λ_post (learn from external reranker)
    - Late training: Low λ_teach, High λ_post (align with LLM's actual needs)
    
    Args:
        current_step: Current training step
        total_steps: Total number of training steps
        lambda_teach_start: Initial teacher weight
        lambda_teach_end: Final teacher weight
        lambda_post_start: Initial posterior weight
        lambda_post_end: Final posterior weight
    
    Returns:
        dict with:
            - lambda_teach: Current teacher weight
            - lambda_post: Current posterior weight
            - progress: Training progress ratio [0, 1]
    
    Formula:
        progress = current_step / total_steps
        lambda_teach(t) = lambda_teach_start + (lambda_teach_end - lambda_teach_start) * progress
        lambda_post(t) = lambda_post_start + (lambda_post_end - lambda_post_start) * progress
    
    Example:
        >>> weights = get_curriculum_weights(5000, 10000)
        >>> # At 50% training: lambda_teach ≈ 0.6, lambda_post ≈ 0.4
    """
    # Compute progress ratio
    progress = min(current_step / max(total_steps, 1), 1.0)
    
    # Linear interpolation
    lambda_teach = lambda_teach_start + (lambda_teach_end - lambda_teach_start) * progress
    lambda_post = lambda_post_start + (lambda_post_end - lambda_post_start) * progress
    
    return {
        "lambda_teach": lambda_teach,
        "lambda_post": lambda_post,
        "progress": progress,
    }


def build_train_subset_mask(
    ranking_logits: Tensor,
    gt_scores: Tensor,
    pool_padding_mask: Optional[Tensor] = None,
    rho_top: float = 0.02,
    l_prime: int = 16,
) -> Tensor:
    """
    Build dynamic training subset mask for Task S.
    
    Constructs: U = I_teacher ∪ I_student
    - I_teacher: Teacher Top-L (top ρ% by gt_scores)
    - I_student: Student Hard Negatives (top l' by ranking_logits, excluding I_teacher)
    
    Purpose:
    - Focus training on most informative fragments
    - I_teacher: Ground truth top fragments (what should be ranked high)
    - I_student: Model's confident mistakes (fragments model ranks high but teacher doesn't)
    
    Args:
        ranking_logits: [batch, K] student predicted scores
        gt_scores: [batch, K] teacher ground truth scores
        pool_padding_mask: [batch, K] valid fragment mask
        rho_top: Teacher Top-L ratio (e.g., 0.02 = top 2%)
        l_prime: Student Hard Negatives count
    
    Returns:
        train_subset_mask: [batch, K] bool mask (True = include in training)
    
    Example:
        >>> ranking_logits = torch.randn(2, 100)
        >>> gt_scores = torch.randn(2, 100)
        >>> mask = build_train_subset_mask(ranking_logits, gt_scores, rho_top=0.05, l_prime=10)
        >>> # mask will have ~10 True values per sample (5 from teacher + 5-10 from student)
    """
    batch_size, K = ranking_logits.shape
    device = ranking_logits.device
    
    # Default mask
    if pool_padding_mask is None:
        pool_padding_mask = torch.ones_like(ranking_logits, dtype=torch.bool)
    
    # Count effective K per sample
    K_eff = pool_padding_mask.sum(dim=-1)  # [batch]
    
    # ========== I_teacher: Teacher Top-L ==========
    # L_curr = ceil(ρ * K_eff)
    L_curr = torch.ceil(rho_top * K_eff.float()).long()  # [batch]
    L_curr = torch.clamp(L_curr, min=1, max=K)  # At least 1, at most K
    
    # Get top-L indices by gt_scores
    gt_scores_masked = gt_scores.masked_fill(~pool_padding_mask, -1e9)
    teacher_indices = torch.argsort(gt_scores_masked, dim=-1, descending=True)  # [batch, K]
    
    # Create teacher mask
    teacher_mask = torch.zeros_like(ranking_logits, dtype=torch.bool)
    for b in range(batch_size):
        L_b = L_curr[b].item()
        teacher_mask[b, teacher_indices[b, :L_b]] = True
    
    # ========== I_student: Student Hard Negatives ==========
    # Exclude teacher region: set teacher fragments to very low score
    student_scores = ranking_logits.clone()
    student_scores[teacher_mask] = -1e9
    student_scores = student_scores.masked_fill(~pool_padding_mask, -1e9)
    
    # Get top-l' indices by student scores (in non-teacher region)
    student_indices = torch.argsort(student_scores, dim=-1, descending=True)  # [batch, K]
    
    # Create student mask
    student_mask = torch.zeros_like(ranking_logits, dtype=torch.bool)
    for b in range(batch_size):
        # Take min(l', remaining valid fragments)
        remaining_valid = (pool_padding_mask[b] & ~teacher_mask[b]).sum().item()
        l_prime_b = min(l_prime, remaining_valid)
        if l_prime_b > 0:
            student_mask[b, student_indices[b, :l_prime_b]] = True
    
    # ========== Union: U = I_teacher ∪ I_student ==========
    train_subset_mask = teacher_mask | student_mask
    
    return train_subset_mask


def compute_ranking_loss(
    ranking_logits: Tensor,
    gt_scores: Tensor,
    posterior_scores: Optional[Tensor] = None,
    pool_padding_mask: Optional[Tensor] = None,
    train_subset_mask: Optional[Tensor] = None,
    lambda_teach: float = 1.0,
    lambda_post: float = 0.0,
    lambda_entropy: float = 0.01,
    tau_pred: float = 1.0,
    tau_gt: float = 1.0,
    alpha_gt: float = 0.7,
) -> dict:
    """
    Compute ranking loss for Task S with dynamic curriculum learning.
    
    Purpose: Train Q-Former to rank fragments from large evidence pools (K=100~5000).
             Combines teacher supervision (external reranker) with posterior alignment (LLM feedback).
    
    Args:
        ranking_logits: [batch, K] predicted ranking scores from FragmentRankingHead
        gt_scores: [batch, K] teacher reranker scores (e.g., BM25, DPR scores)
        posterior_scores: [batch, K] optional LLM posterior scores from Task C (detached)
        pool_padding_mask: [batch, K] valid fragment mask (True=valid, False=padding)
        train_subset_mask: [batch, K] training subset mask (Teacher Top-L ∪ Student Hard Negatives)
        lambda_teach: Weight for teacher supervision loss (decreases over training)
        lambda_post: Weight for posterior alignment loss (increases over training)
        lambda_entropy: Weight for tail entropy regularization
        tau_pred: Temperature for student prediction distribution
        tau_gt: Temperature for teacher target distribution
        alpha_gt: Teacher target distribution Top-L expected cumulative mass (default: 0.7)
    
    Returns:
        dict with:
            - loss: Total weighted loss
            - loss_teach: Teacher supervision loss (ListNet)
            - loss_post: Posterior alignment loss (JS divergence)
            - loss_entropy: Tail entropy regularization
    
    Training Components:
    ===================
    1. Teacher Supervision (L_teach) - ListNet Loss:
       - Convert gt_scores to target distribution via softmax + normalization
       - Apply Cross-Entropy between target and predicted distributions
       - Ensures model learns from external reranker's ranking
    
    2. Posterior Alignment (L_post) - JS Divergence:
       - Align student predictions with LLM's actual fragment usage (from Task C)
       - Uses Jensen-Shannon divergence for symmetric comparison
       - Posterior scores are detached (treated as teacher signal)
    
    3. Tail Entropy Regularization:
       - Encourages high entropy on low-scoring fragments
       - Prevents overconfident predictions on irrelevant fragments
    
    Dynamic Curriculum:
    ==================
    - Early training: High λ_teach, Low λ_post → Learn from external reranker
    - Late training: Low λ_teach, High λ_post → Align with LLM's needs
    - Transition: Linear schedule over training epochs
    
    Formula:
        L_S = λ_teach(t) * L_teach + λ_post(t) * L_post + λ_entropy * L_tail_entropy
    
    Example:
        >>> ranking_logits = torch.randn(2, 100)  # 2 samples, 100 fragments
        >>> gt_scores = torch.randn(2, 100)  # Teacher scores
        >>> posterior_scores = torch.randn(2, 100)  # LLM feedback
        >>> mask = torch.ones(2, 100, dtype=torch.bool)
        >>> subset_mask = torch.zeros(2, 100, dtype=torch.bool)
        >>> subset_mask[:, :20] = True  # Use top 20 for training
        >>> loss_dict = compute_ranking_loss(
        ...     ranking_logits, gt_scores, posterior_scores, mask, subset_mask,
        ...     lambda_teach=0.7, lambda_post=0.3
        ... )
    """
    batch_size, K = ranking_logits.shape
    
    # Default masks
    if pool_padding_mask is None:
        pool_padding_mask = torch.ones_like(ranking_logits, dtype=torch.bool)
    
    if train_subset_mask is None:
        # Use all valid fragments if no subset specified
        train_subset_mask = pool_padding_mask
    
    # Combine masks: only train on valid fragments within subset
    effective_mask = pool_padding_mask & train_subset_mask
    
    # Apply mask to scores (set invalid to large negative for softmax stability)
    ranking_logits_masked = ranking_logits.masked_fill(~effective_mask, -1e4)
    gt_scores_masked = gt_scores.masked_fill(~effective_mask, -1e4)
    
    # ========== Component 1: Teacher Supervision (ListNet Loss) ==========
    # Student prior distribution
    pi_U = F.softmax(ranking_logits_masked / tau_pred, dim=-1)  # [batch, K]
    
    # Teacher target distribution with alpha_gt calibration
    # For each sample, calibrate temperature T_gt* to ensure Top-L cumulative mass ≈ alpha_gt
    P_gt_U_list = []
    for b in range(batch_size):
        sample_mask = effective_mask[b]
        K_eff = sample_mask.sum().item()
        
        if K_eff == 0:
            # No valid fragments, use uniform distribution
            P_gt_sample = torch.zeros(K, device=ranking_logits.device)
            P_gt_U_list.append(P_gt_sample)
            continue
        
        # Get valid gt_scores for this sample
        gt_scores_sample = gt_scores[b]
        
        # Compute Top-L size based on rho_top from train_subset_mask
        # Use build_train_subset_mask's logic: L = ceil(rho_top * K_eff)
        # Since we don't have rho_top here, estimate from train_subset_mask
        train_subset_sample = train_subset_mask[b] & sample_mask
        L_curr = train_subset_sample.sum().item()
        
        if L_curr == 0 or L_curr >= K_eff:
            # Fallback: no temperature calibration needed
            gt_z_scores_sample = gt_scores_sample.masked_fill(~sample_mask, -1e4)
            P_gt_sample = F.softmax(gt_z_scores_sample / tau_gt, dim=-1)
            P_gt_sample = P_gt_sample * sample_mask.float()
            P_gt_sample = P_gt_sample / (P_gt_sample.sum() + 1e-8)
            P_gt_U_list.append(P_gt_sample)
            continue
        
        # Binary search for optimal temperature T_gt*
        # Goal: Top-L cumulative mass ≈ alpha_gt
        T_min, T_max = 1e-3, 1e3
        max_iters = 20
        tolerance = 0.05  # 5% tolerance for alpha_gt
        
        # Get Top-L indices by gt_scores
        gt_scores_valid = gt_scores_sample.clone()
        gt_scores_valid[~sample_mask] = -float('inf')
        topL_indices = torch.argsort(gt_scores_valid, descending=True)[:L_curr]
        
        T_optimal = tau_gt  # Start with default
        for _ in range(max_iters):
            T_mid = (T_min + T_max) / 2.0
            
            # Compute distribution with T_mid
            gt_z_scores_sample = gt_scores_sample.masked_fill(~sample_mask, -1e4)
            P_test = F.softmax(gt_z_scores_sample / T_mid, dim=-1)
            P_test = P_test * sample_mask.float()
            P_test = P_test / (P_test.sum() + 1e-8)
            
            # Compute Top-L cumulative mass
            topL_mass = P_test[topL_indices].sum().item()
            
            if abs(topL_mass - alpha_gt) < tolerance:
                T_optimal = T_mid
                break
            elif topL_mass > alpha_gt:
                # Too much mass on top, increase temperature (flatten distribution)
                T_min = T_mid
            else:
                # Too little mass on top, decrease temperature (sharpen distribution)
                T_max = T_mid
        
        # Use optimal temperature
        gt_z_scores_sample = gt_scores_sample.masked_fill(~sample_mask, -1e4)
        P_gt_sample = F.softmax(gt_z_scores_sample / T_optimal, dim=-1)
        P_gt_sample = P_gt_sample * sample_mask.float()
        P_gt_sample = P_gt_sample / (P_gt_sample.sum() + 1e-8)
        P_gt_U_list.append(P_gt_sample)
    
    P_gt_U = torch.stack(P_gt_U_list, dim=0)  # [batch, K]
    
    # ListNet loss: KL divergence between student and teacher distributions
    # KL(P_gt || pi) = sum(P_gt * log(P_gt / pi))
    loss_teach = -(P_gt_U * torch.log(pi_U + 1e-8)).sum(dim=-1).mean()
    
    # ========== Component 2: Posterior Alignment (JS Divergence) ==========
    loss_post = torch.tensor(0.0, device=ranking_logits.device)
    
    if posterior_scores is not None and lambda_post > 0:
        # Posterior distribution (detached - treat as teacher signal)
        posterior_scores_masked = posterior_scores.masked_fill(~effective_mask, -1e4)
        q_psi_U = F.softmax(posterior_scores_masked.detach(), dim=-1)  # [batch, K]
        q_psi_U = q_psi_U * effective_mask.float()
        q_psi_U = q_psi_U / (q_psi_U.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Jensen-Shannon Divergence: JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)
        # where M = 0.5*(P + Q)
        M = 0.5 * (pi_U + q_psi_U)
        
        # KL(pi_U || M)
        kl_pi_M = F.kl_div(
            torch.log(M + 1e-8), 
            pi_U, 
            reduction='batchmean',
            log_target=False
        )
        
        # KL(q_psi_U || M)
        kl_q_M = F.kl_div(
            torch.log(M + 1e-8), 
            q_psi_U, 
            reduction='batchmean',
            log_target=False
        )
        
        loss_post = 0.5 * (kl_pi_M + kl_q_M)
    
    # ========== Component 3: Tail Entropy Regularization ==========
    # Encourage high entropy on low-scoring (tail) fragments
    # This prevents overconfident predictions on irrelevant fragments
    
    # Find tail fragments (bottom 50% by ranking_logits)
    K_eff = effective_mask.sum(dim=-1, keepdim=True).float()  # [batch, 1]
    threshold = ranking_logits.median(dim=-1, keepdim=True)[0]  # [batch, 1]
    tail_mask = (ranking_logits < threshold) & effective_mask  # [batch, K]
    
    # Entropy of tail distribution
    pi_tail = pi_U * tail_mask.float()
    pi_tail = pi_tail / (pi_tail.sum(dim=-1, keepdim=True) + 1e-8)
    
    # H(p) = -Σ p*log(p)
    entropy_tail = -(pi_tail * torch.log(pi_tail + 1e-8)).sum(dim=-1).mean()
    
    # Maximize entropy → minimize negative entropy
    loss_entropy = -entropy_tail
    
    # ========== Total Loss ==========
    loss_total = (
        lambda_teach * loss_teach +
        lambda_post * loss_post +
        lambda_entropy * loss_entropy
    )
    
    return {
        "loss": loss_total,
        "loss_teach": loss_teach.detach(),
        "loss_post": loss_post.detach() if isinstance(loss_post, Tensor) else loss_post,
        "loss_entropy": loss_entropy.detach(),
    }


def sorting_loss(
    predicted_weights: Optional[object],
    target_weights: Optional[object],
) -> Optional[object]:
    """
    DEPRECATED: Use compute_ranking_loss() instead for Task S.
    
    KL Divergence loss for fragment sorting (Task S).
    
    Purpose: Train CA layer attention weights to match target distribution.
    
    Args:
        predicted_weights: Predicted attention weights [batch, k] from SortingHead
                          Should be probability distribution (sum to 1.0)
        target_weights: Ground-truth soft weights [batch, k] (gt_soft_weights)
                       Probability distribution reflecting fragment importance
                       Example: [0.4, 0.3, 0.2, 0.1] (offline retriever/reranker scores)
    
    Returns:
        loss: KL Divergence scalar - measures distribution mismatch
    """
    # Simple KL implementation (use compute_ranking_loss for production)
    if predicted_weights is None or target_weights is None:
        return None
    
    loss = F.kl_div(
        F.log_softmax(predicted_weights, dim=-1),
        target_weights,
        reduction='batchmean'
    )
    return loss


def reward_margin_loss(
    reward_high: Optional[object],
    reward_low: Optional[object],
    margin: float = 0.5,
) -> Optional[object]:
    """
    Margin-based reward loss for condensing-generation (Task C).
    
    Purpose: Train Q-Former to extract information that improves LLM generation quality.
    
    Contrastive Approach:
    - Generate WITH evidence: answer_with_Z = LLM(Query, Z)
    - Generate WITHOUT evidence: answer_baseline = LLM(Query, Empty_Z or No_Z)
    - Maximize difference: Reward(answer_with_Z) - Reward(answer_baseline)
    
    Args:
        reward_high: Reward for condensed generation [batch]
                    - Primal (QA): ROUGE/BLEU/EM between LLM(Query, Z) and gold_answer
                    - Dual (QG): ROUGE/BLEU between LLM(Answer, Z) and gold_query
        reward_low: Reward for baseline generation [batch]
                   - Primal: ROUGE/BLEU/EM between LLM(Query, Empty_Z) and gold_answer
                   - Dual: ROUGE/BLEU between LLM(Answer, Empty_Z) and gold_query
        margin: Desired minimum reward gap (default: 0.5)
                Encourages evidence-dependent generation
    
    Returns:
        loss: Margin loss encouraging reward_high > reward_low + margin
    
    Formula:
        loss = mean(max(0, margin - (reward_high - reward_low)))
        
        Interpretation:
        - If (reward_high - reward_low) >= margin: loss = 0 (satisfied)
        - If (reward_high - reward_low) < margin: loss > 0 (penalty)
    
    Training Modes:
    ===============
    Primal (QA) - Contrastive Generation:
      1. Forward: Query + Z → LLM → answer_with_Z
      2. Baseline: Query + Empty → LLM → answer_baseline
      3. Compute rewards: ROUGE(answer_*, gold_answer)
      4. Maximize: reward(answer_with_Z) - reward(answer_baseline)
      5. Forces Q-Former to extract answer-relevant evidence
    
    Dual (QG) - Reward Maximization:
      1. Forward: Answer + Z → LLM → query'
      2. Baseline: Answer + Empty → LLM → query_baseline
      3. Compute rewards: BLEU(query*, gold_query)
      4. Maximize: reward(query') - reward(query_baseline)
      5. Forces Q-Former to extract query-relevant evidence
    
    Reward Types:
    =============
    - ROUGE-L: Longest common subsequence (good for answer quality)
    - BLEU: N-gram overlap (good for query/text generation)
    - EM (Exact Match): Binary 0/1 (strict evaluation)
    - F1: Token-level overlap (balanced metric)
    - BERTScore: Semantic similarity (contextual embeddings)
    
    Implementation Notes:
    =====================
    1. Rewards should be normalized to [0, 1] or similar range
    2. Margin should be tuned based on reward scale
    3. Can use different margins for Primal vs Dual
    4. Consider adding penalty for Empty_Z having high reward (prevent shortcuts)
    
    TODO:
    - Implement margin loss with ReLU
    - Add reward normalization if needed
    - Support batched computation
    - Consider adding penalty term: -lambda * reward_low (discourage good baseline)
    - Log reward statistics for monitoring (reward_high mean, reward_low mean, gap)
    """
    # TODO: Implement loss
    # loss = F.relu(margin - (reward_high - reward_low)).mean()
    # 
    # Optional: Add baseline penalty
    # loss = loss - lambda_penalty * reward_low.mean()
    pass
    return None


def kl_divergence_loss(
    log_probs: Optional[object],
    target_probs: Optional[object],
) -> Optional[object]:
    """
    KL divergence loss for distribution matching.
    
    Args:
        log_probs: Log probabilities [batch, k]
        target_probs: Target distribution [batch, k]
    
    Returns:
        loss: KL divergence
    
    TODO:
    - Implement KL divergence
    - Used for regularization or knowledge distillation
    """
    # TODO: Implement loss
    # loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
    pass
    return None


def combined_loss(
    losses: dict,
    weights: Optional[dict] = None,
) -> Optional[object]:
    """
    Combine multiple losses with weights.
    
    Args:
        losses: Dictionary of {loss_name: loss_value}
        weights: Dictionary of {loss_name: weight} (default: equal)
    
    Returns:
        total_loss: Weighted sum of losses
    
    TODO:
    - Implement weighted sum
    - Handle missing losses gracefully
    - Support dynamic weight adjustment
    """
    if weights is None:
        weights = {name: 1.0 for name in losses}
    
    # TODO: Implement combination
    # total_loss = sum(weights.get(name, 1.0) * loss for name, loss in losses.items())
    pass
    
    return None


def compute_lq_entropy_loss(
    ca_raw_scores_per_head: list,
    pool_padding_mask: Tensor,
    target_ratio: float = 0.7,
) -> Tensor:
    """
    LQ-Level Entropy Regularization to prevent over-concentration.
    
    Purpose: Encourage each Learnable Query (LQ) to attend to multiple fragments,
             preventing "winner-takes-all" collapse where one LQ dominates attention.
             This improves representation diversity and facilitates LQ compression (32→16→8).
    
    Motivation:
    ===========
    Without regularization, multiple LQs may learn redundant attention patterns:
        LQ_0: [0.9, 0.05, 0.05]  # Focuses on fragment 0
        LQ_1: [0.85, 0.10, 0.05] # Also focuses on fragment 0
        LQ_2: [0.88, 0.07, 0.05] # Still focuses on fragment 0
    
    This wastes model capacity and makes LQ compression difficult (can't identify
    which LQs are truly important).
    
    With entropy regularization, LQs are encouraged to distribute attention:
        LQ_0: [0.4, 0.3, 0.2, 0.1]   # Moderate distribution
        LQ_1: [0.2, 0.4, 0.3, 0.1]   # Different pattern
        LQ_2: [0.3, 0.2, 0.15, 0.35] # Yet another pattern
    
    Args:
        ca_raw_scores_per_head: List of [B, H, N_lq, K] cross-attention scores per layer
                               Raw QK^T scores before softmax
        pool_padding_mask: [B, K] bool mask (True=valid fragment, False=padding)
        target_ratio: Target entropy ratio relative to uniform (default: 0.7)
                     - 1.0 = uniform distribution (maximum entropy)
                     - 0.7 = conservative (allow 30% concentration)
                     - 0.5 = moderate (allow 50% concentration)
    
    Returns:
        entropy_loss: Scalar loss (minimize to encourage higher entropy)
    
    Formula:
    ========
    1. Average scores across layers and heads:
       ca_scores_avg[b, n, k] = mean_l,h(ca_raw_scores_per_head[l][b, h, n, k])
    
    2. Apply mask and softmax per LQ:
       ca_probs[b, n, k] = softmax_k(ca_scores_avg[b, n, k]) for valid k
    
    3. Compute entropy per LQ:
       H[b, n] = -Σ_k ca_probs[b, n, k] · log(ca_probs[b, n, k])
    
    4. Compute target entropy (uniform over valid K):
       H_target[b] = log(K_eff[b])  where K_eff = number of valid fragments
       H_target_scaled[b] = target_ratio · H_target[b]
    
    5. MSE loss to encourage entropy close to target:
       loss = mean_b,n((H[b, n] - H_target_scaled[b])^2)
    
    Design Choices:
    ===============
    1. Why target_ratio < 1.0?
       - Tasks may inherently need some concentration (e.g., Task E for single-fragment answers)
       - 0.7 allows 30% concentration while preventing collapse
       - Conservative approach: prevent over-regularization
    
    2. Why MSE instead of directly minimizing -H?
       - MSE encourages entropy close to target (not arbitrarily high)
       - Prevents forcing uniform when task requires concentration
       - More stable gradients
    
    3. Why average across layers and heads?
       - LQs should learn consistent patterns across layers
       - Reduces noise from individual layer/head variations
       - Aligns with how FragmentRankingHead aggregates attention
    
    Usage in Training:
    ==================
    Typically used with curriculum learning (high → low weight):
    
    ```python
    # Curriculum weight decay
    lambda_entropy = lambda_start * (1 - 0.9 * step / total_steps)
    
    # Main task loss
    loss_main = compute_focal_loss(...)  # or compute_ranking_loss(...)
    
    # Optional entropy regularization
    if args.enable_lq_entropy_reg:
        loss_entropy = compute_lq_entropy_loss(
            ca_raw_scores_per_head=aux['ca_raw_scores_per_head'],
            pool_padding_mask=pool_padding_mask,
            target_ratio=0.7,  # Conservative
        )
        loss_total = loss_main + lambda_entropy * loss_entropy
    else:
        loss_total = loss_main
    ```
    
    Task-Specific Recommendations:
    ==============================
    - Task E (Entailment): Optional, default OFF (may need concentration)
      - If enabled: target_ratio=0.5, lambda=0.005→0.0005
    
    - Task S (Ranking): Recommended, default ON (diversity important for prior)
      - target_ratio=0.7, lambda=0.01→0.001
    
    - Task C (Condensing): Optional, moderate (diversity vs. posterior alignment)
      - If enabled: target_ratio=0.7, lambda=0.008→0.0001 (fast decay)
    
    Relation to LQ Compression:
    ===========================
    When planning to compress LQs (32→16→8):
    
    1. Training with entropy regularization:
       - Ensures all 32 LQs contribute (no redundancy)
       - Each LQ learns distinct attention pattern
    
    2. Evaluating LQ importance:
       - Compute gradient norms per LQ
       - Or compute attention entropy per LQ
       - Or ablation study (remove LQ, measure performance)
    
    3. Selecting Top-K LQs:
       - Choose K LQs with highest importance scores
       - Entropy reg ensures diverse selection (not all similar)
    
    4. Knowledge distillation:
       - 16-LQ model learns from 32-LQ teacher
       - Distill attention patterns and outputs
    
    5. Fine-tuning compressed model:
       - Disable entropy reg (allow task-specific concentration)
       - Fine-tune with smaller learning rate
    
    Example:
    ========
    >>> # Mock data: 2 samples, 3 layers, 8 heads, 32 LQs, 100 fragments
    >>> ca_scores = [torch.randn(2, 8, 32, 100) for _ in range(3)]
    >>> mask = torch.ones(2, 100, dtype=torch.bool)
    >>> mask[:, 80:] = False  # Last 20 fragments padded
    >>> 
    >>> loss = compute_lq_entropy_loss(ca_scores, mask, target_ratio=0.7)
    >>> # loss will be low if LQs have moderate entropy (not too concentrated)
    """
    # Stack and average across layers and heads
    # ca_raw_scores_per_head: List[Tensor[B, H, N_lq, K]]
    ca_scores = torch.stack(ca_raw_scores_per_head, dim=0)  # [num_layers, B, H, N_lq, K]
    ca_scores_avg = ca_scores.mean(dim=[0, 2])  # Average over layers and heads → [B, N_lq, K]
    
    B, N_lq, K = ca_scores_avg.shape
    
    # Apply padding mask: set invalid fragments to large negative
    # pool_padding_mask: [B, K] → [B, 1, K]
    mask_expanded = pool_padding_mask.unsqueeze(1)  # [B, 1, K]
    ca_scores_masked = ca_scores_avg.masked_fill(~mask_expanded, -1e10)
    
    # Softmax per LQ to get attention probabilities
    ca_probs = F.softmax(ca_scores_masked, dim=-1)  # [B, N_lq, K]
    
    # Compute entropy per LQ: H = -Σ p·log(p)
    entropy_per_lq = -(ca_probs * torch.log(ca_probs + 1e-10)).sum(dim=-1)  # [B, N_lq]
    
    # Compute target entropy: log(K_eff) for uniform distribution
    K_eff = pool_padding_mask.sum(dim=-1, keepdim=True).float()  # [B, 1]
    target_entropy = torch.log(K_eff + 1e-10)  # [B, 1]
    
    # Scale by target_ratio (allow some concentration)
    target_entropy_scaled = target_ratio * target_entropy  # [B, 1]
    
    # Expand to match entropy_per_lq shape
    target_entropy_expanded = target_entropy_scaled.expand(B, N_lq)  # [B, N_lq]
    
    # MSE loss: encourage entropy close to target
    entropy_loss = F.mse_loss(entropy_per_lq, target_entropy_expanded)
    
    return entropy_loss


def compute_condensing_loss(
    nll_with_evidence: Tensor,
    nll_without_evidence: Tensor,
    llm_attention_weights: Optional[Tensor] = None,
    ca_weights: Optional[Tensor] = None,
    subset_indices: Optional[Tensor] = None,
    answer_start_idx: Optional[int] = None,
    softplus_beta: float = 10.0,
    margin_mode: str = 'adaptive',
    margin_fixed: float = 0.5,
    margin_adaptive_ratio: float = 0.5,
    margin_min: float = 0.1,
    margin_max: float = 2.0,
) -> dict:
    """
    Compute Condensing-Generation Loss (Task C) with Contrastive NLL and Posterior Extraction.
    
    Purpose: Train Q-Former to generate knowledge prefix Z that reduces frozen LLM perplexity.
             Uses contrastive NLL loss (with vs. without evidence) and extracts real-time
             posterior importance from LLM attention.
    
    Design Philosophy (v8.0):
    =========================
    - Pure Teacher Forcing: No generative sampling, only measure NLL reduction
    - Dual-Path Forward: Compare NLL with/without evidence prefix Z
    - Posterior Backtracing: Extract fragment importance from LLM→Z attention
    - Subset-Only Posterior: Only compute posterior on training subset U (efficiency)
    
    Args:
        nll_with_evidence: Scalar NLL with Z prefix (Path A)
        nll_without_evidence: Scalar NLL without Z prefix (Path B, detached)
        llm_attention_weights: [batch, n_heads, seq_total, N_lq] LLM→Z attention (Path A only)
                              Captured from answer tokens to Z prefix positions
        ca_weights: [batch, N_lq, K_pool] Q-Former cross-attention weights
        subset_indices: [batch, |U|] dynamic training subset indices
        answer_start_idx: Token position where answer starts (after Z + Query)
        softplus_beta: Softplus sharpness (default: 10.0, higher = closer to ReLU)
        margin_mode: 'fixed' or 'adaptive'
        margin_fixed: Fixed margin value (used if mode='fixed')
        margin_adaptive_ratio: Adaptive margin ratio κ (default: 0.5)
        margin_min: Minimum adaptive margin (default: 0.1)
        margin_max: Maximum adaptive margin (default: 2.0)
    
    Returns:
        dict with keys:
            - 'loss_c': Scalar condensing loss (for backprop)
            - 'nll_gain': NLL reduction G = nll_no - nll_with (detached, for logging)
            - 'margin': Computed margin m (detached, for logging)
            - 'posterior_q_psi_U': [batch, |U|] fragment posterior importance (detached)
                                  Only computed on subset U (None if not available)
    
    Loss Formula:
    =============
    1. Compute NLL Gain:
       G = nll_without_evidence - nll_with_evidence
       (Positive G means evidence helps reduce perplexity)
    
    2. Compute Adaptive Margin:
       If margin_mode == 'adaptive':
           m = clip(μ_G + κ·σ_G, margin_min, margin_max)
           where μ_G = mean(G), σ_G = std(G) over batch
       Else:
           m = margin_fixed
    
    3. Compute Softplus Loss:
       L_C = Softplus(β · (m - G))
           = (1/β) · log(1 + exp(β·(m - G)))
       
       Interpretation:
       - If G > m: Evidence gain exceeds margin → loss ≈ 0
       - If G < m: Evidence gain insufficient → loss increases smoothly
       - β controls steepness (β→∞ approaches ReLU)
    
    Posterior Extraction (Subset-Only):
    ===================================
    Only computed if all of (llm_attention_weights, ca_weights, subset_indices) provided.
    
    1. Average LLM→Z attention over answer tokens and heads:
       w_lq[b, n] = mean(llm_attention_weights[b, :, answer_start:, n])
       Shape: [batch, N_lq]
    
    2. Extract CA weights for subset U:
       ca_weights_U[b, n, u] = ca_weights[b, n, subset_indices[b, u]]
       Shape: [batch, N_lq, |U|]
    
    3. Compute posterior via matrix multiplication:
       q_ψ_U[b, u] = softmax_u(Σ_n w_lq[b,n] · ca_weights_U[b,n,u])
       Shape: [batch, |U|]
    
    4. Detach for use as pseudo-label (no gradient to LLM or this path)
    
    Training Workflow:
    ==================
    1. Path A (With Evidence):
       - Input: [dummy_Z(N), Query(S_q), Answer(S_a)]
       - Embeddings: Replace dummy_Z with actual Z from Q-Former
       - Attention Mask: Prefix-LM (Z sees itself, Q sees Z+itself, A sees Z+Q+itself)
       - Labels: [-100(N), -100(S_q), Answer_tokens(S_a)]
       - Forward: LLM(embeds, mask, labels) → nll_with_evidence
       - Hook: Capture attention[answer_tokens → Z positions]
    
    2. Path B (Without Evidence - Baseline):
       - Same input/embeddings as Path A
       - Attention Mask: Block Q and A from seeing Z (set Z columns to 0 or -inf)
       - Labels: Same as Path A
       - Forward: LLM(embeds, blocked_mask, labels) → nll_without_evidence
       - No gradient (torch.no_grad)
       - No hook needed
    
    3. Loss Computation:
       - Compute G = nll_no - nll_with
       - Compute margin m
       - Compute L_C = Softplus(β(m - G))
       - Extract posterior q_ψ_U from Path A attention
    
    Example:
    ========
    >>> nll_with = torch.tensor(2.5)  # Lower perplexity with evidence
    >>> nll_no = torch.tensor(3.8)    # Higher perplexity without evidence
    >>> G = nll_no - nll_with         # G = 1.3 (positive gain)
    >>> m = 0.5                       # Adaptive margin
    >>> loss = softplus(10.0 * (0.5 - 1.3))  # ≈ 0 (gain exceeds margin)
    
    >>> # If gain insufficient:
    >>> nll_with = torch.tensor(3.5)
    >>> nll_no = torch.tensor(3.7)
    >>> G = 0.2  # Small gain
    >>> loss = softplus(10.0 * (0.5 - 0.2))  # > 0 (penalize insufficient gain)
    
    Key Design Decisions:
    =====================
    1. Why Contrastive NLL?
       - Direct measure of evidence utility (perplexity reduction)
       - No need for reward models or sampling
       - Teacher forcing ensures stable gradients
    
    2. Why Softplus instead of Hinge/ReLU?
       - Smooth gradients everywhere (no dead zones)
       - Differentiable through margin boundary
       - β parameter allows tuning steepness
    
    3. Why Adaptive Margin?
       - Automatically adjusts to NLL scale across batches
       - Prevents overfitting to fixed margin
       - κ·σ_G provides dynamic difficulty scaling
    
    4. Why Subset-Only Posterior?
       - Training subset U typically small (|U| << K)
       - Full K posterior expensive and unnecessary
       - Posterior only used as pseudo-label for Task S integration
    
    TODO (LLM Integration):
    =======================
    - [ ] Implement LLM attention hook registration in FrozenLLM adapter
    - [ ] Ensure hook captures attention[answer_start:, :N_lq] correctly
    - [ ] Handle multi-head averaging (mean over heads)
    - [ ] Test with actual LLM (LLaMA, Mistral, Phi)
    - [ ] Verify attention masking blocks Z correctly in Path B
    """
    # Validate inputs
    assert nll_with_evidence.numel() == 1, "nll_with_evidence must be scalar"
    assert nll_without_evidence.numel() == 1, "nll_without_evidence must be scalar"
    
    # Ensure no gradient flows through baseline NLL
    nll_without_evidence = nll_without_evidence.detach()
    
    # 1. Compute NLL Gain
    nll_gain = nll_without_evidence - nll_with_evidence  # [1]
    
    # 2. Compute Margin
    if margin_mode == 'adaptive':
        # Adaptive margin: m = μ_G + κ·σ_G
        # Note: For scalar, we use nll_gain directly as μ_G
        # In batch setting, compute mean and std over batch
        if nll_gain.dim() == 0:
            # Scalar case (single sample or already aggregated)
            mu_G = nll_gain
            sigma_G = torch.tensor(0.0, device=nll_gain.device)
        else:
            # Batch case
            mu_G = nll_gain.mean()
            sigma_G = nll_gain.std()
        
        margin = mu_G + margin_adaptive_ratio * sigma_G
        margin = torch.clamp(margin, min=margin_min, max=margin_max)
    else:
        # Fixed margin
        margin = torch.tensor(margin_fixed, device=nll_gain.device)
    
    # 3. Compute Softplus Loss
    # L_C = (1/β) · log(1 + exp(β·(m - G)))
    loss_c = F.softplus(softplus_beta * (margin - nll_gain)) / softplus_beta
    
    # 4. Extract Posterior (Subset-Only)
    posterior_q_psi_U = None
    if (llm_attention_weights is not None and 
        ca_weights is not None and 
        subset_indices is not None and
        answer_start_idx is not None):
        
        # LLM attention: [batch, n_heads, seq_total, N_lq]
        # Extract attention from answer tokens to Z
        batch_size = llm_attention_weights.shape[0]
        n_lq = llm_attention_weights.shape[-1]
        
        # Average over answer tokens and heads: [batch, N_lq]
        w_lq = llm_attention_weights[:, :, answer_start_idx:, :].mean(dim=(1, 2))
        
        # CA weights: [batch, N_lq, K_pool]
        # Extract subset: [batch, N_lq, |U|]
        # subset_indices: [batch, |U|]
        subset_size = subset_indices.shape[1]
        ca_weights_U = torch.gather(
            ca_weights, 
            dim=2, 
            index=subset_indices.unsqueeze(1).expand(-1, n_lq, -1)
        )
        
        # Compute posterior: q_ψ_U = softmax(w_lq @ ca_weights_U)
        # w_lq: [batch, N_lq] → [batch, N_lq, 1]
        # ca_weights_U: [batch, N_lq, |U|]
        # Result: [batch, |U|]
        logits_U = torch.bmm(w_lq.unsqueeze(1), ca_weights_U).squeeze(1)  # [batch, |U|]
        posterior_q_psi_U = F.softmax(logits_U, dim=-1).detach()
    
    # Return results
    return {
        'loss_c': loss_c,
        'nll_gain': nll_gain.detach(),
        'margin': margin.detach(),
        'posterior_q_psi_U': posterior_q_psi_U,
    }
