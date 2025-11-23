"""
MACS (Multi-head Attention Consistency Scores) utilities for posterior extraction.

Implements attention aggregation across layers/heads to extract token-to-LQ importance
for use in Stage-2 training (Task S posterior feedback from Task C).

Implementation Variants:
------------------------
This module provides two computation modes controlled by `use_log_space` parameter:

1. **Log-space computation** (use_log_space=True, DEFAULT):
   - Uses log-space cumulative product: log_joint += log(smoothed)
   - Recommended for deep networks (36+ layers like Qwen)
   - Avoids numerical underflow in cumulative products
   - More robust with float16/float32 mixed precision
   - Preserves relative ordering even with tiny values (~e-25)

2. **Original MACS** (use_log_space=False):
   - Uses linear-space cumulative product: joint *= smoothed
   - Matches original MACS paper implementation
   - Faster but may underflow to zero in deep networks
   - Works well for shallow networks (12-24 layers)
   - Useful for ablation studies and comparisons

When to use which:
- Default (use_log_space=True): Production training with Qwen/GPT-4 scale models
- Original (use_log_space=False): Reproducing MACS paper results, shallow LLMs, ablations

Alpha Parameter Selection:
---------------------------
The alpha parameter controls the exponential smoothing rate in cumulative products.
**Critical for deep networks (36+ layers)**: Lower alpha prevents numerical underflow.

Recommended values:
- **alpha=0.5** (DEFAULT): Balanced for 36-layer models, prevents underflow (0.5^36 ≈ 1e-11)
- **alpha=0.3**: Stronger differentiation, good for random/untrained models
- **alpha=0.7-0.8**: Shallow networks (12-24 layers), closer to original MACS paper

Why lower alpha helps:
- In deep networks: smoothed = alpha * att + (1-alpha) → accumulated over 36 layers
- High alpha (0.8): min smoothed = 0.2 → 0.2^36 ≈ 7e-26 (severe underflow)
- Low alpha (0.5): min smoothed = 0.5 → 0.5^36 ≈ 1.5e-11 (numerically stable)
- Effect: Better variance preservation, clearer LQ differentiation

Reference:
    Adapted from MACS_example.py, optimized for training loop integration.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional, List, Union


def compute_macs_to_lqs(
    attentions: Tuple[Tensor, ...],
    num_lqs: int,
    alpha: float = 0.5,
    use_zscore: bool = True,
    use_log_space: bool = True,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Compute MACS saliency map: All tokens → LQs attention aggregation.
    
    Aggregates multi-head, multi-layer attention weights using exponential smoothing
    and max-pooling to identify which LQs are most attended to by each token position.
    
    Algorithm:
    ----------
    1. Extract attention to first num_lqs positions (LQs in sequence)
    2. Max-pool over attention heads (most attentive head wins)
    3. Cumulative product across layers with exponential smoothing:
       - If use_log_space=True (default, recommended for deep networks):
         log_joint[l] = log_joint[l-1] + log(alpha * att[l] + (1-alpha))
       - If use_log_space=False (original MACS):
         joint[l] = joint[l-1] * (alpha * att[l] + (1-alpha))
    4. Optional Z-score normalization for interpretability
    
    Args:
        attentions: Tuple of length num_layers, each [batch, heads, seq_len, seq_len]
                   Output from model(..., output_attentions=True).attentions
        num_lqs: Number of learnable query tokens (LQs) at start of sequence
        alpha: Smoothing coefficient for exponential moving average (0.0-1.0)
              Higher alpha = more weight to current layer, faster decay
              Lower alpha = slower decay, better numerical stability in deep networks
              Recommended values:
              - 0.5: Default, good balance for 36-layer models (Qwen/GPT-4)
              - 0.3: Stronger differentiation, better for random/untrained models
              - 0.7-0.8: Shallow networks (12-24 layers), original MACS paper
              Note: Lower alpha prevents underflow in deep networks (0.5^36 = 1.5e-11 vs 0.8^36 = 7e-26)
        use_zscore: Whether to apply Z-score normalization along LQ dimension
                   Recommended: True (highlights LQs significantly above mean)
        use_log_space: Whether to use log-space cumulative product (default: True)
                      - True: Avoids underflow in deep networks (36+ layers), more stable
                      - False: Original MACS implementation, faster but may underflow
        device: Device for computation (auto-detected if None)
    
    Returns:
        joint_att: [batch, seq_len, num_lqs]
                  MACS importance score for each token→LQ pair
                  Higher values = token attends more to that LQ
    
    Shape Convention:
        Input sequence: [LQ_0, ..., LQ_{N-1}, token_0, ..., token_{T-1}]
        Output joint_att[:, i, j] = importance of LQ_j for token_i
    
    Example:
        >>> # From LLM forward pass
        >>> outputs = model(..., output_attentions=True)
        >>> attentions = outputs.attentions  # Tuple[32] of [B, 32, 512, 512]
        >>> 
        >>> # Compute MACS (assume first 32 tokens are LQs)
        >>> macs_map = compute_macs_to_lqs(attentions, num_lqs=32)
        >>> # macs_map: [B, 512, 32]
        >>> 
        >>> # Extract answer tokens → LQs attention (tokens 100-200)
        >>> answer_to_lqs = macs_map[:, 100:200, :]  # [B, 100, 32]
        >>> lq_importance = answer_to_lqs.mean(dim=1)  # [B, 32] - aggregate over answer
    
    Notes:
        - Attention tensors can be large; consider memory when processing long sequences
        - Z-score normalization along LQ dimension highlights relative importance
        - Alpha=0.8 balances current layer vs accumulated history
        - Max-pooling over heads captures "any head strongly attends" signal
    """
    if device is None:
        device = attentions[0].device
    
    # Step 1: Stack and slice - [num_layers, batch, heads, seq_len, num_lqs]
    # Memory efficient: don't stack all at once if num_layers is huge
    # For typical models (12-32 layers), stack is fine
    # Convert to float32 to avoid underflow in cumulative product (if use_log_space=True)
    # or keep original dtype (if use_log_space=False for original MACS behavior)
    if use_log_space:
        all_layers = torch.stack([att.float() for att in attentions], dim=0)  # [L, B, H, S, S]
    else:
        # Original MACS: use native dtype (may be float16)
        all_layers = torch.stack(attentions, dim=0)  # [L, B, H, S, S]
    
    target_attn = all_layers[..., :num_lqs]      # [L, B, H, S, num_lqs]
    
    # Step 2: Head aggregation (Max) → [num_layers, batch, seq_len, num_lqs]
    layer_max_attn, _ = target_attn.max(dim=2)  # Max over heads
    
    num_layers, batch_size, seq_len, _ = layer_max_attn.shape
    
    # Step 3: Initialize joint attention matrix
    if use_log_space:
        # Log-space: use float32 for numerical stability
        dtype = torch.float32
    else:
        # Original MACS: use native dtype
        dtype = layer_max_attn.dtype
    
    joint_att = torch.ones(
        batch_size, seq_len, num_lqs,
        device=device,
        dtype=dtype
    )
    bias = torch.ones_like(joint_att)
    
    # Step 4: Cumulative product with exponential smoothing
    if use_log_space:
        # Log-space computation to avoid underflow in deep networks (36 layers)
        # Instead of: joint *= smoothed (which underflows)
        # We use: log_joint += log(smoothed) (more stable)
        log_joint_att = torch.zeros(
            batch_size, seq_len, num_lqs,
            device=device,
            dtype=torch.float32
        )
        
        eps = 1e-10  # Small epsilon for numerical stability
        
        for layer_idx in range(num_layers):
            current_layer = layer_max_attn[layer_idx]  # [B, S, num_lqs]
            
            # Exponential moving average smoothing
            smoothed = alpha * current_layer + (1.0 - alpha) * bias
            
            # Log-space cumulative product (avoids underflow)
            log_joint_att = log_joint_att + torch.log(smoothed + eps)
        
        # Convert back from log space
        joint_att = torch.exp(log_joint_att)
    else:
        # Original MACS: Linear-space cumulative product (may underflow in deep networks)
        for layer_idx in range(num_layers):
            current_layer = layer_max_attn[layer_idx]  # [B, S, num_lqs]
            
            # Exponential moving average smoothing
            smoothed = alpha * current_layer + (1.0 - alpha) * bias
            
            # Linear-space cumulative product (original MACS)
            joint_att = joint_att * smoothed
    
    # Step 5: Optional Z-score normalization along LQ dimension
    if use_zscore:
        # For each token position, normalize its attention distribution across LQs
        # Meaning: Which LQs are significantly more important than average?
        mean = joint_att.mean(dim=-1, keepdim=True)      # [B, S, 1]
        std = joint_att.std(dim=-1, keepdim=True)        # [B, S, 1]
        
        if use_log_space:
            # Modified z-score: Apply only if std is large enough (avoid division by near-zero)
            # This prevents noise amplification when all LQs have similar attention
            std_threshold = 1e-4
            valid_std_mask = (std > std_threshold).squeeze(-1)  # [B, S]
            
            # Apply z-score row by row
            joint_att_normalized = torch.zeros_like(joint_att)
            for b in range(batch_size):
                for s in range(seq_len):
                    if valid_std_mask[b, s]:
                        # Apply z-score for this position
                        joint_att_normalized[b, s, :] = (joint_att[b, s, :] - mean[b, s, 0]) / (std[b, s, 0] + 1e-6)
                    else:
                        # Keep original values
                        joint_att_normalized[b, s, :] = joint_att[b, s, :]
            
            joint_att = joint_att_normalized
        else:
            # Original MACS z-score: Apply directly without threshold check
            joint_att = (joint_att - mean) / (std + 1e-6)
    
    return joint_att


def extract_answer_lq_posterior(
    attentions: Tuple[Tensor, ...],
    answer_start_idx: int,
    answer_end_idx: int,
    num_lqs: int,
    alpha: float = 0.5,
    use_zscore: bool = True,
    use_log_space: bool = True,
    aggregation: str = "mean",
) -> Tensor:
    """
    Extract posterior LQ importance distribution from answer tokens' attention.
    
    This is the **SA (Self-Attention) part** of MACS×LQ-CA posterior extraction:
    - MACS aggregates LLM attention to determine which LQs are used during answer generation
    - CA weights (from Q-Former) will later map LQs back to evidence fragments
    
    Workflow:
    ---------
    1. Compute full MACS map (all tokens → LQs)
    2. Slice to answer tokens only: macs_map[answer_start:answer_end, :]
    3. Aggregate over answer tokens (mean/max/sum) → [batch, num_lqs]
    4. Return as posterior LQ importance (input to Task S feedback)
    
    Args:
        attentions: LLM attention tuple from teacher_forcing_dual_path()
        answer_start_idx: Token index where answer begins (typically N_lq + S_q)
        answer_end_idx: Token index where answer ends (exclusive)
        num_lqs: Number of learnable queries at sequence start
        alpha: MACS smoothing coefficient (default: 0.5, see compute_macs_to_lqs for details)
        use_zscore: Whether to normalize MACS scores (default: True)
        use_log_space: Whether to use log-space computation (default: True, recommended)
        aggregation: How to aggregate over answer tokens
                    - "mean": Average importance (default, most stable)
                    - "max": Peak importance (highlights critical tokens)
                    - "sum": Total importance (longer answers get higher weight)
    
    Returns:
        lq_posterior: [batch, num_lqs]
                     Importance of each LQ for answer generation
                     Higher values = LQ was heavily attended during answer
                     Can be softmaxed for probability distribution
    
    Usage in Task S Posterior Feedback:
    ------------------------------------
    ```python
    # In training loop (Task C → Task S feedback)
    llm_outputs = frozen_llm.teacher_forcing_dual_path(z_prefix, query_ids, answer_ids)
    
    # Extract SA posterior (answer → LQs)
    lq_posterior = extract_answer_lq_posterior(
        attentions=llm_outputs['attentions'],
        answer_start_idx=llm_outputs['answer_start_idx'],
        answer_end_idx=llm_outputs['answer_end_idx'],
        num_lqs=32,
    )  # [batch, 32]
    
    # Get CA weights from Q-Former (LQs → evidence, subset U only)
    ca_weights_U = qformer_ca_weights[:, :, subset_indices]  # [batch, 32, |U|]
    
    # Compute evidence posterior: p(e|q,a) = p(LQ|a) × p(e|LQ)
    evidence_posterior = torch.softmax(
        torch.einsum('bn,bnk->bk', lq_posterior, ca_weights_U),
        dim=-1
    )  # [batch, |U|]
    
    # Feed to Task S loss
    loss_s = compute_ranking_loss(
        ...,
        posterior_scores=evidence_posterior.detach(),
        lambda_post=current_lambda_post,
    )
    ```
    
    Example:
        >>> # After LLM forward pass
        >>> attentions = llm_outputs['attentions']  # Tuple of [B, H, 512, 512]
        >>> answer_start = 132  # N_lqs(32) + S_q(100)
        >>> answer_end = 180    # 48 answer tokens
        >>> 
        >>> lq_importance = extract_answer_lq_posterior(
        ...     attentions, answer_start, answer_end, num_lqs=32
        ... )  # [B, 32]
        >>> 
        >>> # Interpret results
        >>> top_lqs = lq_importance.argmax(dim=-1)  # Which LQ most used per sample
    
    Notes:
        - Answer tokens = tokens LLM generates (ground truth in teacher forcing)
        - Posterior is detached - treated as "observed" teacher signal for Task S
        - Aggregation="mean" recommended (stable, length-invariant)
        - Combined with CA weights in compute_evidence_posterior()
    """
    # Step 1: Compute full MACS map
    macs_map = compute_macs_to_lqs(
        attentions=attentions,
        num_lqs=num_lqs,
        alpha=alpha,
        use_zscore=use_zscore,
        use_log_space=use_log_space,
    )  # [batch, seq_len, num_lqs]
    
    # Step 2: Slice to answer tokens only
    answer_to_lqs = macs_map[:, answer_start_idx:answer_end_idx, :]  # [batch, S_a, num_lqs]
    
    # Step 3: Aggregate over answer tokens
    if aggregation == "mean":
        lq_posterior = answer_to_lqs.mean(dim=1)  # [batch, num_lqs]
    elif aggregation == "max":
        lq_posterior, _ = answer_to_lqs.max(dim=1)  # [batch, num_lqs]
    elif aggregation == "sum":
        lq_posterior = answer_to_lqs.sum(dim=1)  # [batch, num_lqs]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Use 'mean', 'max', or 'sum'.")
    
    return lq_posterior


def compute_evidence_posterior(
    lq_posterior: Tensor,
    ca_weights: Tensor,
    subset_indices: Optional[Tensor] = None,
    temperature: float = 1.0,
) -> Tensor:
    """
    Compute evidence posterior distribution: p(evidence | query, answer).
    
    This is the **CA (Cross-Attention) part** of MACS×LQ-CA posterior extraction:
    - LQ posterior (from MACS): Which LQs the LLM used during answer generation
    - CA weights (from Q-Former): Which evidence each LQ attends to
    - Product: Which evidence the LLM implicitly relied on
    
    Formula:
    --------
    p(e_k | q, a) ∝ Σ_j p(LQ_j | a) × p(e_k | LQ_j)
                  = Σ_j lq_posterior[j] × ca_weights[j, k]
    
    Args:
        lq_posterior: [batch, num_lqs]
                     Importance of each LQ from LLM answer generation
                     Output of extract_answer_lq_posterior()
        ca_weights: [batch, num_lqs, K] or [batch, num_lqs, |U|]
                   Cross-attention weights from Q-Former (LQs → evidence)
                   Can be full pool (K) or subset (|U|)
        subset_indices: Optional[batch, |U|] or [|U|]
                       Indices of evidence subset U if ca_weights is subset
                       If None, assume ca_weights covers full pool
        temperature: Softmax temperature for posterior distribution
                    - temp=1.0: Standard softmax
                    - temp<1.0: Sharper distribution (confident)
                    - temp>1.0: Flatter distribution (uncertain)
    
    Returns:
        evidence_posterior: [batch, K] or [batch, |U|]
                           Posterior probability distribution over evidence
                           Sums to 1.0 per sample
                           Ready for JS divergence with Task S predictions
    
    Usage in Task S Loss:
    ---------------------
    ```python
    # Extract LQ posterior from LLM (SA part)
    lq_posterior = extract_answer_lq_posterior(...)  # [batch, 32]
    
    # Get Q-Former CA weights for subset U
    ca_weights_U = qformer_outputs['ca_weights'][:, :, subset_indices]  # [batch, 32, |U|]
    
    # Compute evidence posterior (CA part)
    evidence_posterior = compute_evidence_posterior(
        lq_posterior=lq_posterior,
        ca_weights=ca_weights_U,
        temperature=1.0,
    )  # [batch, |U|]
    
    # Use in Task S loss (detached - treat as teacher)
    loss_s_dict = compute_ranking_loss(
        ranking_logits=student_logits,
        gt_scores=teacher_scores,
        posterior_scores=evidence_posterior.detach(),
        mask=mask,
        subset_mask=subset_mask,
        lambda_teacher=current_lambda_teacher,
        lambda_post=current_lambda_post,
        ...
    )
    ```
    
    Interpretation:
    ---------------
    - High posterior = Evidence strongly attended by important LQs
    - Low posterior = Evidence ignored by LQs LLM actually used
    - Task S learns to align its prior predictions with this posterior
    
    Example:
        >>> lq_post = torch.randn(4, 32).softmax(dim=-1)  # [4, 32] LQ importance
        >>> ca_weights = torch.randn(4, 32, 20)           # [4, 32, 20] LQ→evidence attn
        >>> 
        >>> evidence_post = compute_evidence_posterior(lq_post, ca_weights)
        >>> # evidence_post: [4, 20] probability distribution
        >>> assert evidence_post.sum(dim=-1).allclose(torch.ones(4))
    
    Notes:
        - Evidence posterior is **detached** before feeding to Task S loss
        - Represents "ground truth" of what LLM actually used
        - JS divergence between student prior and this posterior drives alignment
    """
    # Input validation
    batch_size, num_lqs = lq_posterior.shape
    assert ca_weights.shape[:2] == (batch_size, num_lqs), \
        f"Shape mismatch: lq_posterior {lq_posterior.shape} vs ca_weights {ca_weights.shape}"
    
    # Compute weighted sum: Σ_j lq_posterior[j] × ca_weights[j, k]
    # Using einsum for clarity: batch, num_lqs × batch, num_lqs, evidence → batch, evidence
    evidence_logits = torch.einsum('bn,bnk->bk', lq_posterior, ca_weights)
    # Alternative: evidence_logits = (lq_posterior.unsqueeze(-1) * ca_weights).sum(dim=1)
    
    # Apply temperature scaling and softmax
    evidence_posterior = torch.softmax(evidence_logits / temperature, dim=-1)
    
    return evidence_posterior


def compute_evidence_posterior_from_ca(
    ca_weights: Tensor,
    aggregation: str = "last_layer_mean",
    lq_posterior: Optional[Tensor] = None,
    temperature: float = 1.0,
) -> Tensor:
    """
    Compute evidence posterior directly from Q-Former CA attention weights.
    
    This is a specialized version of compute_evidence_posterior() that handles
    Q-Former's multi-layer, multi-head CA weights directly, providing various
    aggregation strategies.
    
    Use Cases:
    ----------
    1. **Without LQ posterior** (uniform LQ importance):
       - Aggregate CA weights directly → evidence importance
       - Assumes all LQs contribute equally
       - Faster, simpler baseline
    
    2. **With LQ posterior** (from MACS SA):
       - Weight CA attention by LQ importance from LLM
       - More accurate: considers which LQs LLM actually used
       - Full MACS(CA) pipeline
    
    Aggregation Strategies:
    ----------------------
    - "last_layer_mean": Mean over heads in last Q-Former layer
      → Fast, focuses on final representation
    
    - "all_layers_mean": Mean over all layers and heads
      → Smooth, considers all reasoning steps
    
    - "all_layers_max": Max over heads, then mean over layers
      → Captures strongest attention signals
    
    - "weighted_layers": Exponentially weighted (recent layers > early layers)
      → Balances early and late reasoning
    
    Args:
        ca_weights: Q-Former CA weights, one of:
                   - Single layer: [batch, heads, num_lqs, K]
                   - Multi-layer: List/Tuple of [batch, heads, num_lqs, K]
        aggregation: Strategy to aggregate multi-head/multi-layer attention
                    See "Aggregation Strategies" above
        lq_posterior: Optional [batch, num_lqs] LQ importance from MACS SA
                     If None, uses uniform distribution (all LQs equal)
        temperature: Softmax temperature for final distribution
                    Lower = sharper, Higher = flatter
    
    Returns:
        evidence_posterior: [batch, K] probability distribution over evidence
                           Sums to 1.0, ready for Task S posterior feedback
    
    Example 1: Uniform LQ importance (baseline)
    -------------------------------------------
    ```python
    # Get CA weights from Q-Former forward pass
    z, aux = qformer(query_embeds=q_emb, p_embeds=fragments)
    ca_weights = aux['ca_attn_weights']  # [batch, heads, 32, 100]
    
    # Simple aggregation - all LQs treated equally
    evidence_post = compute_evidence_posterior_from_ca(
        ca_weights=ca_weights,
        aggregation="last_layer_mean",
    )  # [batch, 100]
    ```
    
    Example 2: Full MACS(CA) with LQ posterior
    ------------------------------------------
    ```python
    # Step 1: Get LQ posterior from LLM (MACS SA)
    llm_outputs = frozen_llm.teacher_forcing_dual_path(z, q_ids, a_ids)
    lq_post = extract_answer_lq_posterior(
        attentions=llm_outputs['attentions'],
        answer_start_idx=llm_outputs['answer_start_idx'],
        num_lqs=32,
    )  # [batch, 32]
    
    # Step 2: Propagate through Q-Former CA (MACS CA)
    evidence_post = compute_evidence_posterior_from_ca(
        ca_weights=all_layer_ca_weights,  # List of [batch, heads, 32, 100]
        aggregation="all_layers_mean",
        lq_posterior=lq_post,
        temperature=1.0,
    )  # [batch, 100]
    
    # Step 3: Use in Task S loss
    loss_s = js_divergence(student_prior, evidence_post.detach())
    ```
    
    Example 3: Ablation study on aggregation
    ----------------------------------------
    ```python
    for agg in ["last_layer_mean", "all_layers_mean", "all_layers_max"]:
        evidence_post = compute_evidence_posterior_from_ca(
            ca_weights=ca_weights_list,
            aggregation=agg,
            lq_posterior=lq_post,
        )
        loss = compute_loss(evidence_post)
        print(f"{agg}: {loss.item():.4f}")
    ```
    
    Notes:
    ------
    - If ca_weights is a single tensor, aggregation applies to heads only
    - If ca_weights is a list/tuple, aggregation applies to layers + heads
    - LQ posterior is optional; if None, uniform distribution is used
    - Output is always detached when used as teacher signal in Task S
    """
    # Handle both single-layer and multi-layer CA weights
    if isinstance(ca_weights, (list, tuple)):
        # Multi-layer: List of [batch, heads, num_lqs, K]
        ca_weights_stack = torch.stack(ca_weights, dim=0)  # [num_layers, batch, heads, num_lqs, K]
        num_layers, batch_size, num_heads, num_lqs, num_evidence = ca_weights_stack.shape
    else:
        # Single layer: [batch, heads, num_lqs, K]
        ca_weights_stack = ca_weights.unsqueeze(0)  # [1, batch, heads, num_lqs, K]
        num_layers, batch_size, num_heads, num_lqs, num_evidence = ca_weights_stack.shape
    
    # Aggregate CA weights based on strategy
    if aggregation == "last_layer_mean":
        # Use last layer, average over heads
        ca_aggregated = ca_weights_stack[-1].mean(dim=1)  # [batch, num_lqs, K]
    
    elif aggregation == "all_layers_mean":
        # Average over all layers and heads
        ca_aggregated = ca_weights_stack.mean(dim=(0, 2))  # [batch, num_lqs, K]
    
    elif aggregation == "all_layers_max":
        # Max over heads, then mean over layers
        ca_max_heads = ca_weights_stack.max(dim=2)[0]  # [num_layers, batch, num_lqs, K]
        ca_aggregated = ca_max_heads.mean(dim=0)  # [batch, num_lqs, K]
    
    elif aggregation == "weighted_layers":
        # Exponentially weighted (recent layers > early layers)
        weights = torch.exp(torch.linspace(0, 1, num_layers, device=ca_weights_stack.device))
        weights = weights / weights.sum()  # Normalize
        
        # Apply weights to each layer
        ca_weighted = ca_weights_stack * weights.view(-1, 1, 1, 1, 1)
        ca_aggregated = ca_weighted.sum(dim=0).mean(dim=1)  # [batch, num_lqs, K]
    
    else:
        raise ValueError(f"Unknown aggregation strategy: {aggregation}")
    
    # Prepare LQ posterior (uniform if not provided)
    if lq_posterior is None:
        lq_posterior = torch.ones(batch_size, num_lqs, device=ca_aggregated.device)
        lq_posterior = lq_posterior / num_lqs  # Uniform distribution
    
    # Compute evidence posterior: Σ_j P(LQ_j) × P(evidence_k | LQ_j)
    evidence_logits = torch.einsum('bn,bnk->bk', lq_posterior, ca_aggregated)
    
    # Apply temperature and softmax
    evidence_posterior = torch.softmax(evidence_logits / temperature, dim=-1)
    
    return evidence_posterior


def extract_span_indices(
    tokenizer,
    input_ids: Tensor,
    question_text: str,
    answer_text: str,
    num_lqs: int,
    remove_special_tokens: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Extract question and answer token span indices from chat-formatted sequence.
    
    For Qwen chat template: [LQs, system, user(question), assistant(answer)]
    Uses apply_chat_template to determine span boundaries.
    
    Args:
        tokenizer: Qwen tokenizer with chat template support
        input_ids: [batch, seq_len] or [seq_len] token IDs (with LQs prepended)
        question_text: Raw question string
        answer_text: Raw answer string
        num_lqs: Number of LQ tokens prepended to sequence
        remove_special_tokens: Whether to exclude <|im_start|>, <|im_end|> tokens
    
    Returns:
        question_indices: List of token positions for question (in [LQs, chat] coordinates)
        answer_indices: List of token positions for answer (in [LQs, chat] coordinates)
    
    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
        >>> question = "What causes rain?"
        >>> answer = "Water vapor condensation."
        >>> input_ids = torch.cat([lq_dummies, chat_ids], dim=1)  # [1, 32+128]
        >>> 
        >>> q_idx, a_idx = extract_span_indices(
        ...     tokenizer, input_ids, question, answer, num_lqs=32
        ... )
        >>> # q_idx: [37, 38, 39, 40]  (question tokens after LQs + system)
        >>> # a_idx: [45, 46, 47, ...]  (answer tokens)
    
    Notes:
        - Uses tokenizer.apply_chat_template() to get consistent spans
        - Handles Qwen special tokens (<|im_start|>, <|im_end|>)
        - Returns indices in full sequence coordinates (including LQs)
    
    TODO:
        - Support other chat templates (LLaMA, Mistral, etc.)
        - Handle batched input_ids
        - Cache span computation if question/answer unchanged
    """
    # This is a simplified version - full implementation depends on tokenizer specifics
    # For production, use the logic from MACS_example.py build_chat_ids_and_spans()
    
    # Placeholder: Return rough estimates based on sequence length
    # Real implementation should use apply_chat_template as in MACS_example.py
    batch_size = input_ids.shape[0] if input_ids.dim() == 2 else 1
    seq_len = input_ids.shape[-1]
    
    # Rough heuristic: question in middle third, answer in last third
    # REPLACE WITH ACTUAL SPAN DETECTION IN PRODUCTION
    q_start = num_lqs + 10  # After LQs + system tokens
    q_len = len(tokenizer.encode(question_text)) - 2  # Exclude BOS/EOS
    a_start = q_start + q_len + 5  # After question + role tokens
    a_len = len(tokenizer.encode(answer_text)) - 2
    
    question_indices = list(range(q_start, q_start + q_len))
    answer_indices = list(range(a_start, min(a_start + a_len, seq_len)))
    
    # TODO: Implement proper span detection using apply_chat_template
    # See MACS_example.py:build_chat_ids_and_spans() for reference
    
    return question_indices, answer_indices


# =============================================================================
# Convenience function for end-to-end posterior extraction
# =============================================================================

def extract_posterior_from_llm_outputs(
    llm_outputs: dict,
    qformer_ca_weights: Tensor,
    subset_indices: Optional[Tensor] = None,
    num_lqs: int = 32,
    alpha: float = 0.5,
    use_log_space: bool = True,
    temperature: float = 1.0,
) -> Tensor:
    """
    End-to-end posterior extraction: LLM attention → Evidence posterior.
    
    Combines extract_answer_lq_posterior() + compute_evidence_posterior()
    for convenient use in training loop.
    
    Args:
        llm_outputs: Dict from FrozenLLM.teacher_forcing_dual_path() with keys:
                    - 'attentions': Tuple of attention tensors
                    - 'answer_start_idx': Start of answer in sequence
                    - 'answer_end_idx': End of answer in sequence (if provided)
        qformer_ca_weights: [batch, num_lqs, K] Q-Former cross-attention weights
        subset_indices: Optional subset indices for dynamic U
        num_lqs: Number of learnable queries
        alpha: MACS smoothing coefficient
        use_log_space: Whether to use log-space computation (default: True)
        temperature: Softmax temperature for posterior
    
    Returns:
        evidence_posterior: [batch, K] or [batch, |U|]
                           Ready for Task S posterior feedback
    
    Usage:
    ------
    ```python
    # In training loop
    llm_outputs = frozen_llm.teacher_forcing_dual_path(z, q_ids, a_ids)
    ca_weights = qformer_outputs['ca_weights']  # [batch, 32, 100]
    
    # One-line posterior extraction
    evidence_posterior = extract_posterior_from_llm_outputs(
        llm_outputs=llm_outputs,
        qformer_ca_weights=ca_weights,
        subset_indices=subset_mask.nonzero(as_tuple=True)[1],
        num_lqs=32,
    )
    
    # Use in Task S loss
    loss_s = compute_ranking_loss(..., posterior_scores=evidence_posterior.detach(), ...)
    ```
    """
    # Extract answer end index (if not provided, use sequence length)
    answer_start = llm_outputs['answer_start_idx']
    answer_end = llm_outputs.get('answer_end_idx', None)
    
    # Get attention tensors
    attentions = llm_outputs['attentions']
    
    # Infer answer_end if not provided
    if answer_end is None:
        seq_len = attentions[0].shape[-1]  # [batch, heads, seq_len, seq_len]
        answer_end = seq_len
    
    # Step 1: MACS SA - Extract LQ posterior from answer attention
    lq_posterior = extract_answer_lq_posterior(
        attentions=attentions,
        answer_start_idx=answer_start,
        answer_end_idx=answer_end,
        num_lqs=num_lqs,
        alpha=alpha,
        use_zscore=True,
        use_log_space=use_log_space,
        aggregation="mean",
    )  # [batch, num_lqs]
    
    # Step 2: Apply subset masking if needed
    if subset_indices is not None:
        batch_size = qformer_ca_weights.shape[0]
        # Gather subset of CA weights
        ca_weights_subset = qformer_ca_weights[:, :, subset_indices]
    else:
        ca_weights_subset = qformer_ca_weights
    
    # Step 3: MACS CA - Compute evidence posterior
    evidence_posterior = compute_evidence_posterior(
        lq_posterior=lq_posterior,
        ca_weights=ca_weights_subset,
        subset_indices=None,  # Already applied
        temperature=temperature,
    )  # [batch, |U|] or [batch, K]
    
    return evidence_posterior


def aggregate_macs_sa_ca(
    lq_posterior: Tensor,
    ca_weights: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    alpha_ca: float = 0.8,
    use_log_space: bool = True,
    use_zscore_ca: bool = True,
    temperature: float = 1.0,
    normalize: bool = True,
) -> Tensor:
    """
    Aggregate MACS SA and MACS CA to compute evidence importance scores.
    
    This function combines:
    - **MACS SA**: LQ posterior from LLM attention (which LQs were used during answer generation)
    - **MACS CA**: Q-Former cross-attention (which evidence each LQ attends to)
    
    To produce final evidence importance scores that match the actual evidence pool dimension.
    
    Mathematical Framework:
    ----------------------
    Linear space:
        evidence_score[k] = Σ_j P(LQ_j | answer) × P(evidence_k | LQ_j)
                          = lq_posterior @ ca_weights
    
    Log space (more stable):
        log_evidence[k] = logsumexp_j(log_lq_posterior[j] + log_ca_weights[j, k])
    
    Output: [batch, K] where K = number of evidence fragments in pool
    
    Why This Function?
    -----------------
    - **Dimension correctness**: Output matches evidence pool size, not LQ count
    - **Numerical stability**: Log-space prevents underflow in deep networks
    - **Interpretability**: Evidence scores directly measure importance for answer
    - **Consistency**: Applies same MACS principles to both SA and CA
    
    Comparison with compute_evidence_posterior():
    ---------------------------------------------
    - compute_evidence_posterior(): Simple weighted sum, no CA MACS aggregation
    - aggregate_macs_sa_ca(): Full MACS pipeline (SA + CA with multi-layer aggregation)
    
    Args:
        lq_posterior: [batch, num_lqs]
                     LQ importance from MACS SA (extract_answer_lq_posterior)
                     Should be positive values (not necessarily normalized)
        ca_weights: Q-Former CA weights, one of:
                   - Single layer: [batch, heads, num_lqs, K]
                   - Multi-layer: List/Tuple of [batch, heads, num_lqs, K]
                   where K = number of evidence fragments
        alpha_ca: Exponential smoothing for CA MACS (default: 0.8)
                 Only used if ca_weights is multi-layer
        use_log_space: Whether to use log-space aggregation (default: True)
                      - True: log-sum-exp (numerically stable, recommended)
                      - False: matrix multiplication (faster, may overflow/underflow)
        use_zscore_ca: Whether to apply z-score to CA MACS (default: True)
                      Filters noise in cross-attention
        temperature: Softmax temperature for final distribution (default: 1.0)
                    Only used if normalize=True
        normalize: Whether to normalize output to probability distribution (default: True)
                  - True: Apply softmax → Σ evidence_score = 1.0
                  - False: Keep raw scores (useful for debugging)
    
    Returns:
        evidence_scores: [batch, K] evidence importance scores
                        If normalize=True: probability distribution (sums to 1.0)
                        If normalize=False: raw importance scores (positive values)
    
    Example 1: Full MACS pipeline (SA + CA)
    ---------------------------------------
    ```python
    # Step 1: Get LQ posterior from LLM (MACS SA)
    llm_outputs = frozen_llm.teacher_forcing_dual_path(z, q_ids, a_ids)
    lq_post = extract_answer_lq_posterior(
        attentions=llm_outputs['attentions'],
        answer_start_idx=llm_outputs['answer_start_idx'],
        answer_end_idx=llm_outputs['answer_end_idx'],
        num_lqs=32,
        alpha=0.5,  # Lower for deep LLM
        use_log_space=True,
    )  # [batch, 32]
    
    # Step 2: Get CA weights from Q-Former
    z, qformer_aux = qformer(query_embeds, evidence_embeds)
    ca_weights = qformer_aux['ca_attn_weights']  # List of [batch, 8, 32, 100]
    
    # Step 3: Aggregate SA + CA → evidence scores
    evidence_scores = aggregate_macs_sa_ca(
        lq_posterior=lq_post,
        ca_weights=ca_weights,
        alpha_ca=0.8,  # Higher for shallow Q-Former
        use_log_space=True,  # Recommended
        normalize=True,  # Get probability distribution
    )  # [batch, 100] - matches evidence pool size!
    
    # Step 4: Use in Task S loss
    loss_s = js_divergence(student_logits.softmax(-1), evidence_scores.detach())
    ```
    
    Example 2: Log-space vs Linear-space comparison
    -----------------------------------------------
    ```python
    # Linear space (faster, may be less stable)
    evidence_linear = aggregate_macs_sa_ca(
        lq_posterior=lq_post,
        ca_weights=ca_weights,
        use_log_space=False,
        normalize=True,
    )
    
    # Log space (recommended, numerically stable)
    evidence_log = aggregate_macs_sa_ca(
        lq_posterior=lq_post,
        ca_weights=ca_weights,
        use_log_space=True,
        normalize=True,
    )
    
    # Compare distributions
    diff = (evidence_linear - evidence_log).abs().max()
    print(f"Max difference: {diff.item():.6f}")  # Should be very small
    ```
    
    Example 3: Raw scores (no normalization)
    ----------------------------------------
    ```python
    # Get raw importance scores for analysis
    raw_scores = aggregate_macs_sa_ca(
        lq_posterior=lq_post,
        ca_weights=ca_weights,
        normalize=False,  # Keep raw scores
    )  # [batch, 100]
    
    # Analyze score distribution
    print(f"Score range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
    print(f"Mean: {raw_scores.mean():.4f}, Std: {raw_scores.std():.4f}")
    
    # Top-k evidence selection
    top_k_indices = raw_scores.topk(k=10, dim=-1).indices  # [batch, 10]
    ```
    
    Notes:
    ------
    - **Dimension guarantee**: Output always matches evidence pool size (K)
    - **Log-space recommended**: Prevents numerical issues in deep networks
    - **Detach for Task S**: Use evidence_scores.detach() as teacher signal
    - **CA MACS optional**: If ca_weights is single-layer, just aggregates heads
    - **Temperature**: Lower → sharper distribution, Higher → flatter distribution
    
    Implementation Details:
    ----------------------
    1. If multi-layer CA: Apply full MACS-CA aggregation (max-heads + smoothing)
    2. If single-layer CA: Simple head averaging
    3. Linear space: evidence = lq_posterior @ ca_aggregated
    4. Log space: evidence = logsumexp(log_lq + log_ca)
    5. Optional softmax normalization
    
    See Also:
    ---------
    - extract_answer_lq_posterior(): Extracts LQ posterior from LLM (MACS SA)
    - compute_evidence_posterior(): Simpler version without CA MACS
    - compute_evidence_posterior_from_ca_macs(): CA-only aggregation
    """
    batch_size, num_lqs = lq_posterior.shape
    
    # Step 1: Aggregate CA weights using MACS-CA
    # This gives us [batch, num_lqs, K] attention matrix
    if isinstance(ca_weights, (list, tuple)):
        # Multi-layer: Apply full MACS-CA algorithm
        ca_weights_list = list(ca_weights)
        num_layers = len(ca_weights_list)
        
        if num_layers == 1:
            # Single layer: just average over heads
            ca_aggregated = ca_weights_list[0].mean(dim=1)  # [batch, num_lqs, K]
        else:
            # Multi-layer: Apply MACS algorithm
            if use_log_space:
                ca_stack = torch.stack([w.float() for w in ca_weights_list], dim=0)
            else:
                ca_stack = torch.stack(ca_weights_list, dim=0)
            
            num_layers, batch_size, num_heads, num_lqs_check, num_evidence = ca_stack.shape
            assert num_lqs_check == num_lqs, f"LQ count mismatch: {num_lqs_check} vs {num_lqs}"
            
            # Max over heads
            ca_max_heads, _ = ca_stack.max(dim=2)  # [num_layers, batch, num_lqs, K]
            
            # Exponential smoothing across layers
            if use_log_space:
                log_joint_ca = torch.zeros(
                    batch_size, num_lqs, num_evidence,
                    device=ca_stack.device,
                    dtype=torch.float32,
                )
                eps = 1e-10
                bias = torch.ones(batch_size, num_lqs, num_evidence,
                                device=ca_stack.device, dtype=torch.float32)
                
                for layer_idx in range(num_layers):
                    current_layer = ca_max_heads[layer_idx]
                    smoothed = alpha_ca * current_layer + (1.0 - alpha_ca) * bias
                    log_joint_ca = log_joint_ca + torch.log(smoothed + eps)
                
                joint_ca = torch.exp(log_joint_ca)
            else:
                joint_ca = torch.ones(
                    batch_size, num_lqs, num_evidence,
                    device=ca_stack.device,
                    dtype=ca_stack.dtype,
                )
                
                for layer_idx in range(num_layers):
                    current_layer = ca_max_heads[layer_idx]
                    joint_ca = joint_ca * (alpha_ca * current_layer + (1 - alpha_ca))
            
            # Optional Z-score normalization
            if use_zscore_ca:
                mean = joint_ca.mean(dim=-1, keepdim=True)
                std = joint_ca.std(dim=-1, keepdim=True)
                
                if use_log_space:
                    std_threshold = 1e-4
                    valid_mask = (std > std_threshold)
                    ca_aggregated = torch.where(
                        valid_mask,
                        (joint_ca - mean) / (std + 1e-8),
                        joint_ca
                    )
                else:
                    ca_aggregated = (joint_ca - mean) / (std + 1e-8)
            else:
                ca_aggregated = joint_ca
    else:
        # Single tensor: [batch, heads, num_lqs, K]
        ca_aggregated = ca_weights.mean(dim=1)  # [batch, num_lqs, K]
    
    # Now ca_aggregated is [batch, num_lqs, K]
    num_evidence = ca_aggregated.shape[-1]
    
    # Step 2: Aggregate SA and CA
    if use_log_space:
        # Log-space aggregation using log-sum-exp
        # log P(evidence_k) = logsumexp_j [log P(LQ_j) + log P(evidence_k | LQ_j)]
        
        # Convert to log space (handle zeros and negatives)
        eps = 1e-10
        
        # Ensure lq_posterior is positive
        lq_posterior_pos = torch.clamp(lq_posterior, min=eps)
        log_lq_posterior = torch.log(lq_posterior_pos)  # [batch, num_lqs]
        
        # Ensure ca_aggregated is positive
        ca_aggregated_pos = torch.clamp(ca_aggregated, min=eps)
        log_ca_aggregated = torch.log(ca_aggregated_pos)  # [batch, num_lqs, K]
        
        # Broadcast and sum: [batch, num_lqs, 1] + [batch, num_lqs, K] → [batch, num_lqs, K]
        log_joint = log_lq_posterior.unsqueeze(-1) + log_ca_aggregated  # [batch, num_lqs, K]
        
        # Log-sum-exp over LQ dimension to get evidence scores
        # evidence[k] = logsumexp_j(log_joint[j, k])
        log_evidence_scores = torch.logsumexp(log_joint, dim=1)  # [batch, K]
        
        # Convert back from log space
        evidence_scores = torch.exp(log_evidence_scores)  # [batch, K]
        
    else:
        # Linear-space aggregation using matrix multiplication
        # evidence[k] = Σ_j P(LQ_j) × P(evidence_k | LQ_j)
        
        # Ensure positive values
        eps = 1e-10
        lq_posterior_pos = torch.clamp(lq_posterior, min=eps)
        ca_aggregated_pos = torch.clamp(ca_aggregated, min=eps)
        
        # Matrix multiplication: [batch, num_lqs] @ [batch, num_lqs, K] → [batch, K]
        evidence_scores = torch.einsum('bn,bnk->bk', lq_posterior_pos, ca_aggregated_pos)
    
    # Step 3: Optional normalization to probability distribution
    if normalize:
        # Apply temperature scaling and softmax
        evidence_scores = torch.softmax(evidence_scores / temperature, dim=-1)
    
    return evidence_scores


def compute_evidence_posterior_from_ca_macs(
    ca_weights: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    lq_posterior: Optional[Tensor] = None,
    alpha: float = 0.8,
    use_zscore: bool = True,
    use_log_space: bool = True,
    temperature: float = 1.0,
) -> Tensor:
    """
    Compute evidence posterior from Q-Former CA using MACS algorithm.
    
    **This is the DEFAULT and RECOMMENDED method** for aggregating Q-Former CA weights.
    It applies the full MACS algorithm (max over heads + exponential smoothing + z-score)
    to Q-Former's cross-attention, extending the MACS framework from LLM SA to Q-Former CA.
    
    Why MACS for CA?
    ----------------
    While compute_evidence_posterior_from_ca() offers simple aggregation strategies
    (mean, max), this function applies the principled MACS algorithm:
    
    1. **Max over heads**: Selects strongest attention signal per layer
    2. **Exponential smoothing**: Captures multi-layer consistency across Q-Former layers
    3. **Z-score normalization**: Filters noise and enhances signal contrast
    4. **Log-space computation**: Avoids underflow in cumulative products (default: True)
    
    Benefits:
    - **Stronger signal**: MACS filtering emphasizes consistent evidence
    - **Noise reduction**: Z-score removes weak, inconsistent attention
    - **Numerical stability**: Log-space prevents underflow in multi-layer products
    - **Principled**: Same algorithm as LLM SA, proven effective in MACS paper
    
    Trade-offs:
    - **Compute cost**: Slightly higher than simple mean (but negligible)
    - **Layer depth**: Q-Former has only 6-8 layers vs LLM's 32-36, so cross-layer
      consistency is less pronounced (but still beneficial)
    
    Args:
        ca_weights: Q-Former CA weights, one of:
                   - Single layer: [batch, heads, num_lqs, K]
                   - Multi-layer: List/Tuple of [batch, heads, num_lqs, K]
        lq_posterior: Optional [batch, num_lqs] LQ importance from MACS SA
                     If None, uses uniform distribution (all LQs equal)
        alpha: Exponential smoothing coefficient (default: 0.8)
               Higher α = more influence from current layer
               Lower α = smoother across layers
               Recommended: 0.5-0.8 for Q-Former (fewer layers than LLM)
        use_zscore: Whether to apply Z-score normalization (default: True)
                   Recommended to keep True for noise filtering
        use_log_space: Whether to use log-space cumulative product (default: True)
                      - True: Avoids underflow, more numerically stable
                      - False: Original linear-space (faster but may underflow)
        temperature: Softmax temperature for final distribution (default: 1.0)
                    Lower = sharper, Higher = flatter
    
    Returns:
        evidence_posterior: [batch, K] probability distribution over evidence
                           Sums to 1.0, ready for Task S posterior feedback
    
    Example 1: Basic usage (uniform LQ importance)
    -----------------------------------------------
    ```python
    # Get CA weights from Q-Former forward pass
    z, aux = qformer(query_embeds=q_emb, p_embeds=fragments)
    ca_weights = aux['ca_attn_weights']  # List of [batch, heads, 32, 100]
    
    # Apply MACS-CA (default method)
    evidence_post = compute_evidence_posterior_from_ca_macs(
        ca_weights=ca_weights,
    )  # [batch, 100]
    ```
    
    Example 2: Full MACS pipeline (SA + CA)
    ---------------------------------------
    ```python
    # Step 1: Get LQ posterior from LLM (MACS SA)
    llm_outputs = frozen_llm.teacher_forcing_dual_path(z, q_ids, a_ids)
    lq_post = extract_answer_lq_posterior(
        attentions=llm_outputs['attentions'],
        answer_start_idx=llm_outputs['answer_start_idx'],
        num_lqs=32,
        alpha=0.5,  # Lower alpha for 36-layer LLM
        use_log_space=True,  # Default, prevents underflow
    )  # [batch, 32]
    
    # Step 2: Apply MACS-CA with LQ weighting
    evidence_post = compute_evidence_posterior_from_ca_macs(
        ca_weights=all_layer_ca_weights,
        lq_posterior=lq_post,  # Weight by LQ importance
        alpha=0.8,  # Can be higher for Q-Former (only 6-8 layers)
        use_zscore=True,
        use_log_space=True,  # Consistent with SA
    )  # [batch, 100]
    
    # Step 3: Use in Task S loss
    loss_s = js_divergence(student_prior, evidence_post.detach())
    ```
    
    Example 3: Ablation study (MACS vs baseline)
    --------------------------------------------
    ```python
    # Baseline: simple mean aggregation
    evidence_baseline = compute_evidence_posterior_from_ca(
        ca_weights=ca_weights,
        aggregation="all_layers_mean",
        lq_posterior=lq_post,
    )
    
    # MACS-CA: principled aggregation
    evidence_macs = compute_evidence_posterior_from_ca_macs(
        ca_weights=ca_weights,
        lq_posterior=lq_post,
    )
    
    # Compare distributions
    js_div = js_divergence(evidence_macs, evidence_baseline)
    print(f"MACS vs Baseline JS divergence: {js_div.mean().item():.4f}")
    ```
    
    Notes:
    ------
    - This is the RECOMMENDED default method for DR-QFormer Stage 2 training
    - Falls back to simple mean if only single-layer CA weights provided
    - Output is always normalized (sums to 1.0) and ready for Task S
    - Use temperature > 1.0 if posterior is too sharp (overconfident)
    
    See Also:
    ---------
    - compute_evidence_posterior_from_ca(): Alternative with simple aggregation
    - compute_macs_to_lqs(): MACS algorithm for LLM SA
    - extract_answer_lq_posterior(): LQ posterior extraction from LLM
    """
    # Handle both single-layer and multi-layer CA weights
    if isinstance(ca_weights, (list, tuple)):
        # Multi-layer: Apply full MACS algorithm
        ca_weights_list = list(ca_weights)
        num_layers = len(ca_weights_list)
        
        if num_layers == 1:
            # Single layer: just average over heads (MACS not applicable)
            ca_aggregated = ca_weights_list[0].mean(dim=1)  # [batch, num_lqs, K]
        else:
            # Multi-layer: Apply MACS algorithm
            # Stack: [num_layers, batch, heads, num_lqs, K]
            # Convert to float32 if using log-space (for numerical stability)
            if use_log_space:
                ca_stack = torch.stack([w.float() for w in ca_weights_list], dim=0)
            else:
                ca_stack = torch.stack(ca_weights_list, dim=0)
            
            num_layers, batch_size, num_heads, num_lqs, num_evidence = ca_stack.shape
            
            # Step 1: Max over heads (select strongest attention per layer)
            ca_max_heads, _ = ca_stack.max(dim=2)  # [num_layers, batch, num_lqs, K]
            
            # Step 2: Exponential smoothing across layers with log-space option
            if use_log_space:
                # Log-space computation (numerically stable for deep networks)
                log_joint_ca = torch.zeros(
                    batch_size, num_lqs, num_evidence,
                    device=ca_stack.device,
                    dtype=torch.float32,
                )
                
                eps = 1e-10  # Small epsilon for numerical stability
                bias = torch.ones(batch_size, num_lqs, num_evidence, 
                                device=ca_stack.device, dtype=torch.float32)
                
                for layer_idx in range(num_layers):
                    current_layer = ca_max_heads[layer_idx]  # [batch, num_lqs, K]
                    # Exponential moving average smoothing
                    smoothed = alpha * current_layer + (1.0 - alpha) * bias
                    # Log-space cumulative product (avoids underflow)
                    log_joint_ca = log_joint_ca + torch.log(smoothed + eps)
                
                # Convert back from log space
                joint_ca = torch.exp(log_joint_ca)
            else:
                # Original linear-space computation (may underflow in deep networks)
                joint_ca = torch.ones(
                    batch_size, num_lqs, num_evidence,
                    device=ca_stack.device,
                    dtype=ca_stack.dtype,
                )
                
                for layer_idx in range(num_layers):
                    current_layer = ca_max_heads[layer_idx]  # [batch, num_lqs, K]
                    # Hadamard product with smoothing: joint *= (α * current + (1-α))
                    joint_ca = joint_ca * (alpha * current_layer + (1 - alpha))
            
            # Step 3: Z-score normalization (optional)
            if use_zscore:
                # Normalize along evidence dimension (K)
                mean = joint_ca.mean(dim=-1, keepdim=True)
                std = joint_ca.std(dim=-1, keepdim=True)
                
                if use_log_space:
                    # Conditional z-score: only apply if std is large enough
                    std_threshold = 1e-4
                    valid_mask = (std > std_threshold)  # [batch, num_lqs, 1]
                    
                    # Apply z-score where valid, keep original otherwise
                    ca_aggregated = torch.where(
                        valid_mask,
                        (joint_ca - mean) / (std + 1e-8),
                        joint_ca
                    )
                else:
                    # Original z-score: apply directly
                    ca_aggregated = (joint_ca - mean) / (std + 1e-8)
            else:
                ca_aggregated = joint_ca
    else:
        # Single tensor: [batch, heads, num_lqs, K]
        # Just average over heads (MACS not applicable to single layer)
        ca_aggregated = ca_weights.mean(dim=1)  # [batch, num_lqs, K]
    
    # Now ca_aggregated is [batch, num_lqs, K]
    # Apply LQ posterior weighting if provided
    if lq_posterior is not None:
        # Weighted sum: evidence_logits = sum_n (lq_posterior[n] * ca[n, k])
        evidence_logits = torch.einsum('bn,bnk->bk', lq_posterior, ca_aggregated)
    else:
        # Uniform LQ importance: simple average over LQs
        evidence_logits = ca_aggregated.mean(dim=1)  # [batch, K]
    
    # Apply temperature scaling and softmax
    evidence_posterior = torch.softmax(evidence_logits / temperature, dim=-1)
    
    return evidence_posterior
