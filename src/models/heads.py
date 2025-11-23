"""
Task-specific heads for DR-QFormer (Fragment-Level Operations).

All three tasks operate on **text fragments/chunks** (not full documents):
- Task E: Fragment-level entailment tagging (蕴含-标注)
- Task S: Fragment-level sorting supervision (排序-监督)
- Task C: Condensing-generation (精炼-生成)

Each task supports both Primal (QA) and Dual (QG) training modes.
All heads operate on Q-Former output Z [batch, N, d].
"""

from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
except ImportError:
    # Dummy fallback
    class nn:
        class Module:
            pass
    Tensor = None


class EntailmentHead(nn.Module):
    """
    Task E: Fragment-Level Entailment Tagging (蕴含-标注).
    
    Purpose: Learn "answerability/entailment" to act as a **fragment-level filter/tagger**.
    
    Architecture:
    - Input: CA attention scores from Q-Former [batch, num_heads, N, k]
    - Output: k logits [batch, k] - one per fragment
    - Uses LogSumExp aggregation over N LQs with temperature scaling
    
    Training Modes:
    - Primal (QA): Q-Former receives query → predicts which fragments entail the answer
    - Dual (QG): Q-Former receives answer → predicts which fragments entail the query
    
    Supervision:
    - gt_labels [batch, k]: Binary vector [0,1,0,1,...] marking golden evidence fragments
    - importance_weights [batch, k]: Fragment-level weights (normal vs longtail)
    
    Loss:
    - Focal Loss with dynamic importance weighting
    
    Metrics:
    - Accuracy, Precision, Recall, F1
    
    Args:
        num_fragments (int): Number of fragments to classify (k). Default: 10
        tau (float): LogSumExp temperature. Default: 0.5
        p_drop_lq (float): Drop-LQ probability during training. Default: 0.1
        focal_gamma (float): Focal loss gamma parameter. Default: 2.0
        focal_alpha (float): Focal loss alpha parameter. Default: 0.25
    """
    
    def __init__(
        self, 
        num_fragments: int = 10,
        tau: float = 0.5,
        p_drop_lq: float = 0.1,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        hidden_dim: Optional[int] = None  # Q-Former hidden dimension (REQUIRED for BLIP-2 style)
    ):
        super().__init__()
        self.num_fragments = num_fragments  # Only used as hint, not enforced
        self.tau = tau
        self.p_drop_lq = p_drop_lq
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
        # BLIP-2 Style: Learnable task-specific projection head
        # This allows the model to learn which aspects of Q-Former output are relevant for entailment
        if hidden_dim is None:
            raise ValueError("hidden_dim is required for EntailmentHead (BLIP-2 paradigm)")
        
        self.hidden_dim = hidden_dim
        
        # Task-specific MLP: Z [B, N, hidden_dim] → fragment scores [B, K]
        # Architecture: Pooling → MLP → Projection
        # 1. Attention-weighted pooling over N learnable queries (learned weights)
        self.query_attention = nn.Linear(hidden_dim, 1)  # Compute attention weights for N queries
        
        # 2. MLP for task-specific transformation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # 3. Final projection to fragment logits (will be dynamically sized)
        # NOTE: We can't pre-define output size due to dynamic K
        # Instead, we'll use a vector that broadcasts across K
        self.fragment_scorer = nn.Linear(hidden_dim // 4, 1)  # Score per fragment
        
        # For numerical stability in manual normalization
        self.eps = 1e-5
        
        print(f"EntailmentHead initialized (BLIP-2 Style with Learnable Projection):")
        print(f"  - hidden_dim: {hidden_dim}")
        print(f"  - tau (temperature): {tau}")
        print(f"  - p_drop_lq: {p_drop_lq}")
        print(f"  - focal_gamma: {focal_gamma}")
        print(f"  - focal_alpha: {focal_alpha}")
        print(f"  ✨ Added learnable projection: {hidden_dim} → {hidden_dim//2} → {hidden_dim//4} → 1")
    
    def forward(
        self, 
        z: Optional[Tensor] = None, 
        ca_raw_scores_per_head: Optional[list] = None,
        pool_padding_mask: Optional[Tensor] = None,
        lq_drop_mask: Optional[Tensor] = None,
        training: bool = True
    ) -> dict:
        """
        Forward pass for entailment tagging using pre-softmax raw CA scores.
        
        Args:
            z: Q-Former output [batch, N, d] (not used, kept for interface compatibility)
            ca_raw_scores_per_head: List of pre-softmax raw CA scores per layer
                                   Each tensor: [batch, num_heads, N, k]
                                   These are QK^T/sqrt(d_head) BEFORE softmax
            pool_padding_mask: [batch, k] bool mask (True=valid, False=padding)
            lq_drop_mask: [batch, N, 1] bool mask for unified Drop-LQ
                         True = keep LQ, False = drop LQ
                         If None, internal Drop-LQ is used (if enabled)
            training: Whether in training mode (affects Drop-LQ)
        
        Returns:
            dict with keys:
                - fragment_logits: [batch, k] entailment scores for each fragment
                - ca_raw_scores_avg: [batch, N, k] head-averaged raw scores (detached, for debugging)
                - ca_raw_scores_per_head: [batch, num_heads, N, k] layer-averaged raw scores (detached, for debugging)
        
        Pipeline (Spec v1.1 - Layer-wise Normalization):
        ------------------------------------------------
        1. For each layer:
           a) Apply pool_padding_mask (set padding to -1e4)
           b) Apply LayerNorm per head along K dimension
           c) Average over heads → [batch, N, k]
        2. Aggregate normalized scores across layers (mean)
        3. Apply Drop-LQ regularization (training only)
        4. LogSumExp aggregation over N LQs → [batch, k]
        """
        if z is None:
            raise ValueError("z (Q-Former output) is required for EntailmentHead")
        
        # z: [batch, N, hidden_dim] - Q-Former learnable query outputs
        batch_size, N, hidden_dim = z.shape
        
        # BLIP-2 Style Pipeline:
        # =====================
        # Step 1: Apply Drop-LQ regularization (training only)
        if training:
            if lq_drop_mask is not None:
                # Use external unified mask (for multi-task training)
                z_dropped = self._apply_drop_lq_on_z(z, mask=lq_drop_mask)
            elif self.p_drop_lq > 0:
                # Use internal random mask (for single-task training)
                z_dropped = self._apply_drop_lq_on_z(z, mask=None)
            else:
                z_dropped = z
        else:
            z_dropped = z
        
        # Step 2: Compute attention weights over N learnable queries
        # This learns "which query embeddings are most important for entailment"
        query_attn_logits = self.query_attention(z_dropped).squeeze(-1)  # [batch, N]
        query_attn_weights = torch.softmax(query_attn_logits, dim=1)  # [batch, N]
        
        # Step 3: Attention-weighted pooling → [batch, hidden_dim]
        pooled_z = torch.einsum('bn,bnh->bh', query_attn_weights, z_dropped)  # [batch, hidden_dim]
        
        # Step 4: Task-specific MLP transformation
        transformed = self.mlp(pooled_z)  # [batch, hidden_dim // 4]
        
        # Step 5: Use CA attention scores to modulate per-fragment predictions
        # This combines learned representation (from Z) with attention patterns (from CA)
        if ca_raw_scores_per_head is not None and len(ca_raw_scores_per_head) > 0:
            # Extract attention-based fragment relevance scores
            ca_raw = ca_raw_scores_per_head[-1]  # Use last layer: [batch, num_heads, N, k]
            _, _, _, k = ca_raw.shape
            
            # Average over heads and queries to get fragment importance
            ca_fragment_scores = ca_raw.mean(dim=(1, 2))  # [batch, k]
            
            # Normalize CA scores per sample
            if pool_padding_mask is not None:
                ca_fragment_scores = ca_fragment_scores.masked_fill(~pool_padding_mask, -1e9)
            ca_fragment_scores = torch.softmax(ca_fragment_scores, dim=-1)  # [batch, k]
        else:
            # Fallback: uniform attention (shouldn't happen in practice)
            k = 10  # Default, will be overridden by pool_padding_mask shape
            if pool_padding_mask is not None:
                k = pool_padding_mask.shape[1]
            ca_fragment_scores = torch.ones(batch_size, k, device=z.device) / k
        
        # Step 6: Combine learned representation with attention scores
        # Broadcast transformed features across fragments and modulate by attention
        transformed_expanded = transformed.unsqueeze(1).expand(-1, k, -1)  # [batch, k, hidden_dim//4]
        
        # Project to scalar logits per fragment
        fragment_logits_raw = self.fragment_scorer(transformed_expanded).squeeze(-1)  # [batch, k]
        
        # Modulate by attention scores (multiplicative gating)
        fragment_logits = fragment_logits_raw * ca_fragment_scores * self.tau  # [batch, k]
        
        # Step 7: Apply final pool_padding_mask to output logits
        if pool_padding_mask is not None:
            fragment_logits = fragment_logits.masked_fill(~pool_padding_mask, -1e4)
        
        # For debug outputs
        ca_raw_first_layer = ca_raw_scores_per_head[0] if ca_raw_scores_per_head else None
        
        # Return dict with debug outputs
        return {
            'fragment_logits': fragment_logits,
            'ca_raw_scores_avg': pooled_z.detach() if pooled_z is not None else None,  # [batch, hidden_dim] pooled representation
            'ca_raw_scores_per_head': ca_raw_first_layer.detach() if ca_raw_first_layer is not None else None  # [batch, num_heads, N, k] first layer raw
        }
    
    def _apply_drop_lq_on_z(self, z: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply Drop-LQ regularization on Q-Former output Z.
        
        Args:
            z: [batch, N, hidden_dim] Q-Former output
            mask: [batch, N, 1] optional external mask (True=keep, False=drop)
        
        Returns:
            z_dropped: [batch, N, hidden_dim] with some LQs zeroed out
        """
        batch_size, n_lqs, hidden_dim = z.shape
        device = z.device
        
        if mask is not None:
            # Use external unified mask
            mask_drop = mask.float()  # [batch, N, 1]
        else:
            # Generate internal dropout mask
            mask_drop = torch.bernoulli(
                torch.full((batch_size, n_lqs, 1), 1.0 - self.p_drop_lq, device=device)
            )
            
            # Safety: prevent all LQs being dropped
            all_dropped = (mask_drop.sum(dim=1, keepdim=True) == 0)
            if all_dropped.any():
                for b in range(batch_size):
                    if all_dropped[b, 0, 0]:
                        random_idx = int(torch.randint(0, n_lqs, (1,), device=device).item())
                        mask_drop[b, random_idx, 0] = 1.0
        
        # Zero out dropped LQs
        return z * mask_drop
    
    def _apply_drop_lq(self, ca_scores: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply Drop-LQ regularization with safety protection.
        
        Args:
            ca_scores: [batch, N, k] attention scores
            mask: [batch, N, 1] optional external mask (True=keep, False=drop)
                 If None, generate internal random mask
        
        Returns:
            ca_scores_dropped: [batch, N, k] with some LQs masked
        """
        batch_size, n_lqs, k = ca_scores.shape
        device = ca_scores.device
        
        if mask is not None:
            # Use external unified mask (for multi-task training)
            mask_drop = mask.float()  # [batch, N, 1], 1.0=keep, 0.0=drop
        else:
            # Generate internal dropout mask: [batch, N, 1]
            # 1.0 = keep, 0.0 = drop
            mask_drop = torch.bernoulli(
                torch.full((batch_size, n_lqs, 1), 1.0 - self.p_drop_lq, device=device)
            )
            
            # Safety: prevent all LQs being dropped for any sample
            all_dropped = (mask_drop.sum(dim=1, keepdim=True) == 0)  # [batch, 1, 1]
            if all_dropped.any():
                # For samples with all LQs dropped, randomly keep one
                for b in range(batch_size):
                    if all_dropped[b, 0, 0]:
                        random_idx_tensor = torch.randint(0, n_lqs, (1,), device=device)
                        random_idx = int(random_idx_tensor.item())
                        mask_drop[b, random_idx, 0] = 1.0
        
        # Apply mask: set dropped LQs to large negative value
        ca_scores_dropped = ca_scores.clone()
        ca_scores_dropped = ca_scores_dropped + (1.0 - mask_drop) * (-1e4)
        
        return ca_scores_dropped
    
    def _logsumexp_aggregate(self, ca_scores: Tensor) -> Tensor:
        """
        Numerically stable LogSumExp aggregation over N LQs.
        
        Args:
            ca_scores: [batch, N, k] attention scores
        
        Returns:
            logits: [batch, k] aggregated scores
        
        Formula:
            logit[b,k] = tau * (m[b,k] + log(sum_i(exp((S[b,i,k]/tau) - m[b,k]))))
            where m[b,k] = max_i(S[b,i,k] / tau)
        """
        # Scale by temperature
        scaled_scores = ca_scores / self.tau  # [batch, N, k]
        
        # Max for numerical stability
        max_scores = scaled_scores.max(dim=1, keepdim=True)[0]  # [batch, 1, k]
        
        # Numerically stable exp
        exp_scores = torch.exp(scaled_scores - max_scores)  # [batch, N, k]
        
        # Sum over LQs
        sum_exp = exp_scores.sum(dim=1)  # [batch, k]
        
        # LogSumExp result
        logits = self.tau * (max_scores.squeeze(1) + torch.log(sum_exp + 1e-8))
        
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters in EntailmentHead."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FragmentRankingHead(nn.Module):
    """
    Task S: Fragment-Level Ranking via Dual-LSE Aggregation (排序-监督).
    
    Purpose: Train Q-Former's cross-attention to precisely rank fragments from 
             large evidence pools (K=100~5000).
    
    Architecture:
    - Input: Pre-softmax CA raw scores from Q-Former [batch, num_heads, N_lq, K]
    - Output: Ranking logits [batch, K] - one score per fragment
    - Uses dual LogSumExp (LSE) aggregation: Head → LQ → Logits
    
    Training Modes:
    - Primal (QA): Q-Former receives query → ranks fragments by answer relevance
    - Dual (QG): Q-Former receives answer → ranks fragments by query relevance
    
    Supervision:
    - gt_scores [batch, K]: Teacher reranker scores (e.g., from BM25, DPR)
    - posterior_scores [batch, K]: Optional LLM feedback scores (from Task C)
    - Dynamic curriculum: Teacher (early) → Posterior (late)
    
    Loss Components:
    - L_teach: ListNet loss vs teacher scores
    - L_post: JS divergence vs posterior scores (detached)
    - L_tail_entropy: Entropy regularization on low-scoring fragments
    
    Key Features:
    - Anti-noise aggregation via dual LSE
    - Dynamic subset construction (Teacher Top-L + Student Hard Negatives)
    - Curriculum learning (teacher → posterior transition)
    
    Args:
        num_fragments (int): Number of fragments to rank (K). Default: 100
        tau_head (float): LSE temperature for head aggregation. Default: 0.1
        tau_lq (float): LSE temperature for LQ aggregation. Default: 0.2
        rho_top (float): Dynamic Teacher Top-L ratio. Default: 0.02 (2%)
        l_prime (int): Student Hard Negatives count. Default: 16
    
    Input:
        ca_raw_scores_per_head: Pre-softmax scores [batch, num_heads, N_lq, K]
        pool_padding_mask: Valid fragment mask [batch, K] (True=valid, False=padding)
        
    Output:
        ranking_logits: [batch, K] ranking scores (higher = more relevant)
    """
    
    def __init__(
        self,
        num_fragments: int = 100,
        tau_head: float = 0.1,
        tau_lq: float = 0.2,
        rho_top: float = 0.02,
        l_prime: int = 16,
        p_drop_lq: float = 0.1,  # Drop-LQ probability (0.0 = disabled)
        hidden_dim: Optional[int] = None  # Accepted for API compatibility, ignored
    ):
        super().__init__()
        self.num_fragments = num_fragments
        self.tau_head = tau_head
        self.tau_lq = tau_lq
        self.rho_top = rho_top
        self.l_prime = l_prime
        self.p_drop_lq = p_drop_lq
        
        print(f"FragmentRankingHead initialized:")
        print(f"  - tau_head (Head LSE temperature): {tau_head}")
        print(f"  - tau_lq (LQ LSE temperature): {tau_lq}")
        print(f"  - rho_top (Teacher Top-L ratio): {rho_top}")
        print(f"  - l_prime (Student Hard Negatives): {l_prime}")
        print(f"  - p_drop_lq (Drop-LQ probability): {p_drop_lq}")
        if hidden_dim is not None:
            print(f"  - hidden_dim provided ({hidden_dim}) but ignored (using raw scores)")
    
    def forward(
        self,
        z: Optional[Tensor] = None,
        ca_raw_scores_per_head: Optional[list] = None,
        pool_padding_mask: Optional[Tensor] = None,
        lq_drop_mask: Optional[Tensor] = None,
        training: bool = True
    ) -> dict:
        """
        Forward pass using dual LogSumExp (LSE) aggregation.
        
        Args:
            z: Q-Former output [batch, N_lq, d] (not used, kept for interface compatibility)
            ca_raw_scores_per_head: List of [batch, num_heads, N_lq, K] per layer
            pool_padding_mask: [batch, K] valid fragment mask
            lq_drop_mask: [batch, N, 1] bool mask for unified Drop-LQ
                         True = keep LQ, False = drop LQ
                         If None, internal Drop-LQ is used (if enabled)
            training: Whether in training mode (affects Drop-LQ)
        
        Returns:
            dict with:
                - ranking_logits: [batch, K] ranking scores
                - ca_raw_scores_avg: [batch, N_lq, K] averaged raw scores (debug)
        
        Dual-LSE Aggregation:
        ====================
        Step 1: Head-level LSE (aggregate multi-head scores)
            scores_h [batch, N_lq, K] = LSE_{h}(raw_scores / tau_head) * tau_head
            
        Step 2: LQ-level LSE (aggregate query token scores)
            ranking_logits [batch, K] = LSE_{lq}(scores_h / tau_lq) * tau_lq
        
        Formula:
            LSE(x) = log(sum(exp(x)))
            
        Properties:
            - Smooth max approximation (as tau → 0, LSE → max)
            - Differentiable everywhere
            - Noise-robust (down-weights outliers)
        """
        if ca_raw_scores_per_head is None or len(ca_raw_scores_per_head) == 0:
            raise ValueError("ca_raw_scores_per_head is required for FragmentRankingHead")
        
        # Use last layer's CA scores (most refined)
        ca_raw_scores = ca_raw_scores_per_head[-1]  # [batch, num_heads, N_lq, K]
        
        batch_size, num_heads, N_lq, K = ca_raw_scores.shape
        
        # Ensure mask type consistency: always convert to bool
        if pool_padding_mask is None:
            pool_padding_mask = torch.ones(batch_size, K, dtype=torch.bool, device=ca_raw_scores.device)
        else:
            pool_padding_mask = pool_padding_mask.to(torch.bool)
        
        # Apply padding mask: set invalid positions to large negative value
        # Expand mask for broadcasting: [batch, 1, 1, K]
        mask_expanded = pool_padding_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, K]
        ca_raw_scores_masked = ca_raw_scores.masked_fill(~mask_expanded, -1e4)
        
        # Step 1: Head-level LSE aggregation
        # LSE over heads: [batch, num_heads, N_lq, K] → [batch, N_lq, K]
        scores_scaled = ca_raw_scores_masked / self.tau_head
        scores_head_lse = torch.logsumexp(scores_scaled, dim=1) * self.tau_head  # [batch, N_lq, K]
        
        # Step 1.5: Apply Drop-LQ regularization (training only)
        if training:
            if lq_drop_mask is not None:
                # Use external unified mask (for multi-task training)
                scores_head_lse = self._apply_drop_lq(scores_head_lse, mask=lq_drop_mask)
            elif self.p_drop_lq > 0:
                # Use internal random mask (for single-task training)
                scores_head_lse = self._apply_drop_lq(scores_head_lse, mask=None)
        
        # Step 2: LQ-level LSE aggregation
        # LSE over LQ tokens: [batch, N_lq, K] → [batch, K]
        scores_lq_scaled = scores_head_lse / self.tau_lq
        ranking_logits = torch.logsumexp(scores_lq_scaled, dim=1) * self.tau_lq  # [batch, K]
        
        # Apply final mask to ranking logits
        ranking_logits = ranking_logits.masked_fill(~pool_padding_mask, -1e4)
        
        # Debug outputs
        ca_raw_scores_avg = ca_raw_scores_masked.mean(dim=1)  # [batch, N_lq, K]
        
        return {
            "ranking_logits": ranking_logits,  # [batch, K]
            "ca_raw_scores_avg": ca_raw_scores_avg.detach(),  # [batch, N_lq, K]
        }
    
    def _apply_drop_lq(self, ca_scores: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply Drop-LQ regularization with safety protection.
        
        Args:
            ca_scores: [batch, N_lq, K] attention scores
            mask: [batch, N_lq, 1] optional external mask (True=keep, False=drop)
                 If None, generate internal random mask
        
        Returns:
            ca_scores_dropped: [batch, N_lq, K] with some LQs masked to -1e4
        
        Implementation:
        - Randomly mask out p_drop_lq proportion of LQ tokens
        - Masked LQs are set to -1e4 (effectively zero after softmax/LSE)
        - Safety: ensure at least one LQ remains active per sample
        """
        batch_size, n_lqs, k = ca_scores.shape
        device = ca_scores.device
        
        if mask is not None:
            # Use external unified mask (for multi-task training)
            mask_drop = mask.float()  # [batch, N_lq, 1], 1.0=keep, 0.0=drop
        else:
            # Generate internal dropout mask: [batch, N_lq, 1]
            # 1.0 = keep, 0.0 = drop
            mask_drop = torch.bernoulli(
                torch.full((batch_size, n_lqs, 1), 1.0 - self.p_drop_lq, device=device)
            )
            
            # Safety: prevent all LQs being dropped for any sample
            all_dropped = (mask_drop.sum(dim=1, keepdim=True) == 0)  # [batch, 1, 1]
            if all_dropped.any():
                # For samples with all LQs dropped, randomly keep one
                for b in range(batch_size):
                    if all_dropped[b, 0, 0]:
                        random_idx_tensor = torch.randint(0, n_lqs, (1,), device=device)
                        random_idx = int(random_idx_tensor.item())
                        mask_drop[b, random_idx, 0] = 1.0
        
        # Apply mask: set dropped LQs to large negative value
        ca_scores_dropped = ca_scores.clone()
        ca_scores_dropped = ca_scores_dropped + (1.0 - mask_drop) * (-1e4)
        
        return ca_scores_dropped
    
    def count_parameters(self) -> int:
        """Count trainable parameters (none - only hyperparameters)."""
        return 0


# Alias for backward compatibility
SortingHead = FragmentRankingHead


class CondenseHead(nn.Module):
    """
    Task C: Condensing-Generation (精炼-生成).
    
    Purpose: Learn to "condense and refine" - ensure Q-Former's extracted Z is useful 
             for frozen LLM generation.
    
    Architecture:
    - Input: Z from Q-Former [batch, N, d]
    - Output: N condensed vectors [batch, N, d_llm] fed as soft prompt prefix to LLM
    - Projection layer to match LLM's embedding dimension if needed
    
    Training Modes:
    ===============
    Primal (QA) - Contrastive Generation Loss:
      - Generate with evidence: answer_with_Z = LLM(Query, Z)
      - Generate without evidence: answer_baseline = LLM(Query, Empty_Z)
      - Maximize: Reward(answer_with_Z) - Reward(answer_baseline)
      - Reward based on ROUGE/BLEU/EM with gold answer
      - Forces Q-Former to extract evidence-dependent information
    
    Dual (QG) - Reward Loss:
      - Generate query from answer: query' = LLM(Answer, Z)
      - Maximize: Similarity(query', gold_query)
      - Similarity based on BLEU/ROUGE/BERTScore
      - Forces Q-Former to extract query-relevant information
    
    Loss:
    - Reward Margin Loss: max(0, margin - (reward_high - reward_low))
    - Encourages evidence-dependent generation
    
    Metrics:
    - ROUGE-L, BLEU, Exact Match (EM), F1
    
    Args:
        hidden_dim (int): Q-Former hidden dimension. Default: 768
        llm_hidden_dim (int): LLM hidden dimension. Default: None (uses hidden_dim)
    
    Input:
        z: Q-Former output [batch, N, d]
        
    Output:
        prefix_embeds: [batch, N, d_llm] soft prompt embeddings for LLM conditioning
    """
    
    def __init__(
        self, 
        hidden_dim: int = 768, 
        llm_hidden_dim: Optional[int] = None,
        p_drop_lq: float = 0.1,  # Drop-LQ probability (0.0 = disabled)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.llm_hidden_dim = llm_hidden_dim or hidden_dim
        self.p_drop_lq = p_drop_lq
        
        # Projection to LLM embedding dimension
        if hidden_dim != self.llm_hidden_dim:
            self.proj = nn.Linear(hidden_dim, self.llm_hidden_dim)
        else:
            self.proj = nn.Identity()
        
        # Layer normalization for stable soft prompt embeddings
        self.norm = nn.LayerNorm(self.llm_hidden_dim)
        
        print(f"CondenseHead initialized:")
        print(f"  - hidden_dim: {hidden_dim}")
        print(f"  - llm_hidden_dim: {self.llm_hidden_dim}")
        print(f"  - p_drop_lq (Drop-LQ probability): {p_drop_lq}")
    
    def forward(
        self, 
        z: Optional[Tensor] = None,
        lq_drop_mask: Optional[Tensor] = None,
        training: bool = True
    ) -> Optional[Tensor]:
        """
        Forward pass for condensing to LLM prefix.
        
        Args:
            z: Q-Former output [batch, N, d]
            lq_drop_mask: [batch, N, 1] bool mask for unified Drop-LQ
                         True = keep LQ, False = drop LQ
                         If None, internal Drop-LQ is used (if enabled)
            training: Whether in training mode (affects Drop-LQ)
        
        Returns:
            prefix_embeds: [batch, N, d_llm] for LLM soft prompt conditioning
        
        Usage with Frozen LLM:
        ======================
        1. Q-Former extracts Z: [batch, N, d]
        2. CondenseHead projects to d_llm: [batch, N, d_llm]
        3. Frozen LLM receives:
           - Soft prompt: prefix_embeds [batch, N, d_llm]
           - Hard text: query_tokens [batch, seq_q, d_llm]
           - Combined input: [batch, N+seq_q, d_llm]
        4. LLM generates answer conditioning on both soft and hard prompts
        
        Implementation:
        ===============
        1. Apply Drop-LQ regularization (training only)
        2. Project Z to LLM dimension: proj(z) → [batch, N, d_llm]
        3. Apply layer norm for stable soft prompt embeddings
        4. Return prefix_embeds for LLM injection
        
        Note: The actual LLM prefix injection is handled by FrozenLLM adapter,
              not by this head. This head only prepares the embeddings.
        """
        if z is None:
            return None
        
        # Project to LLM dimension
        prefix_embeds = self.proj(z)  # [batch, N, d_llm]
        
        # Normalize for stability
        prefix_embeds = self.norm(prefix_embeds)
        
        # Apply Drop-LQ regularization (training only)
        # IMPORTANT: Apply AFTER projection and norm to ensure dropped LQs remain zero
        # (Applying before projection would result in non-zero outputs due to bias)
        if training:
            if lq_drop_mask is not None:
                # Use external unified mask (for multi-task training)
                prefix_embeds = self._apply_drop_lq(prefix_embeds, mask=lq_drop_mask)
            elif self.p_drop_lq > 0:
                # Use internal random mask (for single-task training)
                prefix_embeds = self._apply_drop_lq(prefix_embeds, mask=None)
        
        return prefix_embeds
    
    def _apply_drop_lq(self, z: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply Drop-LQ regularization with safety protection.
        
        Args:
            z: [batch, N_lq, d] Q-Former output embeddings
            mask: [batch, N_lq, 1] optional external mask (True=keep, False=drop)
                 If None, generate internal random mask
        
        Returns:
            z_dropped: [batch, N_lq, d] with some LQs zeroed out
        
        Implementation:
        - Randomly zero out p_drop_lq proportion of LQ embeddings
        - Safety: ensure at least one LQ remains active per sample
        - Uses binary mask (0 or 1) instead of setting to -1e4
        """
        batch_size, n_lqs, d = z.shape
        device = z.device
        
        if mask is not None:
            # Use external unified mask (for multi-task training)
            mask_drop = mask.float()  # [batch, N_lq, 1], 1.0=keep, 0.0=drop
        else:
            # Generate internal dropout mask: [batch, N_lq, 1]
            # 1.0 = keep, 0.0 = drop
            mask_drop = torch.bernoulli(
                torch.full((batch_size, n_lqs, 1), 1.0 - self.p_drop_lq, device=device)
            )
            
            # Safety: prevent all LQs being dropped for any sample
            all_dropped = (mask_drop.sum(dim=1, keepdim=True) == 0)  # [batch, 1, 1]
            if all_dropped.any():
                # For samples with all LQs dropped, randomly keep one
                for b in range(batch_size):
                    if all_dropped[b, 0, 0]:
                        random_idx_tensor = torch.randint(0, n_lqs, (1,), device=device)
                        random_idx = int(random_idx_tensor.item())
                        mask_drop[b, random_idx, 0] = 1.0
        
        # Apply mask: zero out dropped LQs
        z_dropped = z * mask_drop
        
        return z_dropped
    
    def count_parameters(self) -> int:
        """Count trainable parameters in CondenseHead."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return None
