"""
DR-QFormer: BLIP-2-style Query Transformer for RAG.

Cross-Attention Architecture (Online, Query-Relevant):
- N learnable query tokens (LQs) - trainable parameters
- Stage 1: Self-Attention (SA) - LQs fuse with q_embed or a_embed
  * Input: Concat([LQs, q_embed/a_embed]) - length N+1
  * Mask: (N+1) x (N+1) all-ones (bidirectional)
  * Output: LQs_aware [N, d] (query/answer-aware LQs)
- Stage 2: Cross-Attention (CA) - LQs_aware attend to fragment embeddings
  * Query: LQs_aware [N, d]
  * Key/Value: P_embeds [k, d] from frozen retriever
  * Mask: N x k all-ones (each LQ attends to all k fragments)
  * Output: Z [N, d] (knowledge-infused representations)
- Stage 3: Feed-Forward Network (FFN)
  * Output: Z_final [N, d] → fed to task heads or frozen LLM

Training:
- Primal Mode (QA): Receives query embedding, predicts answer-relevant info
- Dual Mode (QG): Receives answer embedding, predicts query-relevant info
- Implicit Dual Constraint: Same parameters trained by both modes
"""

from typing import Optional, Tuple
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
except ImportError:
    # Dummy fallback
    class nn:
        class Module:
            pass
    Tensor = None


class DRQFormer(nn.Module):
    """
    Parameter-efficient Q-Former for RAG middleware (Cross-Attention Architecture).
    
    This module serves as the ONLY trainable reasoning bridge between frozen retriever
    and frozen LLM, operating in online, query-relevant mode.
    
    Architecture (3 Stages):
    =======================
    Stage 1: Self-Attention (SA) - Make LQs query/answer-aware
      - Input: Concat([LQs, q_embed]) for Primal OR Concat([LQs, a_embed]) for Dual
      - Sequence Length: N (LQs) + 1 (query or answer embedding)
      - Mask: (N+1) x (N+1) all-ones matrix (fully bidirectional)
      - Output: LQs_aware [batch, N, d] - contextualized with query/answer
    
    Stage 2: Cross-Attention (CA) - Extract knowledge from fragments
      - Query: LQs_aware [batch, N, d]
      - Key/Value: P_embeds [batch, k, d] from frozen retriever (k fragment embeddings)
      - Mask: N x k all-ones matrix (each LQ attends to all k fragments)
      - Output: Z [batch, N, d] - knowledge-infused representations
    
    Stage 3: Feed-Forward Network (FFN)
      - Standard FFN with layer norm and residual connections
      - Output: Z_final [batch, N, d] → fed to task heads or frozen LLM
    
    Training Modes:
    ===============
    - Primal (QA): Q-Former receives query embedding, predicts answer-relevant info
    - Dual (QG): Q-Former receives answer embedding, predicts query-relevant info
    - Implicit Dual Constraint: Same parameters updated by both mode gradients
    
    Args:
        n_queries (int): Number of learnable query tokens (LQs). Default: 32
        hidden_dim (int): Hidden dimension (d). Default: 768
        num_layers (int): Number of transformer layers. Default: 6
        num_heads (int): Number of attention heads. Default: 8
        max_fragments (int): Maximum k retrieved fragments for CA. Default: 10
        dropout (float): Dropout rate. Default: 0.1
    
    Frozen components (external):
    =============================
    - Retriever (~100-400M params): Contriever, DPR, E5, etc.
    - LLM (~1-10B params): LLaMA, Mistral, Phi, etc.
    
    Trainable components:
    =====================
    - Q-Former parameters (~40-80M params): THIS module
    - Task-specific heads (~1-10M params): EntailmentHead, SortingHead, CondenseHead
    
    Total trainable: ~50-90M params (~1-2% of full system)
    """
    
    def __init__(
        self,
        n_queries: int = 32,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        max_fragments: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_fragments = max_fragments
        
        # Learnable query tokens (LQs) - core trainable parameters
        # Shape: [1, N, d] - will be expanded to batch size during forward
        self.query_tokens = nn.Parameter(torch.randn(1, n_queries, hidden_dim))
        
        # Initialize with small values for stable training (following BLIP-2)
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Stack of Q-Former transformer layers (SA + CA + FFN)
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization (following BLIP-2 design)
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        # Optional: Temperature parameter for Task E (entailment) similarity scaling
        # (similar to BLIP-2's ITC task)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        print(f"DR-QFormer initialized: {self.count_parameters():,} trainable parameters")
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        query_embeds: Optional[Tensor] = None,
        answer_embeds: Optional[Tensor] = None,
        p_embeds: Optional[Tensor] = None,
        sa_mask: Optional[Tensor] = None,
        ca_mask: Optional[Tensor] = None,
        pool_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """
        Forward pass through Q-Former (3-stage cross-attention architecture).
        
        Training Modes:
        - Primal (QA): Pass query_embeds, leave answer_embeds=None
        - Dual (QG): Pass answer_embeds, leave query_embeds=None
        
        Args:
            query_embeds: Query embedding [batch, 1, d] for Primal mode (QA)
            answer_embeds: Answer embedding [batch, 1, d] for Dual mode (QG)
            p_embeds: Fragment embeddings [batch, k, d] from frozen retriever
            sa_mask: Self-attention mask [batch, N+1, N+1] (optional, default: all-ones)
            ca_mask: Cross-attention mask [batch, N, k] (optional, default: all-ones)
        
        Returns:
            z: Contextualized LQ representations [batch, N, d] → fed to task heads
            aux: Auxiliary outputs dict containing:
                - 'sa_attn_weights': Self-attention weights [batch, num_heads, N+1, N+1]
                - 'ca_attn_weights': Cross-attention weights [batch, num_heads, N, k]
                - 'layer_outputs': List of intermediate layer outputs
        
        Forward Pass Stages:
        ====================
        Stage 1: Self-Attention (SA)
          1. Expand LQs to batch size: [batch, N, d]
          2. Concatenate: [LQs, query_embeds] or [LQs, answer_embeds] → [batch, N+1, d]
          3. Apply SA layers: LQs and q/a_embed attend to each other bidirectionally
          4. Extract LQs_aware: [batch, N, d] (first N tokens)
        
        Stage 2: Cross-Attention (CA)
          5. Query: LQs_aware [batch, N, d]
          6. Key/Value: p_embeds [batch, k, d]
          7. Apply CA layers: Each LQ attends to all k fragments
          8. Output: Z [batch, N, d] (knowledge-infused)
        
        Stage 3: Feed-Forward Network (FFN)
          9. Apply FFN with residual connections
          10. Output: Z_final [batch, N, d]
        
        TODO:
        - Expand self.query_tokens (learnable LQs) to batch size
        - Concatenate [LQs, query_embeds or answer_embeds] for SA input
        - Implement N transformer layers with SA + CA + FFN
        - Extract and return LQ representations (not the q/a_embed part)
        - Store attention weights in aux dict for visualization/analysis
        - Handle padding mask for variable-length P_embeds
        - Support gradient checkpointing for memory efficiency
        """
        # Determine mode and get conditioning embedding
        if query_embeds is not None and answer_embeds is not None:
            raise ValueError("Only one of query_embeds or answer_embeds should be provided")
        
        if query_embeds is None and answer_embeds is None:
            raise ValueError("Either query_embeds or answer_embeds must be provided")
        
        # Select conditioning embedding (query for Primal QA, answer for Dual QG)
        cond_embed = query_embeds if query_embeds is not None else answer_embeds
        batch_size = cond_embed.size(0)
        
        # Expand learnable query tokens to batch size
        # [1, N, d] → [batch, N, d]
        lqs = self.query_tokens.expand(batch_size, -1, -1).clone()
        
        # Stage 1 Input: Concatenate [LQs, q/a_embed]
        # [batch, N, d] + [batch, 1, d] → [batch, N+1, d]
        x = torch.cat([lqs, cond_embed], dim=1)
        
        # Default masks: all-ones (full attention)
        if sa_mask is None:
            # Self-attention: (N+1) x (N+1) all-ones (bidirectional)
            # In PyTorch, None means no masking (full attention)
            sa_mask = None
        
        if ca_mask is None and p_embeds is not None:
            # Cross-attention: N x k all-ones (each LQ attends to all fragments)
            ca_mask = None
        
        # Pass through Q-Former layers
        aux = {
            'layer_outputs': [],
            'sa_attn_weights': [],  # List of [batch, num_heads, N+1, N+1] per layer
            'ca_attn_weights': [],  # List of [batch, num_heads, N, k] per layer
            'ca_raw_scores_per_head': [],  # List of [batch, num_heads, N, k] pre-softmax per layer
            'ca_raw_scores_avg': []  # List of [batch, N, k] head-averaged raw scores per layer
        }
        
        for layer_idx, layer in enumerate(self.layers):
            x, layer_aux = layer(
                x=x,
                context=p_embeds,
                sa_mask=sa_mask,
                ca_mask=ca_mask,
                pool_padding_mask=pool_padding_mask
            )
            aux['layer_outputs'].append(x.clone())
            aux['sa_attn_weights'].append(layer_aux['sa_attn_weights'])
            aux['ca_attn_weights'].append(layer_aux['ca_attn_weights'])
            aux['ca_raw_scores_per_head'].append(layer_aux['ca_raw_scores_per_head'])
            aux['ca_raw_scores_avg'].append(layer_aux['ca_raw_scores_avg'])
        
        # Extract final LQ representations (first N tokens)
        # Discard the q/a_embed token (last token)
        z = x[:, :self.n_queries, :]  # [batch, N, d]
        
        # Final layer normalization (following BLIP-2)
        z = self.final_ln(z)
        
        # Store final outputs in aux
        aux['z_raw'] = x  # Full sequence including q/a_embed
        aux['z_final'] = z  # Normalized LQ representations
        
        return z, aux
    
    def get_trainable_params(self):
        """Returns iterator over trainable parameters."""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def freeze(self):
        """Freeze all parameters (shouldn't be called normally)."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all parameters (default behavior)."""
        for param in self.parameters():
            param.requires_grad = True


class QFormerLayer(nn.Module):
    """
    Single Q-Former transformer layer with SA and CA (Cross-Attention Architecture).
    
    This layer implements the core reasoning mechanism:
    1. Self-Attention (SA): Fuse LQs with query/answer embeddings
    2. Cross-Attention (CA): LQs attend to fragment embeddings from retriever
    3. Feed-Forward Network (FFN): Non-linear transformation
    
    Each sublayer uses:
    - Pre-LayerNorm (normalized before attention/FFN)
    - Residual connections (skip connections)
    - Dropout for regularization
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Self-Attention sublayer (Stage 1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-Attention sublayer (Stage 2)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-Forward Network sublayer (Stage 3)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        sa_mask: Optional[Tensor] = None,
        ca_mask: Optional[Tensor] = None,
        pool_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """
        Forward pass through one Q-Former layer.
        
        Args:
            x: Input sequence [batch, N+1, d] - Concat([LQs, q/a_embed])
            context: Fragment embeddings [batch, k, d] from frozen retriever (P_embeds)
            sa_mask: Self-attention mask [batch, N+1, N+1] or [N+1, N+1]
                    (0 = can attend, True/-inf = cannot attend)
            ca_mask: Cross-attention mask [batch, N, k] or [N, k]
                    (0 = can attend, True/-inf = cannot attend)
            pool_padding_mask: Bool [batch, k] - True=valid fragment, False=padding
        
        Returns:
            Output sequence [batch, N+1, d]
            Auxiliary dict with attention weights:
                - sa_attn_weights: [batch, num_heads, N+1, N+1]
                - ca_attn_weights: [batch, num_heads, N, k] or None
                - ca_raw_scores_per_head: [batch, num_heads, N, k] pre-softmax logits
                - ca_raw_scores_avg: [batch, N, k] head-averaged raw scores
        """
        # Stage 1: Self-Attention (SA) - LQs fuse with q/a_embed
        # Input: Concat([LQs, q/a_embed]) [batch, N+1, d]
        x_norm = self.ln1(x)
        
        # PyTorch MultiheadAttention expects attn_mask to be additive (0 or -inf)
        # or boolean (False=attend, True=ignore)
        # need_weights=True, average_attn_weights=False to get per-head weights
        sa_out, sa_weights = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=sa_mask,
            need_weights=True,
            average_attn_weights=False  # Return per-head weights [batch, num_heads, seq, seq]
        )
        x = x + self.dropout1(sa_out)  # Residual connection
        
        # Extract LQs_aware (first N tokens) for Cross-Attention
        # Last token is the q/a_embed, which we preserve
        batch_size, seq_len, d = x.shape
        n_lqs = seq_len - 1  # N learnable queries
        
        lqs_aware = x[:, :n_lqs, :]  # [batch, N, d]
        qa_embed = x[:, n_lqs:, :]   # [batch, 1, d] - preserve q/a embedding
        
        # Stage 2: Cross-Attention (CA) - LQs_aware attend to fragments
        # Compute raw CA scores (pre-softmax) manually for Task E
        ca_weights = None
        ca_raw_scores_per_head = None
        ca_raw_scores_avg = None
        
        if context is not None:
            lqs_norm = self.ln2(lqs_aware)
            
            # Manual CA computation to expose pre-softmax scores
            batch_size = lqs_norm.size(0)
            N_lq = lqs_norm.size(1)
            K_pool = context.size(1)
            d_head = self.hidden_dim // self.num_heads
            
            # Linear projections (Q, K, V)
            # MultiheadAttention has in_proj_weight of shape [3*hidden_dim, hidden_dim]
            # Split into Q, K, V weights
            in_proj_weight = self.cross_attn.in_proj_weight
            in_proj_bias = self.cross_attn.in_proj_bias
            
            # Project query (lqs_norm)
            q = torch.nn.functional.linear(lqs_norm, in_proj_weight[:self.hidden_dim], 
                                           in_proj_bias[:self.hidden_dim] if in_proj_bias is not None else None)
            # Project key (context)
            k = torch.nn.functional.linear(context, in_proj_weight[self.hidden_dim:2*self.hidden_dim],
                                           in_proj_bias[self.hidden_dim:2*self.hidden_dim] if in_proj_bias is not None else None)
            # Project value (context)
            v = torch.nn.functional.linear(context, in_proj_weight[2*self.hidden_dim:],
                                           in_proj_bias[2*self.hidden_dim:] if in_proj_bias is not None else None)
            
            # Reshape to [batch, num_heads, seq, d_head]
            q = q.view(batch_size, N_lq, self.num_heads, d_head).transpose(1, 2)  # [B, H, N, d_head]
            k = k.view(batch_size, K_pool, self.num_heads, d_head).transpose(1, 2)  # [B, H, K, d_head]
            v = v.view(batch_size, K_pool, self.num_heads, d_head).transpose(1, 2)  # [B, H, K, d_head]
            
            # Compute raw attention scores: Q @ K^T / sqrt(d_head)
            ca_raw_scores_per_head = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)  # [B, H, N, K]
            
            # Apply pool_padding_mask before softmax (mask padded keys to -1e4)
            if pool_padding_mask is not None:
                # Expand mask: [B, K] -> [B, 1, 1, K] for broadcasting
                mask_expanded = pool_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K]
                ca_raw_scores_per_head = ca_raw_scores_per_head.masked_fill(~mask_expanded, -1e4)
            
            # Apply ca_mask if provided (additional mask on top of padding)
            if ca_mask is not None:
                # ca_mask should be broadcastable to [B, H, N, K]
                ca_raw_scores_per_head = ca_raw_scores_per_head + ca_mask
            
            # Compute head-averaged raw scores for convenience
            ca_raw_scores_avg = ca_raw_scores_per_head.mean(dim=1)  # [B, N, K]
            
            # Apply softmax to get attention weights
            ca_weights = torch.nn.functional.softmax(ca_raw_scores_per_head, dim=-1)  # [B, H, N, K]
            
            # Compute attention output: weights @ V
            ca_out = torch.matmul(ca_weights, v)  # [B, H, N, d_head]
            ca_out = ca_out.transpose(1, 2).contiguous().view(batch_size, N_lq, self.hidden_dim)  # [B, N, d]
            
            # Apply output projection
            ca_out = torch.nn.functional.linear(ca_out, self.cross_attn.out_proj.weight, 
                                                self.cross_attn.out_proj.bias)
            
            lqs_aware = lqs_aware + self.dropout2(ca_out)  # Residual connection
        
        # Stage 3: Feed-Forward Network (FFN)
        lqs_norm2 = self.ln3(lqs_aware)
        ffn_out = self.ffn(lqs_norm2)
        lqs_aware = lqs_aware + self.dropout3(ffn_out)  # Residual connection
        
        # Concatenate back: [LQs_aware, q/a_embed]
        x = torch.cat([lqs_aware, qa_embed], dim=1)  # [batch, N+1, d]
        
        # Return auxiliary outputs with attention weights and raw scores
        layer_aux = {
            'sa_attn_weights': sa_weights,  # [batch, num_heads, N+1, N+1]
            'ca_attn_weights': ca_weights,  # [batch, num_heads, N, k] or None (post-softmax)
            'ca_raw_scores_per_head': ca_raw_scores_per_head,  # [batch, num_heads, N, k] pre-softmax
            'ca_raw_scores_avg': ca_raw_scores_avg  # [batch, N, k] head-averaged raw scores
        }
        
        return x, layer_aux
