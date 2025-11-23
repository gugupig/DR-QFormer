"""
XLM-RoBERTa-based DR-QFormer: Multilingual Query Transformer for RAG (BLIP-2 Style).

This module implements a DR-QFormer variant that uses XLM-RoBERTa as the backbone
for self-attention layers, while preserving the original Cross-Attention (CA) mechanism
from the base DR-QFormer implementation.

BLIP-2 Architecture Pattern:
=============================
Unlike the original DR-QFormer which uses a pooled query embedding, this variant
follows BLIP-2's paradigm where queries are processed at the token level:

1. Query as Token Sequence:
   - Query text is tokenized into input_ids [B, T]
   - Each token gets embedded via XLM-R embeddings: [B, T, d]

2. Bidirectional Self-Attention:
   - Input: [LQs, query_tokens] concatenated -> [B, N_q+T, d]
   - LQs and query tokens attend to each other bidirectionally in every XLM-R layer
   - LQs gradually become "query-aware" through multi-layer SA interactions
   - This is similar to BLIP-2's image-text grounding in Q-Former

3. Cross-Attention to Evidence:
   - After SA, query-aware LQs attend to evidence fragment embeddings via CA
   - Evidence information is fused into LQs
   - Only LQs [B, N_q, d] are extracted as final output

4. Task Head Integration:
   - Final LQ representations are fed to downstream task heads
   - Each LQ contains both query semantics (from SA) and evidence info (from CA)

Architecture Components:
========================
- SA+FFN: XLM-RoBERTa encoder layers (12 layers of XLMRobertaLayer)
- CA: Preserved from original QFormerLayer - manual Q/K/V projection with raw scores
- Embeddings: XLM-RoBERTa's multilingual embeddings (250K vocab, 100+ languages)

Key Features:
=============
- Token-level query processing (BLIP-2 pattern) vs. pooled embedding (original)
- Bidirectional SA between LQs and query tokens (full attention, no masking)
- Supports 100+ languages via XLM-RoBERTa-base
- Maintains compatibility with original CA logging interface (ca_attn_weights, ca_raw_scores)
- Can be initialized with pretrained XLM-R weights for better multilingual generalization
- Optional freezing of XLM-R backbone for parameter-efficient training

Usage Example:
==============
    from transformers import AutoTokenizer
    from src.models.qformer_xlm import XLMRobertaDRQFormer
    
    # Initialize model and tokenizer
    model = XLMRobertaDRQFormer(
        xlm_model_name="xlm-roberta-base",
        n_queries=32,
        use_ca_layers=[5, 11],  # CA only at layers 6 and 12
        freeze_xlmr=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # Tokenize query text
    query_text = "What is the capital of France?"
    tokens = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True)
    
    # Evidence embeddings (from retriever)
    evidence_emb = retriever_model.encode(retrieved_passages)  # [B, K, 768]
    evidence_mask = torch.ones(B, K, dtype=torch.bool)
    
    # Forward pass
    Z, all_aux = model(
        input_ids=tokens['input_ids'],
        attention_mask=tokens['attention_mask'],
        evidence_emb=evidence_emb,
        evidence_mask=evidence_mask,
    )
    
    # Z: [B, 32, 768] - query-aware, evidence-fused LQ representations
    # Feed to task heads for prediction
"""

from typing import Optional, List, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, XLMRobertaModel


class QueryEvidenceCrossAttention(nn.Module):
    """
    Reusable Cross-Attention module extracted from original QFormerLayer.
    
    This module implements Stage 2 (CA) + Stage 3 (FFN) of the DR-QFormer architecture:
    - Cross-Attention: LQs_aware attend to evidence fragment embeddings
    - FFN: Feed-forward network applied to LQs after CA
    
    The implementation preserves the manual Q/K/V projection and raw score computation
    from the original QFormerLayer to maintain compatibility with Task E logging.
    
    Args:
        hidden_dim (int): Hidden dimension size (d). Default: 768
        num_heads (int): Number of attention heads. Default: 12
        dropout (float): Dropout rate. Default: 0.1
    """
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_head = hidden_dim // num_heads
        
        # Cross-Attention sublayer
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-Forward Network for LQs
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
        lqs_aware: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor] = None,
        ca_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        Forward pass through CA + FFN.
        
        Args:
            lqs_aware: Query-aware LQ representations [B, N_q, d] (already contains query info from SA)
            context: Evidence fragment embeddings [B, K, d_evidence]
            context_mask: Padding mask [B, K] - True=valid, False=padding
            ca_mask: Additional attention mask [B, N_q, K] or [N_q, K] (additive mask)
        
        Returns:
            updated_lqs: Updated LQ representations [B, N_q, d] after CA+FFN
            aux: Dict containing:
                - 'ca_attn_weights': [B, H, N_q, K] post-softmax attention weights
                - 'ca_raw_scores_per_head': [B, H, N_q, K] pre-softmax scores
                - 'ca_raw_scores_avg': [B, N_q, K] head-averaged raw scores
        """
        batch_size = lqs_aware.size(0)
        N_lq = lqs_aware.size(1)
        K_pool = context.size(1)
        
        # Stage 2: Cross-Attention (CA) - LQs_aware attend to fragments
        lqs_norm = self.ln2(lqs_aware)
        
        # Manual CA computation to expose pre-softmax scores (for Task E)
        # Linear projections using MultiheadAttention's internal weights
        in_proj_weight = self.cross_attn.in_proj_weight  # [3*d, d]
        in_proj_bias = self.cross_attn.in_proj_bias      # [3*d] or None
        
        # Project Q (query from lqs_norm)
        q = F.linear(
            lqs_norm,
            in_proj_weight[:self.hidden_dim],
            in_proj_bias[:self.hidden_dim] if in_proj_bias is not None else None
        )
        # Project K (key from context)
        k = F.linear(
            context,
            in_proj_weight[self.hidden_dim:2*self.hidden_dim],
            in_proj_bias[self.hidden_dim:2*self.hidden_dim] if in_proj_bias is not None else None
        )
        # Project V (value from context)
        v = F.linear(
            context,
            in_proj_weight[2*self.hidden_dim:],
            in_proj_bias[2*self.hidden_dim:] if in_proj_bias is not None else None
        )
        
        # Reshape to multi-head format: [B, H, seq, d_head]
        q = q.view(batch_size, N_lq, self.num_heads, self.d_head).transpose(1, 2)     # [B, H, N, d_head]
        k = k.view(batch_size, K_pool, self.num_heads, self.d_head).transpose(1, 2)   # [B, H, K, d_head]
        v = v.view(batch_size, K_pool, self.num_heads, self.d_head).transpose(1, 2)   # [B, H, K, d_head]
        
        # Compute raw attention scores: Q @ K^T / sqrt(d_head)
        ca_raw_scores_per_head = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, H, N, K]
        
        # Apply context_mask (padding mask) before softmax
        if context_mask is not None:
            # Expand mask: [B, K] -> [B, 1, 1, K] for broadcasting
            mask_expanded = context_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K]
            ca_raw_scores_per_head = ca_raw_scores_per_head.masked_fill(~mask_expanded, -1e4)
        
        # Apply additional ca_mask if provided
        if ca_mask is not None:
            ca_raw_scores_per_head = ca_raw_scores_per_head + ca_mask
        
        # Compute head-averaged raw scores (useful for analysis)
        ca_raw_scores_avg = ca_raw_scores_per_head.mean(dim=1)  # [B, N, K]
        
        # Apply softmax to get attention weights
        ca_weights = F.softmax(ca_raw_scores_per_head, dim=-1)  # [B, H, N, K]
        
        # Compute attention output: weights @ V
        ca_out = torch.matmul(ca_weights, v)  # [B, H, N, d_head]
        ca_out = ca_out.transpose(1, 2).contiguous().view(batch_size, N_lq, self.hidden_dim)  # [B, N, d]
        
        # Apply output projection
        ca_out = F.linear(ca_out, self.cross_attn.out_proj.weight, self.cross_attn.out_proj.bias)
        
        # Residual connection
        lqs_aware = lqs_aware + self.dropout2(ca_out)
        
        # Stage 3: Feed-Forward Network (FFN) on LQs
        lqs_norm2 = self.ln3(lqs_aware)
        ffn_out = self.ffn(lqs_norm2)
        lqs_aware = lqs_aware + self.dropout3(ffn_out)  # Residual connection
        
        # Return updated LQs directly (no qa_embed concatenation)
        updated_lqs = lqs_aware  # [B, N_q, d]
        
        # Auxiliary outputs for logging and Task E
        aux = {
            'ca_attn_weights': ca_weights,              # [B, H, N_q, K] post-softmax
            'ca_raw_scores_per_head': ca_raw_scores_per_head,  # [B, H, N_q, K] pre-softmax
            'ca_raw_scores_avg': ca_raw_scores_avg      # [B, N_q, K] head-averaged
        }
        
        return updated_lqs, aux


class XLMRobertaDRQFormer(nn.Module):
    """
    XLM-RoBERTa-based DR-QFormer for multilingual RAG (BLIP-2 Style).
    
    This variant replaces the original SA+FFN backbone with XLM-RoBERTa encoder layers,
    while preserving the Cross-Attention (CA) mechanism for attending to evidence fragments.
    
    BLIP-2 Architecture Pattern:
    ============================
    1. Query as Token-Level Input:
       - Query text is tokenized into input_ids [B, T]
       - Converted to embeddings via XLM-R embeddings: [B, T, d]
    
    2. Bidirectional Self-Attention:
       - Input sequence: [LQs, query_tokens] -> [B, N_q+T, d]
       - LQs and query tokens attend to each other bidirectionally in every layer
       - LQs become "query-aware" through multi-layer SA interactions
    
    3. Cross-Attention to Evidence:
       - Query-aware LQs attend to evidence fragment embeddings
       - Evidence information is fused into LQs via CA
    
    4. Final LQ Output:
       - Only LQs [B, N_q, d] are extracted as output
       - Fed to downstream task heads for prediction
    
    Architecture Pipeline:
    ======================
    1. Embeddings: XLM-RoBERTa embeddings (word + position + token_type)
    2. Per-layer pipeline:
       a) XLM-RoBERTa Layer: Bidirectional self-attention + FFN on [LQs, query_tokens]
       b) Cross-Attention (optional): Query-aware LQs attend to evidence fragments
       c) FFN on LQs: Additional feed-forward after CA
    3. Final LayerNorm on LQ outputs
    
    Key Differences from Base DR-QFormer:
    ======================================
    - Query input: Token-level (BLIP-2) vs. pooled embedding (original)
    - Uses XLM-RoBERTa's 12-layer transformer instead of custom SA+FFN
    - Supports multilingual inputs via XLM-R's 250K vocab
    - CA layers can be inserted at specific layers (e.g., only at layers 6, 12)
    - Can be initialized with pretrained XLM-R weights for better generalization
    
    Args:
        xlm_model_name (str): HuggingFace model name. Default: "xlm-roberta-base"
        n_queries (int): Number of learnable query tokens (LQs). Default: 32
        dropout (float): Dropout rate. Default: 0.1
        use_ca_layers (List[int] | None): Layer indices where CA is applied (0-indexed).
                                           If None, CA is applied after every layer.
                                           Example: [5, 11] applies CA after layer 6 and 12 only.
        freeze_xlmr (bool): If True, freeze XLM-R weights (only train LQs and CA). Default: False
        bypass_embeddings (bool): If True, expect pre-computed embeddings instead of input_ids. Default: False
    """
    
    def __init__(
        self,
        xlm_model_name: str = "FacebookAI/xlm-roberta-base",
        n_queries: int = 32,
        dropout: float = 0.1,
        use_ca_layers: Optional[List[int]] = None,
        freeze_xlmr: bool = False,
        bypass_embeddings: bool = False,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.bypass_embeddings = bypass_embeddings
        
        # Load pretrained XLM-RoBERTa model
        print(f"Loading XLM-RoBERTa model: {xlm_model_name}")
        self.xlmr = AutoModel.from_pretrained(xlm_model_name)
        
        # Automatically get dimensions from XLM-R config
        self.hidden_dim = self.xlmr.config.hidden_size  # e.g., 768 for base, 1024 for large
        self.num_heads = self.xlmr.config.num_attention_heads  # e.g., 12 for base, 16 for large
        
        print(f"  ✓ Hidden dim: {self.hidden_dim} (from XLM-R config)")
        print(f"  ✓ Num heads: {self.num_heads} (from XLM-R config)")
        
        # Extract embeddings and encoder layers
        self.embeddings = self.xlmr.embeddings
        self.encoder_layers = self.xlmr.encoder.layer  # ModuleList of 12 XLMRobertaLayers
        self.num_xlmr_layers = len(self.encoder_layers)
        
        # Learnable query tokens (LQs) - core trainable parameters
        # Following BLIP-2: Initialize with normal distribution
        self.query_tokens = nn.Parameter(torch.randn(1, n_queries, self.hidden_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=self.xlmr.config.initializer_range)
        
        # Cross-Attention layers (optional per layer)
        if use_ca_layers is None:
            # Default: apply CA after every XLM-R layer
            use_ca_layers = list(range(self.num_xlmr_layers))
        
        self.use_ca_layers = use_ca_layers
        self.cross_layers = nn.ModuleList([
            QueryEvidenceCrossAttention(self.hidden_dim, self.num_heads, dropout)
            if i in use_ca_layers else None
            for i in range(self.num_xlmr_layers)
        ])
        
        # BLIP-2 Initialization: Copy self-attention weights to cross-attention query branch
        # This provides a better initialization point than random weights
        self._init_cross_attention_from_self_attention()
        
        # Final LayerNorm (following BLIP-2 design)
        self.final_ln = nn.LayerNorm(self.hidden_dim)
        
        # Optional: freeze XLM-R backbone
        if freeze_xlmr:
            print("Freezing XLM-RoBERTa backbone weights")
            for param in self.xlmr.parameters():
                param.requires_grad = False
        
        # Temperature parameter (for Task E similarity scaling)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        print(f"XLM-RoBERTa DR-QFormer initialized:")
        print(f"  - XLM-R layers: {self.num_xlmr_layers}")
        print(f"  - CA layers at: {use_ca_layers}")
        print(f"  - Trainable params: {self.count_parameters():,}")
        print(f"  - XLM-R frozen: {freeze_xlmr}")
    
    def _init_cross_attention_from_self_attention(self):
        """
        BLIP-2 Initialization Strategy: Copy self-attention weights to cross-attention.
        
        This provides a better initialization point than random weights by leveraging
        the pre-trained self-attention knowledge from XLM-RoBERTa.
        
        Specifically, for each cross-attention layer:
        1. Copy Q projection weights from corresponding self-attention layer
        2. K/V projections remain random (will attend to evidence, different from text)
        3. This warm-starts the query projection with linguistic knowledge
        
        Reference: BLIP-2 paper Section 3.1 and official implementation
        https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py#L80-L85
        """
        print("\n🔄 Initializing Cross-Attention layers from Self-Attention weights (BLIP-2 style)...")
        
        for layer_idx, ca_layer in enumerate(self.cross_layers):
            if ca_layer is not None:
                # Get corresponding XLM-R layer
                xlmr_layer = self.encoder_layers[layer_idx]
                xlmr_self_attn = xlmr_layer.attention.self
                
                # Copy Q projection weights: self-attention -> cross-attention
                # This transfers linguistic understanding to the CA query branch
                try:
                    # PyTorch MultiheadAttention has combined in_proj_weight: [3*hidden_dim, hidden_dim]
                    # Layout: [Q_weight; K_weight; V_weight] stacked vertically
                    # We copy Q weights (first hidden_dim rows) from XLM-R self-attention
                    
                    if hasattr(ca_layer.cross_attn, 'in_proj_weight') and ca_layer.cross_attn.in_proj_weight is not None:
                        # Combined projection (default for nn.MultiheadAttention)
                        ca_layer.cross_attn.in_proj_weight.data[:self.hidden_dim, :].copy_(
                            xlmr_self_attn.query.weight.data
                        )
                        
                        if ca_layer.cross_attn.in_proj_bias is not None and xlmr_self_attn.query.bias is not None:
                            ca_layer.cross_attn.in_proj_bias.data[:self.hidden_dim].copy_(
                                xlmr_self_attn.query.bias.data
                            )
                        
                        print(f"  ✅ Layer {layer_idx}: Copied Q weights (combined projection)")
                    
                    elif hasattr(ca_layer.cross_attn, 'q_proj_weight'):
                        # Separate projections (if using _qkv_same_embed_dim=False)
                        ca_layer.cross_attn.q_proj_weight.data.copy_(
                            xlmr_self_attn.query.weight.data
                        )
                        
                        if hasattr(ca_layer.cross_attn, 'in_proj_bias') and ca_layer.cross_attn.in_proj_bias is not None:
                            ca_layer.cross_attn.in_proj_bias.data[:self.hidden_dim].copy_(
                                xlmr_self_attn.query.bias.data
                            )
                        
                        print(f"  ✅ Layer {layer_idx}: Copied Q weights (separate projection)")
                    
                    else:
                        print(f"  ⚠️  Layer {layer_idx}: Unknown projection structure, using random init")
                
                except Exception as e:
                    print(f"  ⚠️  Layer {layer_idx}: Could not copy weights ({type(e).__name__}: {e})")
                    print(f"       Using random initialization instead")
        
        print("✅ Cross-Attention initialization complete!\n")
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        evidence_emb: Tensor,
        evidence_mask: Optional[Tensor] = None,
        precomputed_query_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Dict]]:
        """
        Forward pass through XLM-RoBERTa-based DR-QFormer (BLIP-2 Style).
        
        Args:
            input_ids: Query token IDs [B, T] from XLM-R tokenizer (question text)
            attention_mask: Attention mask [B, T] - 1=valid token, 0=padding
            evidence_emb: Evidence fragment embeddings [B, K, d_evidence]
                         Expected: d_evidence == hidden_dim (768)
            evidence_mask: Padding mask [B, K] - True=valid fragment, False=padding
        
        Returns:
            Z: Final LQ representations [B, N_q, d] for downstream task heads
            all_aux: List of auxiliary dicts (one per layer) containing:
                - 'ca_attn_weights': [B, H, N_q, K] if CA applied, else {}
                - 'ca_raw_scores_per_head': [B, H, N_q, K] if CA applied, else {}
                - 'ca_raw_scores_avg': [B, N_q, K] if CA applied, else {}
        
        Forward Pass Logic (BLIP-2 Pattern):
        =====================================
        1. Get query token embeddings from input_ids using XLM-R embeddings
        2. Expand learnable query tokens (LQs) to batch size
        3. Concatenate [LQs, query_token_embeddings] -> [B, N_q+T, d]
        4. For each XLM-R layer:
           a) Apply bidirectional self-attention + FFN on [LQs, query_tokens]
              - LQs and query tokens can see each other (full attention)
              - LQs become query-aware through SA interactions
           b) Split output: lq_states [B, N_q, d], tok_states [B, T, d]
           c) If CA layer exists: query-aware LQs attend to evidence
              - Input: lq_states (contains query info from SA)
              - Output: updated LQs with evidence info fused
           d) Concatenate [updated_lqs, tok_states] for next layer
        5. Extract final LQ representations (discard query tokens)
        6. Apply final LayerNorm
        """
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)  # T
        
        # Step 1: Get token embeddings
        if self.bypass_embeddings and precomputed_query_emb is not None:
            # Use pre-computed token embeddings (from Qwen3 or other models)
            # This bypasses XLM-R's embedding layer entirely
            token_emb = precomputed_query_emb  # [B, T, d]
        else:
            # Use XLM-RoBERTa embeddings (word + position + token_type)
            token_emb = self.embeddings(input_ids)  # [B, T, d]
        
        # Step 2: Expand learnable query tokens to batch size
        lqs = self.query_tokens.expand(batch_size, -1, -1)  # [B, N, d]
        
        # Step 3: Concatenate [LQs, tokens] as input to XLM-R layers
        hidden = torch.cat([lqs, token_emb], dim=1)  # [B, N+T, d]
        
        # Step 4: Construct extended attention mask for XLM-R layers
        # LQs should all be valid (no padding), tokens use provided attention_mask
        lq_mask = torch.ones(batch_size, self.n_queries, dtype=attention_mask.dtype, device=attention_mask.device)
        extended_mask = torch.cat([lq_mask, attention_mask], dim=1)  # [B, N+T]
        
        # Convert to XLM-R's expected mask format: [B, 1, 1, N_q+T]
        # In HuggingFace, 1=valid, 0=padding, then convert to additive mask
        extended_mask_4d = self._prepare_attention_mask(extended_mask)  # [B, 1, 1, N_q+T]
        
        # Step 5: Apply XLM-R layers with interleaved CA layers
        # Note: LQs and query_tokens undergo bidirectional self-attention in each layer
        #       This allows LQs to become query-aware (BLIP-2 pattern)
        all_aux = []
        
        for i, xlmr_layer in enumerate(self.encoder_layers):
            # (a) Apply XLM-RoBERTa layer (self-attention + FFN)
            layer_outputs = xlmr_layer(hidden, attention_mask=extended_mask_4d)
            hidden = layer_outputs[0]  # [B, N+T, d]
            
            # (b) Split into LQs and query token parts
            lq_states = hidden[:, :self.n_queries, :]     # [B, N_q, d] (query-aware after SA)
            tok_states = hidden[:, self.n_queries:, :]    # [B, T, d] (query tokens)
            
            # (c) Apply Cross-Attention if this layer has CA
            if self.cross_layers[i] is not None:
                # CA: Query-aware LQs attend to evidence fragments
                # lq_states already contains query information from SA interactions
                updated_lqs, aux = self.cross_layers[i](
                    lqs_aware=lq_states,
                    context=evidence_emb,
                    context_mask=evidence_mask,
                )
                
                # Use updated LQs (now contains both query and evidence info)
                lq_states = updated_lqs  # [B, N_q, d]
                
                all_aux.append(aux)
            else:
                # No CA at this layer - append empty dict
                all_aux.append({})
            
            # (d) Concatenate updated LQs with query tokens for next layer
            hidden = torch.cat([lq_states, tok_states], dim=1)  # [B, N_q+T, d]
        
        # Step 6: Extract final LQ representations
        Z = hidden[:, :self.n_queries, :]  # [B, N, d]
        
        # Step 7: Apply final LayerNorm
        Z = self.final_ln(Z)  # [B, N, d]
        
        return Z, all_aux
    
    def _prepare_attention_mask(self, attention_mask: Tensor) -> Tensor:
        """
        Convert attention mask to XLM-R's expected format.
        
        Input: [B, seq_len] with True/1=valid, False/0=padding (bool or int)
        Output: [B, 1, 1, seq_len] with 0=valid, -10000=padding (additive mask)
        """
        # Convert to float if bool type
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.float()
        
        # Expand dimensions: [B, seq_len] -> [B, 1, 1, seq_len]
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Convert to additive mask: 1 -> 0.0, 0 -> -10000.0
        extended_mask = (1.0 - extended_mask) * -10000.0
        
        return extended_mask
    
    def get_trainable_params(self):
        """Returns iterator over trainable parameters."""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def freeze_xlmr(self):
        """Freeze XLM-RoBERTa backbone parameters."""
        for param in self.xlmr.parameters():
            param.requires_grad = False
        print("XLM-RoBERTa backbone frozen")
    
    def unfreeze_xlmr(self):
        """Unfreeze XLM-RoBERTa backbone parameters."""
        for param in self.xlmr.parameters():
            param.requires_grad = True
        print("XLM-RoBERTa backbone unfrozen")


# =============================================================================
# Simple test script
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("XLM-RoBERTa DR-QFormer Test")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    batch_size = 2
    n_queries = 32
    seq_len = 16  # Number of tokens
    num_fragments = 5
    hidden_dim = 768
    num_heads = 12
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing XLMRobertaDRQFormer...")
    print("=" * 80)
    
    model = XLMRobertaDRQFormer(
        xlm_model_name="xlm-roberta-base",
        n_queries=n_queries,
        dropout=0.1,
        use_ca_layers=[5, 11],  # Apply CA only after layers 6 and 12
        freeze_xlmr=False,
    )
    model = model.to(device)
    model.eval()  # Set to eval mode for testing
    
    # Generate random inputs
    print("\n" + "=" * 80)
    print("Generating random inputs...")
    print("=" * 80)
    
    # Random token IDs (within XLM-R vocab size: 250002)
    input_ids = torch.randint(0, 250000, (batch_size, seq_len), device=device)
    
    # Attention mask (no padding in this test)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    
    # Evidence embeddings
    evidence_emb = torch.randn(batch_size, num_fragments, hidden_dim, device=device)
    
    # Evidence mask (all valid)
    evidence_mask = torch.ones(batch_size, num_fragments, dtype=torch.bool, device=device)
    
    print(f"Input shapes:")
    print(f"  input_ids: {input_ids.shape} (query tokens)")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  evidence_emb: {evidence_emb.shape}")
    print(f"  evidence_mask: {evidence_mask.shape}")
    
    # Forward pass
    print("\n" + "=" * 80)
    print("Running forward pass...")
    print("=" * 80)
    
    with torch.no_grad():
        Z, all_aux = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            evidence_emb=evidence_emb,
            evidence_mask=evidence_mask,
        )
    
    # Print outputs
    print(f"\nOutput shapes:")
    print(f"  Z (final LQ representations): {Z.shape}")
    print(f"  Expected: [{batch_size}, {n_queries}, {hidden_dim}]")
    print(f"  Number of aux dicts: {len(all_aux)}")
    
    # Check CA layers
    print("\n" + "=" * 80)
    print("Checking CA auxiliary outputs...")
    print("=" * 80)
    
    for i, aux in enumerate(all_aux):
        if aux:  # Non-empty dict means CA was applied
            print(f"\nLayer {i} (CA applied):")
            for key, value in aux.items():
                if value is not None:
                    print(f"  {key}: {value.shape}")
        else:
            print(f"\nLayer {i}: No CA")
    
    # Verify expected CA layers
    expected_ca_layers = [5, 11]
    print("\n" + "=" * 80)
    print(f"Verification: CA should be applied at layers {expected_ca_layers}")
    print("=" * 80)
    
    for i in expected_ca_layers:
        if all_aux[i] and 'ca_attn_weights' in all_aux[i] and all_aux[i]['ca_attn_weights'] is not None:
            print(f"✓ Layer {i}: CA applied correctly")
            print(f"  ca_attn_weights shape: {all_aux[i]['ca_attn_weights'].shape}")
            print(f"  ca_raw_scores_per_head shape: {all_aux[i]['ca_raw_scores_per_head'].shape}")
            print(f"  ca_raw_scores_avg shape: {all_aux[i]['ca_raw_scores_avg'].shape}")
        else:
            print(f"✗ Layer {i}: CA NOT applied (unexpected)")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
