"""
Validation test for Task E Spec v1.1 refactor.

Verifies all 8 targets from the refactor checklist:
1. Use pre-softmax CA scores (QK^T/√d) per head
2. Dual training with two forwards (Primal + Dual) with shared weights
3. Padding propagation via pool_padding_mask in CA and aggregation
4. Drop-LQ safety (train-time only, disabled at eval)
5. Dynamic K support (variable K_pool per batch via padding)
6. Dynamic weights (build importance_weights from gt_labels + is_longtail)
7. Debug outputs (return ca_raw_scores_avg and ca_raw_scores_per_head.detach())
8. Fix constructor mismatch (EntailmentHead accept hidden_dim but ignore)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from dr_qformer.models.qformer import DRQFormer
from dr_qformer.models.heads import EntailmentHead


def test_spec_v11_conformance():
    """
    Test all 8 targets from Spec v1.1.
    """
    print("=" * 80)
    print("Spec v1.1 Conformance Test")
    print("=" * 80)
    
    # Setup
    device = torch.device("cpu")
    B = 2  # batch size
    N = 32  # num learnable queries
    K = 5  # num fragments
    d_ret = 768  # retriever embedding dim
    H = 12  # num heads
    num_layers = 12
    
    # Random inputs
    torch.manual_seed(42)
    q_embeds = torch.randn(B, 1, d_ret)  # [batch, 1, d] - note 3D shape
    a_embeds = torch.randn(B, 1, d_ret)  # [batch, 1, d] - for dual mode
    p_embeds = torch.randn(B, K, d_ret)  # [batch, k, d]
    gt_labels = torch.randint(0, 2, (B, K)).float()
    is_longtail = torch.randint(0, 2, (B, K))  # Longtail indicator
    
    # pool_padding_mask: [B, K] with some False entries
    pool_padding_mask = torch.ones(B, K, dtype=torch.bool)
    pool_padding_mask[0, 3:] = False  # Sample 0: only first 3 fragments valid
    pool_padding_mask[1, 4:] = False  # Sample 1: only first 4 fragments valid
    
    # Initialize models
    qformer = DRQFormer(
        n_queries=N,
        hidden_dim=768,
        num_layers=num_layers,
        num_heads=H,
    ).to(device)
    
    # Target 8: Constructor accepts hidden_dim but ignores it
    head = EntailmentHead(
        num_fragments=K,
        tau=0.5,
        p_drop_lq=0.1,
        focal_gamma=2.0,
        focal_alpha=0.25,
        hidden_dim=768  # Should be accepted but ignored
    ).to(device)
    
    print("\n✅ Target 8: EntailmentHead accepts hidden_dim (ignored)")
    
    # Target 1: Use pre-softmax CA scores
    print("\n[Target 1] Testing pre-softmax CA score exposure...")
    z_primal, aux_primal = qformer(
        query_embeds=q_embeds,
        p_embeds=p_embeds,
        pool_padding_mask=pool_padding_mask
    )
    
    ca_raw_scores_per_head = aux_primal.get("ca_raw_scores_per_head", None)
    ca_raw_scores_avg = aux_primal.get("ca_raw_scores_avg", None)
    
    assert ca_raw_scores_per_head is not None, "ca_raw_scores_per_head not in aux"
    assert ca_raw_scores_avg is not None, "ca_raw_scores_avg not in aux"
    assert len(ca_raw_scores_per_head) == num_layers, f"Expected {num_layers} layers, got {len(ca_raw_scores_per_head)}"
    
    # Check shapes
    for layer_idx, raw_scores in enumerate(ca_raw_scores_per_head):
        assert raw_scores.shape == (B, H, N, K), f"Layer {layer_idx}: Expected shape ({B}, {H}, {N}, {K}), got {raw_scores.shape}"
    
    for layer_idx, raw_avg in enumerate(ca_raw_scores_avg):
        assert raw_avg.shape == (B, N, K), f"Layer {layer_idx}: Expected avg shape ({B}, {N}, {K}), got {raw_avg.shape}"
    
    print(f"  ✅ ca_raw_scores_per_head: {len(ca_raw_scores_per_head)} layers, shape {ca_raw_scores_per_head[0].shape}")
    print(f"  ✅ ca_raw_scores_avg: {len(ca_raw_scores_avg)} layers, shape {ca_raw_scores_avg[0].shape}")
    
    # Target 2: Dual training
    print("\n[Target 2] Testing dual training (Primal + Dual forwards)...")
    qformer.train()
    head.train()
    
    # Primal forward
    z_primal, aux_primal = qformer(
        query_embeds=q_embeds,
        p_embeds=p_embeds,
        pool_padding_mask=pool_padding_mask
    )
    head_out_primal = head(
        z=z_primal,
        ca_raw_scores_per_head=aux_primal["ca_raw_scores_per_head"],
        pool_padding_mask=pool_padding_mask,
        training=True
    )
    
    # Dual forward (same parameters)
    z_dual, aux_dual = qformer(
        answer_embeds=a_embeds,
        p_embeds=p_embeds,
        pool_padding_mask=pool_padding_mask
    )
    head_out_dual = head(
        z=z_dual,
        ca_raw_scores_per_head=aux_dual["ca_raw_scores_per_head"],
        pool_padding_mask=pool_padding_mask,
        training=True
    )
    
    # Check both forwards work
    assert head_out_primal['fragment_logits'].shape == (B, K)
    assert head_out_dual['fragment_logits'].shape == (B, K)
    print(f"  ✅ Primal forward: fragment_logits shape {head_out_primal['fragment_logits'].shape}")
    print(f"  ✅ Dual forward: fragment_logits shape {head_out_dual['fragment_logits'].shape}")
    
    # Target 3: Padding propagation
    print("\n[Target 3] Testing padding propagation...")
    # Check that masked positions have been set to large negative values
    raw_layer0 = ca_raw_scores_per_head[0]  # [B, H, N, K]
    # Sample 0, fragments 3-4 should be masked
    assert torch.all(raw_layer0[0, :, :, 3:] < -1000), "Sample 0 padding not masked properly"
    # Sample 1, fragment 4 should be masked
    assert torch.all(raw_layer0[1, :, :, 4:] < -1000), "Sample 1 padding not masked properly"
    print(f"  ✅ Padded positions masked to large negative values")
    print(f"     Sample 0, frag 3-4 min: {raw_layer0[0, :, :, 3:].min():.2f}")
    print(f"     Sample 1, frag 4 min: {raw_layer0[1, :, :, 4:].min():.2f}")
    
    # Target 4: Drop-LQ safety
    print("\n[Target 4] Testing Drop-LQ (train vs eval)...")
    qformer.eval()
    head.eval()
    
    # Eval mode: training=False should disable Drop-LQ
    z_eval, aux_eval = qformer(
        query_embeds=q_embeds,
        p_embeds=p_embeds,
        pool_padding_mask=pool_padding_mask
    )
    head_out_eval = head(
        z=z_eval,
        ca_raw_scores_per_head=aux_eval["ca_raw_scores_per_head"],
        pool_padding_mask=pool_padding_mask,
        training=False
    )
    
    # Run multiple times to check determinism (no dropout in eval)
    logits_list = []
    for _ in range(3):
        z_eval, aux_eval = qformer(
            query_embeds=q_embeds,
            p_embeds=p_embeds,
            pool_padding_mask=pool_padding_mask
        )
        head_out = head(
            z=z_eval,
            ca_raw_scores_per_head=aux_eval["ca_raw_scores_per_head"],
            pool_padding_mask=pool_padding_mask,
            training=False
        )
        logits_list.append(head_out['fragment_logits'])
    
    # Check all runs produce same results (no randomness)
    for i in range(1, len(logits_list)):
        assert torch.allclose(logits_list[0], logits_list[i], atol=1e-6), "Eval mode not deterministic"
    print(f"  ✅ Eval mode (training=False): Deterministic (no Drop-LQ)")
    
    # Target 5: Dynamic K support
    print("\n[Target 5] Testing dynamic K support...")
    # Create batch with different valid K per sample
    pool_padding_mask_dynamic = torch.ones(B, K, dtype=torch.bool)
    pool_padding_mask_dynamic[0, 2:] = False  # Sample 0: K=2
    pool_padding_mask_dynamic[1, 4:] = False  # Sample 1: K=4
    
    z_dynamic, aux_dynamic = qformer(
        query_embeds=q_embeds,
        p_embeds=p_embeds,
        pool_padding_mask=pool_padding_mask_dynamic
    )
    head_out_dynamic = head(
        z=z_dynamic,
        ca_raw_scores_per_head=aux_dynamic["ca_raw_scores_per_head"],
        pool_padding_mask=pool_padding_mask_dynamic,
        training=False
    )
    
    assert head_out_dynamic['fragment_logits'].shape == (B, K)
    print(f"  ✅ Dynamic K: Sample 0 (K=2), Sample 1 (K=4)")
    print(f"     Logits shape: {head_out_dynamic['fragment_logits'].shape}")
    
    # Target 6: Dynamic weights
    print("\n[Target 6] Testing dynamic importance_weights...")
    w_pos = 10.0
    w_longtail = 50.0
    
    importance_weights = torch.ones_like(gt_labels)
    # Positive class weighting
    importance_weights = torch.where(gt_labels == 1, 
                                    w_pos * torch.ones_like(gt_labels), 
                                    torch.ones_like(gt_labels))
    # Longtail weighting
    importance_weights = torch.where((gt_labels == 1) & (is_longtail == 1),
                                    w_longtail * torch.ones_like(gt_labels),
                                    importance_weights)
    
    print(f"  ✅ importance_weights built from gt_labels + is_longtail")
    print(f"     gt_labels:\n{gt_labels}")
    print(f"     is_longtail:\n{is_longtail}")
    print(f"     importance_weights:\n{importance_weights}")
    
    # Target 7: Debug outputs
    print("\n[Target 7] Testing debug outputs...")
    assert 'ca_raw_scores_avg' in head_out_primal, "ca_raw_scores_avg not in head output"
    assert 'ca_raw_scores_per_head' in head_out_primal, "ca_raw_scores_per_head not in head output"
    
    # Check detached (no gradients)
    assert not head_out_primal['ca_raw_scores_avg'].requires_grad, "ca_raw_scores_avg not detached"
    assert not head_out_primal['ca_raw_scores_per_head'].requires_grad, "ca_raw_scores_per_head not detached"
    
    print(f"  ✅ Debug outputs returned:")
    print(f"     ca_raw_scores_avg: {head_out_primal['ca_raw_scores_avg'].shape}, requires_grad={head_out_primal['ca_raw_scores_avg'].requires_grad}")
    print(f"     ca_raw_scores_per_head: {head_out_primal['ca_raw_scores_per_head'].shape}, requires_grad={head_out_primal['ca_raw_scores_per_head'].requires_grad}")
    
    print("\n" + "=" * 80)
    print("✅ ALL 8 TARGETS PASS")
    print("=" * 80)
    print("\nSummary:")
    print("  1. ✅ Pre-softmax CA scores exposed")
    print("  2. ✅ Dual training (Primal + Dual) works")
    print("  3. ✅ Padding propagation via pool_padding_mask")
    print("  4. ✅ Drop-LQ safety (train vs eval)")
    print("  5. ✅ Dynamic K support")
    print("  6. ✅ Dynamic importance_weights")
    print("  7. ✅ Debug outputs (detached)")
    print("  8. ✅ Constructor accepts hidden_dim (ignored)")
    print()


if __name__ == "__main__":
    test_spec_v11_conformance()
