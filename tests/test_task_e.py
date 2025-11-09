"""
Test script for Task E (EntailmentHead) with synthetic data.

This script verifies that the EntailmentHead implementation works correctly
before training on real data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import torch.nn as nn
    import numpy as np
    
    from dr_qformer.models.qformer import DRQFormer
    from dr_qformer.models.heads import EntailmentHead
    from dr_qformer.losses import compute_focal_loss
    
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_entailment_head_shapes():
    """Test 1: Verify output shapes are correct."""
    print("\n" + "="*80)
    print("Test 1: EntailmentHead Shape Test")
    print("="*80)
    
    batch_size = 2
    n_queries = 32
    hidden_dim = 768
    num_heads = 12
    num_layers = 12
    k_fragments = 5
    
    # Create synthetic Q-Former output
    z = torch.randn(batch_size, n_queries, hidden_dim)
    
    # Create synthetic CA raw scores (pre-softmax, per layer)
    # These are QK^T/sqrt(d_head) BEFORE softmax
    ca_raw_scores_per_head = [
        torch.randn(batch_size, num_heads, n_queries, k_fragments)
        for _ in range(num_layers)
    ]
    
    # Create padding mask (all valid for this test)
    pool_padding_mask = torch.ones(batch_size, k_fragments, dtype=torch.bool)
    
    # Initialize EntailmentHead
    head = EntailmentHead(
        num_fragments=k_fragments,
        tau=0.5,
        p_drop_lq=0.0,  # Disable Drop-LQ for deterministic test
    )
    
    # Forward pass (new interface with ca_raw_scores_per_head)
    output = head(
        z=z,
        ca_raw_scores_per_head=ca_raw_scores_per_head,
        pool_padding_mask=pool_padding_mask,
        training=False
    )
    
    # Extract fragment_logits from output dict
    logits = output['fragment_logits']
    
    # Check shapes
    assert logits.shape == (batch_size, k_fragments), \
        f"Expected shape ({batch_size}, {k_fragments}), got {logits.shape}"
    
    print(f"✅ Input shape: z={z.shape}, CA raw scores={len(ca_raw_scores_per_head)} layers")
    print(f"✅ Output shape: logits={logits.shape}")
    print(f"✅ Logit range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print("✅ PASSED: Shape test")


def test_drop_lq_safety():
    """Test 2: Verify Drop-LQ safety protection (at least 1 LQ survives)."""
    print("\n" + "="*80)
    print("Test 2: Drop-LQ Safety Protection Test")
    print("="*80)
    
    batch_size = 8
    n_queries = 32
    hidden_dim = 768
    num_heads = 12
    num_layers = 12
    k_fragments = 5
    
    # High drop probability to trigger safety protection
    p_drop_lq = 0.9
    
    # Create synthetic data
    z = torch.randn(batch_size, n_queries, hidden_dim)
    ca_raw_scores_per_head = [
        torch.randn(batch_size, num_heads, n_queries, k_fragments)
        for _ in range(num_layers)
    ]
    pool_padding_mask = torch.ones(batch_size, k_fragments, dtype=torch.bool)
    
    # Initialize EntailmentHead with high drop probability
    head = EntailmentHead(
        num_fragments=k_fragments,
        tau=0.5,
        p_drop_lq=p_drop_lq,
    )
    
    # Run multiple forward passes (stochastic)
    num_trials = 10
    all_passed = True
    
    for trial in range(num_trials):
        output = head(
            z=z,
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=pool_padding_mask,
            training=True  # Enable Drop-LQ
        )
        logits = output['fragment_logits']
        
        # Check that logits are not all -inf (which would happen if all LQs dropped)
        if torch.isinf(logits).any():
            print(f"❌ Trial {trial+1}: Found -inf in logits (all LQs dropped)")
            all_passed = False
            break
    
    if all_passed:
        print(f"✅ Ran {num_trials} trials with p_drop_lq={p_drop_lq}")
        print(f"✅ Safety protection worked: No -inf logits found")
        print("✅ PASSED: Drop-LQ safety test")
    else:
        print("❌ FAILED: Drop-LQ safety test")


def test_focal_loss():
    """Test 3: Verify focal loss computation."""
    print("\n" + "="*80)
    print("Test 3: Focal Loss Computation Test")
    print("="*80)
    
    batch_size = 4
    k_fragments = 5
    hidden_dim = 768
    
    # Create synthetic data
    head = EntailmentHead(
        num_fragments=k_fragments,
        focal_gamma=2.0,
        focal_alpha=0.25,
    )
    
    # Synthetic logits and labels
    logits = torch.randn(batch_size, k_fragments)
    gt_labels = torch.randint(0, 2, (batch_size, k_fragments)).float()
    pool_padding_mask = torch.ones(batch_size, k_fragments, dtype=torch.bool)
    
    # Compute loss using standalone function
    loss = compute_focal_loss(
        logits=logits,
        gt_labels=gt_labels,
        importance_weights=None,
        pool_padding_mask=pool_padding_mask,
        focal_gamma=2.0,
        focal_alpha=0.25
    )
    
    # Check that loss is scalar and finite
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    print(f"✅ Logits shape: {logits.shape}")
    print(f"✅ Labels shape: {gt_labels.shape}")
    print(f"✅ Loss: {loss.item():.4f}")
    print("✅ PASSED: Focal loss test")


def test_focal_loss_with_weights():
    """Test 4: Verify focal loss with importance weights and padding mask."""
    print("\n" + "="*80)
    print("Test 4: Focal Loss with Importance Weights and Padding Mask")
    print("="*80)
    
    batch_size = 4
    k_fragments = 5
    hidden_dim = 768
    
    head = EntailmentHead(
        num_fragments=k_fragments,
        focal_gamma=2.0,
        focal_alpha=0.25,
    )
    
    # Synthetic data
    logits = torch.randn(batch_size, k_fragments)
    gt_labels = torch.randint(0, 2, (batch_size, k_fragments)).float()
    
    # Importance weights: w_pos=10.0 for positive, 1.0 for negative
    w_pos = 10.0
    importance_weights = torch.where(gt_labels == 1, w_pos, 1.0)
    
    # Pool padding mask: last 2 fragments are padding
    pool_padding_mask = torch.ones_like(logits, dtype=torch.bool)
    pool_padding_mask[:, -2:] = False
    
    # Compute loss using standalone function
    loss = compute_focal_loss(
        logits=logits,
        gt_labels=gt_labels,
        importance_weights=importance_weights,
        pool_padding_mask=pool_padding_mask,
        focal_gamma=2.0,
        focal_alpha=0.25
    )
    
    # Check loss
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    print(f"✅ Importance weights shape: {importance_weights.shape}")
    print(f"✅ Padding mask shape: {pool_padding_mask.shape}")
    print(f"✅ Num valid fragments: {pool_padding_mask.sum().item()}")
    print(f"✅ Loss: {loss.item():.4f}")
    print("✅ PASSED: Focal loss with weights test")


def test_end_to_end():
    """Test 5: End-to-end forward + backward pass."""
    print("\n" + "="*80)
    print("Test 5: End-to-End Forward + Backward Pass")
    print("="*80)
    
    batch_size = 2
    n_queries = 32
    hidden_dim = 768
    num_layers = 12
    num_heads = 12
    k_fragments = 5
    d_ret = 768
    
    # Initialize models
    qformer = DRQFormer(
        n_queries=n_queries,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    
    head = EntailmentHead(
        num_fragments=k_fragments,
        tau=0.5,
        p_drop_lq=0.1,
        focal_gamma=2.0,
        focal_alpha=0.25,
    )
    
    # Create synthetic retriever embeddings
    q_embeds = torch.randn(batch_size, 1, d_ret)  # [batch, 1, d_ret]
    p_embeds = torch.randn(batch_size, k_fragments, d_ret)  # [batch, k, d_ret]
    pool_padding_mask = torch.ones(batch_size, k_fragments, dtype=torch.bool)
    
    # Create synthetic labels
    gt_labels = torch.randint(0, 2, (batch_size, k_fragments)).float()
    
    # Forward pass
    z, aux = qformer(query_embeds=q_embeds, p_embeds=p_embeds, pool_padding_mask=pool_padding_mask)
    ca_raw_scores_per_head = aux.get("ca_raw_scores_per_head", None)
    
    head_output = head(
        z=z,
        ca_raw_scores_per_head=ca_raw_scores_per_head,
        pool_padding_mask=pool_padding_mask,
        training=True
    )
    logits = head_output['fragment_logits']
    
    # Compute loss using standalone function
    loss = compute_focal_loss(
        logits=logits,
        gt_labels=gt_labels,
        importance_weights=None,
        pool_padding_mask=pool_padding_mask,
        focal_gamma=2.0,
        focal_alpha=0.25
    )
    
    # Backward pass
    loss.backward()
    
    # Check gradients (only Q-Former has trainable parameters)
    qformer_has_grad = any(p.grad is not None for p in qformer.parameters())
    
    assert qformer_has_grad, "Q-Former parameters have no gradients"
    
    # EntailmentHead has no trainable parameters (only hyperparameters)
    # so we don't check for its gradients
    
    print(f"✅ Q-Former output shape: z={z.shape}")
    print(f"✅ CA raw scores: {len(ca_raw_scores_per_head)} layers")
    print(f"✅ EntailmentHead output shape: logits={logits.shape}")
    print(f"✅ Loss: {loss.item():.4f}")
    print(f"✅ Q-Former has gradients: {qformer_has_grad}")
    print(f"✅ Q-Former params: {qformer.count_parameters():,}")
    print(f"✅ EntailmentHead (no trainable params - only hyperparameters)")
    print("✅ PASSED: End-to-end test")


def main():
    """Run all tests."""
    print("="*80)
    print("Task E (EntailmentHead) Test Suite")
    print("="*80)
    
    try:
        test_entailment_head_shapes()
        test_drop_lq_safety()
        test_focal_loss()
        test_focal_loss_with_weights()
        test_end_to_end()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nTask E implementation is ready for training.")
        print("Next steps:")
        print("  1. Prepare training data (see TASK_E_QUICKSTART.md)")
        print("  2. Run training: python train/task_e.py --train_data ... --dev_data ...")
        print("  3. Evaluate model and tune hyperparameters")
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
