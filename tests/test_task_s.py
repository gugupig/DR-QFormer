"""
Unit tests for Task S (Fragment Ranking).

Tests:
1. FragmentRankingHead forward pass
2. Dual-LSE aggregation
3. Dynamic training subset construction
4. Curriculum learning weight scheduling
5. Ranking loss computation
6. Ranking metrics computation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import torch.nn as nn
    import numpy as np
    
    from dr_qformer.models.qformer import DRQFormer
    from dr_qformer.models.heads import FragmentRankingHead
    from dr_qformer.losses import (
        compute_ranking_loss,
        build_train_subset_mask,
        get_curriculum_weights
    )
    from dr_qformer.metrics import compute_ranking_metrics
    
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def test_fragmentrankinghead_forward():
    """Test FragmentRankingHead forward pass with dual-LSE aggregation."""
    print("\n" + "="*80)
    print("Test 1: FragmentRankingHead Forward Pass")
    print("="*80)
    
    batch_size = 2
    num_heads = 12
    N_lq = 32
    K = 50
    num_layers = 2
    
    # Initialize head
    head = FragmentRankingHead(
        num_fragments=K,
        tau_head=0.1,
        tau_lq=0.2,
        rho_top=0.02,
        l_prime=16,
    ).to(DEVICE)
    
    # Create mock CA raw scores
    ca_raw_scores_per_head = [
        torch.randn(batch_size, num_heads, N_lq, K, device=DEVICE)
        for _ in range(num_layers)
    ]
    
    # Create padding mask (last 10 fragments are padding)
    pool_padding_mask = torch.ones(batch_size, K, dtype=torch.bool, device=DEVICE)
    pool_padding_mask[:, -10:] = False
    
    # Forward pass
    output = head(
        z=None,  # Not used
        ca_raw_scores_per_head=ca_raw_scores_per_head,
        pool_padding_mask=pool_padding_mask,
        training=True
    )
    
    ranking_logits = output["ranking_logits"]
    ca_raw_scores_avg = output["ca_raw_scores_avg"]
    
    # Assertions
    assert ranking_logits.shape == (batch_size, K), f"Expected shape ({batch_size}, {K}), got {ranking_logits.shape}"
    assert ca_raw_scores_avg.shape == (batch_size, N_lq, K), f"Expected shape ({batch_size}, {N_lq}, {K}), got {ca_raw_scores_avg.shape}"
    
    # Check padding mask is applied (padded positions should have very low scores)
    padded_scores = ranking_logits[:, -10:]
    valid_scores = ranking_logits[:, :-10]
    assert padded_scores.max() < valid_scores.min(), "Padded scores should be lower than valid scores"
    
    # Check no gradients (head has no parameters)
    assert head.count_parameters() == 0, "Head should have no trainable parameters"
    
    print(f"✅ Ranking logits shape: {ranking_logits.shape}")
    print(f"✅ Valid scores range: [{valid_scores.min():.2f}, {valid_scores.max():.2f}]")
    print(f"✅ Padded scores range: [{padded_scores.min():.2f}, {padded_scores.max():.2f}]")
    print("✅ PASSED: FragmentRankingHead forward test")


def test_train_subset_construction():
    """Test dynamic training subset construction."""
    print("\n" + "="*80)
    print("Test 2: Dynamic Training Subset Construction")
    print("="*80)
    
    batch_size = 2
    K = 100
    rho_top = 0.05  # Top 5%
    l_prime = 10  # 10 student hard negatives
    
    # Create synthetic scores
    ranking_logits = torch.randn(batch_size, K, device=DEVICE)
    gt_scores = torch.randn(batch_size, K, device=DEVICE)
    pool_padding_mask = torch.ones(batch_size, K, dtype=torch.bool, device=DEVICE)
    
    # Build subset mask
    train_subset_mask = build_train_subset_mask(
        ranking_logits=ranking_logits,
        gt_scores=gt_scores,
        pool_padding_mask=pool_padding_mask,
        rho_top=rho_top,
        l_prime=l_prime,
    )
    
    # Check shape
    assert train_subset_mask.shape == (batch_size, K), f"Expected shape ({batch_size}, {K}), got {train_subset_mask.shape}"
    
    # Check subset size
    for b in range(batch_size):
        subset_size = train_subset_mask[b].sum().item()
        expected_min = int(rho_top * K)  # Teacher Top-L
        expected_max = expected_min + l_prime  # + Student Hard Negatives
        
        print(f"  Sample {b}: Subset size = {subset_size} (expected: {expected_min}~{expected_max})")
        assert expected_min <= subset_size <= expected_max + 5, f"Subset size {subset_size} out of expected range"
    
    print("✅ PASSED: Training subset construction test")


def test_curriculum_weights():
    """Test curriculum learning weight scheduling."""
    print("\n" + "="*80)
    print("Test 3: Curriculum Learning Weights")
    print("="*80)
    
    total_steps = 10000
    
    # Test at different training stages
    stages = [
        (0, "Start"),
        (2500, "25%"),
        (5000, "50%"),
        (7500, "75%"),
        (10000, "End"),
    ]
    
    for step, stage_name in stages:
        weights = get_curriculum_weights(
            current_step=step,
            total_steps=total_steps,
            lambda_teach_start=1.0,
            lambda_teach_end=0.2,
            lambda_post_start=0.0,
            lambda_post_end=0.8,
        )
        
        lambda_teach = weights["lambda_teach"]
        lambda_post = weights["lambda_post"]
        progress = weights["progress"]
        
        print(f"  {stage_name:6s} (step {step:5d}): "
              f"λ_teach={lambda_teach:.3f}, λ_post={lambda_post:.3f}, progress={progress:.2f}")
        
        # Check monotonic trends
        if step == 0:
            assert abs(lambda_teach - 1.0) < 0.01, "Initial λ_teach should be 1.0"
            assert abs(lambda_post - 0.0) < 0.01, "Initial λ_post should be 0.0"
        elif step == total_steps:
            assert abs(lambda_teach - 0.2) < 0.01, "Final λ_teach should be 0.2"
            assert abs(lambda_post - 0.8) < 0.01, "Final λ_post should be 0.8"
    
    print("✅ PASSED: Curriculum learning weights test")


def test_ranking_loss():
    """Test ranking loss computation."""
    print("\n" + "="*80)
    print("Test 4: Ranking Loss Computation")
    print("="*80)
    
    batch_size = 2
    K = 50
    
    # Create synthetic data
    ranking_logits = torch.randn(batch_size, K, device=DEVICE, requires_grad=True)
    gt_scores = torch.randn(batch_size, K, device=DEVICE)  # Teacher signal (no grad needed)
    posterior_scores = torch.randn(batch_size, K, device=DEVICE)  # Teacher signal (no grad needed)
    pool_padding_mask = torch.ones(batch_size, K, dtype=torch.bool, device=DEVICE)
    pool_padding_mask[:, -10:] = False  # Last 10 are padding
    
    # Build subset mask
    train_subset_mask = build_train_subset_mask(
        ranking_logits=ranking_logits.detach(),
        gt_scores=gt_scores,
        pool_padding_mask=pool_padding_mask,
        rho_top=0.1,
        l_prime=5,
    )
    
    # Compute loss
    loss_dict = compute_ranking_loss(
        ranking_logits=ranking_logits,
        gt_scores=gt_scores,
        posterior_scores=posterior_scores,
        pool_padding_mask=pool_padding_mask,
        train_subset_mask=train_subset_mask,
        lambda_teach=0.7,
        lambda_post=0.3,
        lambda_entropy=0.01,
        tau_pred=1.0,
        tau_gt=1.0,
    )
    
    # Check loss components
    assert "loss" in loss_dict, "Missing 'loss' key"
    assert "loss_teach" in loss_dict, "Missing 'loss_teach' key"
    assert "loss_post" in loss_dict, "Missing 'loss_post' key"
    assert "loss_entropy" in loss_dict, "Missing 'loss_entropy' key"
    
    # Check loss values are finite
    loss = loss_dict["loss"]
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    # Check loss has gradient
    assert loss.requires_grad, "Loss should require gradient"
    
    print(f"✅ Total loss: {loss.item():.4f}")
    print(f"✅ Teacher loss: {loss_dict['loss_teach']:.4f}")
    print(f"✅ Posterior loss: {loss_dict['loss_post']:.4f}")
    print(f"✅ Entropy loss: {loss_dict['loss_entropy']:.4f}")
    print("✅ PASSED: Ranking loss computation test")


def test_ranking_metrics():
    """Test ranking metrics computation."""
    print("\n" + "="*80)
    print("Test 5: Ranking Metrics Computation")
    print("="*80)
    
    batch_size = 3
    K = 50
    
    # Create synthetic data
    ranking_logits = torch.randn(batch_size, K)
    gt_scores = torch.randn(batch_size, K).softmax(dim=-1)
    pool_padding_mask = torch.ones(batch_size, K, dtype=torch.bool)
    
    # Compute metrics
    metrics = compute_ranking_metrics(
        ranking_logits=ranking_logits,
        gt_scores=gt_scores,
        pool_padding_mask=pool_padding_mask,
        k_list=[5, 10, 20],
    )
    
    # Check metric keys
    expected_keys = ["ndcg@5", "ndcg@10", "ndcg@20", "mrr", "map", "spearman"]
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], (int, float)), f"Metric {key} should be numeric"
        assert 0.0 <= metrics[key] <= 1.0 or key == "spearman", f"Metric {key} out of range: {metrics[key]}"
    
    print(f"✅ NDCG@5: {metrics['ndcg@5']:.4f}")
    print(f"✅ NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"✅ MRR: {metrics['mrr']:.4f}")
    print(f"✅ MAP: {metrics['map']:.4f}")
    print(f"✅ Spearman: {metrics['spearman']:.4f}")
    print("✅ PASSED: Ranking metrics computation test")


def test_end_to_end():
    """End-to-end test with Q-Former + FragmentRankingHead."""
    print("\n" + "="*80)
    print("Test 6: End-to-End Forward + Backward")
    print("="*80)
    
    batch_size = 2
    K = 30
    n_queries = 32
    hidden_dim = 768
    d_ret = 768  # Must match hidden_dim for Q-Former
    
    # Initialize models
    qformer = DRQFormer(
        n_queries=n_queries,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=12,
    ).to(DEVICE)
    
    head = FragmentRankingHead(
        num_fragments=K,
        tau_head=0.1,
        tau_lq=0.2,
    ).to(DEVICE)
    
    # Create synthetic data
    q_embeds = torch.randn(batch_size, 1, d_ret, device=DEVICE)  # [batch, 1, d_ret] for single query
    p_embeds = torch.randn(batch_size, K, d_ret, device=DEVICE)
    gt_scores = torch.randn(batch_size, K, device=DEVICE)  # Raw scores
    pool_padding_mask = torch.ones(batch_size, K, dtype=torch.bool, device=DEVICE)
    
    # Forward through Q-Former
    z, aux = qformer(
        query_embeds=q_embeds,
        p_embeds=p_embeds,
        pool_padding_mask=pool_padding_mask
    )
    
    ca_raw_scores_per_head = aux.get("ca_raw_scores_per_head")
    
    # Forward through head
    head_output = head(
        z=z,
        ca_raw_scores_per_head=ca_raw_scores_per_head,
        pool_padding_mask=pool_padding_mask,
        training=True
    )
    
    ranking_logits = head_output["ranking_logits"]
    
    # Compute loss
    loss_dict = compute_ranking_loss(
        ranking_logits=ranking_logits,
        gt_scores=gt_scores,
        pool_padding_mask=pool_padding_mask,
        lambda_teach=1.0,
        lambda_post=0.0,
    )
    
    loss = loss_dict["loss"]
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    qformer_has_grad = any(p.grad is not None for p in qformer.parameters())
    assert qformer_has_grad, "Q-Former should have gradients"
    
    print(f"✅ Q-Former output: z={z.shape}")
    print(f"✅ Ranking logits: {ranking_logits.shape}")
    print(f"✅ Loss: {loss.item():.4f}")
    print(f"✅ Q-Former has gradients: {qformer_has_grad}")
    print("✅ PASSED: End-to-end test")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Task S Unit Tests")
    print("="*80)
    
    try:
        test_fragmentrankinghead_forward()
        test_train_subset_construction()
        test_curriculum_weights()
        test_ranking_loss()
        test_ranking_metrics()
        test_end_to_end()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED (6/6)")
        print("="*80)
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
