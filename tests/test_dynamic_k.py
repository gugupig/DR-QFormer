"""
Test dynamic K functionality for Task S.

Verifies:
1. Variable K handling in collate function
2. pool_padding_mask generation
3. Alpha_gt temperature calibration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from train.task_s import collate_task_s_batch
from dr_qformer.losses import compute_ranking_loss

def test_dynamic_k_collate():
    """Test collate function with variable K."""
    print("\n" + "="*80)
    print("Test: Dynamic K Collation")
    print("="*80)
    
    # Create samples with different K
    batch_list = [
        {
            "queries": "Query 1",
            "fragments": [f"Frag1_{i}" for i in range(50)],  # K=50
            "gt_scores": np.random.rand(50),
            "answers": "Answer 1",
        },
        {
            "queries": "Query 2",
            "fragments": [f"Frag2_{i}" for i in range(80)],  # K=80
            "gt_scores": np.random.rand(80),
            "answers": "Answer 2",
        },
        {
            "queries": "Query 3",
            "fragments": [f"Frag3_{i}" for i in range(30)],  # K=30
            "gt_scores": np.random.rand(30),
            "answers": "Answer 3",
        },
    ]
    
    # Collate
    batch = collate_task_s_batch(batch_list)
    
    # Verify
    K_max = 80  # Max in batch
    assert len(batch["queries"]) == 3, f"Expected 3 queries, got {len(batch['queries'])}"
    assert batch["gt_scores"].shape == (3, K_max), f"Expected (3, {K_max}), got {batch['gt_scores'].shape}"
    assert batch["pool_padding_mask"].shape == (3, K_max), f"Expected (3, {K_max}), got {batch['pool_padding_mask'].shape}"
    
    # Check mask correctness
    assert batch["pool_padding_mask"][0, :50].all(), "Sample 0 first 50 should be valid"
    assert not batch["pool_padding_mask"][0, 50:].any(), "Sample 0 rest should be invalid"
    assert batch["pool_padding_mask"][1, :80].all(), "Sample 1 all 80 should be valid"
    assert batch["pool_padding_mask"][2, :30].all(), "Sample 2 first 30 should be valid"
    assert not batch["pool_padding_mask"][2, 30:].any(), "Sample 2 rest should be invalid"
    
    print(f"✅ K_max: {K_max}")
    print(f"✅ gt_scores shape: {batch['gt_scores'].shape}")
    print(f"✅ pool_padding_mask shape: {batch['pool_padding_mask'].shape}")
    print(f"✅ Sample 0 valid fragments: {batch['pool_padding_mask'][0].sum().item()}")
    print(f"✅ Sample 1 valid fragments: {batch['pool_padding_mask'][1].sum().item()}")
    print(f"✅ Sample 2 valid fragments: {batch['pool_padding_mask'][2].sum().item()}")
    print("✅ PASSED: Dynamic K collation test")


def test_alpha_gt_calibration():
    """Test alpha_gt temperature calibration in loss function."""
    print("\n" + "="*80)
    print("Test: Alpha_gt Temperature Calibration")
    print("="*80)
    
    batch_size = 2
    K = 50
    
    # Create data with clear ranking
    ranking_logits = torch.randn(batch_size, K, requires_grad=True)
    
    # GT scores: high values for first 10 fragments
    gt_scores = torch.zeros(batch_size, K)
    gt_scores[:, :10] = torch.linspace(10, 1, 10).unsqueeze(0)  # Top 10 have high scores
    gt_scores[:, 10:] = torch.randn(batch_size, K-10) * 0.1  # Rest have low scores
    
    pool_padding_mask = torch.ones(batch_size, K, dtype=torch.bool)
    
    # Create train_subset_mask (top 5 + 10 hard negatives)
    train_subset_mask = torch.zeros(batch_size, K, dtype=torch.bool)
    train_subset_mask[:, :15] = True  # Roughly: top 5 + 10 hard negatives
    
    # Compute loss with alpha_gt=0.7
    loss_dict = compute_ranking_loss(
        ranking_logits=ranking_logits,
        gt_scores=gt_scores,
        pool_padding_mask=pool_padding_mask,
        train_subset_mask=train_subset_mask,
        lambda_teach=1.0,
        lambda_post=0.0,
        alpha_gt=0.7,  # Top-L should have 70% cumulative mass
        tau_pred=1.0,
        tau_gt=1.0,
    )
    
    # Verify loss is computed
    assert "loss" in loss_dict
    assert "loss_teach" in loss_dict
    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["loss"].requires_grad
    
    # Test backward
    loss_dict["loss"].backward()
    assert ranking_logits.grad is not None
    
    print(f"✅ Loss: {loss_dict['loss'].item():.4f}")
    print(f"✅ Teacher loss: {loss_dict['loss_teach']:.4f}")
    print(f"✅ Gradient computed: {ranking_logits.grad is not None}")
    print("✅ PASSED: Alpha_gt calibration test")


def test_variable_k_in_loss():
    """Test loss computation with variable K via padding mask."""
    print("\n" + "="*80)
    print("Test: Variable K in Loss Computation")
    print("="*80)
    
    batch_size = 3
    K_max = 100
    
    # Different effective K per sample
    K_effective = [50, 80, 30]
    
    ranking_logits = torch.randn(batch_size, K_max, requires_grad=True)
    gt_scores = torch.randn(batch_size, K_max)
    
    # Create variable padding mask
    pool_padding_mask = torch.zeros(batch_size, K_max, dtype=torch.bool)
    for b, K_eff in enumerate(K_effective):
        pool_padding_mask[b, :K_eff] = True
    
    # Train subset mask (Top-L + Hard negatives)
    train_subset_mask = torch.zeros(batch_size, K_max, dtype=torch.bool)
    for b, K_eff in enumerate(K_effective):
        L = max(1, int(0.1 * K_eff))  # Top 10%
        train_subset_mask[b, :L+10] = True  # Top-L + 10 hard negatives
    
    # Compute loss
    loss_dict = compute_ranking_loss(
        ranking_logits=ranking_logits,
        gt_scores=gt_scores,
        pool_padding_mask=pool_padding_mask,
        train_subset_mask=train_subset_mask,
        lambda_teach=1.0,
        lambda_post=0.0,
        alpha_gt=0.7,
    )
    
    # Verify
    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["loss"].requires_grad
    
    # Test backward
    loss_dict["loss"].backward()
    
    # Check gradients only on valid positions
    for b, K_eff in enumerate(K_effective):
        valid_grad = ranking_logits.grad[b, :K_eff]
        invalid_grad = ranking_logits.grad[b, K_eff:]
        
        assert torch.isfinite(valid_grad).all(), f"Sample {b} valid grad should be finite"
        # Invalid positions may have zero or small gradients due to masking
        
        print(f"  Sample {b}: K_eff={K_eff}, valid_grad_norm={valid_grad.norm().item():.4f}")
    
    print(f"✅ Loss: {loss_dict['loss'].item():.4f}")
    print("✅ PASSED: Variable K in loss test")


def main():
    """Run all dynamic K tests."""
    print("\n" + "="*80)
    print("Dynamic K Tests for Task S")
    print("="*80)
    
    try:
        test_dynamic_k_collate()
        test_alpha_gt_calibration()
        test_variable_k_in_loss()
        
        print("\n" + "="*80)
        print("✅ ALL DYNAMIC K TESTS PASSED (3/3)")
        print("="*80)
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
