"""
Test Drop-LQ functionality in all three task heads.

Validates:
1. Drop-LQ correctly drops LQs during training
2. Drop-LQ is disabled during evaluation
3. Safety mechanism prevents all LQs from being dropped
4. Drop-LQ works with different probabilities
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


def test_entailment_head_drop_lq():
    """Test Drop-LQ in EntailmentHead."""
    print("\n" + "="*80)
    print("TEST 1: EntailmentHead Drop-LQ")
    print("="*80)
    
    from src.models.heads import EntailmentHead
    
    # Test parameters
    batch_size = 4
    num_layers = 3
    num_heads = 8
    n_lqs = 32
    k_fragments = 50
    
    # Create head with Drop-LQ enabled
    head = EntailmentHead(
        num_fragments=k_fragments,
        tau=0.5,
        p_drop_lq=0.3,  # 30% drop rate for visible effect
    )
    
    # Create dummy CA raw scores
    ca_raw_scores_per_head = [
        torch.randn(batch_size, num_heads, n_lqs, k_fragments) 
        for _ in range(num_layers)
    ]
    pool_padding_mask = torch.ones(batch_size, k_fragments, dtype=torch.bool)
    
    print(f"\nInput shape: [batch={batch_size}, num_heads={num_heads}, N_lq={n_lqs}, K={k_fragments}]")
    print(f"Drop-LQ probability: {head.p_drop_lq}")
    
    # Test 1: Training mode (Drop-LQ should be active)
    print("\n--- Training Mode (Drop-LQ Active) ---")
    head.train()
    
    results_train = []
    for trial in range(3):
        result = head(
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=pool_padding_mask,
            training=True
        )
        fragment_logits = result['fragment_logits']
        results_train.append(fragment_logits)
        print(f"Trial {trial+1}: Output shape {fragment_logits.shape}, mean={fragment_logits.mean():.4f}, std={fragment_logits.std():.4f}")
    
    # Check variability (should differ due to random dropout)
    diff_01 = (results_train[0] - results_train[1]).abs().mean()
    diff_12 = (results_train[1] - results_train[2]).abs().mean()
    print(f"\nVariability between trials: {diff_01:.6f}, {diff_12:.6f}")
    assert diff_01 > 0.001, "Drop-LQ should cause variability between training runs"
    print("✅ Drop-LQ causes expected variability in training mode")
    
    # Test 2: Evaluation mode (Drop-LQ should be disabled)
    print("\n--- Evaluation Mode (Drop-LQ Disabled) ---")
    head.eval()
    
    results_eval = []
    for trial in range(3):
        with torch.no_grad():
            result = head(
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                training=False
            )
            fragment_logits = result['fragment_logits']
            results_eval.append(fragment_logits)
            print(f"Trial {trial+1}: Output shape {fragment_logits.shape}, mean={fragment_logits.mean():.4f}")
    
    # Check consistency (should be identical)
    diff_eval_01 = (results_eval[0] - results_eval[1]).abs().max()
    diff_eval_12 = (results_eval[1] - results_eval[2]).abs().max()
    print(f"\nConsistency in eval mode: max diff = {diff_eval_01:.8f}, {diff_eval_12:.8f}")
    assert diff_eval_01 < 1e-6, "Eval mode should be deterministic"
    print("✅ Drop-LQ is correctly disabled in evaluation mode")
    
    print("\n✅ EntailmentHead Drop-LQ: PASSED")


def test_fragment_ranking_head_drop_lq():
    """Test Drop-LQ in FragmentRankingHead."""
    print("\n" + "="*80)
    print("TEST 2: FragmentRankingHead Drop-LQ")
    print("="*80)
    
    from src.models.heads import FragmentRankingHead
    
    # Test parameters
    batch_size = 4
    num_layers = 3
    num_heads = 8
    n_lqs = 32
    k_fragments = 100
    
    # Create head with Drop-LQ enabled
    head = FragmentRankingHead(
        num_fragments=k_fragments,
        tau_head=0.1,
        tau_lq=0.2,
        p_drop_lq=0.25,  # 25% drop rate
    )
    
    # Create dummy CA raw scores
    ca_raw_scores_per_head = [
        torch.randn(batch_size, num_heads, n_lqs, k_fragments) 
        for _ in range(num_layers)
    ]
    pool_padding_mask = torch.ones(batch_size, k_fragments, dtype=torch.bool)
    
    print(f"\nInput shape: [batch={batch_size}, num_heads={num_heads}, N_lq={n_lqs}, K={k_fragments}]")
    print(f"Drop-LQ probability: {head.p_drop_lq}")
    
    # Test 1: Training mode
    print("\n--- Training Mode (Drop-LQ Active) ---")
    head.train()
    
    results_train = []
    for trial in range(3):
        result = head(
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=pool_padding_mask,
            training=True
        )
        ranking_logits = result['ranking_logits']
        results_train.append(ranking_logits)
        print(f"Trial {trial+1}: Output shape {ranking_logits.shape}, mean={ranking_logits.mean():.4f}, std={ranking_logits.std():.4f}")
    
    # Check variability
    diff_01 = (results_train[0] - results_train[1]).abs().mean()
    diff_12 = (results_train[1] - results_train[2]).abs().mean()
    print(f"\nVariability between trials: {diff_01:.6f}, {diff_12:.6f}")
    assert diff_01 > 0.001, "Drop-LQ should cause variability"
    print("✅ Drop-LQ causes expected variability in training mode")
    
    # Test 2: Evaluation mode
    print("\n--- Evaluation Mode (Drop-LQ Disabled) ---")
    head.eval()
    
    results_eval = []
    for trial in range(2):
        with torch.no_grad():
            result = head(
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                training=False
            )
            ranking_logits = result['ranking_logits']
            results_eval.append(ranking_logits)
            print(f"Trial {trial+1}: Output shape {ranking_logits.shape}, mean={ranking_logits.mean():.4f}")
    
    # Check consistency
    diff_eval = (results_eval[0] - results_eval[1]).abs().max()
    print(f"\nConsistency in eval mode: max diff = {diff_eval:.8f}")
    assert diff_eval < 1e-6, "Eval mode should be deterministic"
    print("✅ Drop-LQ is correctly disabled in evaluation mode")
    
    print("\n✅ FragmentRankingHead Drop-LQ: PASSED")


def test_condense_head_drop_lq():
    """Test Drop-LQ in CondenseHead."""
    print("\n" + "="*80)
    print("TEST 3: CondenseHead Drop-LQ")
    print("="*80)
    
    from src.models.heads import CondenseHead
    
    # Test parameters
    batch_size = 4
    n_lqs = 32
    qformer_dim = 768
    llm_dim = 4096
    
    # Create head with Drop-LQ enabled
    head = CondenseHead(
        hidden_dim=qformer_dim,
        llm_hidden_dim=llm_dim,
        p_drop_lq=0.2,  # 20% drop rate
    )
    
    # Create dummy Q-Former output
    z = torch.randn(batch_size, n_lqs, qformer_dim)
    
    print(f"\nInput shape: [batch={batch_size}, N_lq={n_lqs}, dim={qformer_dim}]")
    print(f"Output dim: {llm_dim}")
    print(f"Drop-LQ probability: {head.p_drop_lq}")
    
    # Test 1: Training mode
    print("\n--- Training Mode (Drop-LQ Active) ---")
    head.train()
    
    results_train = []
    for trial in range(3):
        prefix_embeds = head(z, training=True)
        results_train.append(prefix_embeds)
        print(f"Trial {trial+1}: Output shape {prefix_embeds.shape}, mean={prefix_embeds.mean():.4f}, std={prefix_embeds.std():.4f}")
    
    # Check variability
    diff_01 = (results_train[0] - results_train[1]).abs().mean()
    diff_12 = (results_train[1] - results_train[2]).abs().mean()
    print(f"\nVariability between trials: {diff_01:.6f}, {diff_12:.6f}")
    assert diff_01 > 0.001, "Drop-LQ should cause variability"
    print("✅ Drop-LQ causes expected variability in training mode")
    
    # Test 2: Evaluation mode
    print("\n--- Evaluation Mode (Drop-LQ Disabled) ---")
    head.eval()
    
    results_eval = []
    for trial in range(2):
        with torch.no_grad():
            prefix_embeds = head(z, training=False)
            results_eval.append(prefix_embeds)
            print(f"Trial {trial+1}: Output shape {prefix_embeds.shape}, mean={prefix_embeds.mean():.4f}")
    
    # Check consistency
    diff_eval = (results_eval[0] - results_eval[1]).abs().max()
    print(f"\nConsistency in eval mode: max diff = {diff_eval:.8f}")
    assert diff_eval < 1e-6, "Eval mode should be deterministic"
    print("✅ Drop-LQ is correctly disabled in evaluation mode")
    
    print("\n✅ CondenseHead Drop-LQ: PASSED")


def test_drop_lq_safety():
    """Test safety mechanism: at least one LQ always remains active."""
    print("\n" + "="*80)
    print("TEST 4: Drop-LQ Safety Mechanism (No All-Dropped)")
    print("="*80)
    
    from src.models.heads import EntailmentHead
    
    batch_size = 10
    num_layers = 2
    num_heads = 8
    n_lqs = 8  # Small number to increase chance of all-dropped
    k_fragments = 20
    
    # Create head with VERY HIGH drop rate to test safety
    head = EntailmentHead(
        num_fragments=k_fragments,
        tau=0.5,
        p_drop_lq=0.95,  # 95% drop rate! Should trigger safety mechanism
    )
    
    ca_raw_scores_per_head = [
        torch.randn(batch_size, num_heads, n_lqs, k_fragments) 
        for _ in range(num_layers)
    ]
    pool_padding_mask = torch.ones(batch_size, k_fragments, dtype=torch.bool)
    
    print(f"\nInput: batch={batch_size}, N_lq={n_lqs}, K={k_fragments}")
    print(f"Drop-LQ probability: {head.p_drop_lq} (EXTREME!)")
    print("\nRunning 10 trials to verify safety mechanism...")
    
    head.train()
    all_passed = True
    
    for trial in range(10):
        result = head(
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=pool_padding_mask,
            training=True
        )
        fragment_logits = result['fragment_logits']
        
        # Check if output contains any NaNs or Infs (would indicate all-dropped failure)
        has_nan = torch.isnan(fragment_logits).any()
        has_inf = torch.isinf(fragment_logits).any()
        
        if has_nan or has_inf:
            print(f"❌ Trial {trial+1}: NaN or Inf detected! Safety failed.")
            all_passed = False
        else:
            print(f"✅ Trial {trial+1}: Valid output, mean={fragment_logits.mean():.4f}")
    
    assert all_passed, "Safety mechanism should prevent NaN/Inf"
    print("\n✅ Safety mechanism works: No all-dropped samples detected")


def test_drop_lq_probabilities():
    """Test Drop-LQ with different probabilities."""
    print("\n" + "="*80)
    print("TEST 5: Drop-LQ with Different Probabilities")
    print("="*80)
    
    from src.models.heads import FragmentRankingHead
    
    batch_size = 4
    num_layers = 2
    num_heads = 8
    n_lqs = 32
    k_fragments = 50
    
    probabilities = [0.0, 0.1, 0.3, 0.5]
    
    ca_raw_scores_per_head = [
        torch.randn(batch_size, num_heads, n_lqs, k_fragments) 
        for _ in range(num_layers)
    ]
    pool_padding_mask = torch.ones(batch_size, k_fragments, dtype=torch.bool)
    
    print(f"\nTesting Drop-LQ with p = {probabilities}")
    
    for p_drop in probabilities:
        head = FragmentRankingHead(
            num_fragments=k_fragments,
            tau_head=0.1,
            tau_lq=0.2,
            p_drop_lq=p_drop,
        )
        head.train()
        
        # Run multiple trials
        results = []
        for _ in range(5):
            result = head(
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                training=True
            )
            results.append(result['ranking_logits'])
        
        # Compute variance across trials
        results_stacked = torch.stack(results, dim=0)  # [5, batch, K]
        variance = results_stacked.var(dim=0).mean()
        
        print(f"  p_drop={p_drop:.1f}: Variance across 5 trials = {variance:.6f}")
        
        if p_drop == 0.0:
            assert variance < 1e-6, "No dropout should mean no variance"
        else:
            assert variance > 0.001, f"Drop-LQ with p={p_drop} should cause variance"
    
    print("\n✅ Drop-LQ works correctly with different probabilities")


def main():
    """Run all Drop-LQ tests."""
    print("\n" + "="*80)
    print("DR-QFormer Drop-LQ Test Suite")
    print("="*80)
    print("\nTesting Drop-LQ functionality in all three task heads:")
    print("  1. EntailmentHead (Task E)")
    print("  2. FragmentRankingHead (Task S)")
    print("  3. CondenseHead (Task C)")
    print("\nValidating:")
    print("  - Drop-LQ active in training mode")
    print("  - Drop-LQ disabled in evaluation mode")
    print("  - Safety mechanism (no all-dropped)")
    print("  - Different drop probabilities")
    
    try:
        test_entailment_head_drop_lq()
        test_fragment_ranking_head_drop_lq()
        test_condense_head_drop_lq()
        test_drop_lq_safety()
        test_drop_lq_probabilities()
        
        print("\n" + "="*80)
        print("🎉 ALL DROP-LQ TESTS PASSED! 🎉")
        print("="*80)
        print("\nSummary:")
        print("  ✅ EntailmentHead Drop-LQ working")
        print("  ✅ FragmentRankingHead Drop-LQ working")
        print("  ✅ CondenseHead Drop-LQ working")
        print("  ✅ Safety mechanism verified")
        print("  ✅ Multiple probabilities tested")
        print("\nDrop-LQ is ready for use in all three tasks!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
