"""
Unit tests for Task C: Condensing-Generation.

Tests:
1. compute_condensing_loss basic functionality
2. Adaptive margin computation
3. Posterior extraction from LLM attention
4. CondenseHead forward pass
5. End-to-end training step (with dummy LLM)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from dr_qformer.losses import compute_condensing_loss
from dr_qformer.models.heads import CondenseHead


def test_condensing_loss_basic():
    """Test basic condensing loss computation."""
    print("\n" + "=" * 80)
    print("Test 1: Basic Condensing Loss")
    print("=" * 80)
    
    # Scenario: Evidence helps (positive NLL gain)
    nll_with = torch.tensor(2.5)      # Lower perplexity with evidence
    nll_without = torch.tensor(3.8)   # Higher perplexity without evidence
    
    loss_dict = compute_condensing_loss(
        nll_with_evidence=nll_with,
        nll_without_evidence=nll_without,
        softplus_beta=10.0,
        margin_mode='fixed',
        margin_fixed=0.5,
    )
    
    loss = loss_dict['loss_c']
    nll_gain = loss_dict['nll_gain']
    margin = loss_dict['margin']
    
    print(f"NLL with evidence: {nll_with.item():.4f}")
    print(f"NLL without evidence: {nll_without.item():.4f}")
    print(f"NLL Gain (G): {nll_gain.item():.4f}")
    print(f"Margin (m): {margin.item():.4f}")
    print(f"Loss: {loss.item():.4f}")
    
    # Verify
    expected_gain = 3.8 - 2.5  # 1.3
    assert abs(nll_gain.item() - expected_gain) < 1e-5, f"Expected gain {expected_gain}, got {nll_gain.item()}"
    assert abs(margin.item() - 0.5) < 1e-5, "Margin should be 0.5"
    assert loss.item() < 0.1, f"Loss should be near 0 when gain > margin, got {loss.item()}"
    
    print("✅ Test passed: Gain exceeds margin → low loss")
    
    # Scenario: Evidence doesn't help enough (low gain)
    nll_with = torch.tensor(3.5)
    nll_without = torch.tensor(3.7)
    
    loss_dict = compute_condensing_loss(
        nll_with_evidence=nll_with,
        nll_without_evidence=nll_without,
        softplus_beta=10.0,
        margin_mode='fixed',
        margin_fixed=0.5,
    )
    
    loss = loss_dict['loss_c']
    nll_gain = loss_dict['nll_gain']
    
    print(f"\nScenario 2: Insufficient gain")
    print(f"NLL Gain: {nll_gain.item():.4f}")
    print(f"Loss: {loss.item():.4f}")
    
    assert nll_gain.item() < 0.5, "Gain should be < margin"
    assert loss.item() > 0.1, f"Loss should be positive when gain < margin, got {loss.item()}"
    
    print("✅ Test passed: Low gain → higher loss")


def test_adaptive_margin():
    """Test adaptive margin computation."""
    print("\n" + "=" * 80)
    print("Test 2: Adaptive Margin")
    print("=" * 80)
    
    # Single sample (scalar NLL gain)
    nll_with = torch.tensor(2.0)
    nll_without = torch.tensor(3.5)
    
    loss_dict = compute_condensing_loss(
        nll_with_evidence=nll_with,
        nll_without_evidence=nll_without,
        softplus_beta=10.0,
        margin_mode='adaptive',
        margin_adaptive_ratio=0.5,
        margin_min=0.1,
        margin_max=2.0,
    )
    
    nll_gain = loss_dict['nll_gain']
    margin = loss_dict['margin']
    
    print(f"NLL Gain: {nll_gain.item():.4f}")
    print(f"Adaptive Margin: {margin.item():.4f}")
    
    # For scalar, margin = μ_G + κ·σ_G = nll_gain + 0.5*0 = nll_gain
    # Then clipped to [0.1, 2.0]
    expected_margin = min(max(nll_gain.item(), 0.1), 2.0)
    assert abs(margin.item() - expected_margin) < 1e-5, f"Expected {expected_margin}, got {margin.item()}"
    
    print(f"✅ Test passed: Margin = {margin.item():.4f} (within [0.1, 2.0])")


def test_posterior_extraction():
    """Test posterior extraction from LLM attention."""
    print("\n" + "=" * 80)
    print("Test 3: Posterior Extraction")
    print("=" * 80)
    
    batch_size = 2
    n_heads = 8
    seq_total = 200
    N_lq = 32
    K_pool = 50
    subset_size = 10
    answer_start_idx = N_lq + 64  # After Z + Query
    
    # Dummy inputs
    nll_with = torch.tensor(2.5)
    nll_without = torch.tensor(3.5)
    
    # LLM attention: [batch, n_heads, seq_total, N_lq]
    llm_attention = torch.randn(batch_size, n_heads, seq_total, N_lq)
    llm_attention = torch.softmax(llm_attention, dim=-1)
    
    # CA weights: [batch, N_lq, K_pool]
    ca_weights = torch.randn(batch_size, N_lq, K_pool)
    ca_weights = torch.softmax(ca_weights, dim=-1)
    
    # Subset indices: [batch, subset_size]
    subset_indices = torch.randint(0, K_pool, (batch_size, subset_size))
    
    # Compute loss with posterior
    loss_dict = compute_condensing_loss(
        nll_with_evidence=nll_with,
        nll_without_evidence=nll_without,
        llm_attention_weights=llm_attention,
        ca_weights=ca_weights,
        subset_indices=subset_indices,
        answer_start_idx=answer_start_idx,
        softplus_beta=10.0,
        margin_mode='fixed',
        margin_fixed=0.5,
    )
    
    posterior = loss_dict['posterior_q_psi_U']
    
    print(f"LLM Attention shape: {llm_attention.shape}")
    print(f"CA Weights shape: {ca_weights.shape}")
    print(f"Subset Indices shape: {subset_indices.shape}")
    print(f"Posterior q_ψ_U shape: {posterior.shape}")
    
    # Verify posterior
    assert posterior is not None, "Posterior should be computed"
    assert posterior.shape == (batch_size, subset_size), f"Expected ({batch_size}, {subset_size}), got {posterior.shape}"
    
    # Check softmax normalization
    for b in range(batch_size):
        posterior_sum = posterior[b].sum().item()
        print(f"Sample {b} posterior sum: {posterior_sum:.6f}")
        assert abs(posterior_sum - 1.0) < 1e-5, f"Posterior should sum to 1, got {posterior_sum}"
    
    # Check all values are non-negative
    assert torch.all(posterior >= 0), "Posterior should be non-negative"
    assert torch.all(posterior <= 1), "Posterior should be <= 1"
    
    print("✅ Test passed: Posterior extracted and normalized correctly")


def test_condense_head():
    """Test CondenseHead forward pass."""
    print("\n" + "=" * 80)
    print("Test 4: CondenseHead Forward")
    print("=" * 80)
    
    batch_size = 4
    N_lq = 32
    hidden_dim = 768
    llm_hidden_dim = 4096
    
    # Initialize head
    head = CondenseHead(
        hidden_dim=hidden_dim,
        llm_hidden_dim=llm_hidden_dim,
    )
    
    # Input Z from Q-Former
    z = torch.randn(batch_size, N_lq, hidden_dim)
    
    # Forward pass
    z_prefix = head(z)
    
    print(f"Input Z shape: {z.shape}")
    print(f"Output Z_prefix shape: {z_prefix.shape}")
    
    # Verify
    assert z_prefix.shape == (batch_size, N_lq, llm_hidden_dim), \
        f"Expected ({batch_size}, {N_lq}, {llm_hidden_dim}), got {z_prefix.shape}"
    
    # Check parameter count
    param_count = head.count_parameters()
    print(f"Trainable parameters: {param_count:,}")
    
    # Projection layer should have hidden_dim * llm_hidden_dim params
    expected_proj_params = hidden_dim * llm_hidden_dim + llm_hidden_dim  # weights + bias
    expected_norm_params = llm_hidden_dim * 2  # gamma + beta
    expected_total = expected_proj_params + expected_norm_params
    
    assert param_count == expected_total, f"Expected {expected_total}, got {param_count}"
    
    print("✅ Test passed: CondenseHead forward and parameter count correct")
    
    # Test with same dimensions (Identity projection)
    head_identity = CondenseHead(hidden_dim=768, llm_hidden_dim=768)
    z_identity = torch.randn(batch_size, N_lq, 768)
    z_prefix_identity = head_identity(z_identity)
    
    print(f"\nIdentity case (768→768):")
    print(f"  Output shape: {z_prefix_identity.shape}")
    
    assert z_prefix_identity.shape == (batch_size, N_lq, 768)
    print("✅ Test passed: Identity projection works")


def test_end_to_end_dummy():
    """Test end-to-end forward pass with dummy components."""
    print("\n" + "=" * 80)
    print("Test 5: End-to-End (Dummy LLM)")
    print("=" * 80)
    
    batch_size = 2
    N_lq = 32
    hidden_dim = 768
    llm_hidden_dim = 4096
    K_pool = 50
    subset_size = 10
    
    # Dummy Q-Former output
    z = torch.randn(batch_size, N_lq, hidden_dim)
    
    # CondenseHead
    head = CondenseHead(hidden_dim=hidden_dim, llm_hidden_dim=llm_hidden_dim)
    z_prefix = head(z)
    
    print(f"Z shape: {z.shape}")
    print(f"Z_prefix shape: {z_prefix.shape}")
    
    # Dummy LLM outputs (simulating teacher_forcing_dual_path)
    # Create NLL with gradients by using a simple computation involving z_prefix
    # This simulates real training where NLL depends on z_prefix
    # Use squared norm to ensure positive contribution
    dummy_logits = torch.norm(z_prefix) * 0.01  # Small scale
    nll_with = dummy_logits + 2.5  # Depends on z_prefix → has gradient
    nll_without = torch.tensor(3.8).detach()  # Baseline (detached)
    
    llm_attention = torch.randn(batch_size, 8, 200, N_lq)
    llm_attention = torch.softmax(llm_attention, dim=-1)
    
    # Dummy CA weights
    ca_weights = torch.randn(batch_size, N_lq, K_pool)
    ca_weights = torch.softmax(ca_weights, dim=-1)
    
    # Subset indices
    subset_indices = torch.randint(0, K_pool, (batch_size, subset_size))
    
    # Compute loss
    loss_dict = compute_condensing_loss(
        nll_with_evidence=nll_with,
        nll_without_evidence=nll_without,
        llm_attention_weights=llm_attention,
        ca_weights=ca_weights,
        subset_indices=subset_indices,
        answer_start_idx=N_lq + 64,
        softplus_beta=10.0,
        margin_mode='adaptive',
        margin_adaptive_ratio=0.5,
        margin_min=0.1,
        margin_max=2.0,
    )
    
    loss = loss_dict['loss_c']
    
    print(f"Loss: {loss.item():.4f}")
    print(f"NLL Gain: {loss_dict['nll_gain'].item():.4f}")
    print(f"Margin: {loss_dict['margin'].item():.4f}")
    print(f"Posterior shape: {loss_dict['posterior_q_psi_U'].shape}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients flow through CondenseHead
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in head.parameters())
    
    print(f"Gradients computed: {has_grad}")
    assert has_grad, "Gradients should flow through CondenseHead"
    
    print("✅ Test passed: End-to-end forward and backward work")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Task C Unit Tests")
    print("=" * 80)
    
    try:
        test_condensing_loss_basic()
        test_adaptive_margin()
        test_posterior_extraction()
        test_condense_head()
        test_end_to_end_dummy()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED (5/5)")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
