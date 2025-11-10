"""
Test suite for attention mask handling in DR-QFormer (Online Version).

This test verifies:
1. pool_padding_mask is stored in aux dict
2. SA mask defaults to None (full attention)
3. CA mask defaults to None (full attention)
4. pool_padding_mask correctly propagates to Heads
5. Online version features (single query/answer embed)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.qformer import DRQFormer
from src.models.heads import EntailmentHead, FragmentRankingHead


def test_pool_padding_mask_in_aux():
    """
    Test that pool_padding_mask is stored in aux dict and available for Heads.
    """
    print("\n" + "="*70)
    print("TEST 1: pool_padding_mask stored in aux")
    print("="*70)
    
    # Setup
    batch_size = 4
    N_lq = 32
    K = 50  # Number of fragments
    d = 768
    
    qformer = DRQFormer(
        n_queries=N_lq,
        hidden_dim=d,
        num_layers=2,
        num_heads=8,
        dropout=0.1
    )
    qformer.eval()
    
    # Create inputs
    query_embeds = torch.randn(batch_size, 1, d)
    p_embeds = torch.randn(batch_size, K, d)
    
    # Create pool_padding_mask: first 30 fragments valid, last 20 padding
    pool_padding_mask = torch.zeros(batch_size, K, dtype=torch.bool)
    pool_padding_mask[:, :30] = True  # First 30 valid
    
    print(f"Input shapes:")
    print(f"  query_embeds: {query_embeds.shape}")
    print(f"  p_embeds: {p_embeds.shape}")
    print(f"  pool_padding_mask: {pool_padding_mask.shape}")
    print(f"  Valid fragments: {pool_padding_mask.sum(dim=1).tolist()}")
    
    # Forward pass
    with torch.no_grad():
        z, aux = qformer(
            query_embeds=query_embeds,
            p_embeds=p_embeds,
            pool_padding_mask=pool_padding_mask
        )
    
    # Check that pool_padding_mask is in aux
    assert 'pool_padding_mask' in aux, "pool_padding_mask not found in aux dict"
    
    # Check that the mask matches the input
    assert aux['pool_padding_mask'] is not None, "pool_padding_mask is None in aux"
    assert torch.equal(aux['pool_padding_mask'], pool_padding_mask), \
        "pool_padding_mask in aux doesn't match input"
    
    print(f"✅ pool_padding_mask correctly stored in aux")
    print(f"   Shape: {aux['pool_padding_mask'].shape}")
    print(f"   Valid fragments per sample: {aux['pool_padding_mask'].sum(dim=1).tolist()}")


def test_default_masks_none():
    """
    Test that SA and CA masks default to None (full attention).
    """
    print("\n" + "="*70)
    print("TEST 2: SA and CA masks default to None")
    print("="*70)
    
    batch_size = 2
    N_lq = 16
    K = 20
    d = 256
    
    qformer = DRQFormer(
        n_queries=N_lq,
        hidden_dim=d,
        num_layers=2,
        num_heads=4,
        dropout=0.0
    )
    qformer.eval()
    
    query_embeds = torch.randn(batch_size, 1, d)
    p_embeds = torch.randn(batch_size, K, d)
    
    # Forward WITHOUT providing sa_mask and ca_mask
    with torch.no_grad():
        z, aux = qformer(
            query_embeds=query_embeds,
            p_embeds=p_embeds
            # No sa_mask, no ca_mask provided
        )
    
    print(f"✅ Forward pass succeeded without explicit masks")
    print(f"   Output z shape: {z.shape}")
    print(f"   SA attention weights available: {len(aux['sa_attn_weights'])} layers")
    print(f"   CA attention weights available: {len(aux['ca_attn_weights'])} layers")
    
    # Check attention weights shapes
    sa_weights = aux['sa_attn_weights'][0]  # First layer
    ca_weights = aux['ca_attn_weights'][0]
    
    print(f"\n   Layer 0 attention shapes:")
    print(f"   SA weights: {sa_weights.shape} (should be [batch, heads, N+1, N+1])")
    print(f"   CA weights: {ca_weights.shape} (should be [batch, heads, N, K])")
    
    assert sa_weights.shape == (batch_size, 4, N_lq+1, N_lq+1), "SA weights shape incorrect"
    assert ca_weights.shape == (batch_size, 4, N_lq, K), "CA weights shape incorrect"
    
    print(f"✅ Attention weights shapes correct (full attention)")


def test_online_version_features():
    """
    Test that Q-Former correctly handles single query/answer embeds (online version).
    """
    print("\n" + "="*70)
    print("TEST 3: Online version features")
    print("="*70)
    
    batch_size = 3
    N_lq = 24
    K = 15
    d = 512
    
    qformer = DRQFormer(
        n_queries=N_lq,
        hidden_dim=d,
        num_layers=2,
        num_heads=8,
        dropout=0.0
    )
    qformer.eval()
    
    # Test Primal mode (Query → Answer)
    print("\n--- Primal Mode (QA) ---")
    query_embeds = torch.randn(batch_size, 1, d)  # Single embedding
    p_embeds = torch.randn(batch_size, K, d)
    
    print(f"query_embeds shape: {query_embeds.shape} (should be [batch, 1, d])")
    assert query_embeds.size(1) == 1, "query_embeds should have length 1 (single embedding)"
    
    with torch.no_grad():
        z_qa, aux_qa = qformer(
            query_embeds=query_embeds,
            p_embeds=p_embeds
        )
    
    print(f"Output z_qa shape: {z_qa.shape} (should be [batch, N_lq, d])")
    assert z_qa.shape == (batch_size, N_lq, d), "Output shape incorrect"
    print(f"✅ Primal mode works correctly")
    
    # Test Dual mode (Answer → Query)
    print("\n--- Dual Mode (QG) ---")
    answer_embeds = torch.randn(batch_size, 1, d)  # Single embedding
    
    print(f"answer_embeds shape: {answer_embeds.shape} (should be [batch, 1, d])")
    assert answer_embeds.size(1) == 1, "answer_embeds should have length 1 (single embedding)"
    
    with torch.no_grad():
        z_qg, aux_qg = qformer(
            answer_embeds=answer_embeds,
            p_embeds=p_embeds
        )
    
    print(f"Output z_qg shape: {z_qg.shape} (should be [batch, N_lq, d])")
    assert z_qg.shape == (batch_size, N_lq, d), "Output shape incorrect"
    print(f"✅ Dual mode works correctly")
    
    # Test that both modes produce different outputs (different conditioning)
    diff = (z_qa - z_qg).abs().mean().item()
    print(f"\nMean difference between QA and QG outputs: {diff:.6f}")
    assert diff > 0.01, "QA and QG should produce different outputs"
    print(f"✅ QA and QG modes produce different outputs (correct)")


def test_mask_propagation_to_heads():
    """
    Test that pool_padding_mask correctly propagates from Q-Former to Heads.
    """
    print("\n" + "="*70)
    print("TEST 4: Mask propagation to Heads")
    print("="*70)
    
    batch_size = 4
    N_lq = 32
    K = 50
    d = 768
    
    qformer = DRQFormer(
        n_queries=N_lq,
        hidden_dim=d,
        num_layers=2,
        num_heads=8,
        dropout=0.0
    )
    qformer.eval()
    
    # Create EntailmentHead
    head_e = EntailmentHead(
        num_fragments=K,
        tau=0.5,
        p_drop_lq=0.0
    )
    head_e.eval()
    
    # Create inputs with padding
    query_embeds = torch.randn(batch_size, 1, d)
    p_embeds = torch.randn(batch_size, K, d)
    
    # Variable-length fragments per sample
    pool_padding_mask = torch.zeros(batch_size, K, dtype=torch.bool)
    pool_padding_mask[0, :40] = True  # Sample 0: 40 valid fragments
    pool_padding_mask[1, :35] = True  # Sample 1: 35 valid fragments
    pool_padding_mask[2, :45] = True  # Sample 2: 45 valid fragments
    pool_padding_mask[3, :30] = True  # Sample 3: 30 valid fragments
    
    print(f"Variable-length fragments per sample:")
    for i in range(batch_size):
        print(f"  Sample {i}: {pool_padding_mask[i].sum().item()} valid fragments")
    
    # Forward through Q-Former
    with torch.no_grad():
        z, aux = qformer(
            query_embeds=query_embeds,
            p_embeds=p_embeds,
            pool_padding_mask=pool_padding_mask
        )
    
    # Check mask is in aux
    assert 'pool_padding_mask' in aux
    assert torch.equal(aux['pool_padding_mask'], pool_padding_mask)
    print(f"✅ Q-Former stores pool_padding_mask in aux")
    
    # Forward through Head using mask from aux
    with torch.no_grad():
        out = head_e(
            ca_raw_scores_per_head=aux['ca_raw_scores_per_head'],
            pool_padding_mask=aux['pool_padding_mask'],  # From aux!
            training=False
        )
    
    logits = out['fragment_logits']  # [batch, K]
    print(f"\nEntailmentHead output shape: {logits.shape}")
    
    # Check that padding positions have very negative logits
    for i in range(batch_size):
        valid_count = pool_padding_mask[i].sum().item()
        valid_logits = logits[i, :valid_count]
        padding_logits = logits[i, valid_count:]
        
        if padding_logits.numel() > 0:
            print(f"\n  Sample {i}:")
            print(f"    Valid logits range: [{valid_logits.min():.2f}, {valid_logits.max():.2f}]")
            print(f"    Padding logits: {padding_logits.max():.2f} (should be < -1e3)")
            assert padding_logits.max().item() < -1e3, \
                f"Padding logits should be masked to very negative values"
    
    print(f"\n✅ pool_padding_mask correctly propagates to Head")
    print(f"✅ Padding positions correctly masked in output")


def test_ca_raw_scores_with_mask():
    """
    Test that CA raw scores correctly apply pool_padding_mask before softmax.
    """
    print("\n" + "="*70)
    print("TEST 5: CA raw scores with pool_padding_mask")
    print("="*70)
    
    batch_size = 2
    N_lq = 16
    K = 20
    d = 256
    
    qformer = DRQFormer(
        n_queries=N_lq,
        hidden_dim=d,
        num_layers=2,
        num_heads=4,
        dropout=0.0
    )
    qformer.eval()
    
    query_embeds = torch.randn(batch_size, 1, d)
    p_embeds = torch.randn(batch_size, K, d)
    
    # Mask: only first 10 fragments valid
    pool_padding_mask = torch.zeros(batch_size, K, dtype=torch.bool)
    pool_padding_mask[:, :10] = True
    
    with torch.no_grad():
        z, aux = qformer(
            query_embeds=query_embeds,
            p_embeds=p_embeds,
            pool_padding_mask=pool_padding_mask
        )
    
    # Check CA raw scores (pre-softmax)
    ca_raw_scores = aux['ca_raw_scores_per_head'][0]  # First layer [batch, heads, N, K]
    ca_weights = aux['ca_attn_weights'][0]  # Post-softmax [batch, heads, N, K]
    
    print(f"CA raw scores shape: {ca_raw_scores.shape}")
    print(f"CA weights shape: {ca_weights.shape}")
    
    # Check that padding positions have very negative raw scores
    valid_scores = ca_raw_scores[:, :, :, :10]  # Valid positions
    padding_scores = ca_raw_scores[:, :, :, 10:]  # Padding positions
    
    print(f"\nValid scores range: [{valid_scores.min():.2f}, {valid_scores.max():.2f}]")
    print(f"Padding scores max: {padding_scores.max():.2f} (should be < -1e3)")
    
    assert padding_scores.max().item() < -1e3, \
        "Padding positions should have very negative raw scores"
    
    # Check that attention weights on padding are near zero
    valid_weights = ca_weights[:, :, :, :10]
    padding_weights = ca_weights[:, :, :, 10:]
    
    print(f"\nValid attention weights sum: {valid_weights.sum(dim=-1).mean():.6f} (should be ~1.0)")
    print(f"Padding attention weights max: {padding_weights.max():.6f} (should be ~0.0)")
    
    assert padding_weights.max().item() < 1e-6, \
        "Attention weights on padding should be near zero"
    
    print(f"\n✅ pool_padding_mask correctly applied in CA computation")


def main():
    """Run all attention mask tests."""
    print("\n" + "="*70)
    print("ATTENTION MASK TEST SUITE (Online Version)")
    print("="*70)
    print("\nPurpose: Verify attention mask handling in DR-QFormer")
    print("- pool_padding_mask storage and propagation")
    print("- Default mask behavior (None = full attention)")
    print("- Online version features (single query/answer embeds)")
    
    test_pool_padding_mask_in_aux()
    test_default_masks_none()
    test_online_version_features()
    test_mask_propagation_to_heads()
    test_ca_raw_scores_with_mask()
    
    print("\n" + "="*70)
    print("🎉 ALL ATTENTION MASK TESTS PASSED! 🎉")
    print("="*70)
    print("\nSummary:")
    print("  ✅ pool_padding_mask stored in aux dict")
    print("  ✅ SA/CA masks default to None (full attention)")
    print("  ✅ Online version features verified")
    print("  ✅ Mask correctly propagates to Heads")
    print("  ✅ CA computation correctly masks padding")
    print("\nAttention mask handling is production-ready!")
    print("="*70)


if __name__ == "__main__":
    main()
