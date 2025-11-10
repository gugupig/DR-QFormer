"""
Test to verify DR-QFormer's attention masks are correctly differentiated from BLIP-2.

Key Differences:
1. SA Input: BLIP-2 uses only LQs [N, d], DR-QFormer uses [LQs, q/a_embed] [N+1, d]
2. SA Mask: BLIP-2 uses causal mask (lower triangular), DR-QFormer uses full attention
3. Online Mode: DR-QFormer always conditions on query/answer, BLIP-2 learns independently
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from src.models.qformer import DRQFormer


def test_sa_mask_is_bidirectional():
    """
    Verify SA mask allows bidirectional attention (not causal).
    
    Expected:
    - LQ_i can attend to query_embed (position N)
    - query_embed can attend to all LQs
    - All LQs can attend to each other
    
    This is CORRECT for online inference (different from BLIP-2's causal pretraining).
    """
    print("\n" + "="*70)
    print("TEST 1: SA mask is bidirectional (not causal)")
    print("="*70)
    
    qformer = DRQFormer(n_queries=4, hidden_dim=64, num_layers=1, num_heads=2)
    qformer.eval()
    
    batch_size = 2
    query_embeds = torch.randn(batch_size, 1, 64)
    p_embeds = torch.randn(batch_size, 5, 64)
    
    with torch.no_grad():
        z, aux = qformer(
            query_embeds=query_embeds,
            p_embeds=p_embeds,
            sa_mask=None  # Full attention (not causal)
        )
    
    # Extract SA attention weights
    sa_weights = aux['sa_attn_weights'][0]  # [batch, num_heads, N+1, N+1]
    print(f"SA attention weights shape: {sa_weights.shape}")
    print(f"Expected: [batch={batch_size}, heads=2, N+1=5, N+1=5]")
    
    # Check: LQ_0 (position 0) should attend to query_embed (position 4)
    attn_lq0_to_query = sa_weights[0, 0, 0, 4]  # Head 0, LQ_0 -> query
    print(f"\nLQ_0 → query_embed attention: {attn_lq0_to_query.item():.6f}")
    assert attn_lq0_to_query.item() > 0, "LQ_0 should attend to query_embed (bidirectional)"
    
    # Check: query_embed (position 4) should attend to LQ_0 (position 0)
    attn_query_to_lq0 = sa_weights[0, 0, 4, 0]  # Head 0, query -> LQ_0
    print(f"query_embed → LQ_0 attention: {attn_query_to_lq0.item():.6f}")
    assert attn_query_to_lq0.item() > 0, "query_embed should attend to LQ_0 (bidirectional)"
    
    # Check: LQ_1 should attend to LQ_3 (not causal)
    attn_lq1_to_lq3 = sa_weights[0, 0, 1, 3]
    print(f"LQ_1 → LQ_3 attention: {attn_lq1_to_lq3.item():.6f}")
    assert attn_lq1_to_lq3.item() > 0, "LQ_1 should attend to LQ_3 (not causal)"
    
    # Verify it's NOT a causal (lower triangular) mask
    # In causal mask, upper triangle should be zero
    # Here, upper triangle should have non-zero values
    upper_triangle = torch.triu(sa_weights[0, 0], diagonal=1)
    upper_nonzero = (upper_triangle > 1e-6).sum().item()
    print(f"\nUpper triangle non-zero elements: {upper_nonzero} (should be > 0 for bidirectional)")
    assert upper_nonzero > 0, "Upper triangle should have non-zero values (not causal)"
    
    print("\n✅ SA mask is bidirectional (correct for online inference)")
    print("   Different from BLIP-2's causal pretraining mask")


def test_sa_input_includes_condition():
    """
    Verify SA input is [LQs, q/a_embed], not just LQs.
    
    This is the key difference from BLIP-2's independent LQ learning.
    """
    print("\n" + "="*70)
    print("TEST 2: SA input includes query/answer embed")
    print("="*70)
    
    qformer = DRQFormer(n_queries=4, hidden_dim=64, num_layers=1, num_heads=2)
    qformer.eval()
    
    batch_size = 2
    query_embeds = torch.randn(batch_size, 1, 64)
    p_embeds = torch.randn(batch_size, 5, 64)
    
    print(f"Input shapes:")
    print(f"  query_embeds: {query_embeds.shape} (single embed per sample)")
    print(f"  LQs: [batch, N=4, d=64]")
    print(f"  Concatenated: [batch, N+1=5, d=64]")
    
    with torch.no_grad():
        z, aux = qformer(
            query_embeds=query_embeds,
            p_embeds=p_embeds,
        )
    
    # Check: SA attention weights shape should be [batch, H, N+1, N+1]
    # (N+1 because of [LQs, q_embed])
    sa_weights = aux['sa_attn_weights'][0]
    expected_shape = (batch_size, 2, 5, 5)  # N+1 = 4+1 = 5
    print(f"\nSA attention weights shape: {sa_weights.shape}")
    print(f"Expected shape: {expected_shape}")
    assert sa_weights.shape == expected_shape, \
        f"SA weights shape {sa_weights.shape} != {expected_shape}"
    
    # Check: z_raw (full sequence) should include q_embed
    z_raw = aux['z_raw']
    print(f"\nz_raw (full sequence) shape: {z_raw.shape}")
    print(f"Expected: [batch, N+1=5, d=64]")
    assert z_raw.shape == (batch_size, 5, 64), \
        f"z_raw should be [batch, N+1, d], got {z_raw.shape}"
    
    # Check: z_final (extracted LQs) should exclude q_embed
    z_final = aux['z_final']
    print(f"\nz_final (extracted LQs) shape: {z_final.shape}")
    print(f"Expected: [batch, N=4, d=64]")
    assert z_final.shape == (batch_size, 4, 64), \
        f"z_final should be [batch, N, d], got {z_final.shape}"
    
    print("\n✅ SA input includes query/answer embed (online mode correct)")
    print("   BLIP-2 original: LQs learn independently")
    print("   DR-QFormer: LQs conditioned on query/answer")


def test_ca_mask_is_full_attention():
    """
    Verify CA mask allows full attention (same as BLIP-2).
    
    Each LQ should attend to all k fragments.
    """
    print("\n" + "="*70)
    print("TEST 3: CA mask is full attention")
    print("="*70)
    
    qformer = DRQFormer(n_queries=4, hidden_dim=64, num_layers=1, num_heads=2)
    qformer.eval()
    
    batch_size = 2
    query_embeds = torch.randn(batch_size, 1, 64)
    p_embeds = torch.randn(batch_size, 5, 64)
    
    with torch.no_grad():
        z, aux = qformer(
            query_embeds=query_embeds,
            p_embeds=p_embeds,
            ca_mask=None  # Full attention
        )
    
    # Extract CA attention weights
    ca_weights = aux['ca_attn_weights'][0]  # [batch, num_heads, N, k]
    print(f"CA attention weights shape: {ca_weights.shape}")
    print(f"Expected: [batch=2, heads=2, N=4, k=5]")
    
    # Check: All LQs should be able to attend to all fragments
    # Sum across fragment dimension should be ~1.0 (softmax)
    attn_sum = ca_weights.sum(dim=-1)  # [batch, num_heads, N]
    print(f"\nAttention sum across fragments (should be ~1.0):")
    print(f"  Mean: {attn_sum.mean().item():.6f}")
    print(f"  Min: {attn_sum.min().item():.6f}")
    print(f"  Max: {attn_sum.max().item():.6f}")
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), \
        "CA attention weights should sum to 1.0"
    
    # Check: LQ_0 should have non-zero attention to all fragments
    attn_lq0 = ca_weights[0, 0, 0, :]  # Head 0, LQ_0, all fragments
    print(f"\nLQ_0 attention to all fragments:")
    print(f"  {attn_lq0.tolist()}")
    assert (attn_lq0 > 0).all(), "LQ_0 should attend to all fragments (full attention)"
    
    print("\n✅ CA mask is full attention (same as BLIP-2)")
    print("   Each LQ can attend to all k fragments")


def test_pool_padding_mask_handling():
    """
    Verify pool_padding_mask correctly masks padded fragments.
    
    This is DR-QFormer specific (BLIP-2 doesn't need this for fixed-length patches).
    """
    print("\n" + "="*70)
    print("TEST 4: pool_padding_mask handles variable-length fragments")
    print("="*70)
    
    qformer = DRQFormer(n_queries=4, hidden_dim=64, num_layers=1, num_heads=2)
    qformer.eval()
    
    batch_size = 2
    query_embeds = torch.randn(batch_size, 1, 64)
    p_embeds = torch.randn(batch_size, 5, 64)
    
    # Create padding mask: sample 0 has 3 valid fragments, sample 1 has 5
    pool_padding_mask = torch.tensor([
        [True, True, True, False, False],  # 3 valid
        [True, True, True, True, True]     # 5 valid
    ], dtype=torch.bool)
    
    print(f"pool_padding_mask:")
    print(f"  Sample 0: {pool_padding_mask[0].tolist()} (3 valid fragments)")
    print(f"  Sample 1: {pool_padding_mask[1].tolist()} (5 valid fragments)")
    
    with torch.no_grad():
        z, aux = qformer(
            query_embeds=query_embeds,
            p_embeds=p_embeds,
            pool_padding_mask=pool_padding_mask
        )
    
    # Extract CA attention weights
    ca_weights = aux['ca_attn_weights'][0]  # [batch, num_heads, N, k]
    
    # Check: Padded positions should have ~0 attention
    # Sample 0: fragments 3, 4 should have near-zero attention
    attn_sample0_frag3 = ca_weights[0, 0, :, 3]  # All LQs, fragment 3
    attn_sample0_frag4 = ca_weights[0, 0, :, 4]  # All LQs, fragment 4
    
    print(f"\nSample 0 (3 valid fragments):")
    print(f"  Fragment 3 (padded) max attention: {attn_sample0_frag3.max().item():.8f}")
    print(f"  Fragment 4 (padded) max attention: {attn_sample0_frag4.max().item():.8f}")
    assert torch.allclose(attn_sample0_frag3, torch.zeros_like(attn_sample0_frag3), atol=1e-5), \
        "Padded fragment 3 should have ~0 attention"
    assert torch.allclose(attn_sample0_frag4, torch.zeros_like(attn_sample0_frag4), atol=1e-5), \
        "Padded fragment 4 should have ~0 attention"
    
    # Check: Valid positions should have non-zero attention
    attn_sample1_frag4 = ca_weights[1, 0, :, 4]  # Sample 1, all LQs, fragment 4 (valid)
    print(f"\nSample 1 (5 valid fragments):")
    print(f"  Fragment 4 (valid) attention: {attn_sample1_frag4.tolist()}")
    assert (attn_sample1_frag4 > 1e-6).any(), \
        "Valid fragment 4 in sample 1 should have non-zero attention"
    
    print("\n✅ pool_padding_mask correctly handles variable-length fragments")
    print("   BLIP-2: Fixed-length image patches (no padding)")
    print("   DR-QFormer: Variable-length text fragments (requires padding mask)")


def test_primal_dual_modes():
    """
    Verify Primal (QA) and Dual (QG) modes produce different outputs.
    
    This is a key feature of DR-QFormer's online conditioning.
    """
    print("\n" + "="*70)
    print("TEST 5: Primal/Dual modes with online conditioning")
    print("="*70)
    
    qformer = DRQFormer(n_queries=8, hidden_dim=128, num_layers=2, num_heads=4)
    qformer.eval()
    
    batch_size = 2
    p_embeds = torch.randn(batch_size, 10, 128)
    
    # Primal mode: condition on query
    query_embeds = torch.randn(batch_size, 1, 128)
    print(f"\nPrimal Mode (QA):")
    print(f"  Input: query_embeds {query_embeds.shape}")
    
    with torch.no_grad():
        z_qa, aux_qa = qformer(
            query_embeds=query_embeds,
            p_embeds=p_embeds
        )
    print(f"  Output: z_qa {z_qa.shape}")
    
    # Dual mode: condition on answer
    answer_embeds = torch.randn(batch_size, 1, 128)
    print(f"\nDual Mode (QG):")
    print(f"  Input: answer_embeds {answer_embeds.shape}")
    
    with torch.no_grad():
        z_qg, aux_qg = qformer(
            answer_embeds=answer_embeds,
            p_embeds=p_embeds
        )
    print(f"  Output: z_qg {z_qg.shape}")
    
    # Check: Outputs should be different (different conditioning)
    diff = (z_qa - z_qg).abs().mean().item()
    print(f"\nMean absolute difference: {diff:.6f}")
    assert diff > 0.01, "QA and QG should produce different outputs"
    
    print("\n✅ Primal/Dual modes produce different outputs")
    print("   Online conditioning on query/answer is working correctly")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DR-QFORMER vs BLIP-2 DIFFERENTIATION TEST SUITE")
    print("="*70)
    print("\nPurpose: Verify DR-QFormer correctly implements 'online version'")
    print("         and is properly differentiated from BLIP-2 original")
    
    test_sa_mask_is_bidirectional()
    test_sa_input_includes_condition()
    test_ca_mask_is_full_attention()
    test_pool_padding_mask_handling()
    test_primal_dual_modes()
    
    print("\n" + "="*70)
    print("🎉 ALL DR-QFORMER vs BLIP-2 DIFFERENTIATION TESTS PASSED!")
    print("="*70)
    print("\nKey Differences Verified:")
    print("  ✅ SA mask: Bidirectional (not causal like BLIP-2 pretraining)")
    print("  ✅ SA input: [LQs, q/a_embed] (not just LQs)")
    print("  ✅ CA mask: Full attention (same as BLIP-2)")
    print("  ✅ pool_padding_mask: Handles variable-length fragments")
    print("  ✅ Online conditioning: Primal/Dual modes work correctly")
    print("\nDR-QFormer 'online version' is correctly implemented!")
    print("="*70)
