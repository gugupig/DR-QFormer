"""
Test script to verify attention weight export functionality.

This script validates that:
1. Self-attention (SA) weights are exported for each layer
2. Cross-attention (CA) weights are exported for each layer
3. Weights have correct shapes: [batch, num_heads, seq, seq]
4. Weights are not averaged (per-head weights preserved)
5. Can analyze which LQs attend to which fragments
"""

import sys
sys.path.insert(0, 'd:/LLMs/DR-QFormer/DR-QFormer')

import torch
from dr_qformer.models.qformer import DRQFormer

def test_attention_weight_export():
    """Test that attention weights are properly exported."""
    print("=" * 80)
    print("Testing Attention Weight Export")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    n_queries = 32
    hidden_dim = 768
    num_layers = 6
    num_heads = 8
    k_fragments = 10
    
    # Initialize model
    model = DRQFormer(
        n_queries=n_queries,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_fragments=k_fragments,
        dropout=0.0
    )
    model.eval()
    
    # Create dummy inputs
    query_embeds = torch.randn(batch_size, 1, hidden_dim)
    p_embeds = torch.randn(batch_size, k_fragments, hidden_dim)
    
    # Forward pass (Primal mode)
    print("\n1. Testing Primal mode (QA) with attention weight export...")
    with torch.no_grad():
        z, aux = model(
            query_embeds=query_embeds,
            p_embeds=p_embeds
        )
    
    # Verify output shape
    assert z.shape == (batch_size, n_queries, hidden_dim), \
        f"Expected z shape {(batch_size, n_queries, hidden_dim)}, got {z.shape}"
    print(f"   ✓ Output shape: {z.shape}")
    
    # Verify SA attention weights
    print(f"\n2. Verifying Self-Attention (SA) weights...")
    sa_weights = aux['sa_attn_weights']
    assert isinstance(sa_weights, list), "SA weights should be a list (per layer)"
    assert len(sa_weights) == num_layers, \
        f"Expected {num_layers} layers, got {len(sa_weights)}"
    print(f"   ✓ Got SA weights for {len(sa_weights)} layers")
    
    for layer_idx, sa_weight in enumerate(sa_weights):
        expected_shape = (batch_size, num_heads, n_queries + 1, n_queries + 1)
        assert sa_weight.shape == expected_shape, \
            f"Layer {layer_idx}: Expected SA shape {expected_shape}, got {sa_weight.shape}"
        
        # Verify weights sum to 1 along last dimension (attention distribution)
        weight_sums = sa_weight.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
            f"Layer {layer_idx}: SA weights should sum to 1"
        
        print(f"   ✓ Layer {layer_idx}: SA weights shape {sa_weight.shape}")
    
    # Verify CA attention weights
    print(f"\n3. Verifying Cross-Attention (CA) weights...")
    ca_weights = aux['ca_attn_weights']
    assert isinstance(ca_weights, list), "CA weights should be a list (per layer)"
    assert len(ca_weights) == num_layers, \
        f"Expected {num_layers} layers, got {len(ca_weights)}"
    print(f"   ✓ Got CA weights for {len(ca_weights)} layers")
    
    for layer_idx, ca_weight in enumerate(ca_weights):
        if ca_weight is not None:
            expected_shape = (batch_size, num_heads, n_queries, k_fragments)
            assert ca_weight.shape == expected_shape, \
                f"Layer {layer_idx}: Expected CA shape {expected_shape}, got {ca_weight.shape}"
            
            # Verify weights sum to 1 along last dimension (attention distribution)
            weight_sums = ca_weight.sum(dim=-1)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
                f"Layer {layer_idx}: CA weights should sum to 1"
            
            print(f"   ✓ Layer {layer_idx}: CA weights shape {ca_weight.shape}")
        else:
            print(f"   ! Layer {layer_idx}: CA weights are None (no context provided)")
    
    # Analyze attention patterns
    print(f"\n4. Analyzing attention patterns...")
    
    # SA: Which LQs attend to query embedding?
    # Last token is the query embedding (index n_queries)
    last_layer_sa = sa_weights[-1]  # [batch, num_heads, N+1, N+1]
    lqs_to_query = last_layer_sa[:, :, :n_queries, n_queries]  # [batch, num_heads, N]
    
    print(f"   SA Analysis (Layer {num_layers-1}):")
    print(f"   - LQs attention to query embedding: shape {lqs_to_query.shape}")
    print(f"   - Max attention: {lqs_to_query.max().item():.4f}")
    print(f"   - Min attention: {lqs_to_query.min().item():.4f}")
    print(f"   - Mean attention: {lqs_to_query.mean().item():.4f}")
    
    # CA: Which LQs attend to which fragments?
    last_layer_ca = ca_weights[-1]  # [batch, num_heads, N, k]
    
    print(f"\n   CA Analysis (Layer {num_layers-1}):")
    print(f"   - LQs attention to fragments: shape {last_layer_ca.shape}")
    
    # Average across heads for easier analysis
    ca_avg_heads = last_layer_ca.mean(dim=1)  # [batch, N, k]
    
    # Find which LQ attends most to each fragment
    for batch_idx in range(min(1, batch_size)):  # Show first batch only
        print(f"\n   Batch {batch_idx} - Top attending LQ for each fragment:")
        for frag_idx in range(min(5, k_fragments)):  # Show first 5 fragments
            lq_attentions = ca_avg_heads[batch_idx, :, frag_idx]  # [N]
            top_lq = lq_attentions.argmax().item()
            top_attention = lq_attentions[top_lq].item()
            print(f"     Fragment {frag_idx}: LQ {top_lq:2d} (attention: {top_attention:.4f})")
    
    # Find which fragment each LQ attends most to
    print(f"\n   Batch 0 - Top attended fragment for each LQ:")
    for lq_idx in range(min(5, n_queries)):  # Show first 5 LQs
        frag_attentions = ca_avg_heads[0, lq_idx, :]  # [k]
        top_frag = frag_attentions.argmax().item()
        top_attention = frag_attentions[top_frag].item()
        print(f"     LQ {lq_idx:2d}: Fragment {top_frag} (attention: {top_attention:.4f})")
    
    # Test Dual mode
    print(f"\n5. Testing Dual mode (QG) with attention weight export...")
    answer_embeds = torch.randn(batch_size, 1, hidden_dim)
    
    with torch.no_grad():
        z_dual, aux_dual = model(
            answer_embeds=answer_embeds,
            p_embeds=p_embeds
        )
    
    assert z_dual.shape == (batch_size, n_queries, hidden_dim), \
        f"Expected z shape {(batch_size, n_queries, hidden_dim)}, got {z_dual.shape}"
    print(f"   ✓ Dual mode output shape: {z_dual.shape}")
    
    assert len(aux_dual['sa_attn_weights']) == num_layers, \
        "Dual mode should also export SA weights"
    assert len(aux_dual['ca_attn_weights']) == num_layers, \
        "Dual mode should also export CA weights"
    print(f"   ✓ Dual mode exported {num_layers} SA and CA weight layers")
    
    print("\n" + "=" * 80)
    print("✅ All attention weight export tests passed!")
    print("=" * 80)
    
    return True


def test_attention_weight_shapes():
    """Test attention weight shapes with different configurations."""
    print("\n" + "=" * 80)
    print("Testing Attention Weight Shapes with Different Configurations")
    print("=" * 80)
    
    configs = [
        {"n_queries": 16, "num_layers": 3, "num_heads": 4, "k_fragments": 5},
        {"n_queries": 32, "num_layers": 6, "num_heads": 8, "k_fragments": 10},
        {"n_queries": 64, "num_layers": 12, "num_heads": 12, "k_fragments": 20},
    ]
    
    for idx, config in enumerate(configs):
        print(f"\nConfiguration {idx + 1}: {config}")
        
        model = DRQFormer(
            n_queries=config["n_queries"],
            hidden_dim=768,
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            max_fragments=config["k_fragments"],
            dropout=0.0
        )
        model.eval()
        
        query_embeds = torch.randn(2, 1, 768)
        p_embeds = torch.randn(2, config["k_fragments"], 768)
        
        with torch.no_grad():
            z, aux = model(query_embeds=query_embeds, p_embeds=p_embeds)
        
        # Verify shapes
        expected_sa_shape = (2, config["num_heads"], config["n_queries"] + 1, config["n_queries"] + 1)
        expected_ca_shape = (2, config["num_heads"], config["n_queries"], config["k_fragments"])
        
        assert aux['sa_attn_weights'][0].shape == expected_sa_shape, \
            f"SA shape mismatch: expected {expected_sa_shape}, got {aux['sa_attn_weights'][0].shape}"
        assert aux['ca_attn_weights'][0].shape == expected_ca_shape, \
            f"CA shape mismatch: expected {expected_ca_shape}, got {aux['ca_attn_weights'][0].shape}"
        
        print(f"  ✓ SA weights: {aux['sa_attn_weights'][0].shape}")
        print(f"  ✓ CA weights: {aux['ca_attn_weights'][0].shape}")
    
    print("\n" + "=" * 80)
    print("✅ All shape tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    # Run tests
    test_attention_weight_export()
    test_attention_weight_shapes()
    
    print("\n" + "=" * 80)
    print("📊 Summary")
    print("=" * 80)
    print("✅ Attention weight export functionality verified:")
    print("   - SA weights: [batch, num_heads, N+1, N+1] per layer")
    print("   - CA weights: [batch, num_heads, N, k] per layer")
    print("   - Per-head weights preserved (average_attn_weights=False)")
    print("   - Can analyze which LQs attend to which fragments")
    print("   - Works in both Primal (QA) and Dual (QG) modes")
    print("=" * 80)
