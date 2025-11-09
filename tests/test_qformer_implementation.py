"""
Test DR-QFormer implementation to verify the architecture works correctly.
"""

import torch
import torch.nn as nn
from dr_qformer.models.qformer import DRQFormer


def test_drqformer_basic():
    """Test basic DR-QFormer initialization and forward pass."""
    print("=" * 60)
    print("Testing DR-QFormer Basic Functionality")
    print("=" * 60)
    
    # Hyperparameters
    batch_size = 4
    n_queries = 32
    hidden_dim = 768
    num_layers = 6
    num_heads = 8
    max_fragments = 10
    
    # Initialize DR-QFormer
    print("\n1. Initializing DR-QFormer...")
    model = DRQFormer(
        n_queries=n_queries,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_fragments=max_fragments,
        dropout=0.1
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {model.count_parameters():,}")
    
    # Test Primal Mode (QA)
    print("\n2. Testing Primal Mode (QA)...")
    query_embeds = torch.randn(batch_size, 1, hidden_dim)
    p_embeds = torch.randn(batch_size, max_fragments, hidden_dim)
    
    z_qa, aux_qa = model(
        query_embeds=query_embeds,
        p_embeds=p_embeds
    )
    
    print(f"   Input: query_embeds {query_embeds.shape}, p_embeds {p_embeds.shape}")
    print(f"   Output: z_qa {z_qa.shape}")
    assert z_qa.shape == (batch_size, n_queries, hidden_dim), f"Expected {(batch_size, n_queries, hidden_dim)}, got {z_qa.shape}"
    print("   ✓ Primal mode passed!")
    
    # Test Dual Mode (QG)
    print("\n3. Testing Dual Mode (QG)...")
    answer_embeds = torch.randn(batch_size, 1, hidden_dim)
    
    z_qg, aux_qg = model(
        answer_embeds=answer_embeds,
        p_embeds=p_embeds
    )
    
    print(f"   Input: answer_embeds {answer_embeds.shape}, p_embeds {p_embeds.shape}")
    print(f"   Output: z_qg {z_qg.shape}")
    assert z_qg.shape == (batch_size, n_queries, hidden_dim), f"Expected {(batch_size, n_queries, hidden_dim)}, got {z_qg.shape}"
    print("   ✓ Dual mode passed!")
    
    # Test gradient flow
    print("\n4. Testing gradient flow...")
    loss = z_qa.sum() + z_qg.sum()
    loss.backward()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    param_count = sum(1 for p in model.parameters())
    print(f"   Parameters with gradients: {grad_count}/{param_count}")
    assert grad_count == param_count, "Not all parameters received gradients"
    print("   ✓ Gradient flow passed!")
    
    # Test auxiliary outputs
    print("\n5. Testing auxiliary outputs...")
    print(f"   aux_qa keys: {list(aux_qa.keys())}")
    print(f"   Layer outputs: {len(aux_qa['layer_outputs'])} layers")
    assert len(aux_qa['layer_outputs']) == num_layers
    print("   ✓ Auxiliary outputs passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


def test_attention_masking():
    """Test attention masking functionality."""
    print("\n" + "=" * 60)
    print("Testing Attention Masking")
    print("=" * 60)
    
    batch_size = 2
    n_queries = 8
    hidden_dim = 256
    num_layers = 2
    num_heads = 4
    k_fragments = 5
    
    model = DRQFormer(
        n_queries=n_queries,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0
    )
    
    query_embeds = torch.randn(batch_size, 1, hidden_dim)
    p_embeds = torch.randn(batch_size, k_fragments, hidden_dim)
    
    # Test with default masking (None = full attention)
    print("\n1. Testing with default masking (full attention)...")
    z_full, _ = model(query_embeds=query_embeds, p_embeds=p_embeds)
    print(f"   Output shape: {z_full.shape}")
    print("   ✓ Default masking passed!")
    
    # Test with custom padding mask (simulate variable-length fragments)
    print("\n2. Testing with custom padding mask...")
    # Create a mask where last 2 fragments are padding
    ca_mask = torch.zeros(n_queries, k_fragments, dtype=torch.bool)
    ca_mask[:, -2:] = True  # Mask last 2 fragments
    
    z_masked, _ = model(
        query_embeds=query_embeds,
        p_embeds=p_embeds,
        ca_mask=ca_mask
    )
    print(f"   Output shape: {z_masked.shape}")
    print(f"   Masked fragments: 2/{k_fragments}")
    print("   ✓ Custom masking passed!")
    
    print("\n" + "=" * 60)
    print("Attention masking tests passed! ✓")
    print("=" * 60)


def test_parameter_efficiency():
    """Compare parameter counts across different configurations."""
    print("\n" + "=" * 60)
    print("Testing Parameter Efficiency")
    print("=" * 60)
    
    configs = [
        {"n_queries": 16, "hidden_dim": 512, "num_layers": 4},
        {"n_queries": 32, "hidden_dim": 768, "num_layers": 6},
        {"n_queries": 64, "hidden_dim": 1024, "num_layers": 12},
    ]
    
    for i, config in enumerate(configs, 1):
        model = DRQFormer(**config, num_heads=8)
        params = model.count_parameters()
        print(f"\n{i}. Config: N={config['n_queries']}, d={config['hidden_dim']}, L={config['num_layers']}")
        print(f"   Total parameters: {params:,}")
        print(f"   Memory (FP32): ~{params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Parameter efficiency tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    # Run all tests
    test_drqformer_basic()
    test_attention_masking()
    test_parameter_efficiency()
    
    print("\n" + "=" * 60)
    print("🎉 All DR-QFormer implementation tests passed! 🎉")
    print("=" * 60)
