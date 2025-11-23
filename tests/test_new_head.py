"""
Test script to verify new parameterized EntailmentHead works correctly.
"""

import torch
from src.models.heads import EntailmentHead

def test_entailment_head():
    """Test new BLIP-2 style EntailmentHead with learnable parameters."""
    
    print("="*80)
    print("Testing New Parameterized EntailmentHead")
    print("="*80)
    
    # Create head
    head = EntailmentHead(
        hidden_dim=768,
        num_fragments=10,
        tau=0.5,
        p_drop_lq=0.1,
        focal_gamma=1.5,
        focal_alpha=0.85
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\n✅ EntailmentHead created with {num_params:,} trainable parameters")
    
    # Print parameter breakdown
    print("\nParameter breakdown:")
    for name, param in head.named_parameters():
        print(f"  - {name}: {param.shape} ({param.numel():,} params)")
    
    # Create mock inputs
    batch_size = 4
    N = 32  # num learnable queries
    K = 10  # num fragments
    num_heads = 12
    
    # Q-Former output Z
    z = torch.randn(batch_size, N, 768)
    
    # CA raw scores (one layer)
    ca_raw_scores = torch.randn(batch_size, num_heads, N, K)
    ca_raw_scores_per_head = [ca_raw_scores]
    
    # Pool padding mask
    pool_padding_mask = torch.ones(batch_size, K, dtype=torch.bool)
    pool_padding_mask[0, 8:] = False  # First sample has only 8 valid fragments
    
    print(f"\n📊 Input shapes:")
    print(f"  - z: {z.shape}")
    print(f"  - ca_raw_scores: {ca_raw_scores.shape}")
    print(f"  - pool_padding_mask: {pool_padding_mask.shape}")
    
    # Forward pass
    print("\n🔄 Running forward pass...")
    head.train()
    output = head(
        z=z,
        ca_raw_scores_per_head=ca_raw_scores_per_head,
        pool_padding_mask=pool_padding_mask,
        training=True
    )
    
    fragment_logits = output['fragment_logits']
    print(f"✅ Output shape: {fragment_logits.shape}")
    print(f"✅ Output range: [{fragment_logits.min():.4f}, {fragment_logits.max():.4f}]")
    
    # Check gradient flow
    print("\n🔍 Testing gradient flow...")
    loss = fragment_logits.sum()
    loss.backward()
    
    grads_found = []
    for name, param in head.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grads_found.append((name, grad_norm))
    
    if grads_found:
        print(f"✅ Gradients flowing through {len(grads_found)} parameters:")
        for name, grad_norm in grads_found:
            print(f"  - {name}: grad_norm = {grad_norm:.4f}")
    else:
        print("❌ NO GRADIENTS FOUND!")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_entailment_head()
