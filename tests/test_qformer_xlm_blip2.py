"""
Test script for XLM-RoBERTa DR-QFormer with BLIP-2 style token-level query processing.

This test verifies:
1. Query tokens participate in self-attention with LQs
2. LQs become query-aware through SA interactions
3. Query-aware LQs attend to evidence via CA
4. CA diagnostic outputs are preserved
5. Forward pass works with padding in both query and evidence
"""

import torch
from src.models.qformer_xlm import XLMRobertaDRQFormer

def test_blip2_style_qformer():
    print("=" * 80)
    print("BLIP-2 Style XLM-RoBERTa DR-QFormer Test")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    batch_size = 2
    n_queries = 32
    seq_len = 16  # Number of query tokens
    num_fragments = 5
    hidden_dim = 768
    num_heads = 12
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Initialize model
    print("\n" + "-" * 80)
    print("Test 1: Model Initialization")
    print("-" * 80)
    
    model = XLMRobertaDRQFormer(
        xlm_model_name="xlm-roberta-base",
        n_queries=n_queries,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.1,
        use_ca_layers=[5, 11],  # Apply CA only after layers 6 and 12
        freeze_xlmr=False,
    )
    model = model.to(device)
    model.eval()
    
    print("✓ Model initialized successfully")
    print(f"  Total parameters: {model.count_parameters():,}")
    
    # Test 2: Forward pass without padding
    print("\n" + "-" * 80)
    print("Test 2: Forward Pass (No Padding)")
    print("-" * 80)
    
    # Random token IDs (within XLM-R vocab size: 250002)
    input_ids = torch.randint(0, 250000, (batch_size, seq_len), device=device)
    
    # Attention mask (all valid)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    
    # Evidence embeddings
    evidence_emb = torch.randn(batch_size, num_fragments, hidden_dim, device=device)
    
    # Evidence mask (all valid)
    evidence_mask = torch.ones(batch_size, num_fragments, dtype=torch.bool, device=device)
    
    print(f"Input shapes:")
    print(f"  input_ids: {input_ids.shape} (query tokens)")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  evidence_emb: {evidence_emb.shape}")
    print(f"  evidence_mask: {evidence_mask.shape}")
    
    with torch.no_grad():
        Z, all_aux = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            evidence_emb=evidence_emb,
            evidence_mask=evidence_mask,
        )
    
    print(f"\nOutput shapes:")
    print(f"  Z (final LQ representations): {Z.shape}")
    assert Z.shape == (batch_size, n_queries, hidden_dim), \
        f"Expected Z shape [{batch_size}, {n_queries}, {hidden_dim}], got {Z.shape}"
    print(f"  ✓ Shape correct: [{batch_size}, {n_queries}, {hidden_dim}]")
    
    print(f"\n  Number of aux dicts: {len(all_aux)}")
    assert len(all_aux) == 12, f"Expected 12 aux dicts (one per layer), got {len(all_aux)}"
    print(f"  ✓ Correct number of layers: 12")
    
    # Test 3: Verify CA layers
    print("\n" + "-" * 80)
    print("Test 3: CA Layer Verification")
    print("-" * 80)
    
    expected_ca_layers = [5, 11]
    
    for i in expected_ca_layers:
        if all_aux[i] and 'ca_attn_weights' in all_aux[i] and all_aux[i]['ca_attn_weights'] is not None:
            print(f"\n✓ Layer {i}: CA applied correctly")
            ca_weights = all_aux[i]['ca_attn_weights']
            ca_raw_per_head = all_aux[i]['ca_raw_scores_per_head']
            ca_raw_avg = all_aux[i]['ca_raw_scores_avg']
            
            print(f"  ca_attn_weights shape: {ca_weights.shape}")
            assert ca_weights.shape == (batch_size, num_heads, n_queries, num_fragments), \
                f"Expected [{batch_size}, {num_heads}, {n_queries}, {num_fragments}], got {ca_weights.shape}"
            
            print(f"  ca_raw_scores_per_head shape: {ca_raw_per_head.shape}")
            assert ca_raw_per_head.shape == (batch_size, num_heads, n_queries, num_fragments)
            
            print(f"  ca_raw_scores_avg shape: {ca_raw_avg.shape}")
            assert ca_raw_avg.shape == (batch_size, n_queries, num_fragments)
            
            # Check that attention weights sum to 1
            attn_sum = ca_weights.sum(dim=-1)
            assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), \
                "Attention weights should sum to 1 across fragments"
            print(f"  ✓ Attention weights sum to 1")
        else:
            print(f"✗ Layer {i}: CA NOT applied (unexpected)")
            assert False, f"CA should be applied at layer {i}"
    
    # Check that non-CA layers have empty dicts
    for i in range(12):
        if i not in expected_ca_layers:
            assert all_aux[i] == {}, f"Layer {i} should have empty aux dict"
    print(f"\n✓ Non-CA layers have empty aux dicts")
    
    # Test 4: Forward pass with padding
    print("\n" + "-" * 80)
    print("Test 4: Forward Pass (With Padding)")
    print("-" * 80)
    
    # Create attention mask with padding (second half is padding)
    attention_mask_padded = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    attention_mask_padded[:, seq_len//2:] = 0  # Second half is padding
    
    # Create evidence mask with some padding
    evidence_mask_padded = torch.ones(batch_size, num_fragments, dtype=torch.bool, device=device)
    evidence_mask_padded[:, -2:] = False  # Last 2 fragments are padding
    
    print(f"Attention mask: {attention_mask_padded[0].tolist()} (first sample)")
    print(f"Evidence mask: {evidence_mask_padded[0].tolist()} (first sample)")
    
    with torch.no_grad():
        Z_padded, all_aux_padded = model(
            input_ids=input_ids,
            attention_mask=attention_mask_padded,
            evidence_emb=evidence_emb,
            evidence_mask=evidence_mask_padded,
        )
    
    print(f"\nOutput shape: {Z_padded.shape}")
    assert Z_padded.shape == (batch_size, n_queries, hidden_dim)
    print("✓ Forward pass with padding successful")
    
    # Verify CA attention doesn't attend to padded evidence
    for i in expected_ca_layers:
        ca_weights = all_aux_padded[i]['ca_attn_weights']  # [B, H, N_q, K]
        
        # Check that padded positions have near-zero attention
        padded_positions = ~evidence_mask_padded  # [B, K]
        padded_positions_expanded = padded_positions.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K]
        
        # Extract attention to padded positions
        attn_to_padding = ca_weights.masked_select(padded_positions_expanded.expand_as(ca_weights))
        
        # These should be very small (close to 0 after softmax)
        assert attn_to_padding.max() < 0.1, \
            f"Attention to padded evidence should be near 0, got max {attn_to_padding.max()}"
        
        print(f"✓ Layer {i}: Attention to padded evidence is near 0 (max: {attn_to_padding.max():.6f})")
    
    # Test 5: Verify query-awareness
    print("\n" + "-" * 80)
    print("Test 5: Query-Awareness Verification")
    print("-" * 80)
    
    # Create two different query inputs
    input_ids_1 = torch.randint(0, 250000, (1, seq_len), device=device)
    input_ids_2 = torch.randint(100000, 200000, (1, seq_len), device=device)  # Different range
    
    attention_mask_single = torch.ones(1, seq_len, dtype=torch.long, device=device)
    evidence_emb_single = torch.randn(1, num_fragments, hidden_dim, device=device)
    evidence_mask_single = torch.ones(1, num_fragments, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        Z1, _ = model(input_ids_1, attention_mask_single, evidence_emb_single, evidence_mask_single)
        Z2, _ = model(input_ids_2, attention_mask_single, evidence_emb_single, evidence_mask_single)
    
    # LQs should be different for different queries (due to SA interaction)
    difference = (Z1 - Z2).abs().mean().item()
    print(f"Mean absolute difference between LQs for different queries: {difference:.6f}")
    assert difference > 0.01, "LQs should differ for different queries (query-awareness test)"
    print("✓ LQs are query-aware (different for different queries)")
    
    # Test 6: Gradient flow check
    print("\n" + "-" * 80)
    print("Test 6: Gradient Flow Check")
    print("-" * 80)
    
    model.train()
    
    input_ids_grad = torch.randint(0, 250000, (batch_size, seq_len), device=device)
    attention_mask_grad = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    evidence_emb_grad = torch.randn(batch_size, num_fragments, hidden_dim, device=device, requires_grad=True)
    evidence_mask_grad = torch.ones(batch_size, num_fragments, dtype=torch.bool, device=device)
    
    Z_grad, _ = model(input_ids_grad, attention_mask_grad, evidence_emb_grad, evidence_mask_grad)
    
    # Dummy loss
    loss = Z_grad.sum()
    loss.backward()
    
    # Check that evidence_emb received gradients (via CA)
    assert evidence_emb_grad.grad is not None, "Evidence embeddings should receive gradients via CA"
    assert evidence_emb_grad.grad.abs().sum() > 0, "Gradients should be non-zero"
    print("✓ Gradients flow from LQs to evidence via CA")
    
    # Check that query_tokens parameter received gradients
    assert model.query_tokens.grad is not None, "Query tokens should receive gradients"
    print("✓ Gradients flow to learnable query tokens")
    
    model.eval()
    
    # Final summary
    print("\n" + "=" * 80)
    print("All Tests Passed! ✓")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Model initialization successful")
    print("  ✓ Forward pass works correctly")
    print("  ✓ Output shapes are correct")
    print("  ✓ CA layers applied at correct positions")
    print("  ✓ CA diagnostic outputs preserved")
    print("  ✓ Padding handled correctly")
    print("  ✓ LQs are query-aware (BLIP-2 pattern)")
    print("  ✓ Gradients flow correctly")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_blip2_style_qformer()
