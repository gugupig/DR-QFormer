"""
Test script to verify Random Q-Former and XLM-RoBERTa Q-Former process the same input shape.
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.qformer_random_init import DRQFormer
from src.models.qformer_xlm import XLMRobertaDRQFormer


def test_input_shapes():
    """Test that both Q-Formers can handle the same input shapes."""
    print("="*80)
    print("Testing Q-Former Input Shape Compatibility")
    print("="*80)
    
    # Test parameters
    batch_size = 4
    num_tokens = 32  # Query token sequence length
    num_evidence = 10  # K
    hidden_dim = 768
    
    # Random Q-Former
    print("\n1️⃣ Testing Random-Init Q-Former...")
    random_qformer = DRQFormer(
        n_queries=32,
        hidden_dim=hidden_dim,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
    )
    
    # Input: Token-level query embeddings [batch, T, 768]
    query_token_embeddings = torch.randn(batch_size, num_tokens, hidden_dim)
    evidence_embeddings = torch.randn(batch_size, num_evidence, hidden_dim)
    pool_padding_mask = torch.ones(batch_size, num_evidence, dtype=torch.bool)
    
    print(f"   Input shape: query_embeds={query_token_embeddings.shape}")
    print(f"   Evidence shape: p_embeds={evidence_embeddings.shape}")
    
    try:
        Z_random, aux_random = random_qformer(
            query_embeds=query_token_embeddings,
            p_embeds=evidence_embeddings,
            pool_padding_mask=pool_padding_mask,
        )
        print(f"   ✅ Output shape: {Z_random.shape}")
        print(f"   Expected: [batch={batch_size}, N_lq=32, hidden_dim={hidden_dim}]")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # XLM-RoBERTa Q-Former
    print("\n2️⃣ Testing XLM-RoBERTa Q-Former...")
    xlm_qformer = XLMRobertaDRQFormer(
        xlm_model_name="xlm-roberta-base",  # Use alias to avoid re-downloading
        n_queries=32,
        use_ca_layers=[True]*12,
        bypass_embeddings=True,  # Use pre-computed embeddings
    )
    
    # Same input as Random Q-Former
    input_ids = torch.ones(batch_size, num_tokens, dtype=torch.long)
    attention_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)  # Must be bool!
    evidence_mask = torch.ones(batch_size, num_evidence, dtype=torch.bool)  # Must be bool!
    
    print(f"   Input shape: precomputed_query_emb={query_token_embeddings.shape}")
    print(f"   Evidence shape: evidence_emb={evidence_embeddings.shape}")
    
    try:
        Z_xlm, aux_xlm = xlm_qformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            evidence_emb=evidence_embeddings,
            evidence_mask=evidence_mask,
            precomputed_query_emb=query_token_embeddings,
        )
        print(f"   ✅ Output shape: {Z_xlm.shape}")
        print(f"   Expected: [batch={batch_size}, N_lq=32, hidden_dim={hidden_dim}]")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Compare output shapes
    print("\n3️⃣ Comparing outputs...")
    if Z_random.shape == Z_xlm.shape:
        print(f"   ✅ Output shapes match: {Z_random.shape}")
    else:
        print(f"   ❌ Output shapes mismatch!")
        print(f"      Random: {Z_random.shape}")
        print(f"      XLM-R: {Z_xlm.shape}")
        return False
    
    print("\n" + "="*80)
    print("✅ All tests passed! Both Q-Formers handle the same input shapes.")
    print("="*80)
    return True


def test_single_token_backward_compatibility():
    """Test that Random Q-Former still works with single-token input [batch, 1, 768]."""
    print("\n" + "="*80)
    print("Testing Backward Compatibility (Single Token Input)")
    print("="*80)
    
    random_qformer = DRQFormer(
        n_queries=32,
        hidden_dim=768,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
    )
    
    batch_size = 4
    single_token_emb = torch.randn(batch_size, 1, 768)  # [batch, 1, 768]
    evidence_embeddings = torch.randn(batch_size, 10, 768)
    pool_padding_mask = torch.ones(batch_size, 10, dtype=torch.bool)
    
    print(f"   Input: Single token embedding {single_token_emb.shape}")
    
    try:
        Z, aux = random_qformer(
            query_embeds=single_token_emb,
            p_embeds=evidence_embeddings,
            pool_padding_mask=pool_padding_mask,
        )
        print(f"   ✅ Output shape: {Z.shape}")
        print(f"   Backward compatibility maintained!")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n🔍 Q-Former Shape Compatibility Test\n")
    
    # Test 1: Multi-token input
    test1_passed = test_input_shapes()
    
    # Test 2: Single-token backward compatibility
    test2_passed = test_single_token_backward_compatibility()
    
    print("\n" + "="*80)
    print("📊 Test Summary")
    print("="*80)
    print(f"   Multi-token input test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Single-token test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✨ All tests passed! Q-Formers are now compatible.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    print("="*80)
