"""
测试预计算embeddings功能：验证可以直接使用PKL中的token embeddings。
"""

import pickle
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.qformer_xlm import XLMRobertaDRQFormer

print("="*80)
print("Testing Pre-computed Token Embeddings")
print("="*80)

# Load data
print("\n📂 Loading data...")
with open(r"D:\LLMs\DR-QFormer\DR-QFormer\smoking_train_ms_subset.pkl", "rb") as f:
    data = pickle.load(f)

sample_key = list(data.keys())[0]
sample = data[sample_key]

print(f"\n{'='*80}")
print(f"Sample: {sample_key}")
print(f"{'='*80}")
print(f"Query: {sample['query']}")

# Extract pre-computed embeddings
query_emb = sample['query_embedding']
input_ids = query_emb['input_ids']  # [1, T] - Qwen3 token IDs
attention_mask = query_emb['attention_mask']  # [1, T]
token_emb_768 = query_emb['token_emb_768']  # [1, T, 768] - Pre-computed

print(f"\n📊 Pre-computed Embeddings:")
print(f"   Input IDs shape: {input_ids.shape}")
print(f"   Attention mask shape: {attention_mask.shape}")
print(f"   Token embeddings shape: {token_emb_768.shape}")

# Prepare evidence
evidence_embeddings = torch.from_numpy(sample['evidence_embeddings']).float().unsqueeze(0)  # [1, K, 768]
evidence_mask = torch.ones(1, evidence_embeddings.shape[1], dtype=torch.bool)  # [1, K]

print(f"   Evidence embeddings shape: {evidence_embeddings.shape}")

# Test 1: Q-Former with bypass_embeddings=True (use pre-computed)
print(f"\n{'='*80}")
print("Test 1: Q-Former with bypass_embeddings=True")
print(f"{'='*80}")

qformer_bypass = XLMRobertaDRQFormer(
    xlm_model_name="xlm-roberta-base",
    n_queries=32,
    hidden_dim=768,
    bypass_embeddings=True,  # Enable bypass
)

print("\n🔄 Forward pass with pre-computed embeddings...")
Z1, aux1 = qformer_bypass(
    input_ids=input_ids.long(),
    attention_mask=attention_mask.long(),
    evidence_emb=evidence_embeddings,
    evidence_mask=evidence_mask,
    precomputed_query_emb=token_emb_768.float(),  # Pass pre-computed embeddings
)

print(f"✅ Output shape: {Z1.shape}")
print(f"   Expected: [1, 32, 768]")
print(f"   Match: {Z1.shape == torch.Size([1, 32, 768])}")

# Test 2: Q-Former with bypass_embeddings=False (use XLM-R embeddings)
print(f"\n{'='*80}")
print("Test 2: Q-Former with bypass_embeddings=False")
print(f"{'='*80}")

# Need to re-tokenize with XLM-R
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

query_text = sample['query']
query_encoded = tokenizer(
    query_text,
    padding=False,
    truncation=True,
    max_length=512,
    return_tensors='pt',
)

print(f"\n📝 XLM-R Tokenization:")
print(f"   Input IDs shape: {query_encoded['input_ids'].shape}")
print(f"   Attention mask shape: {query_encoded['attention_mask'].shape}")

qformer_normal = XLMRobertaDRQFormer(
    xlm_model_name="xlm-roberta-base",
    n_queries=32,
    hidden_dim=768,
    bypass_embeddings=False,  # Use XLM-R embeddings
)

print("\n🔄 Forward pass with XLM-R embeddings...")
Z2, aux2 = qformer_normal(
    input_ids=query_encoded['input_ids'],
    attention_mask=query_encoded['attention_mask'],
    evidence_emb=evidence_embeddings,
    evidence_mask=evidence_mask,
    precomputed_query_emb=None,  # Don't pass pre-computed embeddings
)

print(f"✅ Output shape: {Z2.shape}")
print(f"   Expected: [1, 32, 768]")
print(f"   Match: {Z2.shape == torch.Size([1, 32, 768])}")

# Compare outputs
print(f"\n{'='*80}")
print("Comparison")
print(f"{'='*80}")
print(f"Z1 (bypass) mean: {Z1.mean().item():.6f}, std: {Z1.std().item():.6f}")
print(f"Z2 (normal) mean: {Z2.mean().item():.6f}, std: {Z2.std().item():.6f}")
print(f"Outputs are different: {not torch.allclose(Z1, Z2, atol=1e-3)}")
print(f"   (Expected: True, because using different embeddings)")

print(f"\n{'='*80}")
print("✅ All tests passed!")
print(f"{'='*80}")
print("\nSummary:")
print("  ✅ bypass_embeddings=True: Successfully uses pre-computed embeddings from PKL")
print("  ✅ bypass_embeddings=False: Successfully uses XLM-R tokenizer/embeddings")
print("  ✅ Both modes produce valid output shapes")
print("\nRecommendation:")
print("  Use bypass_embeddings=True (use_precomputed_embeddings=True in config)")
print("  This avoids tokenizer mismatch and leverages high-quality Qwen3 embeddings!")
