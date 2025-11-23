"""
Quick check: Does precomputed mode correctly use PKL's input_ids and attention_mask?
"""
import pickle
import numpy as np

data_path = r"D:\LLMs\DR-QFormer\DR-QFormer\smoking_train_ms_subset.pkl"

print("Loading data...")
with open(data_path, 'rb') as f:
    data = pickle.load(f)

sample_id = list(data.keys())[0]
sample = data[sample_id]

print(f"\nSample ID: {sample_id}")
print("\n=== query_embedding structure ===")
query_emb = sample['query_embedding']
print(f"Keys: {query_emb.keys()}")
print(f"\ninput_ids shape: {query_emb['input_ids'].shape}")
print(f"attention_mask shape: {query_emb['attention_mask'].shape}")
print(f"token_emb_768 shape: {query_emb['token_emb_768'].shape}")

print(f"\ninput_ids (first 10): {query_emb['input_ids'].squeeze(0)[:10]}")
print(f"attention_mask (first 10): {query_emb['attention_mask'].squeeze(0)[:10]}")

print("\n" + "="*80)
print("ISSUE IDENTIFIED:")
print("="*80)
print("❌ Problem: In precomputed mode (use_precomputed_embeddings=True):")
print("   - input_ids from PKL are Qwen3 token IDs")
print("   - These IDs are passed to Q-Former's input_ids parameter")
print("   - But Q-Former has bypass_embeddings=False by default!")
print("   - So Q-Former's embedding layer treats Qwen3 IDs as XLM-R IDs")
print("   - This causes nonsensical embeddings!")
print()
print("✅ Solution: When use_precomputed_embeddings=True:")
print("   - Q-Former should be initialized with bypass_embeddings=True")
print("   - This tells Q-Former to ignore input_ids and use precomputed_query_emb")
print("="*80)
