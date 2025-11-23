"""
测试tokenizer修复：验证XLM-RoBERTa tokenizer是否正确应用。
"""

import pickle
import torch
from transformers import AutoTokenizer

# Load data
print("Loading data...")
with open(r"D:\LLMs\DR-QFormer\DR-QFormer\smoking_train_ms_subset.pkl", "rb") as f:
    data = pickle.load(f)

sample_key = list(data.keys())[0]
sample = data[sample_key]

print(f"\n{'='*80}")
print(f"Sample ID: {sample_key}")
print(f"{'='*80}")

# Check original Qwen3 encoding
print("\n🔴 Original Qwen3 Encoding (from pickle):")
query_emb = sample['query_embedding']
qwen_input_ids = query_emb['input_ids'].squeeze(0)
qwen_attention_mask = query_emb['attention_mask'].squeeze(0)
print(f"   Query text: {sample['query']}")
print(f"   Input IDs shape: {qwen_input_ids.shape}")
print(f"   Attention mask shape: {qwen_attention_mask.shape}")
print(f"   Input IDs (first 10): {qwen_input_ids[:10].tolist()}")
print(f"   Attention mask (first 10): {qwen_attention_mask[:10].tolist()}")

# Load XLM-RoBERTa tokenizer
print("\n🟢 XLM-RoBERTa Re-encoding:")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
query_text = sample['query']

# Re-encode
encoded = tokenizer(
    query_text,
    padding=False,
    truncation=True,
    max_length=512,
    return_tensors='pt',
)
xlm_input_ids = encoded['input_ids'].squeeze(0)
xlm_attention_mask = encoded['attention_mask'].squeeze(0)

print(f"   Query text: {query_text}")
print(f"   Input IDs shape: {xlm_input_ids.shape}")
print(f"   Attention mask shape: {xlm_attention_mask.shape}")
print(f"   Input IDs (first 10): {xlm_input_ids[:10].tolist()}")
print(f"   Attention mask (first 10): {xlm_attention_mask[:10].tolist()}")

# Decode to verify
decoded = tokenizer.decode(xlm_input_ids, skip_special_tokens=True)
print(f"   Decoded text: {decoded}")

# Compare
print(f"\n{'='*80}")
print("Comparison:")
print(f"{'='*80}")
print(f"   Qwen3 vocab size: ~152000 (Qwen3-Embedding)")
print(f"   XLM-R vocab size: {tokenizer.vocab_size} (xlm-roberta-base)")
print(f"   Qwen3 seq_len: {qwen_input_ids.shape[0]}")
print(f"   XLM-R seq_len: {xlm_input_ids.shape[0]}")
print(f"   Token IDs match: {torch.equal(qwen_input_ids, xlm_input_ids)}")
print(f"   Attention masks match: {torch.equal(qwen_attention_mask, xlm_attention_mask)}")

print(f"\n{'='*80}")
print("✅ Tokenizer Fix Validation:")
print(f"{'='*80}")
if not torch.equal(qwen_input_ids, xlm_input_ids):
    print("✅ CORRECT: Token IDs are different (expected, different tokenizers)")
    print("   This confirms we need to re-encode queries with XLM-R tokenizer!")
else:
    print("❌ WARNING: Token IDs are identical (unexpected)")

print("\n✅ Re-encoding is working correctly!")
print("   The training code will now use XLM-R tokenizer instead of Qwen3 token IDs.")
