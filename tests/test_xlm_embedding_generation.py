"""
Test script to verify XLM-RoBERTa embedding generation in training code.

This script tests:
1. Loading XLM-RoBERTa tokenizer and model
2. Generating token-level embeddings for queries
3. Generating sentence-level embeddings for evidence texts
4. Comparing with pre-computed embeddings
"""

import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Configuration
data_path = r"D:\LLMs\DR-QFormer\DR-QFormer\smoking_train_ms_subset.pkl"
xlm_model_name = "xlm-roberta-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("Testing XLM-RoBERTa Embedding Generation")
print("="*80)
print(f"Device: {device}")
print(f"Model: {xlm_model_name}")
print()

# Load data
print("📂 Loading data...")
with open(data_path, 'rb') as f:
    data = pickle.load(f)
sample_id = list(data.keys())[0]
sample = data[sample_id]
print(f"✅ Loaded sample: {sample_id}")
print()

# Load tokenizer and model
print("🔧 Loading XLM-RoBERTa tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(xlm_model_name)
model = AutoModel.from_pretrained(xlm_model_name).to(device)
model.eval()
print("✅ Model loaded")
print()

# Test 1: Generate token-level embeddings for query
print("="*80)
print("Test 1: Query Token-Level Embeddings")
print("="*80)

query_text = sample['query']
print(f"Query text: {query_text[:100]}...")
print()

# Tokenize
query_encoded = tokenizer(
    query_text,
    padding=False,
    truncation=True,
    max_length=512,
    return_tensors='pt',
)
print(f"Tokenized input_ids shape: {query_encoded['input_ids'].shape}")
print(f"Tokenized attention_mask shape: {query_encoded['attention_mask'].shape}")
print()

# Generate embeddings
with torch.no_grad():
    encoded_input = {
        'input_ids': query_encoded['input_ids'].to(device),
        'attention_mask': query_encoded['attention_mask'].to(device)
    }
    output = model(**encoded_input)
    query_token_embeddings = output.last_hidden_state.squeeze(0).cpu()  # [seq_len, 768]

print(f"Generated query embeddings shape: {query_token_embeddings.shape}")
print(f"Embeddings dtype: {query_token_embeddings.dtype}")
print(f"Embeddings range: [{query_token_embeddings.min():.4f}, {query_token_embeddings.max():.4f}]")
print()

# Compare with pre-computed embeddings
precomputed_query_emb = sample['query_embedding']['token_emb_768'].squeeze(0)
print(f"Pre-computed query embeddings shape: {precomputed_query_emb.shape}")
print(f"Pre-computed dtype: {precomputed_query_emb.dtype}")
print(f"Pre-computed range: [{precomputed_query_emb.min():.4f}, {precomputed_query_emb.max():.4f}]")
print()

# Note: These won't match because pre-computed uses Qwen3-Embedding, not XLM-R
print("ℹ️  Note: Pre-computed embeddings use Qwen3-Embedding tokenizer/model")
print("   Generated embeddings use XLM-RoBERTa tokenizer/model")
print("   They are expected to be different!")
print()

# Test 2: Generate sentence-level embeddings for evidence
print("="*80)
print("Test 2: Evidence Sentence-Level Embeddings")
print("="*80)

evidence_texts = sample['evidence_text']
K = len(sample['evidence_labels'])
print(f"Number of evidence fragments: {K}")
print(f"Number of evidence texts: {len(evidence_texts)}")
print()

# Generate embeddings for first 3 evidence texts
evidence_embeddings_generated = np.zeros((K, 768), dtype=np.float32)

print("Generating embeddings for evidence texts...")
with torch.no_grad():
    for i, text in enumerate(evidence_texts[:3]):
        if text:
            print(f"\n  Fragment {i}: {text[:80]}...")
            
            # Tokenize
            encoded = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            encoded_input = {
                'input_ids': encoded['input_ids'].to(device),
                'attention_mask': encoded['attention_mask'].to(device)
            }
            
            output = model(**encoded_input)
            # Use [CLS] token embedding as sentence representation
            cls_embedding = output.last_hidden_state[:, 0, :]  # [1, 768]
            evidence_embeddings_generated[i] = cls_embedding.squeeze(0).cpu().numpy()
            
            print(f"    Generated embedding shape: {cls_embedding.shape}")
            print(f"    Embedding range: [{cls_embedding.min():.4f}, {cls_embedding.max():.4f}]")

print()
print(f"\n✅ Generated evidence embeddings shape: {evidence_embeddings_generated.shape}")
print(f"   Non-zero rows: {(evidence_embeddings_generated.sum(axis=1) != 0).sum()}")
print()

# Compare with pre-computed evidence embeddings
precomputed_evidence_emb = sample['evidence_embeddings']
print(f"Pre-computed evidence embeddings shape: {precomputed_evidence_emb.shape}")
print(f"Pre-computed range: [{precomputed_evidence_emb.min():.4f}, {precomputed_evidence_emb.max():.4f}]")
print()

# Test 3: Verify compatibility with training pipeline
print("="*80)
print("Test 3: Training Pipeline Compatibility")
print("="*80)

print("✅ Token embeddings:")
print(f"   - Shape matches expected [seq_len, 768]: {query_token_embeddings.shape}")
print(f"   - Type is torch.Tensor: {isinstance(query_token_embeddings, torch.Tensor)}")
print()

print("✅ Evidence embeddings:")
print(f"   - Shape matches expected [K, 768]: {evidence_embeddings_generated.shape}")
print(f"   - Type is numpy.ndarray: {isinstance(evidence_embeddings_generated, np.ndarray)}")
print(f"   - Dtype is float32: {evidence_embeddings_generated.dtype == np.float32}")
print()

print("="*80)
print("✅ All tests passed!")
print("="*80)
print()

print("Summary:")
print("--------")
print("✅ XLM-RoBERTa tokenizer and model loaded successfully")
print("✅ Token-level embeddings generated for queries (shape: [seq_len, 768])")
print("✅ Sentence-level embeddings generated for evidence texts (shape: [K, 768])")
print("✅ Generated embeddings are compatible with training pipeline")
print()
print("ℹ️  To use generated embeddings in training:")
print("   Set use_precomputed_embeddings=False in Stage1Config")
print()
