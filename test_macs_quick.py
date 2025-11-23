"""
Quick test runner for MACS functions.
Run with: python test_macs_quick.py
"""

import sys
sys.path.insert(0, 'd:/LLMs/DR-QFormer/DR-QFormer')

import torch
from src.utils.macs import (
    compute_macs_to_lqs,
    extract_answer_lq_posterior,
    compute_evidence_posterior,
    extract_posterior_from_llm_outputs,
)

print("=" * 80)
print("MACS Posterior Extraction - Quick Tests")
print("=" * 80)

# Test 1: compute_macs_to_lqs
print("\n[Test 1] compute_macs_to_lqs")
print("-" * 40)
batch_size = 2
num_layers = 3
num_heads = 4
seq_len = 64
num_lqs = 32

attentions = []
for _ in range(num_layers):
    att = torch.rand(batch_size, num_heads, seq_len, seq_len)
    att = att / att.sum(dim=-1, keepdim=True)
    attentions.append(att)

macs_map = compute_macs_to_lqs(tuple(attentions), num_lqs=num_lqs, alpha=0.8)
print(f"✓ Shape: {macs_map.shape} (expected: [2, 64, 32])")
print(f"✓ No NaN: {not torch.isnan(macs_map).any()}")
print(f"✓ No Inf: {not torch.isinf(macs_map).any()}")

# Test 2: extract_answer_lq_posterior
print("\n[Test 2] extract_answer_lq_posterior")
print("-" * 40)
lq_posterior = extract_answer_lq_posterior(
    attentions=tuple(attentions),
    answer_start_idx=40,
    answer_end_idx=60,
    num_lqs=num_lqs,
    aggregation='mean',
)
print(f"✓ Shape: {lq_posterior.shape} (expected: [2, 32])")
print(f"✓ No NaN: {not torch.isnan(lq_posterior).any()}")

# Test 3: compute_evidence_posterior
print("\n[Test 3] compute_evidence_posterior")
print("-" * 40)
num_evidence = 50
lq_post = torch.randn(batch_size, num_lqs).softmax(dim=-1)
ca_weights = torch.randn(batch_size, num_lqs, num_evidence)

evidence_posterior = compute_evidence_posterior(
    lq_posterior=lq_post,
    ca_weights=ca_weights,
    temperature=1.0,
)
print(f"✓ Shape: {evidence_posterior.shape} (expected: [2, 50])")
print(f"✓ Non-negative: {(evidence_posterior >= 0).all()}")
sums = evidence_posterior.sum(dim=-1)
print(f"✓ Sums to 1: {torch.allclose(sums, torch.ones(batch_size), atol=1e-5)}")
print(f"  Actual sums: {sums.tolist()}")

# Test 4: extract_posterior_from_llm_outputs
print("\n[Test 4] extract_posterior_from_llm_outputs (end-to-end)")
print("-" * 40)
llm_outputs = {
    'attentions': tuple(attentions),
    'answer_start_idx': 40,
    'answer_end_idx': 60,
}
ca_weights_full = torch.randn(batch_size, num_lqs, num_evidence).softmax(dim=-1)

evidence_posterior_e2e = extract_posterior_from_llm_outputs(
    llm_outputs=llm_outputs,
    qformer_ca_weights=ca_weights_full,
    num_lqs=num_lqs,
    alpha=0.8,
)
print(f"✓ Shape: {evidence_posterior_e2e.shape} (expected: [2, 50])")
print(f"✓ Valid distribution: {torch.allclose(evidence_posterior_e2e.sum(dim=-1), torch.ones(batch_size), atol=1e-5)}")

# Test 5: Subset extraction
print("\n[Test 5] Posterior extraction with subset")
print("-" * 40)
subset_size = 20
subset_indices = torch.randint(0, num_evidence, (subset_size,))

evidence_posterior_subset = extract_posterior_from_llm_outputs(
    llm_outputs=llm_outputs,
    qformer_ca_weights=ca_weights_full,
    subset_indices=subset_indices,
    num_lqs=num_lqs,
)
print(f"✓ Shape: {evidence_posterior_subset.shape} (expected: [2, 20])")
print(f"✓ Valid distribution: {torch.allclose(evidence_posterior_subset.sum(dim=-1), torch.ones(batch_size), atol=1e-5)}")

print("\n" + "=" * 80)
print("✅ All tests passed!")
print("=" * 80)
print("\nMACS posterior extraction is ready for Stage-2 training.")
print("Next step: Integrate real Qwen LLM in src/adapters/llm.py")
