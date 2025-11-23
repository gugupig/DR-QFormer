# MACS Posterior Extraction Guide

**Document**: MACS×LQ-CA Posterior Extraction for Stage-2 Training  
**Author**: DR-QFormer Team  
**Date**: 2025-11-22  
**Status**: Implementation Complete (SA part ready, awaiting LLM integration)

---

## Overview

This document explains the **MACS×LQ-CA** (Multi-head Attention Consistency Scores × Learnable Query Cross-Attention) method for extracting posterior evidence importance from LLM attention patterns during Stage-2 training.

### Purpose

In Stage-2, we implement a **Bayesian-inspired prior-posterior feedback loop**:

1. **Prior** (Task S): Q-Former predicts fragment importance π(e|q) based on query
2. **Posterior** (Task C): Extract actual fragment usage q(e|q,a) from LLM attention during answer generation
3. **Alignment** (Task S Loss): Minimize JS divergence between prior and posterior

**Key Innovation**: Task S learns not just from reranker supervision (teacher), but from **what the LLM actually uses** during generation (posterior).

---

## Algorithm: MACS×LQ-CA in Two Parts

### Part 1: MACS (SA) - Answer Tokens → LQ Importance

**Goal**: Determine which LQs the LLM attended to during answer generation.

**Input**: LLM attention tensors from `teacher_forcing_dual_path()`
- `attentions`: Tuple of [batch, heads, seq_len, seq_len] (one per layer)
- Sequence structure: `[LQ_0, ..., LQ_{N-1}, system, question, answer]`

**Algorithm**:

```python
def compute_macs_to_lqs(attentions, num_lqs=32, alpha=0.8):
    """
    Aggregate attention across layers/heads to get token→LQ importance.
    
    Returns: [batch, seq_len, num_lqs]
    """
    # 1. Extract attention to first num_lqs positions
    target_attn = stack(attentions)[..., :num_lqs]  # [layers, B, H, S, N]
    
    # 2. Max-pool over heads (most attentive head wins)
    layer_max_attn = max(target_attn, dim=heads)  # [layers, B, S, N]
    
    # 3. Cumulative product with exponential smoothing
    joint_att = ones([B, S, N])
    for layer in layers:
        smoothed = alpha * layer_max_attn[layer] + (1-alpha) * 1.0
        joint_att *= smoothed
    
    # 4. Z-score normalization (highlight significant LQs)
    joint_att = (joint_att - mean) / (std + eps)
    
    return joint_att
```

**Extraction**:

```python
# Slice to answer tokens only
answer_start = num_lqs + len(system_tokens) + len(question_tokens)
answer_end = seq_len
answer_to_lqs = macs_map[:, answer_start:answer_end, :]  # [B, S_a, N]

# Aggregate over answer tokens
lq_posterior = answer_to_lqs.mean(dim=1)  # [B, N]
```

**Interpretation**:
- `lq_posterior[b, j]` = importance of LQ_j for generating answer in sample b
- Higher values = LQ was heavily attended during answer generation
- This represents **which LQs the LLM actually used**

---

### Part 2: Q-Former CA - LQ Importance × Evidence Attention → Evidence Posterior

**Goal**: Map LQ importance back to evidence fragments using Q-Former's cross-attention.

**Input**:
- `lq_posterior`: [batch, num_lqs] from Part 1 (MACS)
- `ca_weights`: [batch, num_lqs, K] Q-Former cross-attention weights (LQs → evidence)

**Formula**:

```
p(evidence_k | query, answer) = Σ_j p(LQ_j | answer) × p(evidence_k | LQ_j)
                                = Σ_j lq_posterior[j] × ca_weights[j, k]
```

**Implementation**:

```python
def compute_evidence_posterior(lq_posterior, ca_weights):
    """
    Combine LQ posterior with CA weights to get evidence posterior.
    
    Returns: [batch, K]
    """
    # Weighted sum over LQs
    evidence_logits = einsum('bn,bnk->bk', lq_posterior, ca_weights)
    
    # Softmax to get probability distribution
    evidence_posterior = softmax(evidence_logits, dim=-1)
    
    return evidence_posterior
```

**Interpretation**:
- `evidence_posterior[b, k]` = probability that evidence_k was actually used by LLM
- High posterior = evidence attended by important LQs
- Low posterior = evidence ignored by LQs that LLM used

---

## Integration in Stage-2 Training Loop

### Full Pipeline

```python
# 1. Q-Former Forward
qformer_outputs = qformer(query_embeddings, evidence_embeddings)
ca_weights = qformer_outputs['ca_raw_scores_avg'][-1].softmax(dim=-1)

# 2. Task C: LLM Teacher Forcing
z_prefix = condense_head(qformer_outputs['lqs_after'])
llm_outputs = frozen_llm.teacher_forcing_dual_path(
    z_prefix, query_ids, answer_ids, capture_attention=True
)

# 3. MACS Posterior Extraction (One-liner)
evidence_posterior = extract_posterior_from_llm_outputs(
    llm_outputs=llm_outputs,
    qformer_ca_weights=ca_weights,
    num_lqs=32,
    alpha=0.8,
)  # [batch, K]

# 4. Task S Loss with Posterior Feedback
loss_s = compute_ranking_loss(
    ranking_logits=ranking_outputs['logits'],
    gt_scores=teacher_scores,
    posterior_scores=evidence_posterior.detach(),  # Treat as teacher signal
    lambda_teacher=0.5,  # Weight for reranker supervision
    lambda_post=0.5,     # Weight for LLM posterior feedback
    ...
)
```

### Curriculum Learning Schedule

**Phase 1 (Warmup, 0-1000 steps)**:
- λ_teacher = 1.0 (pure reranker supervision)
- λ_post = 0.0 (no posterior feedback)
- **Why**: Let Q-Former learn basic ranking from strong teacher signal

**Phase 2 (Transition, 1000-5000 steps)**:
- λ_teacher: 1.0 → 0.2 (gradual decay)
- λ_post: 0.0 → 0.8 (gradual increase)
- **Why**: Smooth transition from teacher to posterior alignment

**Phase 3 (Steady, 5000+ steps)**:
- λ_teacher = 0.2 (minimal teacher guidance)
- λ_post = 0.8 (dominant posterior alignment)
- **Why**: Q-Former now aligns with LLM's true evidence usage patterns

---

## Implementation Status

### ✅ Completed

1. **Core MACS Algorithm** (`src/utils/macs.py`)
   - `compute_macs_to_lqs()`: Multi-layer, multi-head attention aggregation
   - `extract_answer_lq_posterior()`: SA part (answer → LQs)
   - `compute_evidence_posterior()`: CA part (LQs → evidence)
   - `extract_posterior_from_llm_outputs()`: End-to-end convenience function

2. **Integration Example** (`examples/stage2_posterior_extraction_example.py`)
   - Complete training step with posterior feedback
   - Curriculum learning schedule
   - Dummy batch demonstration

3. **Documentation**
   - Algorithm explanation with formulas
   - Code comments and docstrings
   - Integration guide

### ⚠️ Pending (LLM Integration)

1. **Real LLM Adapter** (`src/adapters/llm.py`)
   - Load actual Qwen LLM model
   - Implement `teacher_forcing_dual_path()` with real forward pass
   - Register attention hooks to capture `attentions` tuple
   - Construct Prefix-LM masks correctly
   - Test with Qwen-7B/14B

2. **Span Detection** (`src/utils/macs.py:extract_span_indices()`)
   - Currently placeholder (rough heuristic)
   - Need proper implementation using Qwen chat template
   - Identify `<|im_start|>`, `<|im_end|>` positions
   - Exclude special tokens from answer span

---

## Sanity Check Results (MACS_example.py)

**Setup**: Random LQs + Qwen chat template
- Model: Qwen-3-4B-Instruct
- LQs: 32 random embeddings (σ=0.02)
- Question: "What causes rain?"
- Answer: "Rain is caused by the condensation of water vapor into droplets."

**Observations**:
1. ✅ Different roles (user vs assistant) show distinct LQ attention patterns
2. ✅ Most attention mass concentrated on 1-2 LQs (expected for random/untrained LQs)
3. ✅ Heatmap visualization confirms MACS successfully extracts token→LQ relationships
4. ✅ Answer tokens show higher variance in LQ attention than question tokens

**Conclusion**: MACS correctly identifies which LQs are attended during generation, even with random LQs. After training, we expect:
- More uniform LQ usage (load balancing)
- Clear answer→LQ patterns linked to evidence relevance
- Strong correlation between high-posterior evidence and LLM answer quality

---

## Expected Impact on Training

### Quantitative Predictions

**Retrieval Quality (NDCG@10)**:
- Stage-1 (Teacher only): Baseline
- Stage-2 (Teacher + Posterior, early): +2-3% (posterior signal noisy initially)
- Stage-2 (Teacher + Posterior, late): +5-10% (posterior signal refined)

**Reasoning**: Task S learns not just "what reranker prefers" but "what LLM actually uses", directly addressing the retriever-LLM objective mismatch.

**Generation Quality (Answer NLL)**:
- Stage-1: N/A (no Task C)
- Stage-2: Monotonic decrease as Q-Former learns to compress useful evidence

### Qualitative Expectations

1. **Evidence Selection**: Q-Former prioritizes fragments that reduce LLM perplexity (not just reranker score)
2. **Compression**: LQs focus on information-dense spans that LLM attends to
3. **Alignment**: Prior π(e|q) and posterior q(e|q,a) converge (JS divergence decreases)

---

## Usage Examples

### Basic Usage

```python
from src.utils.macs import extract_posterior_from_llm_outputs

# In training loop
llm_outputs = frozen_llm.teacher_forcing_dual_path(z, q_ids, a_ids)
ca_weights = qformer_outputs['ca_weights']

evidence_posterior = extract_posterior_from_llm_outputs(
    llm_outputs=llm_outputs,
    qformer_ca_weights=ca_weights,
    num_lqs=32,
)
```

### Advanced Usage (With Dynamic Subset U)

```python
# Only compute posterior on subset U
subset_indices = subset_mask.nonzero(as_tuple=True)[1]
ca_weights_U = ca_weights[:, :, subset_indices]

evidence_posterior = extract_posterior_from_llm_outputs(
    llm_outputs=llm_outputs,
    qformer_ca_weights=ca_weights_U,
    subset_indices=None,  # Already subsetted
    num_lqs=32,
)

# Feed to Task S loss (only on subset U)
loss_s = compute_ranking_loss(
    ...,
    posterior_scores=evidence_posterior.detach(),
    subset_mask=subset_mask,
    lambda_post=current_lambda_post,
)
```

### Curriculum Learning

```python
def get_curriculum_weights(step, warmup=1000, transition=5000):
    if step < warmup:
        return {'lambda_teacher': 1.0, 'lambda_post': 0.0}
    elif step < transition:
        progress = (step - warmup) / (transition - warmup)
        return {
            'lambda_teacher': 1.0 - 0.8 * progress,  # 1.0 → 0.2
            'lambda_post': 0.8 * progress,           # 0.0 → 0.8
        }
    else:
        return {'lambda_teacher': 0.2, 'lambda_post': 0.8}

# In training loop
weights = get_curriculum_weights(current_step)
loss_s = compute_ranking_loss(..., **weights)
```

---

## Debugging Tips

### Check MACS Map Quality

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize answer→LQ attention
macs_map = compute_macs_to_lqs(attentions, num_lqs=32)
answer_to_lqs = macs_map[0, answer_start:answer_end, :]  # [S_a, 32]

plt.figure(figsize=(10, 6))
sns.heatmap(answer_to_lqs.cpu().numpy(), cmap='viridis')
plt.xlabel('LQ Index')
plt.ylabel('Answer Token')
plt.title('MACS: Answer → LQ Attention')
plt.show()

# Expected: Clear patterns (not uniform noise)
```

### Verify Posterior Distribution

```python
# Posterior should be a valid probability distribution
assert evidence_posterior.min() >= 0.0
assert torch.allclose(evidence_posterior.sum(dim=-1), torch.ones(batch_size))

# Check entropy (too low = overconfident, too high = flat)
entropy = -(evidence_posterior * evidence_posterior.log()).sum(dim=-1)
print(f"Posterior entropy: {entropy.mean():.3f}")
# Healthy range: 2.0-4.0 (for K=64)
```

### Monitor JS Divergence

```python
# Track convergence of prior and posterior
js_div = loss_s_dict['loss_post']
print(f"JS divergence (prior || posterior): {js_div:.4f}")

# Should decrease over training:
# Early: ~0.3-0.5 (prior and posterior disagree)
# Late:  ~0.05-0.1 (convergence)
```

---

## Next Steps

### Priority 1: LLM Integration

1. Implement `FrozenLLM.teacher_forcing_dual_path()` with real Qwen model
2. Test attention hook registration on Qwen architecture
3. Verify `attentions` tuple shape and format
4. Validate MACS extraction on real LLM outputs

### Priority 2: Full Stage-2 Training

1. Create `train/stage2_joint.py` (E + S + C)
2. Implement curriculum learning scheduler
3. Add validation loop with posterior quality metrics
4. Test on MS-MARCO subset (G=1-11)

### Priority 3: Scale-Up

1. Extend to larger evidence pools (G=64, 128, 256)
2. Implement dynamic subset U sampling
3. Optimize memory (gradient checkpointing, mixed precision)
4. Profile training speed (ensure posterior extraction < 10% overhead)

---

## References

- **MACS Paper**: "Attention Consistency for LLMs Explanation"
- **BLIP-2**: "Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"
- **Tang et al. (2017)**: "Question Answering and Question Generation as Dual Tasks"
- **DR-QFormer MACS Example**: `src/utils/MACS_example.py`

---

**End of Document**
