# LQ-Level Entropy Regularization

**Status**: ✅ Implemented (Optional Feature)  
**Version**: v1.0  
**Date**: 2024-01-XX

---

## 1. Overview

**Purpose**: Prevent Learnable Query (LQ) representation collapse by encouraging diversity in cross-attention distributions.

**Problem**: Without regularization, multiple LQs may learn redundant attention patterns:
```
LQ_0: [0.9, 0.05, 0.05]  # Focuses on fragment 0
LQ_1: [0.85, 0.10, 0.05] # Also focuses on fragment 0
LQ_2: [0.88, 0.07, 0.05] # Still focuses on fragment 0
```

This wastes model capacity and makes LQ compression (32→16→8) difficult.

**Solution**: Add entropy regularization to encourage moderate distribution:
```
LQ_0: [0.4, 0.3, 0.2, 0.1]   # Moderate distribution
LQ_1: [0.2, 0.4, 0.3, 0.1]   # Different pattern
LQ_2: [0.3, 0.2, 0.15, 0.35] # Yet another pattern
```

---

## 2. Implementation

### 2.1 Loss Function

**Location**: `src/losses.py::compute_lq_entropy_loss`

**Formula**:
```
1. Average attention scores across layers and heads:
   ca_scores_avg[b, n, k] = mean_{l,h}(ca_raw_scores_per_head[l][b, h, n, k])

2. Apply mask and softmax per LQ:
   ca_probs[b, n, k] = softmax_k(ca_scores_avg[b, n, k]) for valid k

3. Compute entropy per LQ:
   H[b, n] = -Σ_k ca_probs[b, n, k] · log(ca_probs[b, n, k])

4. Target entropy (uniform over valid fragments):
   H_target[b] = target_ratio · log(K_eff[b])
   where K_eff = number of valid fragments

5. MSE loss:
   loss = mean_{b,n}((H[b, n] - H_target[b])^2)
```

**Key Design Choices**:
- **Target ratio < 1.0**: Allows task-specific concentration (conservative mode)
- **MSE instead of -H**: Encourages entropy close to target (not arbitrarily high)
- **Average across layers/heads**: Reduces noise, consistent with FragmentRankingHead

---

### 2.2 Curriculum Learning

**Strategy**: Linear decay from high exploration (early) to low task-driven (late)

```python
# Curriculum weight
progress = current_step / total_steps
lambda_entropy = lambda_start * (1 - 0.9 * progress)
lambda_entropy = max(lambda_entropy, lambda_end)
```

**Rationale**:
- **Early training**: High weight → encourage exploration and diversity
- **Late training**: Low weight → allow task-specific concentration
- **Fast decay**: Prevents over-regularization interfering with main task

---

## 3. Task-Specific Configuration

| Task | Recommendation | Reason | Weight Schedule | Target Ratio |
|------|---------------|--------|-----------------|--------------|
| **Task E** | ⚠️ Optional | May need concentrated attention for binary classification | 0.005 → 0.0005 | 0.5 (50% concentration) |
| **Task S** | ✅ Recommended | Ranking requires multi-perspective evaluation (prior distribution) | 0.01 → 0.001 | 0.7 (30% concentration) |
| **Task C** | ⚠️ Moderate | Compression needs diversity but must align with LLM posterior | 0.008 → 0.0001 | 0.7 (fast decay) |

### 3.1 Task E (Fragment Entailment)

**Command-line flags**:
```bash
python train/task_e.py \
  --enable_lq_entropy_reg \
  --lambda_entropy_start 0.005 \
  --lambda_entropy_end 0.0005 \
  --entropy_target_ratio 0.5
```

**Why optional?**:
- Task E is binary classification (entailment vs. non-entailment)
- May require concentrated attention on single relevant fragment
- Use conservative target (0.5) to allow 50% concentration

**When to enable**:
- If LQs show high redundancy during ablation
- If planning LQ compression (32→16→8)
- If validation shows overfitting to specific fragments

---

### 3.2 Task S (Fragment Ranking)

**Command-line flags**:
```bash
python train/task_s.py \
  --enable_lq_entropy_reg \
  --lambda_entropy_start 0.01 \
  --lambda_entropy_end 0.001 \
  --entropy_target_ratio 0.7
```

**Why recommended?**:
- Task S learns **prior distribution** π(p|q) over fragments
- Ranking requires multi-perspective evaluation (different LQs assess different aspects)
- Prior should be broad (diversity) before posterior refinement
- Helps prevent "winner-takes-all" collapse

**Expected benefits**:
- Better diversity in learned prior distribution
- Improved robustness to long-tail queries
- Easier LQ compression (distinct attention patterns)
- Better generalization to unseen fragments

---

### 3.3 Task C (Knowledge Condensing)

**Command-line flags**:
```bash
python train/task_c.py \
  --enable_lq_entropy_reg \
  --lambda_entropy_start 0.008 \
  --lambda_entropy_end 0.0001 \
  --entropy_target_ratio 0.7
```

**Why moderate?**:
- Task C produces knowledge prefix Z for LLM
- Must balance:
  - **Diversity**: Compress information from multiple fragments
  - **Alignment**: Match LLM's posterior attention distribution q_ψ(p|q,a)
- **Fast decay**: Strong regularization early (exploration), minimal late (posterior alignment)

**Trade-off**:
- Too strong: Forces uniform distribution → conflicts with LLM posterior
- Too weak: Collapses to single fragment → loses compression benefit
- **Solution**: Fast decay schedule (0.008 → 0.0001)

---

## 4. Relation to LQ Compression

### 4.1 Motivation

**Goal**: Compress 32 LQs → 16 LQs → 8 LQs for efficiency

**Challenge**: Which LQs to keep?
- Without entropy reg: Many LQs are redundant (hard to identify important ones)
- With entropy reg: Each LQ learns distinct pattern (clear importance ranking)

### 4.2 Compression Pipeline

```
1. Training with entropy regularization
   └─> Ensures all 32 LQs contribute (no redundancy)
   
2. Evaluate LQ importance
   ├─> Gradient norms per LQ during validation
   ├─> Attention entropy per LQ (higher = more active)
   └─> Ablation study (remove LQ, measure performance)

3. Select Top-K LQs
   └─> Choose K LQs with highest importance scores
   
4. Knowledge distillation
   ├─> Train K-LQ student from 32-LQ teacher
   └─> Distill attention patterns + outputs

5. Fine-tuning compressed model
   ├─> Disable entropy reg (allow task-specific concentration)
   └─> Fine-tune with smaller learning rate
```

### 4.3 Evaluation Metrics

**Before compression** (32 LQs with entropy reg):
```python
# Compute per-LQ importance
importance_scores = []
for lq_idx in range(32):
    # Method 1: Gradient norm
    grad_norm = torch.norm(z[:, lq_idx].grad)
    
    # Method 2: Attention entropy
    entropy = -torch.sum(attn_weights[:, lq_idx] * torch.log(attn_weights[:, lq_idx]))
    
    # Method 3: Ablation performance
    ablation_score = evaluate_without_lq(lq_idx)
    
    importance_scores.append((grad_norm, entropy, ablation_score))

# Rank LQs
top_k_indices = torch.argsort(importance_scores, descending=True)[:K]
```

**After compression** (K LQs):
```python
# Compare performance
results = {
    "32_LQs_no_reg": baseline_score,
    "32_LQs_with_reg": entropy_reg_score,
    "16_LQs_distilled": compressed_16_score,
    "8_LQs_distilled": compressed_8_score,
}

# Ideal outcome:
# - 32_LQs_with_reg ≈ 32_LQs_no_reg (no harm)
# - 16_LQs_distilled ≈ 32_LQs_with_reg (minimal loss)
# - 8_LQs_distilled > baseline_at_8_LQs (better than naive compression)
```

---

## 5. Usage Examples

### 5.1 Task E (Default: Disabled)

```bash
# Baseline (no entropy reg)
python train/task_e.py \
  --mode primal \
  --batch_size 8 \
  --epochs 10

# With entropy reg (optional)
python train/task_e.py \
  --mode primal \
  --enable_lq_entropy_reg \
  --lambda_entropy_start 0.005 \
  --lambda_entropy_end 0.0005 \
  --entropy_target_ratio 0.5
```

---

### 5.2 Task S (Recommended)

```bash
# With entropy reg (recommended for prior learning)
python train/task_s.py \
  --enable_lq_entropy_reg \
  --lambda_entropy_start 0.01 \
  --lambda_entropy_end 0.001 \
  --entropy_target_ratio 0.7 \
  --batch_size 8 \
  --epochs 10
```

---

### 5.3 Task C (Moderate - Fast Decay)

```bash
# With entropy reg (moderate, fast decay)
python train/task_c.py \
  --enable_lq_entropy_reg \
  --lambda_entropy_start 0.008 \
  --lambda_entropy_end 0.0001 \
  --entropy_target_ratio 0.7 \
  --batch_size 4 \
  --epochs 10
```

---

## 6. Monitoring and Debugging

### 6.1 Log Metrics

**During training**:
```python
# Log entropy loss
if args.enable_lq_entropy_reg:
    logger.log({
        "loss_main": loss_main.item(),
        "loss_entropy": entropy_loss.item(),
        "lambda_entropy": lambda_entropy,
        "avg_entropy": avg_entropy.item(),  # Average entropy across LQs
        "target_entropy": target_entropy.item(),
    })
```

**Expected trends**:
- `loss_entropy` should decrease over time (LQs learn moderate distributions)
- `avg_entropy` should stabilize near `target_entropy`
- `lambda_entropy` should decay linearly
- `loss_main` should NOT significantly increase (no over-regularization)

---

### 6.2 Visualization

**Plot attention heatmaps**:
```python
import matplotlib.pyplot as plt

# Compare with/without entropy reg
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Without entropy reg
axes[0].imshow(attn_weights_baseline, cmap='viridis', aspect='auto')
axes[0].set_title("Baseline (No Entropy Reg)")
axes[0].set_xlabel("Fragments")
axes[0].set_ylabel("LQs")

# With entropy reg
axes[1].imshow(attn_weights_entropy, cmap='viridis', aspect='auto')
axes[1].set_title("With Entropy Reg")
axes[1].set_xlabel("Fragments")
axes[1].set_ylabel("LQs")

plt.colorbar()
plt.show()
```

**Expected patterns**:
- **Baseline**: Darker bands (multiple LQs attend to same fragments)
- **With entropy reg**: More uniform colors (LQs distribute attention more evenly)

---

### 6.3 Ablation Study

**Compare with/without entropy reg**:
```bash
# Run both versions
python train/task_s.py --enable_lq_entropy_reg --save_dir ./ckpt_entropy
python train/task_s.py --save_dir ./ckpt_baseline

# Evaluate both checkpoints
python eval/evaluate.py --checkpoint ./ckpt_entropy/best.pt
python eval/evaluate.py --checkpoint ./ckpt_baseline/best.pt
```

**Expected results**:
- Similar or slightly better validation metrics with entropy reg
- Better LQ diversity (measured by attention entropy)
- Easier LQ compression (top-K selection more stable)

---

## 7. Theoretical Justification

### 7.1 Information-Theoretic View

**Maximum Entropy Principle**:
```
Under uncertainty, choose distribution with maximum entropy
subject to known constraints (e.g., task loss)
```

**Application to LQs**:
- Task loss: Ensure accurate fragment scoring (main objective)
- Entropy constraint: Among all solutions with same task loss, prefer higher entropy (diversity)
- Implementation: Weighted sum of task loss + entropy regularization

**Benefits**:
- **Robustness**: Diverse LQs are less sensitive to perturbations
- **Generalization**: Avoid overfitting to specific fragment patterns
- **Interpretability**: Each LQ learns distinct semantic role

---

### 7.2 Comparison to Other Regularization

| Method | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **Dropout** | Randomly drop LQs during training | Simple, prevents co-adaptation | Stochastic (training instability) |
| **L2 Regularization** | Penalize large weights | Prevents overfitting | Doesn't encourage diversity |
| **Orthogonal Constraint** | Force LQ embeddings orthogonal | Strong diversity guarantee | Hard constraint (optimization difficulty) |
| **Entropy Regularization** | Encourage uniform attention | Soft constraint, curriculum learning | Requires tuning target ratio |

**Why entropy regularization?**:
- **Soft constraint**: Balances task loss and diversity (no hard enforcement)
- **Curriculum learning**: Strong early (exploration) → weak late (task-driven)
- **Interpretable**: Direct control via target_ratio (0.5 = allow 50% concentration)

---

### 7.3 Relation to Variational Inference

**Bayesian View**:
```
Posterior: q(z|x) ∝ p(x|z) · p(z)
           ↑        ↑         ↑
           Task    Likelihood Prior (entropy)
```

**Application**:
- `p(x|z)`: Fragment scoring accuracy (task loss)
- `p(z)`: Prior over LQ attention distributions (entropy regularization)
- `q(z|x)`: Learned LQ representations

**Objective**:
```
ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
       ↑                  ↑
       Task loss          Entropy regularization
```

**Implementation**:
- Use uniform distribution as prior `p(z) = Uniform(1/K_eff)`
- Minimize KL divergence → maximize entropy of `q(z|x)`
- Equivalent to entropy regularization with `target_ratio = 1.0`

---

## 8. Future Work

### 8.1 Adaptive Target Ratio

**Current**: Fixed `target_ratio` (0.5 for Task E, 0.7 for Task S/C)

**Proposal**: Learn target ratio per LQ dynamically
```python
# Per-LQ target ratios (learnable parameters)
target_ratios = nn.Parameter(torch.ones(N_lq) * 0.7)

# Compute per-LQ target entropy
target_entropy_per_lq = target_ratios * log(K_eff)

# MSE loss per LQ
loss_entropy = F.mse_loss(entropy_per_lq, target_entropy_per_lq)
```

**Benefits**:
- Some LQs can be concentrated (specific roles)
- Other LQs can be distributed (broad coverage)
- Learned from data (no manual tuning)

---

### 8.2 Multi-Level Entropy

**Current**: Entropy over fragments (K_eff)

**Proposal**: Add entropy over LQs (N_lq)
```python
# Current: Entropy per LQ (over fragments)
H_per_lq[n] = -Σ_k p[n,k] · log(p[n,k])

# New: Entropy per fragment (over LQs)
H_per_fragment[k] = -Σ_n p[n,k] · log(p[n,k])

# Combined loss
loss_entropy = loss_lq_level + λ_fragment * loss_fragment_level
```

**Benefits**:
- Prevents fragment-level collapse (all LQs ignore some fragments)
- Encourages balanced attention (each fragment attended by multiple LQs)
- Useful for Task C (ensure all relevant fragments used)

---

### 8.3 Conditional Entropy

**Current**: Marginal entropy over fragments

**Proposal**: Conditional entropy given query type
```python
# Cluster queries into K types (e.g., factoid, list, yes/no)
query_type = classify_query(query)

# Compute target entropy based on query type
target_entropy_dict = {
    "factoid": 0.3 * log(K_eff),    # Concentrated (single answer)
    "list": 0.9 * log(K_eff),        # Distributed (multiple answers)
    "yes_no": 0.5 * log(K_eff),      # Moderate (evidence aggregation)
}
target_entropy = target_entropy_dict[query_type]

# MSE loss with query-specific target
loss_entropy = F.mse_loss(entropy_per_lq, target_entropy)
```

**Benefits**:
- Task-adaptive regularization (different queries need different attention patterns)
- Better alignment with query intent
- Improved performance on diverse query types

---

## 9. Related Work

### 9.1 Attention Regularization

**Papers**:
1. "Attention is not Explanation" (Jain & Wallace, 2019)
   - Shows attention weights can be arbitrary without regularization
   
2. "Learning to Deceive with Attention-Based Explanations" (Pruthi et al., 2020)
   - Proposes attention supervision to prevent degenerate solutions

3. "Entropy Regularized Reinforcement Learning" (Haarnoja et al., 2017)
   - Maximum entropy RL for exploration-exploitation trade-off

**Difference**:
- Our work: Apply entropy regularization to Learnable Queries (cross-attention)
- Related work: Attention heads (self-attention) or policy distributions (RL)

---

### 9.2 Multi-Head Attention Collapse

**Papers**:
1. "Analyzing Multi-Head Self-Attention" (Voita et al., 2019)
   - Shows many attention heads are redundant
   - Proposes pruning based on head importance

2. "Are Sixteen Heads Really Better than One?" (Michel et al., 2019)
   - Most attention heads can be removed without performance loss

**Connection**:
- LQs analogous to attention heads (multiple slots for information)
- Entropy regularization prevents LQ redundancy (similar to head pruning)
- Our method: Encourage diversity during training (preventive)
- Pruning methods: Remove redundant heads after training (corrective)

---

### 9.3 Variational Information Bottleneck

**Papers**:
1. "Deep Variational Information Bottleneck" (Alemi et al., 2017)
   - Minimize mutual information between input and representation
   - Maximize mutual information between representation and output

2. "The Information Bottleneck Method" (Tishby et al., 2000)
   - Compress input while preserving task-relevant information

**Connection**:
- LQs act as information bottleneck (compress K fragments into N_lq slots)
- Entropy regularization controls compression rate (diversity)
- Trade-off: Task performance (minimal sufficient statistics) vs. diversity (generalization)

---

## 10. Summary

### 10.1 Key Takeaways

✅ **Implemented**: LQ-Level Entropy Regularization across all 3 tasks  
✅ **Optional**: Disabled by default (enable via `--enable_lq_entropy_reg`)  
✅ **Task-Specific**: Different weights and target ratios per task  
✅ **Curriculum Learning**: Linear decay from high (exploration) to low (task-driven)  
✅ **Preparation**: Facilitates future LQ compression (32→16→8)

---

### 10.2 Command Summary

| Task | Command | Recommendation |
|------|---------|----------------|
| **Task E** | `--enable_lq_entropy_reg --lambda_entropy_start 0.005 --lambda_entropy_end 0.0005 --entropy_target_ratio 0.5` | Optional (may need concentration) |
| **Task S** | `--enable_lq_entropy_reg --lambda_entropy_start 0.01 --lambda_entropy_end 0.001 --entropy_target_ratio 0.7` | ✅ Recommended (diversity important) |
| **Task C** | `--enable_lq_entropy_reg --lambda_entropy_start 0.008 --lambda_entropy_end 0.0001 --entropy_target_ratio 0.7` | Moderate (balance with posterior) |

---

### 10.3 Files Modified

**Core Implementation**:
- ✅ `src/losses.py`: Added `compute_lq_entropy_loss` function (200+ lines)

**Training Scripts**:
- ✅ `train/task_e.py`: Added command-line flags + integration
- ✅ `train/task_s.py`: Added command-line flags + integration
- ✅ `train/task_c.py`: Added command-line flags + integration

**Documentation**:
- ✅ `documents/LQ_ENTROPY_REGULARIZATION.md`: This file (comprehensive guide)

---

### 10.4 Next Steps

**Immediate**:
1. ✅ Test integration (run simple training loop)
2. ✅ Verify entropy loss decreases during training
3. ✅ Compare with/without entropy reg on validation set

**Short-Term**:
1. Ablation study on Task S (recommended use case)
2. Visualize attention heatmaps (with/without entropy reg)
3. Measure per-LQ importance scores

**Long-Term**:
1. LQ compression pipeline (32→16→8)
2. Adaptive target ratio (learnable per-LQ)
3. Multi-level entropy (per-LQ + per-fragment)
4. Conditional entropy (query-type-specific)

---

## References

1. **DR-QFormer Specification v1.1**
   - Task E: Fragment Entailment Tagging
   - Task S: Fragment Ranking (Prior Distribution)
   - Task C: Knowledge Condensing (Posterior Extraction)

2. **Information Theory**
   - Cover & Thomas, "Elements of Information Theory" (2006)
   - Maximum Entropy Principle

3. **Variational Inference**
   - Blei et al., "Variational Inference: A Review for Statisticians" (2017)
   - ELBO optimization

4. **Attention Mechanisms**
   - Vaswani et al., "Attention is All You Need" (2017)
   - Voita et al., "Analyzing Multi-Head Self-Attention" (2019)

---

**Document Version**: v1.0  
**Last Updated**: 2024-01-XX  
**Author**: DR-QFormer Team  
**Status**: ✅ Complete
