# Joint Training Guide for DR-QFormer (Tasks E+S+C)

## Overview

This document describes the **joint multi-task training system** for DR-QFormer, implementing simultaneous training of:
- **Task E**: Entailment tagging (high-recall filter)
- **Task S**: Fragment ranking (curriculum learning)
- **Task C**: Condensing-generation (posterior extraction)

### Core Philosophy

Following **BLIP-2's "shared forward + multi-objective" paradigm**:
1. **ONE Q-Former forward pass** produces shared representations
2. **Three task heads** compute task-specific outputs in parallel
3. **Bayesian closed loop** feeds Task C posterior back to Task S
4. **Curriculum learning** transitions from prior-only to posterior-dominated

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Frozen Retriever Adapter                      │
│              (Encode queries/passages → p_embeds)               │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Q-Former (Shared, Trainable)                   │
│   • N learnable queries (LQs)                                   │
│   • L transformer layers                                        │
│   • Cross-attention to p_embeds (pool)                          │
│   • Global LQ-drop mask (shared across tasks)                   │
│                                                                 │
│   Outputs:                                                      │
│     - Z: [batch, N_lq, d] (knowledge prefix)                    │
│     - ca_raw_scores_per_head: List[[B,H,N,K]] (attention)       │
│     - lq_drop_mask: [N_lq] (global mask)                        │
└─────────────┬───────────────┬─────────────────┬─────────────────┘
              │               │                 │
              ▼               ▼                 ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐
    │ Entailment  │  │  Ranking    │  │  Condense Head          │
    │ Head (E)    │  │  Head (S)   │  │  (C)                    │
    │             │  │             │  │                         │
    │ • Pool CA   │  │ • Aggregate │  │ • Project Z → LLM dim   │
    │ • LQ-drop   │  │   CA scores │  │ • Teacher forcing       │
    │ • Logits    │  │ • Softmax   │  │ • Dual-path NLL         │
    │   [B, K]    │  │ • Logits    │  │ • Posterior extraction  │
    │             │  │   [B, K]    │  │   q_ψ_U: [B, |U|]       │
    └──────┬──────┘  └──────┬──────┘  └────────┬────────────────┘
           │                │                  │
           ▼                ▼                  ▼
    ┌──────────┐   ┌─────────────────┐ ┌──────────────┐
    │ Focal    │   │ ListNet         │ │ Contrastive  │
    │ Loss     │   │ + JS Div        │ │ NLL Loss     │
    │ (L_E)    │   │ + Tail Entropy  │ │ (L_C)        │
    │          │   │ (L_S)           │ │              │
    └──────┬───┘   └────┬────────────┘ └──┬───────────┘
           │            │                  │
           └────────────┴──────────────────┘
                        │
                        ▼
              ┌──────────────────────────┐
              │ Weighted Loss            │
              │ L = w_E·L_E +            │
              │     w_S·L_S +            │
              │     w_C·L_C              │
              └──────────────────────────┘
```

---

## Training Phases

### Phase 1: Warm-up (0-10% steps)
**Goal**: Establish entailment filter and teacher ranking baseline

```yaml
w_E: 1.0 → 0.5  (linear decay)
w_S: 1.0        (constant)
w_C: 0.0        (disabled)

λ_teach: 1.0    (teacher-only)
λ_post: 0.0     (no posterior)
```

**Behavior**:
- Task E learns entailment discrimination with focal loss
- Task S learns to match teacher scores (ListNet)
- Task C disabled (no LLM gradient)

### Phase 2: Bridge (10-70% steps)
**Goal**: Gradual integration of posterior feedback

```yaml
w_E: 0.5        (stable)
w_S: 1.0        (stable)
w_C: 0.5 → 1.0  (linear increase)

λ_teach: 1.0 → 0.2  (decay)
λ_post: 0.0 → 0.8   (increase)
λ_entropy: 0.01 → 0.001  (decay)
```

**Behavior**:
- Task C starts producing posterior q_ψ_U
- Task S gradually weights posterior over teacher
- Bayesian closed loop begins to activate

### Phase 3: Closed-loop (70-100% steps)
**Goal**: Posterior-dominated ranking with LLM alignment

```yaml
w_E: 0.5        (stable anchor)
w_S: 1.0        (stable)
w_C: 1.0        (stable)

λ_teach: 0.2    (residual teacher)
λ_post: 0.8     (posterior-dominated)
λ_entropy: 0.001  (tail regularization)
```

**Behavior**:
- Task S primarily follows Task C posterior
- Teacher acts as prior regularization
- Full Bayesian loop active

---

## Loss Function

```python
L_total = w_E(t)·[L_E_focal + λ_E_entropy·L_E_lq_entropy] +
          w_S(t)·[λ_teach·L_ListNet + λ_post·L_JS + λ_entropy·L_tail + λ_S_entropy·L_S_lq_entropy] +
          w_C(t)·[L_C_contrastive_NLL + λ_C_entropy·L_C_lq_entropy]
```

### Task E: Entailment Tagging

```python
L_E_focal = FocalLoss(logits_E, gt_entailment, γ=2.0, α=0.25)
```

- **Importance weights**:
  - Positive samples: 10x
  - Longtail positive: 50x (extra emphasis)
- **Goal**: High recall entailment filter

### Task S: Fragment Ranking

```python
L_S = λ_teach·ListNet(pred, teacher) + 
      λ_post·JS_div(pred, posterior) + 
      λ_entropy·TailEntropy(pred)
```

- **Training subset U**: Teacher Top-L ∪ Student Top-l'
- **ListNet**: Permutation probability divergence
- **JS divergence**: Align with Task C posterior q_ψ_U
- **Tail entropy**: Discourage uniform distribution on low-scored fragments

### Task C: Condensing-Generation

```python
L_C = softplus(NLL_without - NLL_with - margin, β=10.0)

margin = adaptive_margin(NLL_without - NLL_with, ratio=0.5, min=0.1, max=2.0)
```

- **Dual-path teacher forcing**:
  - **Path 1**: LLM with Z prefix → NLL_with
  - **Path 2**: LLM without Z → NLL_without
- **Contrastive NLL**: Encourage Z to reduce answer NLL
- **Posterior extraction**:
  ```python
  q_ψ_U = softmax(LLM_attention_to_Z @ CA_weights, dim=-1)  # [B, |U|]
  ```
  - Backtraced from LLM attention to Z
  - Projected back to fragment space via CA weights
  - Detached before feeding to Task S

---

## Configuration

### Model Architecture
```yaml
model:
  n_queries: 32          # Number of learnable queries (LQs)
  hidden_dim: 768        # Q-Former hidden dimension
  num_layers: 12         # Q-Former transformer layers
  num_heads: 12          # Attention heads
  
  task_e:
    tau: 0.5             # Temperature for softmax
    p_drop_lq: 0.1       # LQ-drop probability
    focal_gamma: 2.0     # Focal loss focusing parameter
    focal_alpha: 0.25    # Focal loss balancing parameter
  
  task_s:
    tau_head: 0.1        # Head aggregation temperature
    tau_lq: 0.2          # LQ aggregation temperature
    rho_top: 0.02        # Top-L cumulative mass threshold
    l_prime: 16          # Student Top-l' count
  
  task_c:
    llm_hidden_dim: 4096  # LLM hidden dimension
    softplus_beta: 10.0   # Softplus sharpness
```

### Loss Weights Schedule
```yaml
loss_weights:
  w_E:
    start: 1.0
    end: 0.5
    phase_end_ratio: 0.3  # Decay during first 30% of training
  
  w_S:
    start: 1.0
    end: 1.0              # Constant
  
  w_C:
    start: 0.5
    end: 1.0
    phase_end_ratio: 0.3  # Increase during first 30%
```

### Curriculum (Task S)
```yaml
curriculum_task_s:
  lambda_teach:
    start: 1.0
    end: 0.2
    phase_end_ratio: 0.7  # Decay over 70% of training
  
  lambda_post:
    start: 0.0
    end: 0.8
    phase_end_ratio: 0.7  # Increase over 70%
  
  lambda_entropy:
    start: 0.01
    end: 0.001
    phase_end_ratio: 0.7  # Decay over 70%
  
  tau_pred: 1.0           # Softmax temperature for predictions
  tau_gt: 1.0             # Softmax temperature for ground truth
  alpha_gt: 0.7           # Teacher Top-L cumulative mass
```

### LQ Entropy Regularization (Optional)
```yaml
lq_entropy_regularization:
  task_e:
    enabled: true
    start: 0.01
    end: 0.001
    schedule: "cosine"     # Cosine decay
    target_ratio: 0.5      # Target attention concentration (50%)
  
  task_s:
    enabled: true
    start: 0.01
    end: 0.001
    schedule: "cosine"
    target_ratio: 0.7      # Conservative diversity (70%)
  
  task_c:
    enabled: true
    start: 0.01
    end: 0.001
    schedule: "cosine"
    target_ratio: 0.7
```

---

## Data Format

### Input JSON/JSONL

```json
{
  "query": "What is the capital of France?",
  "answer": "Paris is the capital and most populous city of France.",
  "fragments": [
    "Paris is the capital of France.",
    "Lyon is the second largest city in France.",
    "France is a country in Western Europe.",
    ...
  ],
  "gt_entailment": [1, 0, 0, ...],   // Task E: Binary labels [K]
  "is_longtail": [1, 0, 0, ...],     // Task E: Longtail indicator [K]
  "gt_scores": [0.95, 0.23, 0.18, ...],  // Task S: Teacher scores [K]
  "posterior_scores": [0.88, 0.31, 0.12, ...]  // Task S: Optional posterior [K]
}
```

### Dynamic K Support

- **K range**: 10-5000 fragments per sample
- **Padding**: Automatically pad to max K in batch
- **Mask**: `pool_padding_mask` [batch, K_max] indicates valid fragments

---

## Usage

### Basic Training

```bash
python scripts/train_joint.py --config configs/joint_train.yaml
```

### Resume from Checkpoint

```bash
python scripts/train_joint.py \
    --config configs/joint_train.yaml \
    --resume checkpoints/checkpoint_epoch10.pt
```

### Custom Configuration

```bash
# Modify configs/joint_train.yaml
# Then run:
python scripts/train_joint.py --config configs/joint_train.yaml
```

---

## Key Implementation Details

### 1. Shared Q-Former Forward

**ONE forward pass** per training step:

```python
# Shared forward (all tasks use same Z, CA scores, LQ-drop mask)
z, aux = qformer(
    query_embeds=q_embeds,
    p_embeds=p_embeds,
    pool_padding_mask=pool_padding_mask,
)

ca_raw_scores_per_head = aux['ca_raw_scores_per_head']  # List[[B,H,N,K]]
lq_drop_mask = aux['lq_drop_mask']  # [N_lq] global mask
```

### 2. Training Subset U (Task S)

Build subset U from **Teacher Top-L ∪ Student Top-l'**:

```python
train_subset_mask = build_train_subset_mask(
    ranking_logits=ranking_logits.detach(),
    gt_scores=gt_scores,
    pool_padding_mask=pool_padding_mask,
    rho_top=0.02,    # Teacher Top-L: cumsum > 0.02
    l_prime=16,      # Student Top-16
)
```

**Rationale**: Focus ranking loss on high-scoring regions (avoid wasting gradient on noise)

### 3. Posterior Extraction (Task C)

Backtrace LLM attention to fragments:

```python
# LLM attention to Z: [B, H, S_a, N]
# CA weights: [B, N, K]
posterior_q_psi_U = softmax(
    einsum('b h s n, b n k -> b k', llm_attention_to_z, ca_weights),
    dim=-1
)  # [B, K]

# Detach before feeding to Task S
posterior_q_psi_U = posterior_q_psi_U.detach()
```

**Rationale**: Prevent LLM gradient flow while using its posterior belief

### 4. Curriculum Weight Application

```python
# Get weights at current step
weights = scheduler.get_weights(self.global_step)

# Apply to losses
loss_total = (
    weights['w_E'] * loss_e +
    weights['w_S'] * loss_s +
    weights['w_C'] * loss_c
)
```

---

## Placeholders (TODO)

### Retriever Adapter
```python
# Current: Mock encoding
q_embeds = torch.randn(batch_size, 1, hidden_dim)
p_embeds = torch.randn(batch_size, K, hidden_dim)

# TODO: Replace with actual retriever
from src.adapters.retriever import RetrieverAdapter
retriever = RetrieverAdapter(model_name='facebook/contriever')
q_embeds = retriever.encode_queries(queries)
p_embeds = retriever.encode_passages(fragments_flat)
```

### LLM Adapter
```python
# Current: Mock NLL values
nll_with = torch.tensor(2.5)
nll_without = torch.tensor(3.8)

# TODO: Replace with actual LLM
from src.adapters.llm import FrozenLLM
llm = FrozenLLM(model_name='microsoft/phi-2')
outputs = llm.teacher_forcing_dual_path(
    z_prefix=z_prefix,
    answer_tokens=answer_tokens,
    pool_padding_mask=pool_padding_mask,
)
nll_with = outputs['nll_with']
nll_without = outputs['nll_without']
llm_attention_to_z = outputs['attention_to_z']
```

---

## Troubleshooting

### Issue: Task C loss not decreasing

**Possible causes**:
1. LLM not frozen properly (check `requires_grad=False`)
2. Margin too large (reduce `margin_adaptive_ratio` or set fixed margin)
3. Z projection not properly initialized (check CondenseHead)

**Solution**:
```yaml
margin_adaptive:
  mode: "fixed"
  fixed_margin: 0.3  # Reduce from default 0.5
```

### Issue: Task S ignores posterior

**Possible causes**:
1. `λ_post` not increasing (check curriculum schedule)
2. Posterior too noisy (Task C not converged)
3. Temperature `tau_pred` too high (predictions too smooth)

**Solution**:
```yaml
curriculum_task_s:
  lambda_post:
    start: 0.1  # Non-zero start for earlier integration
    end: 0.8
```

### Issue: Entailment recall too low

**Possible causes**:
1. Focal loss `focal_gamma` too high (over-focusing on hard examples)
2. Importance weights not properly set
3. Positive samples too few

**Solution**:
```yaml
model:
  task_e:
    focal_gamma: 1.5  # Reduce from default 2.0
    focal_alpha: 0.35  # Increase positive weight
```

### Issue: LQ attention collapses

**Possible causes**:
1. LQ-drop disabled or too low
2. No entropy regularization
3. CA scores saturating

**Solution**:
```yaml
lq_entropy_regularization:
  task_*:
    enabled: true
    start: 0.05  # Increase from default 0.01

model:
  task_*:
    p_drop_lq: 0.15  # Increase from default 0.1
```

---

## Evaluation Metrics

### Task E (Entailment)
- **Precision**: Among predicted positive, how many are correct
- **Recall**: Among ground truth positive, how many are found (PRIMARY METRIC)
- **F1**: Harmonic mean of precision and recall
- **Longtail Recall**: Recall on longtail samples (high-value metric)

### Task S (Ranking)
- **NDCG@k**: Normalized Discounted Cumulative Gain (k=5,10,20)
- **MRR**: Mean Reciprocal Rank of first relevant fragment
- **Recall@k**: Proportion of relevant fragments in Top-k

### Task C (Condensing)
- **NLL Gain**: `NLL_without - NLL_with` (higher is better)
- **Answer Perplexity**: `exp(NLL_with / answer_length)` (lower is better)
- **Posterior JS Divergence**: JS(q_ψ_U || teacher) (monitor alignment)

---

## References

- BLIP-2: [Li et al., 2023] - "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"
- Q-Former Architecture: Shared forward pass with multi-objective learning
- Curriculum Learning: [Bengio et al., 2009] - "Curriculum Learning"
- Focal Loss: [Lin et al., 2017] - "Focal Loss for Dense Object Detection"

---

## Version History

- **v1.0** (2024-01): Initial joint training implementation
  - Shared Q-Former forward
  - 3-phase curriculum learning
  - Bayesian closed loop (Task C → Task S)
  - Optional LQ entropy regularization
  - Dynamic K support (10-5000)

---

## Contact

For questions or issues, please refer to:
- Main documentation: `documents/`
- Implementation details: `src/models/`, `src/losses.py`
- Training code: `train/task_joint.py`, `scripts/train_joint.py`
