# Joint Training Guide for DR-QFormer

**Tasks**: E (Entailment), S (Ranking), C (Contrastive NLL with Posterior Feedback)

Following **BLIP-2 Stage-1 philosophy**: Shared Q-Former forward pass supporting multiple objectives with dynamic scheduling.

---

## Overview

### Core Innovation

**One-pass Q-Former forward → Multi-task learning**:
- Same `LQs_aware`, `CA raw scores`, and `Z` feed into E/S/C
- **Task E**: Entailment filtering (Focal Loss) for high-recall hard filtering
- **Task S**: Ranking with teacher→posterior transition (ListNet + JS divergence)
- **Task C**: Contrastive NLL (dual-path Teacher Forcing) + posterior extraction

### Key Features

1. **Unified Drop-LQ**: Same LQ drop mask across all tasks → consistent representation
2. **Dynamic Scheduling**: Gradual shift from prior (teacher) to posterior (LLM feedback)
3. **Closed-loop Feedback**: Task C extracts posterior qψ_U → feeds back to Task S (detached)
4. **BLIP-2 Alignment**: Task E/S don't enter closed loop (like BLIP-2's "anchor tasks")

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Required packages
pip install torch transformers sentence-transformers pyyaml tqdm
```

### 2. Prepare Data

**Expected data format** (see `train/joint_data.py`):

```json
[
  {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "fragments": ["Paris is the capital...", "France is a country..."],
    "entailment_labels": [1, 0],  // Task E: 1=entailed, 0=not entailed
    "ranking_scores": [0.95, 0.45]  // Task S: BM25/relevance scores
  },
  ...
]
```

**TODO**: Replace placeholders in `train/joint_data.py`:
- `retriever`: BM25 or dense retriever
- `embedder`: Sentence-BERT or similar
- `tokenizer`: LLM tokenizer (e.g., LlamaTokenizer)

### 3. Configure Training

Edit `configs/joint_train.yaml`:

```yaml
model:
  d: 768
  num_lqs: 32
  num_heads: 12

schedule:
  total_steps: 50000
  warmup_frac: 0.1    # Prior-only phase
  bridge_frac: 0.6    # Teacher→Posterior transition
  closedloop_frac: 0.3  # Posterior-dominant

losses:
  focal:
    gamma: 2.0
    alpha: 0.25
  ranking:
    lambda_teach_start: 1.0
    lambda_teach_end: 0.2
    lambda_post_start: 0.0
    lambda_post_end: 0.8

data:
  train_path: "data/train.json"
  batch_size: 8
  max_fragments: 100
```

### 4. Run Training

```bash
python scripts/train_joint.py --config configs/joint_train.yaml
```

---

## Training Phases

Following the training schedule (see `train/schedule.py`):

### Phase 1: Warm-up (~10% steps)

**Goal**: Stabilize E/S with prior (teacher) only

- **Task E**: Full Focal Loss on entailment labels
- **Task S**: ListNet with teacher scores (λ_teach=1.0, λ_post=0.0)
- **Task C**: Records NLL only (no gradient, prepares for posterior)

**Why**: Let entailment filtering establish "recall anchors", and ranking align with teacher distribution before introducing LLM feedback.

### Phase 2: Bridge (~60% steps)

**Goal**: Gradually shift from teacher to posterior

- **Task E**: Continue with Focal Loss (weight decays to 0.5)
- **Task S**: Linear crossover from teacher to posterior
  - λ_teach: 1.0 → 0.2
  - λ_post: 0.0 → 0.8
- **Task C**: Enable dual-path Teacher Forcing, extract qψ_U, feed to Task S

**Why**: Smooth transition prevents sudden gradient shifts. Task S learns to trust LLM's "actual needs" over static teacher scores.

### Phase 3: Closed-loop (~30% steps)

**Goal**: Posterior-dominant, full feedback loop

- **Task E**: Lower weight (0.5) to avoid over-filtering
- **Task S**: Dominated by posterior (λ_post=0.8), teacher provides baseline (λ_teach=0.2)
- **Task C**: Full contrastive NLL with adaptive margin

**Why**: By now, ranking should follow LLM's attention patterns closely. Task E remains as "hard filter" but doesn't enter closed loop (keeps true positives as anchor).

---

## Task Details

### Task E: Entailment Head

**Purpose**: High-recall hard filter

**Loss**: Focal Loss
```
L_E = -α(1-p)^γ log(p)
```
- `γ=2.0`: Focus on hard negatives
- `α=0.25`: Balance class weights
- Importance weights: Boost long-tail fragments

**Aggregation**: LogSumExp across LQs
```
logit_k = LSE_i(CA_raw_score[i,k])
```

**No closed loop**: Uses ground-truth labels, doesn't take posterior feedback

### Task S: Fragment Ranking Head

**Purpose**: Rank fragments by "usefulness to LLM"

**Loss**: Multi-component
```
L_S = λ_teach * L_ListNet + λ_post * L_JS + λ_ent * L_entropy
```

1. **L_ListNet** (Teacher): Align with teacher scores (e.g., BM25)
   - Computed on training subset U: Teacher Top-L ∪ Student Top-l'
   - Temperature-calibrated teacher distribution

2. **L_JS** (Posterior): JS divergence with posterior from Task C
   - Posterior qψ_U: Backtracked from LLM→Z attention
   - Detached (no gradient from LLM)

3. **L_entropy** (Tail regularization): Penalize over-confident tail
   - Annealedλ_ent: 0.01 → 0.001

**Aggregation**: Nested LSE
```
logit_k = LSE_i(LSE_h(CA_raw_score[h,i,k]))
```
- First LSE across heads → per-LQ scores
- Second LSE across LQs → final ranking logit

**Training subset U**: Dynamic Top-L (teacher) + fixed Top-l' (student, hard negatives)

### Task C: Contrastive NLL with Posterior

**Purpose**: Align Z with LLM's generation needs, extract posterior for Task S

**Dual-path Teacher Forcing**:

1. **Path-A**: `[Z, Q_tokens, A_tokens]`
   - Full attention
   - Compute NLL_A
   - Extract LLM→Z attention

2. **Path-B**: `[Z_dummy, Q_tokens, A_tokens]`
   - Mask Q/A→Z attention (fair baseline)
   - Compute NLL_B (no_grad)

**Loss**: Contrastive NLL
```
L_C = Softplus(β * (m - G))
```
- `G = NLL_B - NLL_A`: Gain from using Z
- `m`: Adaptive margin = μ_G + κσ_G (κ=0.5)
- `β=10`: Sharpness of Softplus

**Posterior extraction**:
```
qψ_U = CA_weights @ (A_tokens → Z attention)  [detached]
```
- Backtrack A_tokens→Z attention to fragments
- Only on subset U (same as Task S)
- Detached before feeding to Task S

**TODO**: Implement dual-path logic in `train/task_joint.py` (currently placeholder)

---

## Loss Function (Total)

```
L_total = w_E(t) * L_E 
        + w_S(t) * [λ_teach(t)*L_ListNet + λ_post(t)*L_JS + λ_ent(t)*L_entropy]
        + w_C(t) * L_C
        + λ_dual * [L_E_QG + L_S_QG + L_C_QG]  (optional)
```

**Task weights** (time-dependent):
- `w_E`: 1.0 → 0.5 (decay over first 30%)
- `w_S`: 1.0 (constant)
- `w_C`: 0.5 → 1.0 (ramp up over first 30%)

**Ranking lambdas** (time-dependent):
- `λ_teach`: 1.0 → 0.2 (linear decay in Bridge phase)
- `λ_post`: 0.0 → 0.8 (linear ramp in Bridge phase)
- `λ_ent`: 0.01 → 0.001 (exponential decay)

---

## Implementation Checklist

### ✅ Completed

- [x] Data module: `train/joint_data.py`
  - JointBatch with pool_padding_mask
  - Variable-length K handling
  - Collate function
  
- [x] Schedule module: `train/schedule.py`
  - Dynamic weight scheduling (w_E, w_S, w_C)
  - Lambda scheduling (λ_teach, λ_post, λ_ent)
  - Phase management (warmup/bridge/closedloop)
  
- [x] Trainer: `train/task_joint.py`
  - JointTrainer class
  - Shared Q-Former forward
  - Task E/S loss computation
  - Unified Drop-LQ mask
  
- [x] Config: `configs/joint_train.yaml`
  - All hyperparameters
  - Schedule parameters
  - Placeholders for LLM/retriever/embedder
  
- [x] Script: `scripts/train_joint.py`
  - Training loop
  - Checkpoint saving
  - Logging

### ⚠️ TODO (Placeholders)

- [ ] **Task C dual-path Teacher Forcing**
  - Path-A/B construction
  - Attention masking
  - RNG synchronization
  - Posterior extraction qψ_U
  
- [ ] **LLM integration**
  - Load from HuggingFace
  - 8-bit quantization
  - Freeze parameters
  - Device mapping
  
- [ ] **Retriever**
  - BM25 or dense retriever
  - Fragment retrieval
  - Teacher score generation
  
- [ ] **Embedding model**
  - Sentence-BERT or similar
  - Query/answer/fragment embedding
  
- [ ] **Tokenizer**
  - LLM tokenizer (e.g., LlamaTokenizer)
  - Q/A tokenization for Task C
  
- [ ] **Validation loop**
  - Metrics: NDCG, Accuracy, Perplexity
  - Early stopping
  
- [ ] **Experiment tracking**
  - Wandb or TensorBoard
  - Loss curves
  - Schedule curves
  
- [ ] **Distributed training**
  - DistributedDataParallel
  - Multi-GPU support

---

## Key Design Choices

### Why Unified Drop-LQ?

**Problem**: If E/S/C each drop different LQ slots, they see inconsistent representations.

**Solution**: Generate one global `lq_drop_mask` in `shared_forward()`, pass to all tasks.

**Benefit**: E/S/C learn on same "reduced capacity" state → better alignment.

### Why Task E Doesn't Enter Closed Loop?

**BLIP-2 analogy**: Task E is like BLIP-2's "image-text matching" — it uses ground-truth labels as **anchor** to prevent drift.

**Reason**: If E also follows posterior, it might over-adapt to LLM's current attention (which is still learning). Ground-truth entailment provides stable "recall anchor".

**Result**: E maintains high recall, filters out obvious negatives, while S/C handle nuanced ranking and generation alignment.

### Why Training Subset U?

**Problem**: ListNet/JS on full K=1000-5000 is expensive.

**Solution**: Define subset U = Teacher Top-L ∪ Student Top-l'
- Top-L: Dynamic size, chosen s.t. cumulative prob ≈ α_gt (0.9)
- Top-l': Fixed hard negatives from student

**Benefit**: 10-20x speedup, focuses on informative fragments.

### Why Detach Posterior?

**Problem**: If posterior has gradients from LLM, it brings LLM's noise into Q-Former.

**Solution**: `posterior_scores.detach()` before feeding to Task S.

**Benefit**: Task S treats posterior as "fixed teacher signal", avoids unstable gradients from frozen LLM.

---

## Hyperparameter Recommendations

### Small-scale (K≤100, steps≤10k)

```yaml
schedule:
  total_steps: 10000
  warmup_frac: 0.05
  bridge_frac: 0.5

optimizer:
  lr: 2e-4

data:
  batch_size: 16
```

### Medium-scale (K≤500, steps≤50k)

```yaml
schedule:
  total_steps: 50000
  warmup_frac: 0.1
  bridge_frac: 0.6

optimizer:
  lr: 1e-4

data:
  batch_size: 8
```

### Large-scale (K≤5000, steps≥100k)

```yaml
schedule:
  total_steps: 100000
  warmup_frac: 0.15
  bridge_frac: 0.7

optimizer:
  lr: 5e-5

training:
  gradient_accumulation_steps: 8
  use_amp: true
  amp_dtype: "bf16"

data:
  batch_size: 4
```

---

## Troubleshooting

### Loss Explodes in Bridge Phase

**Cause**: Posterior is noisy early on.

**Fix**: Slow down λ_post ramp:
```yaml
schedule:
  warmup_frac: 0.15  # Longer warm-up
  lambda_post_end: 0.6  # Lower final weight
```

### Task E Over-filters (Low Recall)

**Cause**: w_E too high, or γ too large.

**Fix**:
```yaml
losses:
  focal:
    gamma: 1.5  # Lower γ
schedule:
  w_E_end: 0.3  # Reduce Task E weight faster
```

### Task S Ignores Posterior

**Cause**: λ_post ramps too slowly, or posterior is weak.

**Fix**:
```yaml
schedule:
  bridge_frac: 0.5  # Shorter bridge
  lambda_post_end: 0.9  # Higher posterior weight
```

### OOM on Large K

**Fix**:
```yaml
training:
  gradient_accumulation_steps: 8
  use_amp: true

losses:
  ranking:
    top_lprime: 5  # Reduce hard negatives
```

---

## Comparison: BLIP-2 vs DR-QFormer

| Aspect | BLIP-2 Stage-1 | DR-QFormer Joint Training |
|--------|----------------|---------------------------|
| **Input** | Image patches | Text fragments |
| **Tasks** | ITC, ITM, ITG | E (Entailment), S (Ranking), C (Generation) |
| **Shared forward** | ✅ Q-Former once | ✅ Q-Former once |
| **Anchor task** | ITM (image-text matching) | Task E (entailment) |
| **Posterior feedback** | ❌ No | ✅ Yes (Task C → Task S) |
| **Drop-LQ** | ❌ No | ✅ Yes (unified) |
| **Training subset** | Full attention | Subset U (Top-L + Top-l') |

**Key difference**: DR-QFormer adds **closed-loop posterior feedback** (Task C → Task S) while keeping Task E as anchor, enabling dynamic ranking optimization beyond static teacher scores.

---

## Next Steps

1. **Implement Task C dual-path**:
   - See `compute_task_c_loss()` in `train/task_joint.py`
   - Follow implementation notes in `src/losses.py` (already has outline)

2. **Integrate real LLM**:
   - Replace `create_llm_placeholder()` in `scripts/train_joint.py`
   - Load Llama-2-7b with 8-bit quantization

3. **Add retriever**:
   - Replace `JointDataset._load_placeholder_data()`
   - BM25 or dense retriever (e.g., DPR)

4. **Add embedder**:
   - Replace `embedder` placeholder in `joint_data.py`
   - Use sentence-transformers or similar

5. **Add validation**:
   - Compute NDCG@10/20, Accuracy, Perplexity
   - Early stopping on validation loss

6. **Add tracking**:
   - Wandb or TensorBoard
   - Log loss curves, schedule curves, attention maps

---

## Citation

If you use this joint training framework, please cite:

```bibtex
@software{drqformer_joint,
  title = {DR-QFormer Joint Training Framework},
  author = {Your Name},
  year = {2025},
  note = {Based on BLIP-2 Stage-1 training philosophy}
}
```

---

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].

**Happy Training! 🚀**
