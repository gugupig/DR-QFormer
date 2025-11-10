# DR-QFormer Joint Training - Quick Reference

## 📋 Summary

**Framework**: Joint training for Tasks E/S/C following BLIP-2 Stage-1 philosophy  
**Key Innovation**: Shared Q-Former forward + closed-loop posterior feedback (C→S)  
**Training Phases**: Warm-up (prior) → Bridge (transition) → Closed-loop (posterior)

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install torch transformers sentence-transformers pyyaml tqdm

# 2. Configure (edit configs/joint_train.yaml)
# - Set model dimensions (d, N, num_heads)
# - Set schedule (total_steps, phase fractions)
# - Set data paths (train/val)

# 3. Train
python scripts/train_joint.py --config configs/joint_train.yaml
```

---

## 📂 File Structure

```
train/
├── joint_data.py       # Data loading, batching, pool_padding_mask
├── schedule.py         # Dynamic weight scheduling (w_E/w_S/w_C, λs)
└── task_joint.py       # JointTrainer: shared forward, E/S/C losses

scripts/
└── train_joint.py      # Training entry point

configs/
└── joint_train.yaml    # All hyperparameters

documents/
└── JOINT_TRAINING.md   # Full documentation
```

---

## 🎯 Three Tasks

| Task | Purpose | Loss | Input | Output |
|------|---------|------|-------|--------|
| **E** | Entailment filtering | Focal Loss | CA raw scores | fragment_logits [batch, K] |
| **S** | Fragment ranking | ListNet + JS + Entropy | CA raw scores | ranking_logits [batch, K] |
| **C** | Generation alignment | Contrastive NLL | Z (LM-projected) | NLL, posterior qψ_U |

---

## 📊 Training Schedule

### Phase Timeline (Default: 50k steps)

```
|←  Warm-up  →|←        Bridge        →|← Closed-loop →|
0            5k                      35k              50k
Prior-only    Teacher→Posterior       Posterior-dominant
```

### Weight Evolution

```
w_E:  1.0 ━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━ 0.5
w_S:  1.0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0
w_C:  0.5 ━━━━━━━━━━▲━━━━━━━━━━━━━━━━━━━━━━━━ 1.0

λ_teach: 1.0 ━━━━━━━┓                    ┏━━━ 0.2
                     ╲                  ╱
λ_post:  0.0 ━━━━━━━━┗━━━━━━━━━━━━━━━━┛━━━ 0.8
         (Warm-up)   (Bridge transition)  (Closed)
```

---

## 🔧 Key Hyperparameters

### Model

```yaml
model:
  d: 768                    # Hidden dimension
  num_lqs: 32               # Number of LQ slots
  num_heads: 12             # Attention heads
  drop_lq_rate: 0.1         # Unified Drop-LQ
```

### Losses

```yaml
losses:
  focal:
    gamma: 2.0              # Focus on hard negatives
    alpha: 0.25             # Class balance
  
  ranking:
    lambda_teach_end: 0.2   # Final teacher weight
    lambda_post_end: 0.8    # Final posterior weight
    lambda_ent_start: 0.01  # Entropy regularization
    top_lprime: 10          # Hard negatives
  
  contrastive:
    softplus_beta: 10.0     # Sharpness
    adaptive_margin: true   # μ_G + κσ_G
```

### Schedule

```yaml
schedule:
  total_steps: 50000
  warmup_frac: 0.1          # Prior-only (10%)
  bridge_frac: 0.6          # Transition (60%)
  closedloop_frac: 0.3      # Posterior (30%)
```

---

## 🔄 Training Flow (Per Step)

```python
# 1. Shared Q-Former Forward (once)
forward_out = shared_forward(batch, mode="primal")
# → z, z_lm, ca_raw_scores, ca_weights, sa_weights, lq_drop_mask

# 2. Task E: Entailment
loss_e = task_e_head(forward_out) + focal_loss(labels)

# 3. Task C: Contrastive NLL + Posterior
loss_c, posterior_qψ_U = task_c(forward_out, llm) if enable_posterior else 0

# 4. Task S: Ranking (with posterior if enabled)
loss_s = task_s_head(forward_out) + ListNet(teacher) + JS(posterior_qψ_U)

# 5. Combined Loss
loss_total = w_E*loss_e + w_S*loss_s + w_C*loss_c

# 6. Backward
optimizer.zero_grad()
loss_total.backward()
optimizer.step()
```

---

## ⚠️ TODO (Placeholders)

### Critical

- [ ] **Task C dual-path Teacher Forcing**
  - Path-A: `[Z, Q, A]` with full attention
  - Path-B: `[Z_dummy, Q, A]` with masked Q/A→Z
  - Extract posterior qψ_U from LLM→Z attention
  - Location: `train/task_joint.py` line 275-300

- [ ] **LLM Integration**
  - Load Llama-2-7b or similar
  - 8-bit quantization
  - Freeze parameters
  - Location: `scripts/train_joint.py` line 96-115

### Data Pipeline

- [ ] **Retriever** (BM25 or dense)
  - Location: `train/joint_data.py` line 82-85

- [ ] **Embedder** (Sentence-BERT)
  - Location: `train/joint_data.py` line 87-90

- [ ] **Tokenizer** (LLM tokenizer)
  - Location: `train/joint_data.py` line 92-95

### Evaluation

- [ ] **Validation Loop**
  - Metrics: NDCG, Accuracy, Perplexity
  - Location: Add to `scripts/train_joint.py` line 300+

- [ ] **Experiment Tracking**
  - Wandb or TensorBoard
  - Location: Add to `train/task_joint.py` line 480+

---

## 🔍 Testing

### Test Data Module

```bash
cd train
python joint_data.py
# Output: Batch inspection with dummy data
```

### Test Schedule

```bash
cd train
python schedule.py
# Output: Schedule evolution at key checkpoints
```

### Test Trainer Structure

```bash
cd train
python task_joint.py
# Output: One training step with dummy models
```

---

## 📈 Monitoring

### Key Metrics

**Task E (Entailment)**:
- `e_accuracy`: Classification accuracy
- `e_precision`: Precision on positive class
- `e_recall`: Recall on positive class (should be high!)

**Task S (Ranking)**:
- `s_ndcg`: NDCG@K on teacher scores
- `s_subset_size`: Training subset U size

**Task C (Generation)**:
- `c_nll_a`: NLL with Z (Path-A)
- `c_nll_b`: NLL without Z (Path-B)
- `c_gain`: G = NLL_B - NLL_A (should be positive!)

**Schedule**:
- `phase`: warmup/bridge/closedloop
- `w_E`, `w_S`, `w_C`: Task weights
- `lambda_teach`, `lambda_post`: Ranking sub-loss weights

---

## 🐛 Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Loss explodes in Bridge | Posterior too noisy | Longer warm-up, lower λ_post_end |
| Task E over-filters | γ too high | Lower focal γ to 1.5 |
| Task S ignores posterior | λ_post too low | Increase λ_post_end to 0.9 |
| OOM on large K | Too many fragments | Gradient accumulation, reduce top_lprime |

---

## 📚 Key Concepts

### Unified Drop-LQ
- **What**: Same LQ drop mask across E/S/C
- **Why**: Consistent representation, better alignment
- **Where**: Generated in `shared_forward()`, passed to all tasks

### Training Subset U
- **What**: Teacher Top-L ∪ Student Top-l'
- **Why**: 10-20x speedup, focus on informative fragments
- **How**: Dynamic L (cumulative prob ≈ α_gt), fixed l' (hard negatives)

### Posterior Feedback (C→S)
- **What**: LLM→Z attention backtracked to fragments
- **Why**: Task S learns "what LLM actually needs"
- **How**: Extract from Path-A, detach, feed to Task S as posterior_scores

### Anchor Task (E)
- **What**: Task E uses ground-truth labels, no posterior
- **Why**: Prevents drift, maintains high recall
- **Analogy**: Like BLIP-2's image-text matching

---

## 🎓 BLIP-2 Comparison

| Aspect | BLIP-2 Stage-1 | DR-QFormer Joint |
|--------|----------------|------------------|
| Shared forward | ✅ | ✅ |
| Multiple objectives | ✅ (ITC/ITM/ITG) | ✅ (E/S/C) |
| Anchor task | ITM | Task E |
| Posterior feedback | ❌ | ✅ (C→S) |
| Drop-LQ | ❌ | ✅ |
| Training subset | Full | Subset U |

---

## 📞 Support

- **Documentation**: `documents/JOINT_TRAINING.md`
- **Config**: `configs/joint_train.yaml`
- **Tests**: Run `python train/*.py` for each module

---

## ✅ Checklist Before Training

- [ ] Installed dependencies (`torch`, `transformers`, etc.)
- [ ] Configured `configs/joint_train.yaml` (paths, hyperparameters)
- [ ] Prepared data in expected format (Q/A/fragments/labels)
- [ ] Replaced retriever placeholder in `joint_data.py`
- [ ] Replaced embedder placeholder in `joint_data.py`
- [ ] Replaced tokenizer placeholder in `joint_data.py`
- [ ] Implemented Task C dual-path in `task_joint.py` (or accepted placeholder for initial testing)
- [ ] Loaded actual LLM in `train_joint.py` (or accepted placeholder)
- [ ] Set `save_dir` for checkpoints
- [ ] (Optional) Configured wandb/tensorboard

---

**Ready to train? Run:**

```bash
python scripts/train_joint.py --config configs/joint_train.yaml
```

**Good luck! 🚀**
