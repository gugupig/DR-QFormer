# Task E Quick Start Guide

## What is Task E?

**Task E (Fragment-level Entailment Tagging)** trains the Q-Former to predict which retrieved fragments are relevant/entailed by a query.

**Purpose**: Learn to tag fragments as entailment (relevant) or non-entailment (irrelevant) to improve RAG retrieval quality.

---

## Quick Start

### 1. Basic Training

```bash
python train/task_e.py \
    --train_data data/train_entailment.json \
    --dev_data data/dev_entailment.json \
    --retriever_model facebook/contriever \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4
```

### 2. With Custom Hyperparameters

```bash
python train/task_e.py \
    --train_data data/train_entailment.json \
    --dev_data data/dev_entailment.json \
    --retriever_model facebook/contriever \
    --mode both \
    --tau 0.5 \
    --p_drop_lq 0.1 \
    --focal_gamma 2.0 \
    --focal_alpha 0.25 \
    --w_pos 10.0 \
    --w_longtail 50.0 \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4 \
    --save_dir ./checkpoints/task_e
```

---

## Data Format

Your training data should be in JSON format:

```json
[
    {
        "query": "What is the capital of France?",
        "fragments": [
            "Paris is the capital and most populous city of France.",
            "The Eiffel Tower is located in Paris.",
            "London is the capital of the United Kingdom.",
            "France is a country in Western Europe.",
            "The French Revolution began in 1789."
        ],
        "gt_labels": [1, 1, 0, 1, 0],
        "importance_weights": [10.0, 10.0, 1.0, 10.0, 1.0],
        "mode": "primal"
    },
    {
        "query": "Where is the Eiffel Tower?",
        "fragments": [
            "The Eiffel Tower is located in Paris.",
            "Paris is the capital of France.",
            "The tower was built in 1889.",
            "Tokyo Tower is in Japan.",
            "France is in Europe."
        ],
        "gt_labels": [1, 1, 1, 0, 0],
        "importance_weights": [10.0, 10.0, 10.0, 1.0, 1.0],
        "mode": "primal"
    }
]
```

**Fields**:
- `query`: Query string
- `fragments`: List of k retrieved fragments (typically k=5)
- `gt_labels`: Binary labels (1=entailment/relevant, 0=non-entailment/irrelevant)
- `importance_weights`: Weight per fragment (w_pos=10.0 for positive, w_longtail=50.0 for longtail)
- `mode`: "primal" (QA) or "dual" (QG)

---

## Training Modes

### Primal Mode (QA)
- Input: **Query** → Predict fragment relevance
- Example: "What is X?" → Tag fragments about X as relevant

### Dual Mode (QG)
- Input: **Generated Question** → Predict fragment relevance
- Example: "What topic does this fragment discuss?" → Tag source fragment as relevant

### Both Mode (Recommended)
- Alternates between Primal and Dual modes
- Shares parameters between both modes
- Better generalization

---

## Hyperparameters

### Core Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_queries` | 32 | 16-64 | Number of learnable queries (N) |
| `k_fragments` | 5 | 3-10 | Number of retrieved fragments |
| `hidden_dim` | 768 | 512-1024 | Q-Former hidden dimension |
| `num_layers` | 12 | 6-12 | Number of Q-Former layers |

### EntailmentHead Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `tau` | 0.5 | 0.1-1.0 | LogSumExp temperature (lower = more selective) |
| `p_drop_lq` | 0.1 | 0.0-0.3 | Drop-LQ probability during training |
| `focal_gamma` | 2.0 | 1.0-3.0 | Focal loss focusing parameter |
| `focal_alpha` | 0.25 | 0.1-0.5 | Focal loss class balance |
| `w_pos` | 10.0 | 5.0-20.0 | Positive class importance weight |
| `w_longtail` | 50.0 | 20.0-100.0 | Longtail example importance weight |

### Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `batch_size` | 8 | 4-32 | Batch size |
| `epochs` | 10 | 5-50 | Number of epochs |
| `lr` | 1e-4 | 5e-5 to 5e-4 | Learning rate |
| `grad_clip` | 1.0 | 0.5-2.0 | Gradient clipping threshold |

---

## Understanding the Pipeline

### 1. Input Processing
```
Query + k Fragments
    ↓ (Frozen Retriever: Contriever/DPR/E5/BGE)
Query Embedding [d_ret] + Fragment Embeddings [k, d_ret]
```

### 2. Q-Former Forward
```
Q-Former(q_embed, p_embeds)
    ↓
z: [batch, N=32, d=768]  (Learned query representations)
aux: {ca_attn_weights: [...]}  (Cross-attention weights per layer)
```

### 3. EntailmentHead Forward
```
EntailmentHead(z, ca_attn_weights)
    ↓ (4-step pipeline)
    1. Normalize CA scores: LayerNorm per fragment
    2. Apply Drop-LQ: Bernoulli mask (training only)
    3. Aggregate: LogSumExp with temperature τ
    4. Output: fragment_logits [batch, k]
```

### 4. Loss Computation
```
Focal Loss with Importance Weighting
    - Positive class weight: w_pos = 10.0
    - Longtail weight: w_longtail = 50.0
    - Pool padding mask: Ignore padded fragments
```

---

## Evaluation

After training, evaluate your model:

```python
import torch
from dr_qformer.models.qformer import DRQFormer
from dr_qformer.models.heads import EntailmentHead
from dr_qformer.adapters.retriever import RetrieverAdapter

# Load model
checkpoint = torch.load("checkpoints/task_e/best.pt")
qformer = DRQFormer(n_queries=32, hidden_dim=768)
head = EntailmentHead(hidden_dim=768, num_fragments=5)
qformer.load_state_dict(checkpoint["qformer_state_dict"])
head.load_state_dict(checkpoint["head_state_dict"])

# Inference
retriever = RetrieverAdapter("facebook/contriever")
query = "What is the capital of France?"
fragments = ["Paris is the capital...", "London is...", ...]

q_embed = retriever.encode_queries([query])
p_embeds = retriever.encode_passages(fragments)
p_embeds = p_embeds.unsqueeze(0)  # [1, k, d_ret]

z, aux = qformer(q_embed, p_embeds)
logits = head(z, aux["ca_attn_weights"])
probs = torch.sigmoid(logits)  # [1, k]

print(f"Fragment relevance scores: {probs[0]}")
```

---

## Troubleshooting

### Issue: Loss not decreasing
- **Solution**: Lower learning rate (try 5e-5 instead of 1e-4)
- **Solution**: Increase w_pos to emphasize positive class
- **Solution**: Check data quality (are labels correct?)

### Issue: Model predicts all 0 or all 1
- **Solution**: Adjust focal_alpha (try 0.5 for balanced classes)
- **Solution**: Check class distribution in training data
- **Solution**: Increase w_pos or w_longtail

### Issue: Training too slow
- **Solution**: Increase batch size (if GPU memory allows)
- **Solution**: Use a smaller retriever model
- **Solution**: Reduce k_fragments (try 3 instead of 5)

### Issue: Overfitting
- **Solution**: Increase p_drop_lq (try 0.2 instead of 0.1)
- **Solution**: Add more training data
- **Solution**: Early stopping based on dev loss

---

## Next Steps

1. **Test with synthetic data** to verify setup
2. **Prepare real dataset** with entailment labels
3. **Run training** with default hyperparameters
4. **Tune hyperparameters** based on dev set performance
5. **Evaluate** with precision, recall, F1, AUC-ROC
6. **Integrate** with Task S and Task C for full pipeline

---

## Resources

- **Full Implementation Guide**: `TASK_E_IMPLEMENTATION.md`
- **Q-Former Architecture**: `DR_QFORMER_IMPLEMENTATION.md`
- **Attention Analysis**: `ATTENTION_ANALYSIS_GUIDE.md`
- **Training Script**: `train/task_e.py`
- **EntailmentHead Code**: `dr_qformer/models/heads.py`

**Happy Training! 🚀**
