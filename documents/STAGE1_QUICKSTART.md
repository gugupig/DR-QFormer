# Stage-1 Training Quick Start Guide

## Overview

This guide explains how to run Stage-1 training for DR-QFormer, which implements joint TASK E + TASK S training without LLM integration.

## What is Stage-1?

**Stage-1 (MVP)** is a simplified training phase that:
- ✅ Trains TASK E (Fragment-level Entailment Tagging)
- ✅ Trains TASK S (Fragment-level Ranking/Sorting)
- ❌ Does NOT include TASK C (no LLM integration)
- ✅ Uses small evidence pools (K_max = 10)
- ✅ Uses pre-computed embeddings from `smoking_train_with_NoE.pkl`
- ✅ Supports batch processing with dynamic K padding

## Data Format

The training script expects `smoking_train_with_NoE.pkl` with the following structure per sample:

```python
{
    'sample_id': {
        'query': str,  # Query text
        'answer': str,  # Answer text
        'query_embedding': {
            'input_ids': tensor[1, seq_len],
            'attention_mask': tensor[1, seq_len],
            'token_emb_768': tensor[1, seq_len, 768]
        },
        'evidence_labels': ndarray[11],  # Binary labels (0/1)
        'evidence_text': list[10],  # Fragment texts
        'evidence_embeddings': ndarray[11, 768],  # Pre-computed embeddings
        'evidence_ranking': list[11],  # (idx, score) tuples
    }
}
```

## Quick Start

### 1. Prepare Data

Ensure `smoking_train_with_NoE.pkl` is in the project root or update the path in config:

```python
config = Stage1Config(
    train_data_path="path/to/smoking_train_with_NoE.pkl",
)
```

### 2. Run Training

```bash
cd train
python stage1_train.py
```

### 3. Monitor Progress

The script will:
- Load and split data (90% train, 10% val)
- Initialize XLM-RoBERTa-based Q-Former + Task Heads
- Train for specified epochs with automatic checkpointing
- Display progress bars with real-time metrics

Example output:
```
================================================================================
Epoch 1/10
================================================================================
Epoch 1: 100%|████████| 112/112 [02:15<00:00, loss=0.8234, loss_e=0.4521, loss_s=0.3713]

📈 Train Metrics:
   loss_total: 0.8234
   loss_e: 0.4521
   loss_s: 0.3713
   loss_s_teach: 0.3215
   loss_s_entropy: 0.0498
   lambda_teach: 0.9500
   lambda_post: 0.0000

📊 Validation Metrics:
   val_loss: 0.7654
   val_loss_e: 0.4123
   val_loss_s: 0.3531

✨ New best model! Val loss: 0.7654
💾 Checkpoint saved: ./checkpoints/stage1/best.pt
```

## Configuration

Key hyperparameters in `Stage1Config`:

### Model Architecture
- `n_queries`: Number of learnable queries (default: 32)
- `hidden_dim`: Hidden dimension (default: 768)
- `freeze_xlmr`: Freeze XLM-RoBERTa backbone (default: False)

### Task E (Entailment)
- `task_e_tau`: Temperature for LogSumExp (default: 0.5)
- `task_e_focal_gamma`: Focal loss gamma (default: 2.0)
- `task_e_w_pos`: Positive class weight (default: 10.0)

### Task S (Ranking)
- `task_s_tau_head`: Head temperature (default: 0.1)
- `task_s_rho_top`: Teacher top-L ratio (default: 0.2 for K=10)
- `task_s_l_prime`: Student hard negatives (default: 3)

### Training
- `batch_size`: Batch size (default: 8)
- `num_epochs`: Number of epochs (default: 10)
- `max_steps`: Maximum training steps (default: 50000)
- `lr`: Learning rate (default: 1e-4)

### Multi-task Weights
- `w_task_e`: Weight for TASK E loss (default: 1.0)
- `w_task_s`: Weight for TASK S loss (default: 1.0)

## Advanced Usage

### Custom Configuration

```python
from train.stage1_train import Stage1Config, Stage1Trainer, load_and_split_data

# Create custom config
config = Stage1Config(
    train_data_path="my_data.pkl",
    batch_size=16,
    num_epochs=20,
    lr=5e-5,
    freeze_xlmr=True,  # Freeze backbone for faster training
    w_task_e=1.5,  # Emphasize entailment task
    w_task_s=1.0,
)

# Load data
train_dataset, val_dataset = load_and_split_data(
    config.train_data_path,
    val_split=0.15,  # 15% validation
    shuffle=True,
    seed=42,
)

# Train
trainer = Stage1Trainer(config)
# ... training loop ...
```

### Resume from Checkpoint

```python
trainer = Stage1Trainer(config)
trainer.load_checkpoint("checkpoints/stage1/step_10000.pt")

# Continue training
for epoch in range(config.num_epochs):
    trainer.train_epoch(train_loader, epoch)
```

### Evaluation Only

```python
trainer = Stage1Trainer(config)
trainer.load_checkpoint("checkpoints/stage1/best.pt")

# Evaluate
metrics = trainer.evaluate(val_loader)
print(f"Validation metrics: {metrics}")
```

## Key Features

### 1. Dynamic K Padding
The script automatically handles variable evidence pool sizes:
- Finds max K in each batch
- Pads to K_max with appropriate masking
- Uses `pool_padding_mask` to ignore padded fragments

### 2. Unified Drop-LQ
Multi-task training uses a shared Drop-LQ mask:
- Generated once per forward pass
- Shared by both Task E and Task S heads
- Ensures consistent LQ dropout across tasks

### 3. Curriculum Learning for Task S
Task S uses curriculum learning to transition from teacher supervision:
- Early training: λ_teach = 1.0, λ_post = 0.0
- Late training: λ_teach = 0.2, λ_post = 0.0 (no posterior in Stage-1)
- Linear schedule over max_steps

### 4. No Subset Masking
Since K_max = 10 is small, all fragments are used for training:
- `train_subset_mask = None` in Task S
- No need for subset selection (Top-L + Hard Negatives)
- Simplifies training loop

## Output Files

Training produces the following checkpoints:

```
checkpoints/stage1/
├── best.pt              # Best model by validation loss
├── step_1000.pt         # Periodic checkpoints
├── step_2000.pt
└── ...
```

Each checkpoint contains:
- Q-Former state dict
- Task E head state dict
- Task S head state dict
- Optimizer state dict
- Global step counter
- Configuration

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or freeze XLM-RoBERTa:
```python
config = Stage1Config(
    batch_size=4,  # Reduce from 8
    freeze_xlmr=True,  # Freeze backbone
)
```

### Slow Training

Enable XLM-RoBERTa freezing and reduce CA layers:
```python
config = Stage1Config(
    freeze_xlmr=True,
    use_ca_layers=[5, 11],  # Only layers 6 and 12
)
```

### Data Loading Errors

Check pickle file format:
```python
import pickle
with open('smoking_train_with_NoE.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f"Loaded {len(data)} samples")
    sample_id = list(data.keys())[0]
    print(f"Sample structure: {data[sample_id].keys()}")
```

### Loss Not Decreasing

1. Check learning rate (may be too high/low)
2. Verify data quality (labels should be meaningful)
3. Monitor task-specific losses separately
4. Try adjusting task weights

## Next Steps

After Stage-1 training:

1. **Evaluate Model Performance**
   - Check entailment tagging accuracy
   - Analyze ranking metrics (NDCG, MRR)
   - Visualize attention weights

2. **Prepare for Stage-2**
   - Integrate LLM adapter (Frozen LLM)
   - Add TASK C (Condensing-generation)
   - Enable posterior extraction from LLM attention

3. **Hyperparameter Tuning**
   - Grid search over task weights
   - Optimize Drop-LQ rate
   - Tune curriculum schedule

4. **Model Compression**
   - Reduce LQs (32 → 16 → 8)
   - Distill to smaller Q-Former
   - Quantize for deployment

## References

- `train/stage1_train.py`: Main training script
- `src/models/qformer_xlm.py`: XLM-RoBERTa-based Q-Former
- `src/models/heads.py`: Task E and Task S heads
- `src/losses.py`: Loss functions and curriculum schedules
- `documents/JOINT_TRAINING_QUICKREF.md`: Architecture reference
