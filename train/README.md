# DR-QFormer Training Scripts

This directory contains training scripts for DR-QFormer at different stages.

## Available Training Scripts

### 1. Stage-1 Training (MVP) - `stage1_train.py` ✅ **NEW**

**Status**: Complete and Ready  
**Purpose**: Joint TASK E + TASK S without LLM integration

**Features**:
- ✅ TASK E: Fragment-level entailment tagging
- ✅ TASK S: Fragment-level ranking with curriculum learning
- ❌ TASK C: Not included (no LLM in Stage-1)
- ✅ Small evidence pools (K_max = 10)
- ✅ Batch processing with dynamic padding
- ✅ Pre-computed embeddings support

**Quick Start**:
```bash
# Test components first
python ../tests/test_stage1.py

# Run training
python stage1_train.py
```

**Documentation**: See `../documents/STAGE1_QUICKSTART.md`

**Use Cases**:
- Initial DR-QFormer training without LLM
- Fast prototyping and debugging
- Learning basic fragment-level operations
- Preparing for Stage-2 (with LLM)

---

### 2. Task E Training - `task_e.py`

**Purpose**: Single-task training for entailment tagging

**Features**:
- Fragment-level binary classification
- Focal loss with importance weighting
- Optional dual mode (QA + QG)
- LQ entropy regularization

**Quick Start**:
```bash
python task_e.py \
    --train_data path/to/train.jsonl \
    --dev_data path/to/dev.jsonl \
    --mode primal
```

**Use Cases**:
- Isolated Task E development
- Hyperparameter tuning
- Ablation studies

---

### 3. Task S Training - `task_s.py`

**Purpose**: Single-task training for fragment ranking

**Features**:
- Large evidence pool support (K=100-5000)
- Curriculum learning (teacher → posterior)
- Dynamic subset selection (Top-L + Hard Negatives)
- Optional dual mode

**Quick Start**:
```bash
python task_s.py \
    --train_data path/to/train.jsonl \
    --dev_data path/to/dev.jsonl
```

**Use Cases**:
- Isolated Task S development
- Ranking algorithm testing
- Curriculum schedule tuning

---

### 4. Joint Training - `task_joint.py`

**Purpose**: Full multi-task training (E+S+C)

**Features**:
- All three tasks (E, S, C)
- LLM integration (frozen)
- Posterior extraction from LLM attention
- Bayesian closed-loop learning
- Large-scale evidence pools

**Status**: Requires LLM adapter implementation

**Use Cases**:
- Full DR-QFormer training
- End-to-end RAG pipeline
- Production model training

---

## Training Progression

Recommended training order:

```
Stage-1 (MVP)  →  Task E/S tuning  →  Full Joint Training
     ↓                    ↓                    ↓
   No LLM           Single tasks         All tasks + LLM
   K=10             K=100                K=100-5000
   Quick test       Fine-tuning          Production
```

### Stage-1 → Stage-2 Migration

After Stage-1 converges:

1. **Add LLM Adapter**:
   ```python
   from src.adapters.frozen_llm import FrozenLLM
   llm = FrozenLLM("meta-llama/Llama-2-7b-hf")
   ```

2. **Enable TASK C**:
   ```python
   config.w_task_c = 1.0  # Enable Task C loss
   ```

3. **Enable Posterior**:
   ```python
   config.task_s_lambda_post_end = 0.8  # Target posterior weight
   ```

4. **Expand Evidence Pool**:
   ```python
   config.k_max = 100  # Or larger
   ```

## Data Format

### Stage-1 Format (smoking_train_with_NoE.pkl)
```python
{
    'sample_id': {
        'query': str,
        'answer': str,
        'query_embedding': {...},  # Pre-computed
        'evidence_labels': ndarray[11],
        'evidence_embeddings': ndarray[11, 768],
        'evidence_ranking': list[11],
    }
}
```

### Standard JSONL Format (task_e.py, task_s.py)
```json
{
    "query": "What is X?",
    "answer": "Y",
    "fragments": [
        {"text": "...", "score": 0.9, "entailment_label": 1},
        ...
    ],
    "gt_k": [1, 1, 0, ...],
    "gt_soft_weights": [0.4, 0.3, 0.2, ...]
}
```

See `../documents/DATA_FORMAT.md` for complete specification.

## Configuration Files

Training configurations in `../configs/`:

- `drqf_qa.yaml` - QA mode settings
- `drqf_qg.yaml` - QG mode settings
- `joint_train.yaml` - Multi-task settings

## Checkpointing

All scripts save checkpoints to `../checkpoints/{task_name}/`:

```
checkpoints/
├── stage1/
│   ├── best.pt
│   ├── step_1000.pt
│   └── ...
├── task_e/
├── task_s/
└── joint/
```

Load checkpoint:
```python
trainer.load_checkpoint("../checkpoints/stage1/best.pt")
```

## Monitoring

### Metrics Tracked

**Task E**:
- loss_e, accuracy, precision, recall, F1

**Task S**:
- loss_s, loss_teach, loss_post, NDCG, MRR

**Task C** (when enabled):
- loss_c, nll_gain, margin, posterior quality

### Logging

Use progress bars for real-time monitoring:
```
Epoch 1: 100%|████| 112/112 [02:15<00:00, loss=0.82, loss_e=0.45, loss_s=0.37]
```

## Hyperparameter Tuning

### Quick Reference

| Parameter | Task E | Task S | Typical Range |
|-----------|--------|--------|---------------|
| tau | 0.5 | 0.1-0.2 | 0.1-1.0 |
| focal_gamma | 2.0 | - | 1.0-3.0 |
| rho_top | - | 0.02-0.2 | 0.01-0.3 |
| l_prime | - | 3-16 | 1-32 |
| w_pos | 10.0 | - | 5.0-50.0 |
| p_drop_lq | 0.1 | 0.1 | 0.0-0.3 |

### Tuning Tips

1. **Start with defaults** in Stage-1
2. **Single-task tuning** for Task E/S separately
3. **Multi-task balancing** adjust w_E, w_S, w_C
4. **Curriculum schedule** tune lambda_teach/post transitions
5. **Regularization** adjust Drop-LQ and entropy weights

## Troubleshooting

### Common Issues

**Out of Memory**:
```python
config.batch_size = 4  # Reduce
config.freeze_xlmr = True  # Freeze backbone
```

**Loss Not Decreasing**:
- Check learning rate (try 1e-5 to 1e-3)
- Verify data quality
- Monitor individual task losses
- Check gradient norms

**NaN Loss**:
- Reduce learning rate
- Add gradient clipping
- Check for invalid data
- Verify mask usage

**Slow Training**:
- Use GPU (`device='cuda'`)
- Freeze XLM-RoBERTa backbone
- Reduce num_layers or n_queries
- Reduce CA layers (use_ca_layers=[5, 11])

## Testing

Run tests before training:

```bash
# Stage-1 components
python ../tests/test_stage1.py

# Task E components
python ../tests/test_task_e.py

# Task S components
python ../tests/test_task_s.py

# Full integration
python ../tests/test_joint_training.py
```

## Documentation

Detailed guides in `../documents/`:

1. **STAGE1_QUICKSTART.md** - Stage-1 training guide
2. **STAGE1_IMPLEMENTATION_SUMMARY.md** - Implementation details
3. **JOINT_TRAINING_QUICKREF.md** - Architecture reference
4. **DATA_FORMAT.md** - Data format specification
5. **TASK_E_IMPLEMENTATION.md** - Task E details
6. **TASK_S_MODIFICATIONS.md** - Task S details

## Support Files

### Schedulers (`schedule.py`)

Learning rate and curriculum schedulers:
```python
from train.schedule import get_lr_schedule, JointTrainingScheduler

lr_scheduler = get_lr_schedule(optimizer, warmup_steps, total_steps)
curriculum = JointTrainingScheduler(config)
```

### Data Loaders (`joint_data.py`)

Dataset and collate functions:
```python
from train.joint_data import JointTrainingDataset, collate_joint_batch

dataset = JointTrainingDataset(data_path)
loader = DataLoader(dataset, collate_fn=collate_joint_batch)
```

### Common Utils (`common.py`)

Shared utilities across training scripts:
```python
from train.common import set_seed, count_parameters, save_args
```

## Quick Command Reference

```bash
# Stage-1 training
python train/stage1_train.py

# Task E training
python train/task_e.py --train_data data/train.jsonl --dev_data data/dev.jsonl

# Task S training
python train/task_s.py --train_data data/train.jsonl --dev_data data/dev.jsonl

# Resume training
python train/stage1_train.py --resume checkpoints/stage1/step_5000.pt

# Evaluate only
python train/stage1_train.py --eval_only --checkpoint checkpoints/stage1/best.pt

# Custom config
python train/stage1_train.py --config configs/custom.yaml
```

## Best Practices

1. **Start Simple**: Begin with Stage-1, then expand
2. **Test Components**: Run test suite before full training
3. **Monitor Closely**: Watch loss curves and metrics
4. **Save Frequently**: Use small save_interval initially
5. **Ablate Carefully**: Change one thing at a time
6. **Document Results**: Keep training logs and notes
7. **Version Control**: Commit configs and code changes

## Contributing

When adding new training scripts:

1. Follow existing naming convention (`task_*.py`)
2. Include comprehensive docstrings
3. Add test script in `../tests/`
4. Update this README
5. Add documentation in `../documents/`
6. Test with both mock and real data

## License

See main repository LICENSE file.

## Citation

If using this training code, please cite:

```bibtex
@software{drqformer2024,
  title={DR-QFormer: Dense Retrieval Query Former},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/DR-QFormer}
}
```

---

**Last Updated**: 2024-11-21  
**Status**: Stage-1 complete, Stage-2 in progress
