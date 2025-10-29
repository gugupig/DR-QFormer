# Quick Start Guide

Get started with DR-QFormer in 5 minutes!

## 📋 Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU support)

## 🔧 Installation

### Option 1: Basic Installation

```bash
# Clone repository
git clone https://github.com/gugupig/DR-QFormer.git
cd DR-QFormer

# Install core dependencies
pip install torch transformers numpy pyyaml

# Install package in editable mode
pip install -e .
```

### Option 2: Full Installation

```bash
# Clone repository
git clone https://github.com/gugupig/DR-QFormer.git
cd DR-QFormer

# Install all dependencies
pip install -r requirements.txt

# Install package with all extras
pip install -e ".[dev,eval,retrieval,logging]"
```

## 🎮 Run Demo

Test that everything is working:

```bash
python scripts/demo_cli.py
```

This will:
- Initialize model components
- Print parameter counts
- Echo sample queries

Expected output:
```
================================================================================
DR-QFormer Demo
================================================================================

Initializing models...

✓ DRQFormer: 0 trainable parameters
✓ EntailmentHead: 0 trainable parameters
...
```

## 📊 Project Structure Overview

```
DR-QFormer/
├── dr_qformer/          # Core library
│   ├── models/          # Q-Former & heads
│   ├── adapters/        # Retriever & LLM
│   ├── data/            # Data interfaces
│   └── utils/           # Masks & checkpoints
├── train/               # Training scripts
├── eval/                # Evaluation scripts
├── configs/             # YAML configs
└── tests/               # Unit tests
```

## 🚂 Training Workflow

### 1. Prepare Your Data

Create a JSONL file with examples (see `DATA_FORMAT.md`):

```json
{"query": "What is AI?", "fragments": [{"text": "AI is..."}], "answer": "..."}
```

### 2. Configure Training

Edit `configs/drqf_qa.yaml`:

```yaml
model:
  n_queries: 32
  hidden_dim: 768

training:
  lr: 1.0e-4
  batch_size: 8
  epochs: 10

data:
  train_data: "data/train.jsonl"
  dev_data: "data/dev.jsonl"
```

### 3. Start Training

```bash
# Train entailment tagging
python -m train.task_e --cfg configs/drqf_qa.yaml

# Train fragment sorting
python -m train.task_s --cfg configs/drqf_qa.yaml

# Train condensing-generation
python -m train.task_c --cfg configs/drqf_qg.yaml
```

### 4. Monitor Training

Training logs will show:
- Loss values
- Learning rate
- GPU memory usage
- Estimated time remaining

### 5. Evaluate Model

```bash
python -m eval.evaluate \
    --checkpoint outputs/checkpoint.ckpt \
    --test_data data/test.jsonl \
    --task_type qa
```

## 🔍 Understanding the Architecture

### Components

1. **Retriever** (Frozen)
   - Retrieves k fragments for each query
   - Produces embeddings P_embeds [k, d]
   - Example: Contriever, DPR, E5

2. **Q-Former** (Trainable)
   - N learnable query tokens (LQs)
   - Self-attention over [LQs, query text]
   - Cross-attention over P_embeds
   - Outputs Z [N, d]

3. **Task Heads** (Trainable)
   - EntailmentHead: Binary classification
   - SortingHead: Ranking scores
   - CondenseHead: LLM prefix

4. **LLM** (Frozen)
   - Generates text conditioned on Z
   - Example: LLaMA, Mistral, Phi

### Data Flow

```
Query → Retriever → Fragments (P_embeds)
                         ↓
Query Text → Q-Former → Z
                         ↓
                    Task Head → Output
                         ↓
                    LLM → Generated Text
```

## 🎯 Choosing a Task

### Task E: Entailment Tagging
**Use when**: You need to filter irrelevant fragments
**Output**: Binary labels [k] for each fragment
**Metrics**: Accuracy, Precision, Recall, F1

```bash
python -m train.task_e --cfg configs/drqf_qa.yaml
```

### Task S: Fragment Sorting
**Use when**: You need to rank fragments by relevance
**Output**: Ranking scores [k] for each fragment
**Metrics**: NDCG, MAP, Kendall's Tau

```bash
python -m train.task_s --cfg configs/drqf_qa.yaml
```

### Task C: Condensing-Generation
**Use when**: You need to improve LLM generation with retrieval
**Output**: Generated text from LLM
**Metrics**: ROUGE, BLEU, Exact Match

```bash
python -m train.task_c --cfg configs/drqf_qg.yaml
```

## 🧪 Running Tests

```bash
# Test shapes and forward passes
python tests/test_shapes.py

# Test parameter freezing
python tests/test_freezing.py

# Run all tests with pytest
pytest tests/
```

## 📝 Modifying Configurations

Common configuration changes:

### Increase Model Capacity
```yaml
model:
  n_queries: 64        # More query tokens
  hidden_dim: 1024     # Larger dimension
  num_layers: 12       # Deeper model
```

### Adjust Training
```yaml
training:
  lr: 5.0e-5          # Lower learning rate
  batch_size: 16      # Larger batches
  epochs: 20          # More epochs
  grad_clip: 0.5      # Gradient clipping
```

### Change Models
```yaml
retriever:
  model_name: "facebook/dpr-ctx_encoder-single-nq-base"

llm:
  model_name: "meta-llama/Llama-2-7b-hf"
  max_length: 512
```

## 🐛 Common Issues

### Issue: Import errors
```bash
# Solution: Install dependencies
pip install torch transformers
```

### Issue: CUDA out of memory
```yaml
# Solution: Reduce batch size
training:
  batch_size: 4  # or smaller
```

### Issue: Model not loading
```bash
# Solution: Download models first
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/contriever')"
```

## 📚 Next Steps

1. **Read the full README**: `README.md`
2. **Check data format**: `DATA_FORMAT.md`
3. **Review architecture**: `assets/diagram.md`
4. **Explore code**: Start with `dr_qformer/models/qformer.py`
5. **Contribute**: See `CONTRIBUTING.md`

## 💡 Tips

- Start with small models for testing
- Use small datasets for quick iterations
- Monitor GPU memory with `nvidia-smi`
- Use gradient checkpointing for large models
- Save checkpoints frequently
- Validate on dev set regularly

## 🆘 Getting Help

- Check existing issues: GitHub Issues
- Read documentation: All `.md` files
- Review code comments: Look for `# TODO` notes
- Ask questions: GitHub Discussions

## 🎉 You're Ready!

You now have everything you need to start using DR-QFormer. Happy training! 🚀

For detailed implementation, see:
- Training loops: `train/task_*.py`
- Model architecture: `dr_qformer/models/qformer.py`
- Data handling: `dr_qformer/data/interfaces.py`
