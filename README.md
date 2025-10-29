# DR-QFormer

**Parameter-Efficient Middleware for Retrieval-Augmented Generation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DR-QFormer is a BLIP-2-style parameter-efficient architecture that bridges frozen retrievers and frozen large language models (LLMs) for RAG tasks. Only the Q-Former and task-specific heads are trained, achieving efficient adaptation while leveraging powerful pretrained components.

## 🎯 Key Features

- **Parameter Efficiency**: Train only ~1-2% of total parameters (Q-Former + heads)
- **Frozen Components**: Retriever and LLM remain frozen throughout training
- **Modular Design**: Swap retrievers, LLMs, and task heads independently
- **Multiple Tasks**: Support for entailment tagging, fragment sorting, and condensing-generation
- **BLIP-2 Architecture**: Learnable query tokens with self-attention and cross-attention

## 📐 Architecture

```
Query → [Frozen Retriever] → Fragments (P_embeds)
                                    ↓
                              [Q-Former] ← Learnable Query Tokens (LQs)
                                    ↓
                        Self-Attention (SA) over [LQs, query/answer]
                                    ↓
                        Cross-Attention (CA) over P_embeds
                                    ↓
                              Output: Z [N, d]
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
            EntailmentHead    SortingHead    CondenseHead
                    ↓               ↓               ↓
              [k] labels      [k] scores    [Frozen LLM] → Text
```

See [`assets/diagram.md`](assets/diagram.md) for detailed architecture diagram.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/gugupig/DR-QFormer.git
cd DR-QFormer

# Install dependencies (TODO: create requirements.txt)
pip install torch transformers
```

### Training

Train on different tasks using the provided training scripts:

```bash
# Task E: Fragment-level Entailment Tagging
python -m train.task_e --cfg configs/drqf_qa.yaml

# Task S: Fragment-level Sorting Supervision
python -m train.task_s --cfg configs/drqf_qa.yaml

# Task C: Condensing-Generation (Reward Margin)
python -m train.task_c --cfg configs/drqf_qg.yaml
```

### Evaluation

```bash
python -m eval.evaluate --checkpoint path/to/checkpoint.ckpt --test_data data/test.json
```

### Demo

```bash
python scripts/demo_cli.py
```

## 📁 Project Structure

```
dr_qformer/
  models/
    qformer.py          # DR-QFormer core model
    heads.py            # Task-specific heads (Entailment, Sorting, Condense)
  adapters/
    retriever.py        # Frozen retriever adapter
    llm.py              # Frozen LLM adapter
  data/
    interfaces.py       # Data structures (Fragment, Example, Dataset)
  utils/
    masks.py            # Attention mask utilities
    checkpoint.py       # Checkpoint save/load utilities
  losses.py             # Loss functions (BCE, ranking, reward margin)
  metrics.py            # Evaluation metrics (EM, F1, ROUGE, BLEU, NDCG)

train/
  task_e.py             # Training script for entailment tagging
  task_s.py             # Training script for fragment sorting
  task_c.py             # Training script for condensing-generation
  common.py             # Common utilities (arg parsing, setup, Trainer)

eval/
  evaluate.py           # Evaluation script

scripts/
  demo_cli.py           # Demo script

configs/
  drqf_qa.yaml          # Configuration for QA tasks
  drqf_qg.yaml          # Configuration for generation tasks

tests/
  test_shapes.py        # Shape/smoke tests
  test_freezing.py      # Parameter freezing tests
```

## 🎓 Training Tasks

### Task E: Entailment Tagging
Trains Q-Former + EntailmentHead to predict binary relevance labels for retrieved fragments.

**Loss**: Binary cross-entropy  
**Metrics**: Accuracy, Precision, Recall, F1

### Task S: Fragment Sorting
Trains Q-Former + SortingHead to rank retrieved fragments by relevance.

**Loss**: Ranking loss (ListMLE, pairwise margin)  
**Metrics**: NDCG, MAP, Kendall's Tau

### Task C: Condensing-Generation
Trains Q-Former + CondenseHead to produce condensed representations that improve LLM generation quality.

**Loss**: Reward margin (ROUGE/BLEU comparison)  
**Metrics**: ROUGE, BLEU, Exact Match, F1

## 🔧 Configuration

Edit YAML config files in `configs/` to customize:

- Model architecture (N, d, layers, heads)
- Training hyperparameters (lr, batch_size, epochs)
- Retriever model (Contriever, DPR, E5, etc.)
- LLM model (LLaMA, Mistral, Phi, etc.)
- Data paths and output directories

Example (`configs/drqf_qa.yaml`):

```yaml
model:
  n_queries: 32              # Number of learnable query tokens
  hidden_dim: 768            # Hidden dimension
  num_layers: 6              # Q-Former layers
  k_fragments: 10            # Max retrieved fragments

retriever:
  model_name: "facebook/contriever"
  freeze: true               # Always frozen

llm:
  model_name: "microsoft/phi-2"
  freeze: true               # Always frozen

training:
  lr: 1.0e-4
  batch_size: 8
  epochs: 10
```

## 📊 Attention Mechanism

**Self-Attention (SA)**:
- LQs attend to: `[LQs, query tokens, answer tokens]`
- Bidirectional attention
- Aggregates query/answer information

**Cross-Attention (CA)**:
- LQs attend to: retriever fragment embeddings (P_embeds)
- Bidirectional attention
- Extracts relevant information from retrieved fragments

**Output**: Contextualized LQ representations `Z [N, d]` fed to task heads or LLM

## 🧪 Testing

Run tests to verify shapes and parameter freezing:

```bash
# Test attention masks and forward pass shapes
python tests/test_shapes.py

# Test that only Q-Former + heads are trainable
python tests/test_freezing.py
```

## 🔄 Swappability

DR-QFormer is designed for easy component swapping:

**Retrievers**: Contriever, DPR, E5, ColBERT, etc.
```python
retriever = Retriever(model_name="facebook/contriever")
```

**LLMs**: LLaMA, Mistral, Phi, Flan-T5, etc.
```python
llm = FrozenLLM(model_name="microsoft/phi-2")
```

**Task Heads**: Implement custom heads by subclassing and defining forward pass
```python
class CustomHead(nn.Module):
    def forward(self, z):
        # Your logic here
        return output
```

## 📝 TODO

This is a **barebones scaffold**. Key areas to implement:

- [ ] Q-Former forward pass (attention layers, masks)
- [ ] Task head forward passes (pooling, projection)
- [ ] Retriever integration (corpus loading, vector search)
- [ ] LLM integration (soft prompt prefix injection)
- [ ] Dataset loading and preprocessing
- [ ] Training loops (forward, loss, backward, optimization)
- [ ] Evaluation loops (inference, metric computation)
- [ ] Checkpoint save/load implementation
- [ ] Logging (TensorBoard, Weights & Biases)
- [ ] Distributed training support

See `# TODO` comments throughout the codebase for detailed implementation notes.

## 📚 Citation

If you use DR-QFormer in your research, please cite:

```bibtex
@software{drqformer2025,
  title = {DR-QFormer: Parameter-Efficient Middleware for RAG},
  author = {DR-QFormer Contributors},
  year = {2025},
  url = {https://github.com/gugupig/DR-QFormer},
  version = {0.1.0},
  license = {MIT}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! This is a barebones scaffold designed for extension. Key areas:

1. **Model Implementation**: Complete Q-Former forward pass and attention mechanisms
2. **Data Pipelines**: Add dataset loaders for common RAG benchmarks
3. **Training Infrastructure**: Implement distributed training, mixed precision
4. **Evaluation**: Add more metrics and benchmark support
5. **Documentation**: Expand tutorials and examples

## 🙏 Acknowledgments

- **BLIP-2**: Architecture inspiration for Q-Former design
- **Hugging Face**: Transformers library for model integration
- **PyTorch**: Deep learning framework

## 📞 Contact

For questions or issues, please open a GitHub issue in the repository.