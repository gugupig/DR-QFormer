# DR-QFormer

**Parameter-Efficient Middleware for Retrieval-Augmented Generation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DR-QFormer is a BLIP-2-style parameter-efficient architecture that bridges frozen retrievers and frozen large language models (LLMs) for RAG tasks. Only the Q-Former and task-specific heads are trained, achieving efficient adaptation while leveraging powerful pretrained components.

## 💡 Motivation

### Problems with Traditional RAG
- **Pipeline Fragmentation**: Retrieval, (implicit) ranking, and generation are independent stages with no end-to-end optimization, leading to poor inter-module alignment
- **Lack of Unified Optimization**: Retriever doesn't know what generator needs; generator isn't optimized to handle retrieval noise
- **Long Context & LLM Burden**: Feeding multiple complete text fragments (thousands of tokens) forces LLM to handle filtering, sorting, and reasoning - inefficient and suboptimal

### Problems with Existing Differentiable RAG
Existing methods like RAG-DDR and Stochastic RAG have limitations:
- **Poor Parameter Efficiency**: Still require training/finetuning large retrievers or generator LLMs
- **Training Instability**: Introducing stochasticity (e.g., Gumbel-top-k) can cause high gradient variance and training difficulties

### DR-QFormer's Solution
- **Parameter Efficient**: Only train Q-Former (~40-80M) + heads (~1-10M), keep retriever (~100-400M) and LLM (~1-10B) frozen
- **End-to-End Differentiable**: Q-Former serves as the trainable reasoning bridge between frozen components
- **Shortened LLM Input**: Compress k fragments into N condensed vectors, drastically reducing LLM context length
- **Stable Training**: Cross-attention architecture with supervised tasks (E, S, C) provides stable gradients without stochastic sampling

## 🎯 Key Features

- **Parameter Efficiency**: Train only ~1-2% of total parameters (Q-Former + heads ~40-90M), while keeping retriever (~100-400M) and LLM (~1-10B) frozen
- **Frozen Components**: Retriever and LLM remain frozen throughout training - only Q-Former and task heads are trainable
- **Cross-Attention Architecture**: BLIP-2-style design adapted for online, query-relevant RAG with SA (self-attention) and CA (cross-attention) stages
- **Fragment-Level Operations**: All three tasks operate on text fragments/chunks, not full documents
- **Dual Training (Primal & Dual)**: Each task supports both QA (Primal) and QG (Dual) modes with implicit duality constraints through parameter sharing
- **Modular Design**: Swap retrievers, LLMs, and task heads independently

## 📐 Architecture (Cross-Attention Design)

DR-QFormer adopts a parameter-efficient BLIP-2 paradigm adapted for **online, query-relevant RAG**:

### Workflow
```
1. Retrieval:
   Query → [Frozen Retriever] → Evidence Pool (P): k text fragments
                                                  ↓
                                           P_embeds [batch, k, d]

2. Q-Former Reasoning (Cross-Attention):
   ┌─────────────────────────────────────────────────────────┐
   │ Stage 1: Self-Attention (SA)                            │
   │   Input: Concat([LQs, q_embed] or [LQs, a_embed])       │
   │          Length: N + 1 (Primal: N LQs + query)          │
   │                       (Dual: N LQs + answer)            │
   │   Mask: (N+1) x (N+1) all-ones (bidirectional)         │
   │   Output: LQs_aware [batch, N, d] (query/answer-aware)  │
   │                                                          │
   │ Stage 2: Cross-Attention (CA)                           │
   │   Query: LQs_aware [batch, N, d]                        │
   │   Key/Value: P_embeds [batch, k, d]                     │
   │   Mask: N x k all-ones (LQs attend to all fragments)   │
   │   Output: Z [batch, N, d] (knowledge-infused)           │
   │                                                          │
   │ Stage 3: Feed-Forward Network (FFN)                     │
   │   Output: Z_final [batch, N, d]                         │
   └─────────────────────────────────────────────────────────┘

3. Task Heads (Fragment-Level):
   Z_final → ┌───────────────┼──────────────┐
             ↓               ↓              ↓
     EntailmentHead    SortingHead    CondenseHead
     (Fragment Tags)   (Attn Weights) (LLM Prefix)
             ↓               ↓              ↓
       [k] logits      [k] scores     [N, d_llm]
             ↓               ↓              ↓
      BCE vs gt_k    KL vs gt_soft   [Frozen LLM]
                                           ↓
                                      Generated Text
```

### Key Design Principles
- **Online**: Q-Former receives query embedding at inference time (not precomputed)
- **Query-Relevant**: SA stage fuses LQs with query/answer context before CA
- **Fragment-Level**: All tasks operate on k text fragments, not full documents
- **Frozen Pipeline**: Only Q-Former (~40-80M params) + heads (~1-10M params) are trainable

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

## 🎓 Training Tasks (Primal & Dual)

All three tasks jointly train the Q-Former with both **Primal (QA)** and **Dual (QG)** modes, operating on **text fragments (k chunks)**:

### Task E: Fragment-Level Entailment Tagging (蕴含-标注)
**Purpose**: Learn "answerability/entailment" to act as a **fragment-level filter/tagger**.

**Architecture**:
- Input: Z from Q-Former [batch, N, d]
- Output: k logits [batch, k] - one per fragment, derived from CA layer attention scores (e.g., max-pooled)
- Supervision: `gt_k` [batch, k] - binary vector marking which fragments are golden evidence (offline generated)

**Training Modes**:
- **Primal (QA)**: Q-Former receives query embedding → predicts which fragments entail the answer
- **Dual (QG)**: Q-Former receives answer embedding → predicts which fragments entail the query

**Loss**: Binary Cross-Entropy (BCE) vs `gt_k`  
**Metrics**: Accuracy, Precision, Recall, F1

---

### Task S: Fragment-Level Sorting Supervision (排序-监督)
**Purpose**: Learn **fragment-level ranking behavior** by training CA layer attention weight allocation.

**Architecture**:
- Input: Z from Q-Former [batch, N, d]
- Output: k attention weights [batch, k] - CA layer softmax output (A_weights) showing relative importance
- Supervision: `gt_soft_weights` [batch, k] - target probability distribution reflecting fragment relative importance (generated from offline retriever/reranker scores)

**Training Modes**:
- **Primal (QA)**: Q-Former receives query embedding → learns to rank fragments by answer relevance
- **Dual (QG)**: Q-Former receives answer embedding → learns to rank fragments by query relevance

**Loss**: KL Divergence vs `gt_soft_weights` (distribution matching)  
**Metrics**: NDCG, MAP, Kendall's Tau, Spearman's ρ

---

### Task C: Condensing-Generation (精炼-生成)
**Purpose**: Learn to "condense and refine" - ensure Q-Former's extracted Z is useful for frozen LLM generation.

**Architecture**:
- Input: Z from Q-Former [batch, N, d]
- Output: N condensed vectors [batch, N, d_llm] fed as soft prompt prefix to LLM

**Training Modes**:
- **Primal (QA)**: 
  - **Contrastive Generation Loss**: Maximize `Reward(LLM(Query, Z))` vs `Reward(LLM(Query, Empty_Z))`
  - Forces evidence dependency, rewards improve ROUGE/BLEU/EM with gold answer
- **Dual (QG)**: 
  - **Reward Loss**: Maximize `Similarity(LLM(Answer, Z) → Query', Gold_Query)`
  - Similarity based on BLEU/ROUGE between generated and gold queries

**Loss**: Reward Margin Loss (contrastive for Primal, similarity for Dual)  
**Metrics**: ROUGE, BLEU, Exact Match, F1

## � Implicit Dual Constraint (隐式对偶约束)

### Inspiration from Tang et al. (2017)
DR-QFormer borrows the core idea of leveraging QA-QG duality ($P(a|q)$ and $P(q|a)$) from Tang et al.'s work on dual learning for question answering and generation.

### Our Implementation: Implicit Parameter Sharing
Unlike Tang et al.'s **explicit probabilistic consistency loss** ($L_{dual}$), DR-QFormer implements duality constraints **implicitly through parameter sharing**:

- **Shared Q-Former**: The same Q-Former parameters (especially attention layers and matching logic) are updated by gradients from **both** Primal (QA) and Dual (QG) tasks
- **Joint Training**: For each task (E, S, C), we train with both:
  - **E_qa, S_qa, C_qa** (Primal): Query → Predict answer-relevant fragments/generation
  - **E_qg, S_qg, C_qg** (Dual): Answer → Predict query-relevant fragments/generation

### Effect
This bidirectional training forces Q-Former to learn a more **robust, generalizable bidirectional logic** for `Query ↔ Evidence Pool ↔ Answer` relationships, rather than overfitting to single-direction tasks.

### Optional Explicit Constraint
Consider adding **Attention Consistency Loss** ($L_{AC}$) as an explicit (but non-probabilistic) constraint between Primal and Dual attention distributions:

```python
# Pseudo-code
L_AC = KL(Attn_primal || Attn_dual) + KL(Attn_dual || Attn_primal)
```

This encourages similar attention patterns for symmetric (query, answer) pairs.

---

## �🔧 Configuration

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

## 📊 Attention Mechanism & Masking

### Self-Attention (SA) Stage
**Purpose**: Make LQs "query/answer-aware" by fusing them with query or answer embeddings.

- **Input Sequence**: `Concat([LQs, q_embed])` or `Concat([LQs, a_embed])` - length N+1
- **Mask (`M_SA`)**: **(N+1) x (N+1) all-ones** (or no mask) - fully bidirectional
- **Behavior**: LQs and query/answer tokens attend to each other freely
- **Output**: `LQs_aware [batch, N, d]` - query/answer-contextualized LQs

### Cross-Attention (CA) Stage  
**Purpose**: Extract relevant information from retrieved fragments into LQs_aware.

- **Query Sequence**: `LQs_aware [batch, N, d]`
- **Key/Value Sequence**: `P_embeds [batch, k, d]` - k fragment embeddings
- **Mask (`M_CA`)**: **N x k all-ones** (or no mask) - each LQ can attend to all k fragments
- **Behavior**: Each LQs_aware attends to all fragments, weights learned via task losses
- **Output**: `Z [batch, N, d]` - knowledge-infused representations

### Padding Mask
**Required**: Handles variable-length `P_embeds` or batch padding. Ensures model ignores invalid padding tokens (typically handled automatically by Transformer library).

### Masking Philosophy
- **Permissive Masks**: All connections allowed by default (all-ones matrices)
- **Learning via Loss**: Tasks E, S, C train attention mechanism to learn effective filtering and ranking
- **No Hard Constraints**: Unlike causal LM, no triangle masks - Q-Former is bidirectional

**Final Output**: Contextualized LQ representations `Z [batch, N, d]` fed to task heads or frozen LLM

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

- [x] **Q-Former Core Architecture** (`dr_qformer/models/qformer.py`) ✅
  - [x] DRQFormer main class with learnable query tokens (LQs)
  - [x] QFormerLayer with SA + CA + FFN stages
  - [x] Primal (QA) and Dual (QG) mode support
  - [x] Attention masking (SA and CA)
  - [x] Auxiliary outputs for task heads
  - [x] ~57M parameters (medium config)
- [ ] Task head forward passes (pooling, projection)
- [ ] Retriever integration (corpus loading, vector search)
- [ ] LLM integration (soft prompt prefix injection)
- [ ] Dataset loading and preprocessing
- [ ] Training loops (forward, loss, backward, optimization)
- [ ] Evaluation loops (inference, metric computation)
- [ ] Checkpoint save/load implementation
- [ ] Logging (TensorBoard, Weights & Biases)
- [ ] Distributed training support

### 🎉 Recent Updates

**2025-11-01: Core Q-Former Implementation Complete** ✅
- Fully implemented DR-QFormer cross-attention architecture
- Tested with Primal (QA) and Dual (QG) modes
- All tests passing (see `simple_test_qformer.py`)
- Comprehensive documentation and visualization tools added
- See `IMPLEMENTATION_SUMMARY.md` for details

**Next Steps**: Implement task heads (E, S, C) and integrate frozen models.

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