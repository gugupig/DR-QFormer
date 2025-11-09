# Changelog

All notable changes to DR-QFormer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-01-23

### ✨ Added - Task E Implementation

#### Task-Specific Heads
- **EntailmentHead** (`dr_qformer/models/heads.py`)
  - Fragment-level binary entailment classification
  - LayerNorm for attention score normalization
  - Drop-LQ regularization with safety protection (at least 1 LQ survives)
  - LogSumExp aggregation with temperature τ (default: 0.5, range: 0.1-1.0)
  - Focal loss with importance weighting (w_pos=10.0, w_longtail=50.0)
  - Support for `pool_padding_mask` (variable-length fragment pools)
  - ~4K trainable parameters (mostly LayerNorm)

#### Training Scripts
- **`train/task_e.py`** - Task E training pipeline
  - Dual training mode: Primal (QA) + Dual (QG) with shared parameters
  - Frozen retriever integration (Contriever/DPR/E5/BGE)
  - Focal loss with importance weighting
  - Gradient clipping and cosine LR scheduling
  - Checkpointing and evaluation
  - Configurable hyperparameters (tau, p_drop_lq, focal_gamma/alpha, w_pos, w_longtail)

#### Documentation
- **`TASK_E_IMPLEMENTATION.md`** - Complete implementation guide
  - Component descriptions and usage
  - Data format specification
  - Integration with Q-Former
  - Testing strategy
  - Next steps and roadmap

#### Bug Fixes
- Fixed indexing bug in EntailmentHead Drop-LQ safety protection
  - Issue: `random_idx` from `torch.randint().item()` needed explicit `int()` cast
  - Fixed in `_apply_drop_lq()` method

### 🔄 Modified
- **`dr_qformer/models/heads.py`**
  - Replaced TODO placeholder with full EntailmentHead implementation
  - Added comprehensive docstrings
  - Added parameter counting method

## [0.2.1] - 2025-11-02

### ✨ Added - Attention Weight Export

#### Core Changes
- **Attention Weight Export** (`dr_qformer/models/qformer.py`)
  - Modified `QFormerLayer.forward()` to return attention weights
  - Set `need_weights=True, average_attn_weights=False` in MultiheadAttention
  - SA weights: `[batch, num_heads, N+1, N+1]` per layer
  - CA weights: `[batch, num_heads, N, k]` per layer
  - Per-head weights preserved (not averaged) for fine-grained analysis

#### Analysis Tools
- **`test_attention_weights.py`** - Validation tests
  - Tests SA/CA weight shapes across configurations
  - Verifies attention distributions sum to 1
  - Tests both Primal (QA) and Dual (QG) modes
  
- **`analyze_attention.py`** - Comprehensive analysis
  - LQ-to-query/answer embedding attention
  - Fragment attention statistics
  - Attention selectivity and diversity (entropy)
  - Per-head attention patterns
  - LQ-to-fragment mappings ("哪几个LQ关注了哪几段")
  - Primal vs Dual mode comparison
  - Export weights to `.npz` files

#### Analysis Capabilities
- ✅ Which LQs attend to which fragments
- ✅ Attention flow through layers
- ✅ Attention collapse detection
- ✅ Cross-attention head specialization
- ✅ Query/answer embedding influence

### 🔧 Changed
- **QFormerLayer return signature**: Now returns `(output, layer_aux)` tuple
- **DRQFormer forward pass**: Collects per-layer SA/CA weights in `aux` dict

## [0.2.0] - 2025-11-01

### ✨ Added - Core Q-Former Implementation

#### Architecture
- **DRQFormer main class** (`dr_qformer/models/qformer.py`)
  - Learnable query tokens (LQs) - 32 parameterized vectors
  - 6-layer Transformer stack with SA + CA + FFN
  - Final LayerNorm layer
  - Temperature parameter for Task E
  - ~57M trainable parameters (medium config)

- **QFormerLayer implementation**
  - Stage 1: Self-Attention (SA) - LQs fuse with query/answer embeddings
  - Stage 2: Cross-Attention (CA) - LQs attend to fragment embeddings
  - Stage 3: Feed-Forward Network (FFN) - Non-linear transformation
  - Pre-LayerNorm + residual connections for all sublayers

#### Training Modes
- Primal Mode (QA): Query → Answer prediction
- Dual Mode (QG): Answer → Query generation
- Implicit dual constraint via parameter sharing

#### Features
- Attention masking support (SA and CA)
- Padding mask handling for variable-length fragments
- Auxiliary outputs: layer outputs, attention weights, raw sequences
- Gradient flow verification

#### Testing & Documentation
- `simple_test_qformer.py` - Functional tests (all passing ✅)
- `visualize_drqformer.py` - Architecture visualization
- `DR_QFORMER_IMPLEMENTATION.md` - Detailed implementation guide
- `IMPLEMENTATION_SUMMARY.md` - Complete summary

### 🎯 Specifications
- **Model Size**: 56,736,769 parameters
- **Memory**: ~216 MB (FP32), ~108 MB (FP16)
- **Efficiency**: ~1-2% of total system parameters
- **Architecture**: Cross-attention (BLIP-2-inspired, adapted for RAG)

### 📝 Next Steps
- [ ] Implement task heads (EntailmentHead, SortingHead, CondenseHead)
- [ ] Integrate frozen retriever (Contriever, DPR, E5, BGE)
- [ ] Integrate frozen LLM (LLaMA, Mistral, Phi)
- [ ] Implement training tasks (E, S, C) with dual mode
- [ ] Data preparation utilities
- [ ] Training and evaluation loops

## [Unreleased]

### To Be Implemented
- Q-Former forward pass implementation
- Task head forward passes
- Training loop implementations
- Data loading and preprocessing
- Retriever corpus integration
- LLM soft prompt prefix injection
- Checkpoint save/load functionality
- Evaluation metrics computation
- Distributed training support

## [0.1.0] - 2025-10-29

### Added
- Initial barebones scaffold
- Project structure with all directories and files
- Core model interfaces:
  - `DRQFormer` class with docstrings
  - `EntailmentHead`, `SortingHead`, `CondenseHead` interfaces
  - `Retriever` adapter stub
  - `FrozenLLM` adapter stub
- Training scripts:
  - Task E (Entailment Tagging)
  - Task S (Fragment Sorting)
  - Task C (Condensing-Generation)
  - Common utilities and Trainer stub
- Loss functions (stubs):
  - Binary cross-entropy for entailment
  - Ranking loss for sorting
  - Reward margin loss for generation
- Metrics (stubs):
  - EM, F1, ROUGE, BLEU
  - NDCG, MAP for ranking
  - Classification metrics
- Data interfaces:
  - `Fragment` and `Example` dataclasses
  - `DRQFormerDataset` base class
- Utilities:
  - Attention mask builders (stubs)
  - Checkpoint save/load (stubs)
- Configuration files:
  - `drqf_qa.yaml` for QA tasks
  - `drqf_qg.yaml` for generation tasks
- Testing:
  - Shape tests
  - Parameter freezing tests
- Documentation:
  - Comprehensive README
  - Quick start guide
  - Data format specification
  - Contributing guidelines
  - Architecture diagram
  - Citation metadata
- Project setup:
  - `setup.py` for package installation
  - `requirements.txt` with dependencies
  - `.gitignore` configuration
  - MIT License
- Demo script (`demo_cli.py`)

### Architecture
- BLIP-2-style Q-Former with learnable query tokens
- Self-attention over [LQs, query/answer text]
- Cross-attention over retriever embeddings
- Frozen retriever and LLM components
- Parameter-efficient design (~1-2% trainable params)

### Notes
- All implementations are stubs with comprehensive TODOs
- Type hints and docstrings throughout
- Modular design for easy component swapping
- Ready for community contributions

---

## Release Notes Template

### [Version] - YYYY-MM-DD

#### Added
- New features

#### Changed
- Changes to existing functionality

#### Deprecated
- Soon-to-be removed features

#### Removed
- Removed features

#### Fixed
- Bug fixes

#### Security
- Security updates
