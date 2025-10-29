# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
