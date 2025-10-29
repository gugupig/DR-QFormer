# Contributing to DR-QFormer

Thank you for your interest in contributing to DR-QFormer! This is a barebones scaffold designed for community extension and improvement.

## 🎯 High-Priority Areas

### 1. Model Implementation
- **Q-Former Core**: Complete the transformer layers with self-attention and cross-attention
- **Attention Masks**: Implement mask generation functions in `dr_qformer/utils/masks.py`
- **Head Forward Passes**: Implement pooling and projection logic in task heads

### 2. Data Pipelines
- **Dataset Loaders**: Support common RAG benchmarks (NQ, MS MARCO, HotpotQA, etc.)
- **Collation Functions**: Batch processing with proper padding and masking
- **Data Preprocessing**: Tokenization, embedding preparation

### 3. Training Infrastructure
- **Training Loops**: Complete implementation in `train/task_*.py`
- **Optimization**: AdamW, learning rate scheduling, gradient clipping
- **Distributed Training**: Multi-GPU support with DDP
- **Mixed Precision**: FP16/BF16 training support

### 4. Adapter Integration
- **Retriever Adapter**: Complete corpus loading, vector search (FAISS)
- **LLM Adapter**: Implement soft prompt prefix injection
- **Model Loading**: Handle different model formats and quantization

### 5. Evaluation & Metrics
- **Metrics Implementation**: Complete metric functions in `dr_qformer/metrics.py`
- **Benchmark Evaluation**: Add evaluation on standard RAG benchmarks
- **Analysis Tools**: Error analysis, visualization utilities

## 🚀 Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/DR-QFormer.git
   cd DR-QFormer
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Make your changes**
   - Follow existing code structure
   - Add docstrings and type hints
   - Include TODOs for incomplete parts

5. **Test your changes**
   ```bash
   python tests/test_shapes.py
   python tests/test_freezing.py
   ```

6. **Submit a pull request**

## 📝 Code Guidelines

### Style
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Include docstrings for classes and functions
- Keep line length ≤ 100 characters

### Structure
- Maintain modular design (swap-ability is key)
- Keep frozen components (retriever, LLM) separate from trainable components
- Use try/except for optional dependencies

### Documentation
- Update README.md if adding new features
- Add inline comments for complex logic
- Include usage examples in docstrings

### Example Docstring Format
```python
def function_name(arg1: type1, arg2: type2) -> return_type:
    """
    Brief description of function.
    
    Longer description if needed, explaining the purpose,
    algorithm, or usage patterns.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ExceptionType: When this exception is raised
    
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
    
    TODO:
        - Future improvements
        - Known issues to fix
    """
    pass
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_shapes.py
```

### Adding Tests
- Add test files to `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names
- Include both positive and negative test cases

## 📦 Pull Request Process

1. **Update documentation** if you're changing functionality
2. **Add tests** for new features
3. **Ensure all tests pass**
4. **Update CHANGELOG** (if exists) with your changes
5. **Request review** from maintainers

### PR Title Format
```
[Type] Brief description

Types: Feature, Bugfix, Docs, Refactor, Test, Chore
```

Examples:
- `[Feature] Add FAISS-based retrieval support`
- `[Bugfix] Fix attention mask broadcasting issue`
- `[Docs] Update training tutorial with examples`

## 🐛 Reporting Issues

### Bug Reports
Include:
- Python version
- PyTorch version
- Operating system
- Full error traceback
- Minimal reproducible example

### Feature Requests
Include:
- Use case description
- Proposed API (if applicable)
- Why this feature is important
- Potential implementation approach

## 💡 Ideas for Contributions

### Easy (Good First Issues)
- Add more unit tests
- Improve documentation and examples
- Fix TODO items in existing code
- Add type hints to untyped functions

### Medium
- Implement dataset loaders for specific benchmarks
- Add new evaluation metrics
- Implement checkpointing and resumption
- Add logging integrations (TensorBoard, W&B)

### Hard
- Complete Q-Former forward pass implementation
- Implement distributed training support
- Add quantization support for frozen models
- Optimize memory usage for large retrievers

## 📞 Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Pull Requests**: Code contributions

## 🙏 Attribution

Contributors will be acknowledged in:
- README.md contributors section
- CITATION.cff authors list
- Release notes

Thank you for contributing to DR-QFormer!
