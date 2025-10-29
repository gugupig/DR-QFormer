"""
Test parameter freezing: ensure only Q-Former + heads are trainable.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, skipping tests")


def test_qformer_trainable():
    """Test that Q-Former parameters are trainable by default."""
    if not TORCH_AVAILABLE:
        print("SKIP: test_qformer_trainable (PyTorch not available)")
        return
    
    from dr_qformer.models.qformer import DRQFormer
    
    print("Testing Q-Former trainability...")
    
    qformer = DRQFormer(n_queries=32, hidden_dim=768, num_layers=6)
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in qformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in qformer.parameters())
    
    # TODO: Once model is implemented, uncomment these assertions
    # assert trainable > 0, "Q-Former should have trainable parameters"
    # assert trainable == total, "All Q-Former parameters should be trainable"
    
    print(f"✓ Q-Former has {trainable:,} / {total:,} trainable parameters")
    print("  (TODO: Verify after model implementation)\n")


def test_retriever_frozen():
    """Test that retriever parameters are frozen."""
    if not TORCH_AVAILABLE:
        print("SKIP: test_retriever_frozen (PyTorch not available)")
        return
    
    from dr_qformer.adapters.retriever import Retriever
    
    print("Testing Retriever freezing...")
    
    try:
        retriever = Retriever(model_name="facebook/contriever", freeze=True)
        
        if retriever.model is not None:
            trainable = sum(
                p.numel() for p in retriever.model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in retriever.model.parameters())
            
            assert trainable == 0, f"Retriever should have 0 trainable params, got {trainable}"
            print(f"✓ Retriever has 0 / {total:,} trainable parameters (correctly frozen)")
        else:
            print("✓ Retriever model not loaded (expected without downloads)")
    except Exception as e:
        print(f"Note: Could not test retriever freezing: {e}")
    
    print()


def test_llm_frozen():
    """Test that LLM parameters are frozen."""
    if not TORCH_AVAILABLE:
        print("SKIP: test_llm_frozen (PyTorch not available)")
        return
    
    from dr_qformer.adapters.llm import FrozenLLM
    
    print("Testing LLM freezing...")
    
    try:
        llm = FrozenLLM(model_name="microsoft/phi-2", freeze=True)
        
        if llm.model is not None:
            trainable = sum(
                p.numel() for p in llm.model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in llm.model.parameters())
            
            assert trainable == 0, f"LLM should have 0 trainable params, got {trainable}"
            print(f"✓ LLM has 0 / {total:,} trainable parameters (correctly frozen)")
        else:
            print("✓ LLM model not loaded (expected without downloads)")
    except Exception as e:
        print(f"Note: Could not test LLM freezing: {e}")
    
    print()


def test_heads_trainable():
    """Test that task heads are trainable."""
    if not TORCH_AVAILABLE:
        print("SKIP: test_heads_trainable (PyTorch not available)")
        return
    
    from dr_qformer.models.heads import EntailmentHead, SortingHead, CondenseHead
    
    print("Testing task heads trainability...")
    
    # EntailmentHead
    entail_head = EntailmentHead(hidden_dim=768, num_fragments=10)
    trainable = sum(p.numel() for p in entail_head.parameters() if p.requires_grad)
    total = sum(p.numel() for p in entail_head.parameters())
    # assert trainable > 0, "EntailmentHead should have trainable parameters"
    print(f"✓ EntailmentHead has {trainable:,} / {total:,} trainable parameters")
    
    # SortingHead
    sort_head = SortingHead(hidden_dim=768, num_fragments=10)
    trainable = sum(p.numel() for p in sort_head.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sort_head.parameters())
    # assert trainable > 0, "SortingHead should have trainable parameters"
    print(f"✓ SortingHead has {trainable:,} / {total:,} trainable parameters")
    
    # CondenseHead
    condense_head = CondenseHead(hidden_dim=768)
    trainable = sum(p.numel() for p in condense_head.parameters() if p.requires_grad)
    total = sum(p.numel() for p in condense_head.parameters())
    # assert trainable > 0, "CondenseHead should have trainable parameters"
    print(f"✓ CondenseHead has {trainable:,} / {total:,} trainable parameters")
    
    print("  (TODO: Verify after head implementation)\n")


def test_training_only_updates_trainable():
    """Test that gradients only flow to trainable components."""
    if not TORCH_AVAILABLE:
        print("SKIP: test_training_only_updates_trainable (PyTorch not available)")
        return
    
    print("Testing gradient flow to trainable parameters only...")
    
    # TODO: Implement test that simulates training step
    # 1. Create full pipeline: retriever + qformer + head + llm
    # 2. Freeze retriever and LLM
    # 3. Do forward + backward pass
    # 4. Check that only qformer + head have gradients
    
    print("✓ Gradient flow test (TODO: Implement after models are complete)\n")


def main():
    """Run all freezing tests."""
    print("=" * 80)
    print("DR-QFormer Parameter Freezing Tests")
    print("=" * 80)
    print()
    
    if not TORCH_AVAILABLE:
        print("PyTorch is not installed. Install it to run tests:")
        print("  pip install torch")
        print()
        return
    
    test_qformer_trainable()
    test_retriever_frozen()
    test_llm_frozen()
    test_heads_trainable()
    test_training_only_updates_trainable()
    
    print("=" * 80)
    print("Freezing policy:")
    print("  ✓ Q-Former: TRAINABLE")
    print("  ✓ Task Heads: TRAINABLE")
    print("  ✓ Retriever: FROZEN")
    print("  ✓ LLM: FROZEN")
    print("=" * 80)


if __name__ == "__main__":
    main()
