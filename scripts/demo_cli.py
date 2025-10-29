"""
Demo CLI for DR-QFormer.

Simple demonstration of model parameter counts and query echo.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dr_qformer.models.qformer import DRQFormer
    from dr_qformer.models.heads import EntailmentHead, SortingHead, CondenseHead
    from dr_qformer.adapters.retriever import Retriever
    from dr_qformer.adapters.llm import FrozenLLM
except ImportError as e:
    print(f"Warning: Could not import DR-QFormer modules: {e}")


def count_parameters(model) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        num_params: Number of trainable parameters
    """
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception:
        return 0


def demo():
    """Run simple demo."""
    print("=" * 80)
    print("DR-QFormer Demo")
    print("=" * 80)
    print()
    
    # Initialize models (with dummy components for demo)
    print("Initializing models...")
    print()
    
    try:
        # Q-Former
        qformer = DRQFormer(
            n_queries=32,
            hidden_dim=768,
            num_layers=6,
        )
        qformer_params = count_parameters(qformer)
        print(f"✓ DRQFormer: {qformer_params:,} trainable parameters")
        
        # Heads
        entailment_head = EntailmentHead(hidden_dim=768, num_fragments=10)
        sorting_head = SortingHead(hidden_dim=768, num_fragments=10)
        condense_head = CondenseHead(hidden_dim=768)
        
        entail_params = count_parameters(entailment_head)
        sort_params = count_parameters(sorting_head)
        condense_params = count_parameters(condense_head)
        
        print(f"✓ EntailmentHead: {entail_params:,} trainable parameters")
        print(f"✓ SortingHead: {sort_params:,} trainable parameters")
        print(f"✓ CondenseHead: {condense_params:,} trainable parameters")
        print()
        
        total_trainable = qformer_params + entail_params + sort_params + condense_params
        print(f"Total trainable (Q-Former + heads): {total_trainable:,} parameters")
        print()
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        print("(This is expected if PyTorch is not installed)")
        print()
    
    # Initialize adapters
    print("Adapter components (frozen):")
    try:
        retriever = Retriever(model_name="facebook/contriever")
        print(f"✓ {retriever}")
        
        llm = FrozenLLM(model_name="microsoft/phi-2")
        print(f"✓ {llm}")
        print()
        
    except Exception as e:
        print(f"Note: Adapters require models to be downloaded")
        print()
    
    # Echo query demo
    print("Query echo demo:")
    print("-" * 80)
    
    queries = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "How does photosynthesis work?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"{i}. Query: {query}")
        print(f"   → [TODO: Forward through Q-Former → Generate with LLM]")
        print()
    
    print("=" * 80)
    print("Demo completed!")
    print()
    print("To train models, use:")
    print("  python -m train.task_e --cfg configs/drqf_qa.yaml")
    print("  python -m train.task_s --cfg configs/drqf_qa.yaml")
    print("  python -m train.task_c --cfg configs/drqf_qa.yaml")
    print()
    print("To evaluate:")
    print("  python -m eval.evaluate --checkpoint path/to.ckpt")


if __name__ == "__main__":
    demo()
