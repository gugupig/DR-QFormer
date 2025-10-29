"""
Task E: Fragment-level Entailment Tagging.

Train Q-Former + EntailmentHead to predict which retrieved fragments
are entailed by/relevant to the query.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dr_qformer.models.qformer import DRQFormer
    from dr_qformer.models.heads import EntailmentHead
    from dr_qformer.adapters.retriever import Retriever
    from dr_qformer.losses import entailment_loss
    from dr_qformer.metrics import entailment_metrics
    from train.common import parse_args, setup_training, Trainer
except ImportError as e:
    print(f"Warning: Could not import DR-QFormer modules: {e}")


def train_entailment(config: dict):
    """
    Train entailment tagging model.
    
    Args:
        config: Training configuration dictionary
    
    Workflow:
    1. Load frozen retriever
    2. Initialize DRQFormer + EntailmentHead (trainable)
    3. Load dataset with entailment labels
    4. Training loop:
       - Get query + k fragments from retriever
       - Forward through Q-Former → EntailmentHead
       - Compute BCE loss vs. ground-truth labels
       - Update only Q-Former + head parameters
    5. Evaluate on dev set
    6. Save checkpoint
    
    TODO:
    - Implement data loading with entailment labels
    - Set up model, optimizer, scheduler
    - Implement training loop
    - Add logging and checkpointing
    - Add early stopping
    """
    print("=" * 80)
    print("Task E: Fragment-level Entailment Tagging")
    print("=" * 80)
    
    # TODO: Initialize components
    # retriever = Retriever(model_name=config["retriever_model"])
    # qformer = DRQFormer(
    #     n_queries=config["n_queries"],
    #     hidden_dim=config["hidden_dim"],
    # )
    # head = EntailmentHead(
    #     hidden_dim=config["hidden_dim"],
    #     num_fragments=config["k_fragments"],
    # )
    
    # TODO: Load dataset
    # from dr_qformer.data.interfaces import load_dataset
    # train_data = load_dataset(config["train_data"], task_type="entailment")
    # dev_data = load_dataset(config["dev_data"], task_type="entailment")
    
    # TODO: Set up optimizer (only trainable params)
    # trainable_params = list(qformer.parameters()) + list(head.parameters())
    # optimizer = torch.optim.AdamW(trainable_params, lr=config["lr"])
    
    # TODO: Training loop
    # for epoch in range(config["epochs"]):
    #     for batch in train_loader:
    #         # Forward pass
    #         z, _ = qformer(batch["query_embeds"], p_embeds=batch["p_embeds"])
    #         logits = head(z)
    #         
    #         # Compute loss
    #         loss = entailment_loss(logits, batch["labels"])
    #         
    #         # Backward pass (only updates trainable params)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    
    print("TODO: Implement training loop")
    print("Entailment training completed!")


def main():
    """Main entry point."""
    config = parse_args()
    setup_training(config)
    train_entailment(config)


if __name__ == "__main__":
    main()
