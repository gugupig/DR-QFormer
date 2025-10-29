"""
Task C: Condensing-Generation with Reward Margin.

Train Q-Former + CondenseHead to produce condensed representations
that improve LLM generation quality (measured by reward).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dr_qformer.models.qformer import DRQFormer
    from dr_qformer.models.heads import CondenseHead
    from dr_qformer.adapters.retriever import Retriever
    from dr_qformer.adapters.llm import FrozenLLM
    from dr_qformer.losses import reward_margin_loss
    from dr_qformer.metrics import rouge_score, bleu_score
    from train.common import parse_args, setup_training, Trainer
except ImportError as e:
    print(f"Warning: Could not import DR-QFormer modules: {e}")


def train_condensing(config: dict):
    """
    Train condensing model with reward-based supervision.
    
    Args:
        config: Training configuration dictionary
    
    Workflow:
    1. Load frozen retriever and frozen LLM
    2. Initialize DRQFormer + CondenseHead (trainable)
    3. Load dataset with queries and answers
    4. Training loop:
       - Get query + k fragments from retriever
       - Forward through Q-Former → CondenseHead → Z
       - Generate with LLM using Z as prefix (condensed)
       - Generate with LLM without Z (baseline)
       - Compute reward (ROUGE/BLEU vs. reference answer)
       - Compute margin loss: reward(condensed) > reward(baseline)
       - Update only Q-Former + head parameters
    5. Evaluate generation quality
    6. Save checkpoint
    
    Reward computation:
    - ROUGE-L or BLEU between generated text and reference answer
    - Higher reward = better generation
    - Goal: condensed representation improves LLM output
    
    TODO:
    - Implement data loading with query-answer pairs
    - Set up retriever, Q-Former, head, LLM
    - Implement two-path generation (condensed vs. baseline)
    - Compute rewards and margin loss
    - Handle reward normalization
    - Add logging and checkpointing
    """
    print("=" * 80)
    print("Task C: Condensing-Generation (Reward Margin)")
    print("=" * 80)
    
    # TODO: Initialize components
    # retriever = Retriever(model_name=config["retriever_model"])
    # qformer = DRQFormer(
    #     n_queries=config["n_queries"],
    #     hidden_dim=config["hidden_dim"],
    # )
    # head = CondenseHead(
    #     hidden_dim=config["hidden_dim"],
    #     llm_hidden_dim=config["llm_hidden_dim"],
    # )
    # llm = FrozenLLM(model_name=config["llm_model"])
    
    # TODO: Load dataset
    # from dr_qformer.data.interfaces import load_dataset
    # train_data = load_dataset(config["train_data"], task_type="condense")
    # dev_data = load_dataset(config["dev_data"], task_type="condense")
    
    # TODO: Set up optimizer (only trainable params)
    # trainable_params = list(qformer.parameters()) + list(head.parameters())
    # optimizer = torch.optim.AdamW(trainable_params, lr=config["lr"])
    
    # TODO: Training loop
    # for epoch in range(config["epochs"]):
    #     for batch in train_loader:
    #         # Forward pass
    #         z, _ = qformer(batch["query_embeds"], p_embeds=batch["p_embeds"])
    #         prefix = head(z)
    #         
    #         # Generate with condensed prefix
    #         output_condensed = llm.generate_with_prefix(
    #             batch["query"], z=prefix
    #         )
    #         
    #         # Generate without prefix (baseline)
    #         output_baseline = llm.generate_with_prefix(batch["query"], z=None)
    #         
    #         # Compute rewards (ROUGE vs. reference)
    #         reward_condensed = compute_reward(output_condensed, batch["answer"])
    #         reward_baseline = compute_reward(output_baseline, batch["answer"])
    #         
    #         # Compute margin loss
    #         loss = reward_margin_loss(reward_condensed, reward_baseline)
    #         
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    
    print("TODO: Implement training loop")
    print("Condensing training completed!")


def compute_reward(generated: str, reference: str) -> float:
    """
    Compute reward for generated text.
    
    Args:
        generated: Generated text
        reference: Reference answer
    
    Returns:
        reward: Scalar reward (higher = better)
    
    TODO:
    - Compute ROUGE-L or BLEU score
    - Normalize reward to [0, 1] range
    - Consider other reward metrics (F1, etc.)
    """
    # Placeholder
    reward = rouge_score([generated], [reference], rouge_type="rougeL")
    return reward


def main():
    """Main entry point."""
    config = parse_args()
    setup_training(config)
    train_condensing(config)


if __name__ == "__main__":
    main()
