"""
Task C: Condensing-Generation (Posterior Distribution Extraction).

Core Training Objective:
- Produce knowledge prefix Z that reduces frozen LLM perplexity (NLL)
- Extract **posterior distribution** q_ψ(p|q,a) from LLM attention during generation
- Contrastive NLL: Compare LLM(with Z) vs. LLM(without Z)
- Feed posterior back to Task S for Bayesian-inspired closed loop

Implementation:
- Dual-path teacher forcing: WITH evidence vs. WITHOUT evidence (contrastive learning)
- Adaptive margin based on NLL gain statistics
- Posterior extraction from LLM cross-attention weights
- Only Q-Former + CondenseHead trainable (LLM frozen)

Note: "Dual-path" here refers to WITH/WITHOUT evidence comparison, 
NOT Primal/Dual (QA/QG) training modes.

Design: v8.0 - Pure Teacher Forcing, Contrastive NLL, Subset Posterior
"""

import sys
from pathlib import Path
from typing import Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import torch.nn.functional as F
    from torch import Tensor
except ImportError:
    print("Warning: PyTorch not available")
    Tensor = None

try:
    from dr_qformer.models.qformer import DRQFormer
    from dr_qformer.models.heads import CondenseHead
    from dr_qformer.adapters.retriever import Retriever
    from dr_qformer.adapters.llm import FrozenLLM
    from dr_qformer.losses import compute_condensing_loss
    from train.common import setup_training
except ImportError as e:
    print(f"Warning: Could not import DR-QFormer modules: {e}")


class KnowledgeCondenser(nn.Module):
    """
    Knowledge Condenser for Task C.
    
    Architecture:
    - Q-Former: Extract knowledge from fragments
    - CondenseHead: Project to LLM embedding dimension
    - FrozenLLM: Compute contrastive NLL (frozen, eval mode)
    
    Training:
    - Dual-path teacher forcing (with/without evidence prefix Z)
    - Contrastive NLL loss with adaptive margin
    - Posterior extraction from LLM attention
    - Only Q-Former + CondenseHead are trainable
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Q-Former (trainable)
        self.qformer = DRQFormer(
            n_queries=args.n_queries,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
        
        # CondenseHead (trainable)
        self.condense_head = CondenseHead(
            hidden_dim=args.hidden_dim,
            llm_hidden_dim=args.llm_hidden_dim,
        )
        
        # Frozen LLM (eval mode, no gradients)
        print(f"[INFO] Initializing FrozenLLM: {args.llm_model_name}")
        print("       LLM remains frozen throughout training (placeholder)")
        self.frozen_llm = FrozenLLM(
            model_name=args.llm_model_name,
            device=args.device,
            freeze=True,
        )
        
        # Ensure LLM is in eval mode
        if self.frozen_llm.model is not None:
            self.frozen_llm.model.eval()
    
    def forward(
        self,
        q_embeds: Tensor,
        p_embeds: Tensor,
        query_input_ids: Tensor,
        answer_input_ids: Tensor,
        subset_indices: Optional[Tensor] = None,
        pool_padding_mask: Optional[Tensor] = None,
    ) -> dict:
        """
        Forward pass for Task C.
        
        Args:
            q_embeds: Query embeddings [batch, dim]
            p_embeds: Fragment embeddings [batch, K, dim]
            query_input_ids: Query token IDs [batch, S_q]
            answer_input_ids: Answer token IDs [batch, S_a]
            subset_indices: Training subset indices [batch, |U|]
            pool_padding_mask: Valid fragment mask [batch, K]
        
        Returns:
            dict with keys:
                - 'loss_c': Condensing loss (scalar)
                - 'nll_gain': NLL reduction (detached, for logging)
                - 'margin': Computed margin (detached, for logging)
                - 'posterior_q_psi_U': Fragment posterior [batch, |U|] (detached)
        """
        # 1. Q-Former forward pass
        z, ca_outputs = self.qformer(
            q_embeds=q_embeds,
            p_embeds=p_embeds,
            pool_padding_mask=pool_padding_mask,
        )
        # z: [batch, N_lq, hidden_dim]
        # ca_outputs['ca_scores']: [batch, n_heads, N_lq, K]
        
        # 2. CondenseHead: Project to LLM dimension
        z_prefix = self.condense_head(z)  # [batch, N_lq, d_llm]
        
        # 3. Extract CA weights for posterior computation
        ca_weights = ca_outputs.get('ca_weights', None)
        # ca_weights: [batch, N_lq, K] (post-softmax attention)
        
        # 4. Dual-path teacher forcing with frozen LLM
        # TODO: When LLM is integrated, this will call actual LLM forward
        llm_outputs = self.frozen_llm.teacher_forcing_dual_path(
            z_prefix=z_prefix,
            query_input_ids=query_input_ids,
            answer_input_ids=answer_input_ids,
            capture_attention=True,
        )
        
        # 5. Compute condensing loss
        loss_dict = compute_condensing_loss(
            nll_with_evidence=llm_outputs['nll_with_evidence'],
            nll_without_evidence=llm_outputs['nll_without_evidence'],
            llm_attention_weights=llm_outputs['llm_attention_to_z'],
            ca_weights=ca_weights,
            subset_indices=subset_indices,
            answer_start_idx=llm_outputs['answer_start_idx'],
            softplus_beta=self.args.softplus_beta,
            margin_mode=self.args.margin_mode,
            margin_fixed=self.args.margin_fixed,
            margin_adaptive_ratio=self.args.margin_adaptive_ratio,
            margin_min=self.args.margin_min,
            margin_max=self.args.margin_max,
        )
        
        return loss_dict
    
    def count_trainable_parameters(self) -> dict:
        """Count trainable parameters."""
        qformer_params = sum(p.numel() for p in self.qformer.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.condense_head.parameters() if p.requires_grad)
        
        return {
            'qformer': qformer_params,
            'condense_head': head_params,
            'total': qformer_params + head_params,
        }


class DummyCondenseDataset(Dataset):
    """
    Dummy dataset for testing Task C pipeline.
    
    TODO: Replace with actual QA dataset loader
    - Load queries, fragments, answers
    - Support dynamic subset selection
    - Handle variable-length inputs
    """
    
    def __init__(self, num_samples=100, k_fragments=50, max_seq_len=128):
        self.num_samples = num_samples
        self.k_fragments = k_fragments
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Dummy data
        return {
            'q_embeds': torch.randn(768),  # Query embedding
            'p_embeds': torch.randn(self.k_fragments, 768),  # Fragment embeddings
            'query_input_ids': torch.randint(0, 32000, (self.max_seq_len,)),
            'answer_input_ids': torch.randint(0, 32000, (self.max_seq_len,)),
            'subset_indices': torch.randint(0, self.k_fragments, (10,)),  # |U| = 10
            'pool_padding_mask': torch.ones(self.k_fragments, dtype=torch.bool),
        }


def collate_fn(batch_list):
    """Collate function for Task C batch."""
    return {
        'q_embeds': torch.stack([item['q_embeds'] for item in batch_list]),
        'p_embeds': torch.stack([item['p_embeds'] for item in batch_list]),
        'query_input_ids': torch.stack([item['query_input_ids'] for item in batch_list]),
        'answer_input_ids': torch.stack([item['answer_input_ids'] for item in batch_list]),
        'subset_indices': torch.stack([item['subset_indices'] for item in batch_list]),
        'pool_padding_mask': torch.stack([item['pool_padding_mask'] for item in batch_list]),
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Task C: Condensing-Generation")
    
    # Model architecture
    parser.add_argument('--n_queries', type=int, default=32, help='Number of learnable queries')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Q-Former hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Q-Former layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Q-Former attention heads')
    parser.add_argument('--llm_hidden_dim', type=int, default=4096, help='LLM hidden dimension')
    parser.add_argument('--llm_model_name', type=str, default='microsoft/phi-2', help='LLM model name')
    
    # Loss hyperparameters
    parser.add_argument('--softplus_beta', type=float, default=10.0, help='Softplus sharpness')
    parser.add_argument('--margin_mode', type=str, default='adaptive', choices=['fixed', 'adaptive'])
    parser.add_argument('--margin_fixed', type=float, default=0.5, help='Fixed margin value')
    parser.add_argument('--margin_adaptive_ratio', type=float, default=0.5, help='Adaptive margin ratio κ')
    parser.add_argument('--margin_min', type=float, default=0.1, help='Min adaptive margin')
    parser.add_argument('--margin_max', type=float, default=2.0, help='Max adaptive margin')
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Dataset
    parser.add_argument('--num_samples', type=int, default=100, help='Dummy dataset size')
    parser.add_argument('--k_fragments', type=int, default=50, help='Number of fragments')
    
    args = parser.parse_args()
    return args


def train_condensing(args):
    """
    Train condensing model with contrastive NLL loss.
    
    Workflow:
    1. Initialize KnowledgeCondenser (Q-Former + CondenseHead + FrozenLLM)
    2. Load dataset with query-fragment-answer triplets
    3. Training loop:
       - Forward: Q-Former → CondenseHead → Z prefix
       - Dual-path: LLM(with Z) vs. LLM(without Z)
       - Compute contrastive NLL loss with adaptive margin
       - Extract posterior from LLM attention
       - Backward: Update Q-Former + CondenseHead only
    4. Log metrics: NLL gain, margin, posterior
    5. Save checkpoint
    """
    print("=" * 80)
    print("Task C: Condensing-Generation (Contrastive NLL)")
    print("=" * 80)
    print(f"[INFO] Q-Former: {args.n_queries} queries, {args.hidden_dim}D, {args.num_layers} layers")
    print(f"[INFO] LLM: {args.llm_model_name} (frozen, eval mode)")
    print(f"[INFO] Loss: {args.margin_mode} margin, β={args.softplus_beta}")
    print("=" * 80)
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Initialize model
    model = KnowledgeCondenser(args).to(args.device)
    
    # Count parameters
    param_counts = model.count_trainable_parameters()
    print(f"[INFO] Trainable Parameters:")
    print(f"       Q-Former: {param_counts['qformer']:,}")
    print(f"       CondenseHead: {param_counts['condense_head']:,}")
    print(f"       Total: {param_counts['total']:,}")
    print(f"[INFO] LLM parameters: FROZEN (not counted)")
    print()
    
    # Dataset and loader
    train_dataset = DummyCondenseDataset(
        num_samples=args.num_samples,
        k_fragments=args.k_fragments,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"[INFO] Dataset: {len(train_dataset)} samples, batch_size={args.batch_size}")
    
    # Optimizer (only trainable parameters)
    trainable_params = list(model.qformer.parameters()) + list(model.condense_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    print(f"[INFO] Optimizer: AdamW, lr={args.lr}")
    print()
    
    # Training loop
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    for epoch in range(args.epochs):
        model.train()
        # Keep LLM in eval mode
        if model.frozen_llm.model is not None:
            model.frozen_llm.model.eval()
        
        epoch_loss = 0.0
        epoch_nll_gain = 0.0
        epoch_margin = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            loss_dict = model(
                q_embeds=batch['q_embeds'],
                p_embeds=batch['p_embeds'],
                query_input_ids=batch['query_input_ids'],
                answer_input_ids=batch['answer_input_ids'],
                subset_indices=batch['subset_indices'],
                pool_padding_mask=batch['pool_padding_mask'],
            )
            
            loss = loss_dict['loss_c']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            if loss_dict['nll_gain'] is not None:
                epoch_nll_gain += loss_dict['nll_gain'].item()
            if loss_dict['margin'] is not None:
                epoch_margin += loss_dict['margin'].item()
            num_batches += 1
            
            # Log batch
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | "
                      f"NLL Gain: {loss_dict['nll_gain'].item():.4f} | "
                      f"Margin: {loss_dict['margin'].item():.4f}")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_nll_gain = epoch_nll_gain / num_batches
        avg_margin = epoch_margin / num_batches
        
        print("-" * 80)
        print(f"Epoch [{epoch+1}/{args.epochs}] Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg NLL Gain: {avg_nll_gain:.4f}")
        print(f"  Avg Margin: {avg_margin:.4f}")
        print("-" * 80)
    
    print("=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print("[NOTE] This is a placeholder implementation using dummy LLM outputs.")
    print("       Actual LLM integration requires:")
    print("       - Implementing FrozenLLM.teacher_forcing_dual_path()")
    print("       - Loading real LLM model (LLaMA, Mistral, Phi, etc.)")
    print("       - Registering attention capture hooks")
    print("       - Constructing Prefix-LM attention masks")
    print("       - Testing with actual QA datasets")
    
    return model


def main():
    """Main entry point."""
    args = parse_args()
    setup_training(args)
    model = train_condensing(args)


if __name__ == "__main__":
    main()
