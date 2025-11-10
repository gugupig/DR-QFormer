"""
Task S: Fragment-Level Sorting Supervision (Prior Distribution Learning).

Core Training Objective:
- Learn **prior distribution** π_θ(p|q): Q-Former's fragment importance prediction
- Curriculum learning: Teacher signal (reranker) → Posterior signal (LLM attention)
- **Bayesian-inspired closed loop**: Minimize JS divergence with posterior from Task C

Optional Regularization:
- Dual training (Primal QA + Dual QG) can be enabled with --dual_mode
- Expected gain: ~1-3% improvement (auxiliary, not core mechanism)
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    
    from dr_qformer.models.qformer import DRQFormer
    from dr_qformer.models.heads import FragmentRankingHead
    from dr_qformer.losses import compute_ranking_loss, build_train_subset_mask, get_curriculum_weights
    from dr_qformer.metrics import compute_ranking_metrics
    from dr_qformer.adapters.retriever import Retriever
    import numpy as np
    
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


@dataclass
class TaskSArgs:
    """Arguments for Task S training."""
    # Model hyperparameters
    n_queries: int = 32
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    
    # Task S specific
    num_fragments: int = 100  # K fragments per query
    tau_head: float = 0.1  # Head LSE temperature
    tau_lq: float = 0.2  # LQ LSE temperature
    rho_top: float = 0.02  # Teacher Top-L ratio (2%)
    l_prime: int = 16  # Student Hard Negatives count
    
    # Curriculum learning
    lambda_teach_start: float = 1.0
    lambda_teach_end: float = 0.2
    lambda_post_start: float = 0.0
    lambda_post_end: float = 0.8
    lambda_entropy: float = 0.01
    
    # Loss temperatures
    tau_pred: float = 1.0  # Student prediction temperature
    tau_gt: float = 1.0  # Teacher target temperature
    alpha_gt: float = 0.7  # Teacher Top-L cumulative mass
    
    # Training
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 10
    batch_size: int = 8
    max_steps: int = 10000
    
    # Optional Dual mode (regularization only, not core mechanism)
    dual_mode: bool = False  # Enable optional Primal + Dual training (default: disabled)
    dual_weight: float = 0.1  # Weight for dual loss (small regularization coefficient)
    
    # LQ Entropy Regularization (Recommended for Task S - diversity important for prior)
    enable_lq_entropy_reg: bool = False  # Enable LQ-level entropy regularization
    lambda_entropy_start: float = 0.01  # Initial entropy weight (Task S: 0.01)
    lambda_entropy_end: float = 0.001  # Final entropy weight (Task S: 0.001)
    entropy_target_ratio: float = 0.7  # Target entropy ratio (0.7=conservative)
    
    # Paths
    retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    train_data: str = "data/train.json"
    dev_data: str = "data/dev.json"
    save_dir: str = "checkpoints/task_s"
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TaskSTrainer:
    """Trainer for Task S (Fragment Ranking)."""
    
    def __init__(self, args: TaskSArgs):
        self.args = args
        self.device = torch.device(args.device)
        self.global_step = 0
        
        # Initialize models
        print("\n" + "="*80)
        print("Initializing Task S Training")
        print("="*80)
        
        self.qformer = DRQFormer(
            n_queries=args.n_queries,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        ).to(self.device)
        
        self.head = FragmentRankingHead(
            num_fragments=args.num_fragments,
            tau_head=args.tau_head,
            tau_lq=args.tau_lq,
            rho_top=args.rho_top,
            l_prime=args.l_prime,
        ).to(self.device)
        
        # Retriever (frozen) - mock implementation
        print("📦 Loading retriever (mock)...")
        self.retriever_dim = args.hidden_dim  # Must match Q-Former's hidden_dim
        
        # Optimizer (only Q-Former parameters, head has none)
        self.optimizer = torch.optim.AdamW(
            self.qformer.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        
        print(f"\n✅ Models initialized on {self.device}")
        print(f"   Q-Former parameters: {sum(p.numel() for p in self.qformer.parameters()):,}")
        print(f"   FragmentRankingHead parameters: {self.head.count_parameters():,}")
    
    def encode_texts(self, texts: list) -> torch.Tensor:
        """Mock text encoding. Returns [batch, 1, d_ret] for Q-Former."""
        return torch.randn(len(texts), 1, self.retriever_dim, device=self.device)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """Train for one epoch."""
        self.qformer.train()
        
        total_loss = 0.0
        total_loss_teach = 0.0
        total_loss_post = 0.0
        total_loss_entropy = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Get curriculum weights
            curriculum = get_curriculum_weights(
                current_step=self.global_step,
                total_steps=self.args.max_steps,
                lambda_teach_start=self.args.lambda_teach_start,
                lambda_teach_end=self.args.lambda_teach_end,
                lambda_post_start=self.args.lambda_post_start,
                lambda_post_end=self.args.lambda_post_end,
            )
            
            # Move batch to device
            queries = batch["queries"]
            fragments = batch["fragments"]
            gt_scores = batch["gt_scores"].to(self.device)
            pool_padding_mask = batch.get("pool_padding_mask")
            if pool_padding_mask is not None:
                pool_padding_mask = pool_padding_mask.to(self.device)
            
            posterior_scores = batch.get("posterior_scores")
            if posterior_scores is not None:
                posterior_scores = posterior_scores.to(self.device)
            
            # === Primal Mode ===
            loss_primal = self._forward_step(
                queries=queries,
                fragments=fragments,
                gt_scores=gt_scores,
                posterior_scores=posterior_scores,
                pool_padding_mask=pool_padding_mask,
                lambda_teach=curriculum["lambda_teach"],
                lambda_post=curriculum["lambda_post"],
            )
            
            # === Dual Mode ===
            loss_dual = torch.tensor(0.0, device=self.device)
            if self.args.dual_mode and "answers" in batch:
                answers = batch["answers"]
                loss_dual = self._forward_step(
                    queries=answers,
                    fragments=fragments,
                    gt_scores=gt_scores,
                    posterior_scores=posterior_scores,
                    pool_padding_mask=pool_padding_mask,
                    lambda_teach=curriculum["lambda_teach"],
                    lambda_post=curriculum["lambda_post"],
                )
            
            # Combined loss
            if self.args.dual_mode:
                loss = loss_primal["loss"] + self.args.dual_weight * loss_dual["loss"]
            else:
                loss = loss_primal["loss"]
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qformer.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_loss_teach += loss_primal["loss_teach"].item()
            total_loss_post += loss_primal["loss_post"]
            total_loss_entropy += loss_primal["loss_entropy"].item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % self.args.log_interval == 0:
                print(f"Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {total_loss/num_batches:.4f} | "
                      f"λ_teach: {curriculum['lambda_teach']:.3f} | "
                      f"λ_post: {curriculum['lambda_post']:.3f}")
            
            if self.global_step >= self.args.max_steps:
                break
        
        return {
            "loss": total_loss / max(num_batches, 1),
            "loss_teach": total_loss_teach / max(num_batches, 1),
            "loss_post": total_loss_post / max(num_batches, 1),
            "loss_entropy": total_loss_entropy / max(num_batches, 1),
        }
    
    def _forward_step(
        self,
        queries: list,
        fragments: list,
        gt_scores: torch.Tensor,
        posterior_scores: Optional[torch.Tensor],
        pool_padding_mask: Optional[torch.Tensor],
        lambda_teach: float,
        lambda_post: float,
    ) -> dict:
        """Single forward step."""
        # Encode
        q_embeds = self.encode_texts(queries)  # [batch, 1, d]
        
        # Flatten fragments (all samples to single list)
        fragments_flat = [frag for frags in fragments for frag in frags]
        if not fragments_flat:
            return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True),
                    "loss_teach": torch.tensor(0.0, device=self.device),
                    "loss_post": 0.0,
                    "loss_entropy": torch.tensor(0.0, device=self.device)}
        
        p_embeds_flat = self.encode_texts(fragments_flat)  # [batch*K_max, 1, d]
        batch_size = len(fragments)
        K_max = len(fragments[0])  # All samples padded to same K_max by collate_fn
        # Reshape: [batch*K_max, 1, d] -> [batch, K_max, d]
        p_embeds = p_embeds_flat.squeeze(1).reshape(batch_size, K_max, -1)
        
        # Q-Former
        z, aux = self.qformer(query_embeds=q_embeds, p_embeds=p_embeds, pool_padding_mask=pool_padding_mask)
        ca_raw_scores_per_head = aux.get("ca_raw_scores_per_head")
        
        # Ranking Head
        head_output = self.head(z=z, ca_raw_scores_per_head=ca_raw_scores_per_head,
                                pool_padding_mask=pool_padding_mask, training=True)
        ranking_logits = head_output["ranking_logits"]
        
        # Build subset mask
        train_subset_mask = build_train_subset_mask(
            ranking_logits=ranking_logits.detach(),
            gt_scores=gt_scores,
            pool_padding_mask=pool_padding_mask,
            rho_top=self.args.rho_top,
            l_prime=self.args.l_prime,
        )
        
        # Loss with alpha_gt constraint
        loss_dict = compute_ranking_loss(
            ranking_logits=ranking_logits,
            gt_scores=gt_scores,
            posterior_scores=posterior_scores,
            pool_padding_mask=pool_padding_mask,
            train_subset_mask=train_subset_mask,
            lambda_teach=lambda_teach,
            lambda_post=lambda_post,
            lambda_entropy=self.args.lambda_entropy,
            tau_pred=self.args.tau_pred,
            tau_gt=self.args.tau_gt,
            alpha_gt=self.args.alpha_gt,  # Teacher Top-L cumulative mass constraint
        )
        
        # === Optional LQ Entropy Regularization ===
        if self.args.enable_lq_entropy_reg and ca_raw_scores_per_head is not None:
            # Curriculum weight (linear decay)
            progress = self.global_step / self.args.max_steps
            lambda_entropy_curr = self.args.lambda_entropy_start * (1 - 0.9 * progress)
            lambda_entropy_curr = max(lambda_entropy_curr, self.args.lambda_entropy_end)
            
            from dr_qformer.losses import compute_lq_entropy_loss
            entropy_loss = compute_lq_entropy_loss(
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                target_ratio=self.args.entropy_target_ratio,
            )
            
            # Add to total loss
            loss_dict["loss"] = loss_dict["loss"] + lambda_entropy_curr * entropy_loss
            loss_dict["loss_entropy_lq"] = entropy_loss.item()
        
        return loss_dict


def collate_task_s_batch(batch_list):
    """
    Custom collate function for Task S batches with dynamic K padding.
    
    Handles variable-length fragment lists by:
    1. Finding max K in batch
    2. Padding all samples to K_max
    3. Creating pool_padding_mask to indicate valid fragments
    """
    batch_size = len(batch_list)
    
    # Find max K in this batch
    K_max = max(len(item["fragments"]) for item in batch_list)
    
    # Prepare padded tensors
    queries = [item["queries"] for item in batch_list]
    answers = [item["answers"] for item in batch_list]
    fragments_padded = []
    gt_scores_padded = []
    pool_padding_mask = torch.zeros(batch_size, K_max, dtype=torch.bool)
    
    # Pad each sample
    for b, item in enumerate(batch_list):
        K_curr = len(item["fragments"])
        
        # Pad fragments
        fragments = item["fragments"]
        if K_curr < K_max:
            fragments = fragments + ["<PAD>"] * (K_max - K_curr)
        fragments_padded.append(fragments)
        
        # Pad gt_scores
        gt_scores = item["gt_scores"]
        if K_curr < K_max:
            # Pad with zeros (will be masked out)
            padding = np.zeros(K_max - K_curr, dtype=gt_scores.dtype)
            gt_scores = np.concatenate([gt_scores, padding])
        gt_scores_padded.append(gt_scores)
        
        # Set mask (True for valid fragments)
        pool_padding_mask[b, :K_curr] = True
    
    # Stack gt_scores
    gt_scores_tensor = torch.from_numpy(np.stack(gt_scores_padded)).float()
    
    # Handle posterior_scores if present (currently not in dummy dataset)
    posterior_scores = None
    if "posterior_scores" in batch_list[0]:
        posterior_scores_padded = []
        for b, item in enumerate(batch_list):
            K_curr = len(item["fragments"])
            post_scores = item["posterior_scores"]
            if K_curr < K_max:
                padding = np.zeros(K_max - K_curr, dtype=post_scores.dtype)
                post_scores = np.concatenate([post_scores, padding])
            posterior_scores_padded.append(post_scores)
        posterior_scores = torch.from_numpy(np.stack(posterior_scores_padded)).float()
    
    return {
        "queries": queries,
        "fragments": fragments_padded,
        "gt_scores": gt_scores_tensor,
        "pool_padding_mask": pool_padding_mask,
        "posterior_scores": posterior_scores,
        "answers": answers,
    }


def create_dummy_dataset(args: TaskSArgs) -> Dataset:
    """Create dummy dataset for testing."""
    class DummyRankingDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            K = 100  # Match args.num_fragments
            return {
                "queries": f"What is the capital of country {idx}?",  # Note: singular key but named plural
                "fragments": [f"Fragment {i} about country {idx}" for i in range(K)],
                "gt_scores": torch.randn(K).softmax(dim=0).numpy(),
                "answers": f"Capital {idx}",  # Note: singular key but named plural
            }
    
    return DummyRankingDataset()


def main():
    """Main training loop."""
    args = TaskSArgs()
    
    print("\n" + "="*80)
    print("Task S: Fragment-Level Sorting Supervision")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - K fragments: {args.num_fragments}")
    print(f"  - ρ (Teacher Top-L): {args.rho_top}")
    print(f"  - l' (Student Hard Negatives): {args.l_prime}")
    print(f"  - Dual mode: {args.dual_mode}")
    
    # Datasets
    print("\n📊 Loading datasets...")
    train_dataset = create_dummy_dataset(args)
    dev_dataset = create_dummy_dataset(args)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_task_s_batch,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_task_s_batch,
    )
    
    print(f"✅ Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")
    
    # Train
    trainer = TaskSTrainer(args)
    
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")
        
        metrics = trainer.train_epoch(train_loader, epoch)
        print(f"\nTrain: Loss={metrics['loss']:.4f}, Teach={metrics['loss_teach']:.4f}")
        
        if trainer.global_step >= args.max_steps:
            break
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
