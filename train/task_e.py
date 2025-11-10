"""
Task E: Fragment-level Entailment Tagging.

Train Q-Former + EntailmentHead to predict which retrieved fragments
are entailed by/relevant to the query.

Core Training Objective (Specification v1.1):
- Learn fragment-level answerability/entailment scores
- Acts as trainable filter/tagger for downstream tasks
- Focal loss with importance weighting (longtail, positive class)
- Drop-LQ regularization during training

Optional Dual Training:
- Primal mode (QA): query → predict fragment relevance (default)
- Dual mode (QG): answer → predict fragment relevance (optional regularization)
- Use --mode=both to enable dual training (~1-3% gain)
- Default: --mode=primal (core functionality only)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import numpy as np
    from tqdm import tqdm
    
    from dr_qformer.models.qformer import DRQFormer
    from dr_qformer.models.heads import EntailmentHead
    from dr_qformer.adapters.retriever import RetrieverAdapter
    from dr_qformer.losses import compute_focal_loss
    from dr_qformer.metrics import compute_entailment_metrics
    from dr_qformer.data.collate import collate_task_e
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    torch = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Task E: Fragment Entailment Tagging")
    
    # Model hyperparameters
    parser.add_argument("--n_queries", type=int, default=32, help="Number of learnable queries")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of Q-Former layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--k_fragments", type=int, default=5, help="Number of retrieved fragments")
    
    # EntailmentHead hyperparameters (from spec v1.1)
    parser.add_argument("--tau", type=float, default=0.5, help="LogSumExp temperature (0.1-1.0)")
    parser.add_argument("--p_drop_lq", type=float, default=0.1, help="Drop-LQ probability")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha")
    parser.add_argument("--w_pos", type=float, default=10.0, help="Positive class weight")
    parser.add_argument("--w_longtail", type=float, default=50.0, help="Longtail example weight")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    
    # Data paths
    parser.add_argument("--train_data", type=str, required=True, help="Training data path")
    parser.add_argument("--dev_data", type=str, required=True, help="Dev data path")
    parser.add_argument("--retriever_model", type=str, default="facebook/contriever", 
                        help="Frozen retriever model")
    
    # Training modes (Optional dual training for regularization)
    parser.add_argument("--mode", type=str, choices=["primal", "dual", "both"], default="primal",
                        help="Training mode: primal (QA only, default), dual (QG only), or both (optional regularization)")
    
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="./checkpoints/task_e",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


class TaskETrainer:
    """
    Trainer for Task E: Fragment-level Entailment Tagging.
    
    Implements dual training strategy (Primal + Dual modes) with shared parameters.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize trainer."""
        self.args = args
        self.device = torch.device(args.device)
        
        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Initialize models
        self.retriever = RetrieverAdapter(model_name=args.retriever_model, device=self.device)
        self.retriever.eval()  # Frozen
        
        self.qformer = DRQFormer(
            n_queries=args.n_queries,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        ).to(self.device)
        
        self.head = EntailmentHead(
            hidden_dim=args.hidden_dim,
            num_fragments=args.k_fragments,
            tau=args.tau,
            p_drop_lq=args.p_drop_lq,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
        ).to(self.device)
        
        # Optimizer (only trainable parameters)
        trainable_params = list(self.qformer.parameters()) + list(self.head.parameters())
        self.optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
        
        # Learning rate scheduler (cosine annealing with warmup)
        # TODO: Implement warmup + cosine schedule
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        
        # Checkpointing
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_dev_loss = float('inf')
        
        print(f"Initialized Task E Trainer:")
        print(f"  Q-Former: {self.qformer.count_parameters():,} trainable params")
        print(f"  EntailmentHead: {self.head.count_parameters():,} trainable params")
        print(f"  Retriever: Frozen ({args.retriever_model})")
        print(f"  Device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.qformer.train()
        self.head.train()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            queries = batch["queries"]  # List of strings
            fragments = batch["fragments"]  # List of lists of strings [batch, k]
            gt_labels = batch["gt_labels"].to(self.device)  # [batch, k]
            importance_weights = batch.get("importance_weights", None)  # [batch, k] or None
            pool_padding_mask = batch.get("pool_padding_mask", None)  # [batch, k] or None
            mode = batch.get("mode", "primal")  # "primal" or "dual"
            
            if importance_weights is not None:
                importance_weights = importance_weights.to(self.device)
            if pool_padding_mask is not None:
                pool_padding_mask = pool_padding_mask.to(self.device)
            
            # Embed query and fragments with frozen retriever
            with torch.no_grad():
                q_embeds_2d = self.retriever.encode_queries(queries)  # [batch, d_ret]
                # Expand to 3D: [batch, 1, d_ret] as required by qformer
                q_embeds = q_embeds_2d.unsqueeze(1)  # [batch, 1, d_ret]
                
                # Flatten fragments list and encode (dynamic K_pool per batch)
                flat_fragments = [frag for frag_list in fragments for frag in frag_list]
                p_embeds_flat = self.retriever.encode_passages(flat_fragments)  # [total_frags, d_ret]
                
                # Reshape to [batch, K_pool, d_ret] where K_pool is max K in this batch
                # Note: fragments is already padded to max K in this batch by dataloader
                batch_size = len(queries)
                K_pool = len(fragments[0])  # All samples in batch have same length after padding
                p_embeds = p_embeds_flat.view(batch_size, K_pool, -1)  # [batch, K_pool, d_ret]
                
                # Get answer embeddings for Dual mode (if available in batch)
                if "answers" in batch:
                    answers = batch["answers"]  # List of answer strings
                    a_embeds_2d = self.retriever.encode_queries(answers)  # [batch, d_ret]
                    a_embeds = a_embeds_2d.unsqueeze(1)  # [batch, 1, d_ret]
                else:
                    # Fallback: use query embeddings as placeholder (should not happen in production)
                    a_embeds = q_embeds
            
            # Build dynamic importance weights from gt_labels + is_longtail
            if importance_weights is None:
                importance_weights = torch.ones_like(gt_labels)
                # Positive class weighting
                importance_weights = torch.where(gt_labels == 1, 
                                                self.args.w_pos * torch.ones_like(gt_labels), 
                                                torch.ones_like(gt_labels))
                # Longtail weighting (if provided)
                if "is_longtail" in batch:
                    is_longtail = batch["is_longtail"].to(self.device)  # [batch, k]
                    importance_weights = torch.where((gt_labels == 1) & (is_longtail == 1),
                                                    self.args.w_longtail * torch.ones_like(gt_labels),
                                                    importance_weights)
            
            # === Primal Forward (query_embeds) ===
            loss_primal = torch.tensor(0.0, device=self.device)
            if self.args.mode in ["primal", "both"]:
                z_primal, aux_primal = self.qformer(
                    query_embeds=q_embeds,  
                    p_embeds=p_embeds,
                    pool_padding_mask=pool_padding_mask
                )
                
                ca_raw_scores_per_head_primal = aux_primal.get("ca_raw_scores_per_head", None)
                
                head_out_primal = self.head(
                    z=z_primal,
                    ca_raw_scores_per_head=ca_raw_scores_per_head_primal,
                    pool_padding_mask=pool_padding_mask,
                    training=True
                )
                fragment_logits_primal = head_out_primal['fragment_logits']
                
                loss_primal = compute_focal_loss(
                    logits=fragment_logits_primal,
                    gt_labels=gt_labels,
                    importance_weights=importance_weights,
                    pool_padding_mask=pool_padding_mask,
                    focal_gamma=self.args.focal_gamma,
                    focal_alpha=self.args.focal_alpha,
                )
            
            # === Optional Dual Forward (answer_embeds) ===
            loss_dual = torch.tensor(0.0, device=self.device)
            if self.args.mode in ["dual", "both"]:
                z_dual, aux_dual = self.qformer(
                    answer_embeds=a_embeds,
                    p_embeds=p_embeds,
                    pool_padding_mask=pool_padding_mask
                )
                
                ca_raw_scores_per_head_dual = aux_dual.get("ca_raw_scores_per_head", None)
                
                head_out_dual = self.head(
                    z=z_dual,
                    ca_raw_scores_per_head=ca_raw_scores_per_head_dual,
                    pool_padding_mask=pool_padding_mask,
                    training=True
                )
                fragment_logits_dual = head_out_dual['fragment_logits']
                
                loss_dual = compute_focal_loss(
                    logits=fragment_logits_dual,
                    gt_labels=gt_labels,
                    importance_weights=importance_weights,
                    pool_padding_mask=pool_padding_mask,
                    focal_gamma=self.args.focal_gamma,
                    focal_alpha=self.args.focal_alpha,
                )
            
            # Combined loss (dual training uses small weight by default)
            if self.args.mode == "both":
                loss = loss_primal + 0.1 * loss_dual  # Small regularization weight
            else:
                loss = loss_primal + loss_dual
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.qformer.parameters()) + list(self.head.parameters()),
                    self.args.grad_clip
                )
            
            self.optimizer.step()
            
            # Update metrics
            batch_size = len(queries)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / total_samples:.4f}",
                "mode": mode,
            })
            
            # Save checkpoint
            if self.global_step % self.args.save_every == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")
        
        # Epoch metrics
        avg_loss = total_loss / total_samples
        return {"loss": avg_loss}
    
    @torch.no_grad()
    def evaluate(self, dev_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on dev set.
        
        Args:
            dev_loader: Dev data loader
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.qformer.eval()
        self.head.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(dev_loader, desc="Evaluating"):
            queries = batch["queries"]
            fragments = batch["fragments"]
            gt_labels = batch["gt_labels"].to(self.device)
            importance_weights = batch.get("importance_weights", None)
            pool_padding_mask = batch.get("pool_padding_mask", None)
            
            if importance_weights is not None:
                importance_weights = importance_weights.to(self.device)
            if pool_padding_mask is not None:
                pool_padding_mask = pool_padding_mask.to(self.device)
            
            # Embed with frozen retriever
            q_embeds_2d = self.retriever.encode_queries(queries)  # [batch, d_ret]
            q_embeds = q_embeds_2d.unsqueeze(1)  # [batch, 1, d_ret] - expand to 3D
            
            # Dynamic K_pool per batch
            flat_fragments = [frag for frag_list in fragments for frag in frag_list]
            p_embeds_flat = self.retriever.encode_passages(flat_fragments)
            batch_size = len(queries)
            K_pool = len(fragments[0])  # Max K in this batch (after padding)
            p_embeds = p_embeds_flat.view(batch_size, K_pool, -1)  # [batch, K_pool, d_ret]
            
            # Forward (Primal mode only during eval)
            z, aux = self.qformer(
                query_embeds=q_embeds,
                p_embeds=p_embeds,
                pool_padding_mask=pool_padding_mask
            )
            ca_raw_scores_per_head = aux.get("ca_raw_scores_per_head", None)
            
            # EntailmentHead forward (training=False disables Drop-LQ)
            head_out = self.head(
                z=z,
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                training=False
            )
            fragment_logits = head_out['fragment_logits']
            
            # Compute loss using standalone loss function
            loss = compute_focal_loss(
                logits=fragment_logits,
                gt_labels=gt_labels,
                importance_weights=importance_weights,
                pool_padding_mask=pool_padding_mask,
                focal_gamma=self.args.focal_gamma,
                focal_alpha=self.args.focal_alpha,
            )
            
            # Update metrics
            batch_size = len(queries)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Collect predictions
            all_preds.append(fragment_logits)
            all_labels.append(gt_labels)
        
        # Aggregate metrics
        avg_loss = total_loss / total_samples
        
        # Stack all predictions and labels
        all_preds_tensor = torch.cat(all_preds, dim=0)  # [total_samples, k]
        all_labels_tensor = torch.cat(all_labels, dim=0)  # [total_samples, k]
        
        # Compute binary classification metrics using standalone metrics function
        metrics = compute_entailment_metrics(
            logits=all_preds_tensor,
            gt_labels=all_labels_tensor,
            pool_padding_mask=None,  # Already filtered during collection
            threshold=0.5,
        )
        
        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / filename
        torch.save({
            "global_step": self.global_step,
            "qformer_state_dict": self.qformer.state_dict(),
            "head_state_dict": self.head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "args": self.args,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.qformer.load_state_dict(checkpoint["qformer_state_dict"])
        self.head.load_state_dict(checkpoint["head_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        print(f"Loaded checkpoint from: {checkpoint_path}")


def main():
    """Main training loop."""
    args = parse_args()
    
    print("=" * 80)
    print("Task E: Fragment-level Entailment Tagging")
    print("=" * 80)
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Initialize trainer
    trainer = TaskETrainer(args)
    
    # TODO: Load datasets
    # train_dataset = TaskEDataset(args.train_data, mode=args.mode)
    # dev_dataset = TaskEDataset(args.dev_data, mode="primal")  # Always eval in primal mode
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\nTODO: Implement data loading")
    print("Expected data format:")
    print("  - queries: List of query strings")
    print("  - fragments: List of lists of fragment strings [batch, k]")
    print("  - gt_labels: Binary labels [batch, k] (1 = entailment, 0 = no entailment)")
    print("  - importance_weights: [batch, k] (w_pos for positive, w_longtail for longtail)")
    print("  - pool_padding_mask: [batch, k] (True = valid fragment, False = padding)")
    print("  - mode: 'primal' (QA) or 'dual' (QG)")
    print()
    
    # Training loop
    # for epoch in range(args.epochs):
    #     train_metrics = trainer.train_epoch(train_loader, epoch)
    #     print(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
    #     
    #     dev_metrics = trainer.evaluate(dev_loader)
    #     print(f"Epoch {epoch} - Dev Loss: {dev_metrics['loss']:.4f}")
    #     
    #     # Save best model
    #     if dev_metrics['loss'] < trainer.best_dev_loss:
    #         trainer.best_dev_loss = dev_metrics['loss']
    #         trainer.save_checkpoint("best.pt")
    #     
    #     trainer.scheduler.step()
    
    print("Task E training script ready!")
    print("Next steps:")
    print("  1. Implement TaskEDataset for data loading")
    print("  2. Run training with real data")
    print("  3. Add evaluation metrics (precision, recall, F1, AUC-ROC)")
    print("  4. Implement warmup + cosine LR schedule")


if __name__ == "__main__":
    main()
