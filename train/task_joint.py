"""
Joint Training for DR-QFormer: Task E, S, C

Following BLIP-2 Stage-1 philosophy:
- Shared Q-Former forward pass for all tasks
- Multi-objective training with dynamic scheduling
- Gradual shift from prior (teacher) to posterior (LLM feedback)

Key innovations:
- Task E: Entailment filtering (Focal Loss, high recall)
- Task S: Ranking with teacher→posterior transition (ListNet + JS divergence)
- Task C: Contrastive NLL with posterior extraction (dual-path Teacher Forcing)
- Unified Drop-LQ mask across all tasks
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from .joint_data import JointBatch
from .schedule import TrainingScheduler, ScheduleConfig
from ..models.qformer import DRQFormer
from ..models.heads import EntailmentHead, FragmentRankingHead
from ..losses import (
    compute_entailment_loss,
    compute_ranking_loss,
    compute_contrastive_nll_with_posterior,
)
from ..utils.masks import generate_lq_drop_mask


class JointTrainer:
    """
    Trainer for joint E/S/C training with shared Q-Former forward.
    
    Training flow (per step):
    1. Q-Former forward (once) → Z, CA raw scores, SA weights
    2. Task E forward + loss
    3. Task S forward + training subset U + loss
    4. Task C dual-path Teacher Forcing + posterior extraction + loss
    5. Combined backward
    """
    
    def __init__(
        self,
        model: DRQFormer,
        task_e_head: EntailmentHead,
        task_s_head: FragmentRankingHead,
        llm_model: nn.Module,  # PLACEHOLDER: Add actual LLM
        tokenizer: Any,        # PLACEHOLDER: Add actual tokenizer
        optimizer: torch.optim.Optimizer,
        scheduler: TrainingScheduler,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # Loss hyperparameters
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        softplus_beta: float = 10.0,
        adaptive_margin: bool = True,
        margin_kappa: float = 0.5,
        # Drop-LQ parameters
        drop_lq_rate: float = 0.1,
        lq_entropy_reg_lambda: float = 0.01,
        # Logging
        log_interval: int = 100,
        save_dir: str = "./checkpoints",
    ):
        self.model = model
        self.task_e_head = task_e_head
        self.task_s_head = task_s_head
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Loss hyperparameters
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.softplus_beta = softplus_beta
        self.adaptive_margin = adaptive_margin
        self.margin_kappa = margin_kappa
        
        # Drop-LQ
        self.drop_lq_rate = drop_lq_rate
        self.lq_entropy_reg_lambda = lq_entropy_reg_lambda
        
        # Logging
        self.log_interval = log_interval
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Move models to device
        self.model = self.model.to(device)
        self.task_e_head = self.task_e_head.to(device)
        self.task_s_head = self.task_s_head.to(device)
        self.llm_model = self.llm_model.to(device)
        
        # Freeze LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        print("=" * 80)
        print("JointTrainer Initialized")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Q-Former params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Task E Head params: {sum(p.numel() for p in task_e_head.parameters() if p.requires_grad):,}")
        print(f"Task S Head params: {sum(p.numel() for p in task_s_head.parameters() if p.requires_grad):,}")
        print(f"LLM params (frozen): {sum(p.numel() for p in llm_model.parameters()):,}")
        print(f"Drop-LQ rate: {drop_lq_rate}")
        print(f"Focal Loss: γ={focal_gamma}, α={focal_alpha}")
        print(f"Contrastive NLL: β={softplus_beta}, adaptive_margin={adaptive_margin}")
        print("=" * 80)
    
    def shared_forward(
        self,
        batch: JointBatch,
        mode: str = "primal",
    ) -> Dict[str, torch.Tensor]:
        """
        Shared Q-Former forward pass for all tasks.
        
        Args:
            batch: JointBatch
            mode: "primal" (QA) or "dual" (QG)
        
        Returns:
            Dict with keys:
                - z: [batch, N, d] - Final LQ representations
                - z_lm: [batch, num_lm_tokens, d_lm] - LLM-projected Z
                - ca_raw_scores: [batch, heads, N, K] - Pre-softmax CA scores
                - ca_weights: [batch, heads, N, K] - Post-softmax CA weights
                - sa_weights: [batch, heads, N+1, N+1] - SA attention weights
                - lq_drop_mask: [batch, N] - Unified Drop-LQ mask
                - pool_padding_mask: [batch, K] - Fragment padding mask
        """
        # Select embeddings based on mode
        if mode == "primal":
            cond_embeds = batch.query_embeds  # QA: condition on query
        else:
            cond_embeds = batch.answer_embeds  # QG: condition on answer
        
        # Generate unified Drop-LQ mask (shared across E/S/C)
        batch_size, N = self.model.num_lqs, self.model.num_lqs
        lq_drop_mask = generate_lq_drop_mask(
            batch_size=batch.batch_size,
            num_lqs=N,
            drop_rate=self.drop_lq_rate,
            device=self.device,
        )
        
        # Q-Former forward
        z_lm, aux = self.model(
            query_embeds=cond_embeds,
            fragment_embeds=batch.fragment_embeds,
            pool_padding_mask=batch.pool_padding_mask,
            lq_drop_mask=lq_drop_mask,
            return_lm_proj=True,
        )
        
        return {
            'z': aux['z_final'],  # [batch, N, d]
            'z_lm': z_lm,         # [batch, num_lm_tokens, d_lm]
            'ca_raw_scores': aux.get('ca_raw_scores_per_head'),  # [batch, heads, N, K]
            'ca_weights': aux.get('ca_weights_per_head'),        # [batch, heads, N, K]
            'sa_weights': aux.get('sa_weights_per_head'),        # [batch, heads, N+1, N+1]
            'lq_drop_mask': lq_drop_mask,                        # [batch, N]
            'pool_padding_mask': batch.pool_padding_mask,        # [batch, K]
        }
    
    def compute_task_e_loss(
        self,
        forward_out: Dict[str, torch.Tensor],
        batch: JointBatch,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Task E: Entailment classification with Focal Loss.
        
        Args:
            forward_out: Output from shared_forward
            batch: JointBatch
        
        Returns:
            loss: Scalar tensor
            metrics: Dict with accuracy, precision, recall
        """
        # Task E Head forward
        fragment_logits = self.task_e_head(
            z=forward_out['z'],
            ca_raw_scores=forward_out['ca_raw_scores'],
            pool_padding_mask=forward_out['pool_padding_mask'],
            lq_drop_mask=forward_out['lq_drop_mask'],
        )
        
        # Compute loss
        loss, metrics = compute_entailment_loss(
            logits=fragment_logits,
            labels=batch.entailment_labels,
            pool_padding_mask=batch.pool_padding_mask,
            importance_weights=batch.entailment_weights,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
        )
        
        return loss, metrics
    
    def compute_task_s_loss(
        self,
        forward_out: Dict[str, torch.Tensor],
        batch: JointBatch,
        posterior_scores: Optional[torch.Tensor] = None,
        lambda_teach: float = 1.0,
        lambda_post: float = 0.0,
        lambda_ent: float = 0.01,
        temperature: float = 1.0,
        alpha_gt: float = 0.9,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Task S: Fragment ranking with teacher→posterior transition.
        
        Args:
            forward_out: Output from shared_forward
            batch: JointBatch
            posterior_scores: [batch, K] - Posterior from Task C (detached)
            lambda_teach: Weight for teacher (ListNet)
            lambda_post: Weight for posterior (JS divergence)
            lambda_ent: Weight for tail entropy regularization
            temperature: For adaptive teacher distribution
            alpha_gt: Target cumulative probability for Top-L
        
        Returns:
            loss: Scalar tensor
            metrics: Dict with NDCG, subset size, etc.
        """
        # Task S Head forward
        ranking_logits = self.task_s_head(
            z=forward_out['z'],
            ca_raw_scores=forward_out['ca_raw_scores'],
            pool_padding_mask=forward_out['pool_padding_mask'],
            lq_drop_mask=forward_out['lq_drop_mask'],
        )
        
        # Compute loss
        loss, metrics = compute_ranking_loss(
            ranking_logits=ranking_logits,
            gt_scores=batch.ranking_scores,
            pool_padding_mask=batch.pool_padding_mask,
            posterior_scores=posterior_scores,
            lambda_teach=lambda_teach,
            lambda_post=lambda_post,
            lambda_entropy=lambda_ent,
            temperature=temperature,
            alpha_gt=alpha_gt,
            top_l_dynamic=True,
            top_lprime=10,  # Hard negatives
        )
        
        return loss, metrics
    
    def compute_task_c_loss(
        self,
        forward_out: Dict[str, torch.Tensor],
        batch: JointBatch,
        mode: str = "primal",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """
        Task C: Contrastive NLL with dual-path Teacher Forcing.
        
        Computes:
        - Path-A: [Z, Q, A] with full attention → extract posterior qψ_U
        - Path-B: [Z_dummy, Q, A] with Q/A→Z masked (baseline)
        - Loss: Softplus(β * (margin - G))
        
        Args:
            forward_out: Output from shared_forward
            batch: JointBatch
            mode: "primal" (QA) or "dual" (QG)
        
        Returns:
            loss: Scalar tensor
            posterior_scores: [batch, K] - Detached posterior for Task S
            metrics: Dict with NLL_A, NLL_B, gain G, margin
        """
        # PLACEHOLDER: Implement dual-path Teacher Forcing
        # This requires:
        # 1. Construct inputs_embeds: [Z_lm, Q_tokens, A_tokens]
        # 2. Path-A: Full forward, extract LLM→Z attention
        # 3. Path-B: Mask Q/A→Z attention, no_grad
        # 4. Compute contrastive NLL
        # 5. Backtrack attention to get qψ_U (detached)
        
        print("[Task C] PLACEHOLDER: Dual-path Teacher Forcing not yet implemented")
        print("[Task C] TODO: Integrate with LLM, extract attention, compute gain")
        
        # Dummy loss and posterior
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        posterior_scores = None
        
        metrics = {
            'nll_path_a': 0.0,
            'nll_path_b': 0.0,
            'gain': 0.0,
            'margin': 0.0,
        }
        
        return loss, posterior_scores, metrics
    
    def train_step(self, batch: JointBatch) -> Dict[str, float]:
        """
        Single training step: joint E/S/C.
        
        Args:
            batch: JointBatch
        
        Returns:
            metrics: Dict with all losses and metrics
        """
        self.model.train()
        self.task_e_head.train()
        self.task_s_head.train()
        
        # Move batch to device
        batch.query_embeds = batch.query_embeds.to(self.device)
        batch.answer_embeds = batch.answer_embeds.to(self.device)
        batch.fragment_embeds = batch.fragment_embeds.to(self.device)
        batch.pool_padding_mask = batch.pool_padding_mask.to(self.device)
        batch.entailment_labels = batch.entailment_labels.to(self.device)
        if batch.entailment_weights is not None:
            batch.entailment_weights = batch.entailment_weights.to(self.device)
        batch.ranking_scores = batch.ranking_scores.to(self.device)
        batch.question_tokens = batch.question_tokens.to(self.device)
        batch.answer_tokens = batch.answer_tokens.to(self.device)
        
        # Get current schedule weights
        weights = self.scheduler.get_all_weights()
        
        # =====================================================================
        # 1. Shared Q-Former Forward (Primal Mode: QA)
        # =====================================================================
        forward_out = self.shared_forward(batch, mode="primal")
        
        # =====================================================================
        # 2. Task E: Entailment
        # =====================================================================
        loss_e, metrics_e = self.compute_task_e_loss(forward_out, batch)
        
        # =====================================================================
        # 3. Task C: Contrastive NLL + Posterior Extraction
        # =====================================================================
        enable_posterior = weights['enable_posterior']
        if enable_posterior:
            loss_c, posterior_scores, metrics_c = self.compute_task_c_loss(
                forward_out, batch, mode="primal"
            )
        else:
            # Warm-up: C records NLL only, no gradient
            with torch.no_grad():
                loss_c, posterior_scores, metrics_c = self.compute_task_c_loss(
                    forward_out, batch, mode="primal"
                )
            loss_c = loss_c * 0.0  # Zero out gradient
        
        # =====================================================================
        # 4. Task S: Ranking (with posterior if enabled)
        # =====================================================================
        loss_s, metrics_s = self.compute_task_s_loss(
            forward_out,
            batch,
            posterior_scores=posterior_scores,
            lambda_teach=weights['lambda_teach'],
            lambda_post=weights['lambda_post'],
            lambda_ent=weights['lambda_ent'],
        )
        
        # =====================================================================
        # 5. Optional: Dual Mode (QG)
        # =====================================================================
        loss_dual = torch.tensor(0.0, device=self.device)
        if weights['enable_dual']:
            # Repeat E/S/C for dual mode
            forward_out_dual = self.shared_forward(batch, mode="dual")
            loss_e_dual, _ = self.compute_task_e_loss(forward_out_dual, batch)
            loss_s_dual, _ = self.compute_task_s_loss(forward_out_dual, batch)
            loss_c_dual, _, _ = self.compute_task_c_loss(forward_out_dual, batch, mode="dual")
            
            loss_dual = loss_e_dual + loss_s_dual + loss_c_dual
        
        # =====================================================================
        # 6. Combined Loss
        # =====================================================================
        loss_total = (
            weights['w_E'] * loss_e +
            weights['w_S'] * loss_s +
            weights['w_C'] * loss_c +
            weights['lambda_dual'] * loss_dual
        )
        
        # =====================================================================
        # 7. Backward
        # =====================================================================
        self.optimizer.zero_grad()
        loss_total.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.task_e_head.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.task_s_head.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # =====================================================================
        # 8. Update Scheduler
        # =====================================================================
        self.scheduler.step()
        self.global_step += 1
        
        # =====================================================================
        # 9. Collect Metrics
        # =====================================================================
        metrics = {
            # Losses
            'loss_total': loss_total.item(),
            'loss_e': loss_e.item(),
            'loss_s': loss_s.item(),
            'loss_c': loss_c.item(),
            'loss_dual': loss_dual.item(),
            # Task E
            'e_accuracy': metrics_e.get('accuracy', 0.0),
            'e_precision': metrics_e.get('precision', 0.0),
            'e_recall': metrics_e.get('recall', 0.0),
            # Task S
            's_ndcg': metrics_s.get('ndcg', 0.0),
            's_subset_size': metrics_s.get('subset_size', 0),
            # Task C
            'c_nll_a': metrics_c.get('nll_path_a', 0.0),
            'c_nll_b': metrics_c.get('nll_path_b', 0.0),
            'c_gain': metrics_c.get('gain', 0.0),
            # Weights
            'w_E': weights['w_E'],
            'w_S': weights['w_S'],
            'w_C': weights['w_C'],
            'lambda_teach': weights['lambda_teach'],
            'lambda_post': weights['lambda_post'],
            'lambda_ent': weights['lambda_ent'],
            # Metadata
            'phase': weights['phase'],
            'progress': weights['progress'],
        }
        
        return metrics
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with JointBatch
        
        Returns:
            avg_metrics: Dict with averaged metrics
        """
        self.epoch += 1
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        epoch_metrics = []
        
        for batch in pbar:
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss_total']:.4f}",
                'E': f"{metrics['loss_e']:.3f}",
                'S': f"{metrics['loss_s']:.3f}",
                'C': f"{metrics['loss_c']:.3f}",
                'phase': metrics['phase'],
            })
            
            # Log
            if self.global_step % self.log_interval == 0:
                self.log_metrics(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            if isinstance(epoch_metrics[0][key], (int, float)):
                avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
            else:
                avg_metrics[key] = epoch_metrics[-1][key]  # Take last value for strings
        
        return avg_metrics
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics (extend with wandb/tensorboard as needed)."""
        print(f"\n[Step {self.global_step}] Phase: {metrics['phase']}")
        print(f"  Total Loss: {metrics['loss_total']:.4f}")
        print(f"  Task E: {metrics['loss_e']:.4f} | Acc: {metrics['e_accuracy']:.3f}")
        print(f"  Task S: {metrics['loss_s']:.4f} | NDCG: {metrics['s_ndcg']:.3f}")
        print(f"  Task C: {metrics['loss_c']:.4f} | Gain: {metrics['c_gain']:.3f}")
        print(f"  Weights: E={metrics['w_E']:.2f}, S={metrics['w_S']:.2f}, C={metrics['w_C']:.2f}")
        print(f"  Lambdas: teach={metrics['lambda_teach']:.2f}, post={metrics['lambda_post']:.2f}")
    
    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save checkpoint."""
        if path is None:
            path = os.path.join(self.save_dir, f"checkpoint_step_{self.global_step}.pt")
        
        torch.save({
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'task_e_head_state_dict': self.task_e_head.state_dict(),
            'task_s_head_state_dict': self.task_s_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
        }, path)
        
        print(f"✅ Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.task_e_head.load_state_dict(checkpoint['task_e_head_state_dict'])
        self.task_s_head.load_state_dict(checkpoint['task_s_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.current_step = checkpoint['scheduler_step']
        
        print(f"✅ Checkpoint loaded: {path}")
        print(f"   Resuming from step {self.global_step}, epoch {self.epoch}")


# ============================================================================
# Entry point for testing trainer structure
# ============================================================================

if __name__ == "__main__":
    print("Testing JointTrainer structure...")
    
    from .joint_data import create_joint_dataloader
    from .schedule import get_default_schedule
    
    # Create dummy components
    model = DRQFormer(d=64, N=4, num_heads=2)
    task_e_head = EntailmentHead(d=64, aggregation_mode="lse")
    task_s_head = FragmentRankingHead(d=64)
    
    # PLACEHOLDER: LLM
    llm_model = nn.Identity()
    tokenizer = None
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) +
        list(task_e_head.parameters()) +
        list(task_s_head.parameters()),
        lr=1e-4
    )
    
    schedule_config = get_default_schedule(total_steps=1000)
    scheduler = TrainingScheduler(schedule_config)
    
    trainer = JointTrainer(
        model=model,
        task_e_head=task_e_head,
        task_s_head=task_s_head,
        llm_model=llm_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    
    # Create dummy dataloader
    dataloader = create_joint_dataloader(
        data_path="dummy.json",
        batch_size=2,
        num_workers=0,
    )
    
    # Test one step
    batch = next(iter(dataloader))
    metrics = trainer.train_step(batch)
    
    print("\n✅ Trainer test complete!")
    print(f"   Total loss: {metrics['loss_total']:.4f}")
    print("\n⚠️  Remember to implement:")
    print("   - Task C dual-path Teacher Forcing")
    print("   - LLM integration")
    print("   - Posterior extraction qψ_U")
