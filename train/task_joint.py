"""
Joint Training for Multi-Task DR-QFormer (E+S+C).

Core Design (BLIP-2 Inspired):
================================
- **ONE Q-Former forward pass** produces:
  * Z: Knowledge prefix for LLM [batch, N_lq, d]
  * CA raw scores: Per-head, pre-softmax [batch, H, N_lq, K]
  * LQs aware: Final learnable query states

- **Task E**: Entailment tagging (Focal Loss)
- **Task S**: Fragment ranking (Teacher→Posterior curriculum)
- **Task C**: Condensing-generation (Contrastive NLL)

- **Bayesian Closed Loop**: Task C posterior → Task S alignment

Training Flow (Per Step):
==========================
1. Q-Former forward (ONE pass, shared LQ-drop mask)
2. Task E forward + loss
3. Task S forward + train subset U + loss
4. Task C forward (dual-path) + posterior extract + loss
5. Weighted loss combination: w_E*L_E + w_S*L_S + w_C*L_C
6. Backward + update Q-Former + Heads (LLM frozen)

Key Features:
=============
- Shared forward pass (efficiency)
- Dynamic K per sample (padding mask)
- Curriculum learning (λ_teach → λ_post)
- Posterior backtracing (LLM→Z attention)
- Optional dual mode (QA+QG regularization)
- Optional LQ entropy reg (prevent collapse)

TODO (Placeholders):
====================
- [ ] Retriever integration (replace mock encoding)
- [ ] LLM integration (replace mock NLL/attention)
- [ ] Real dataset loading
- [ ] Distributed training support
- [ ] Evaluation metrics logging
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x  # Fallback: no progress bar
    
    from src.models.qformer import DRQFormer
    from src.models.heads import EntailmentHead, FragmentRankingHead, CondenseHead
    from src.losses import (
        compute_focal_loss,
        compute_ranking_loss,
        compute_condensing_loss,
        build_train_subset_mask,
        compute_lq_entropy_loss,
    )
    from train.schedule import JointTrainingScheduler, ScheduleConfig, get_lr_schedule
    from train.joint_data import collate_joint_batch
    
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


@dataclass
class JointTrainingConfig:
    """Configuration for joint training."""
    # Model architecture
    n_queries: int = 32
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    
    # Task heads
    task_e_tau: float = 0.5
    task_e_p_drop_lq: float = 0.1
    task_e_focal_gamma: float = 2.0
    task_e_focal_alpha: float = 0.25
    
    task_s_tau_head: float = 0.1
    task_s_tau_lq: float = 0.2
    task_s_rho_top: float = 0.02
    task_s_l_prime: int = 16
    
    task_c_llm_hidden_dim: int = 4096
    task_c_softplus_beta: float = 10.0
    task_c_margin_mode: str = "adaptive"
    task_c_margin_fixed: float = 0.5
    task_c_margin_adaptive_ratio: float = 0.5
    task_c_margin_min: float = 0.1
    task_c_margin_max: float = 2.0
    
    # Unified Drop-LQ for multi-task training
    p_drop_lq_unified: float = 0.1  # Global Drop-LQ rate (0.0 = disabled)
    
    # Training
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 50000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Schedule
    schedule_config: Optional[ScheduleConfig] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class JointTrainer:
    """
    Trainer for joint multi-task learning (E+S+C).
    
    Architecture:
        - Q-Former (shared, trainable)
        - EntailmentHead (Task E, trainable)
        - FragmentRankingHead (Task S, trainable)
        - CondenseHead (Task C, trainable)
        - Retriever (frozen, placeholder)
        - LLM (frozen, placeholder)
    """
    
    def __init__(self, config: JointTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.global_step = 0
        
        print("="*80)
        print("Initializing Joint Training (E+S+C)")
        print("="*80)
        
        # Initialize Q-Former (shared)
        self.qformer = DRQFormer(
            n_queries=config.n_queries,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
        ).to(self.device)
        
        # Task E: EntailmentHead
        self.head_e = EntailmentHead(
            hidden_dim=config.hidden_dim,
            num_fragments=1000,  # Dynamic K handled by padding
            tau=config.task_e_tau,
            p_drop_lq=0.0,  # Disable internal Drop-LQ (use unified mask)
            focal_gamma=config.task_e_focal_gamma,
            focal_alpha=config.task_e_focal_alpha,
        ).to(self.device)
        
        # Task S: FragmentRankingHead
        self.head_s = FragmentRankingHead(
            num_fragments=1000,  # Dynamic K
            tau_head=config.task_s_tau_head,
            tau_lq=config.task_s_tau_lq,
            rho_top=config.task_s_rho_top,
            l_prime=config.task_s_l_prime,
            p_drop_lq=0.0,  # Disable internal Drop-LQ (use unified mask)
        ).to(self.device)
        
        # Task C: CondenseHead
        self.head_c = CondenseHead(
            hidden_dim=config.hidden_dim,
            llm_hidden_dim=config.task_c_llm_hidden_dim,
            p_drop_lq=0.0,  # Disable internal Drop-LQ (use unified mask)
        ).to(self.device)
        
        # Optimizer (trainable parameters only)
        trainable_params = (
            list(self.qformer.parameters()) +
            list(self.head_e.parameters()) +
            list(self.head_s.parameters()) +
            list(self.head_c.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # LR scheduler
        self.lr_scheduler = get_lr_schedule(
            self.optimizer,
            num_warmup_steps=int(config.max_steps * 0.1),
            num_training_steps=config.max_steps,
        )
        
        # Curriculum scheduler
        if config.schedule_config is None:
            config.schedule_config = ScheduleConfig(max_steps=config.max_steps)
        self.curriculum_scheduler = JointTrainingScheduler(config.schedule_config)
        
        # Print model info
        print(f"✅ Models initialized on {self.device}")
        print(f"   Q-Former: {sum(p.numel() for p in self.qformer.parameters()):,} params")
        print(f"   EntailmentHead: {sum(p.numel() for p in self.head_e.parameters()):,} params")
        print(f"   FragmentRankingHead: {sum(p.numel() for p in self.head_s.parameters()):,} params")
        print(f"   CondenseHead: {sum(p.numel() for p in self.head_c.parameters()):,} params")
        print(f"   Total trainable: {sum(p.numel() for p in trainable_params):,} params")
        print("="*80)
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step (forward + backward).
        
        Flow:
        1. ONE Q-Former forward (shared LQ-drop mask)
        2. Task E: Entailment tagging
        3. Task S: Fragment ranking (with subset U)
        4. Task C: Condensing-generation (with posterior)
        5. Weighted loss combination
        6. Backward + optimizer step
        
        Args:
            batch: Collated batch from joint_data.collate_joint_batch()
        
        Returns:
            metrics: Dict of loss components
        """
        # Get curriculum weights
        weights = self.curriculum_scheduler.get_weights(self.global_step)
        
        # Move batch to device
        queries = batch['queries']  # List[str]
        answers = batch['answers']  # List[str]
        fragments = batch['fragments']  # List[List[str]]
        gt_entailment = batch['gt_entailment'].to(self.device)  # [batch, K]
        is_longtail = batch['is_longtail'].to(self.device)  # [batch, K]
        gt_scores = batch['gt_scores'].to(self.device)  # [batch, K]
        posterior_scores = batch['posterior_scores']  # [batch, K] or None
        if posterior_scores is not None:
            posterior_scores = posterior_scores.to(self.device)
        pool_padding_mask = batch['pool_padding_mask'].to(self.device)  # [batch, K]
        
        batch_size = len(queries)
        K_max = pool_padding_mask.shape[1]
        
        # ========== 1. Encode texts (Placeholder) ==========
        # TODO: Replace with actual retriever adapter
        q_embeds = torch.randn(batch_size, 1, self.config.hidden_dim, device=self.device)
        a_embeds = torch.randn(batch_size, 1, self.config.hidden_dim, device=self.device)
        
        # Flatten fragments and encode
        fragments_flat = [frag for frags in fragments for frag in frags]
        p_embeds_flat = torch.randn(len(fragments_flat), self.config.hidden_dim, device=self.device)
        p_embeds = p_embeds_flat.reshape(batch_size, K_max, self.config.hidden_dim)
        
        # ========== 1.5. Generate Unified Drop-LQ Mask (Training Only) ==========
        lq_drop_mask = None
        p_drop_lq_unified = getattr(self.config, 'p_drop_lq_unified', 0.1)  # Default 0.1
        if self.qformer.training and p_drop_lq_unified > 0:
            # Generate unified mask: True = keep, False = drop
            lq_drop_mask = torch.rand(batch_size, self.config.n_queries, 1, device=self.device) > p_drop_lq_unified
            # Shape: [batch, N_lq, 1] - bool tensor
            
            # Safety: Ensure at least 1 LQ is kept per sample
            all_dropped = (lq_drop_mask.sum(dim=1, keepdim=True) == 0)  # [batch, 1, 1]
            if all_dropped.any():
                for b in range(batch_size):
                    if all_dropped[b, 0, 0]:
                        # Randomly keep one LQ
                        random_idx = torch.randint(0, self.config.n_queries, (1,), device=self.device)
                        lq_drop_mask[b, random_idx, 0] = True
        
        # ========== 2. Q-Former Forward (ONE PASS) ==========
        z, aux = self.qformer(
            query_embeds=q_embeds,
            p_embeds=p_embeds,
            pool_padding_mask=pool_padding_mask,
            lq_drop_mask=lq_drop_mask,  # Pass unified mask
        )
        # z: [batch, N_lq, hidden_dim]
        # aux: Contains 'ca_raw_scores_per_head', 'lq_drop_mask', etc.
        
        ca_raw_scores_per_head = aux.get('ca_raw_scores_per_head', None)
        # lq_drop_mask already generated above (not from aux)
        
        # ========== 3. Task E: Entailment Tagging ==========
        head_e_out = self.head_e(
            z=z,
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=pool_padding_mask,
            training=True,
            lq_drop_mask=lq_drop_mask,  # Use shared mask
        )
        fragment_logits_e = head_e_out['fragment_logits']  # [batch, K]
        
        # Build importance weights (positive + longtail)
        importance_weights = torch.ones_like(gt_entailment)
        importance_weights = torch.where(gt_entailment == 1, 10.0 * torch.ones_like(gt_entailment), importance_weights)
        importance_weights = torch.where((gt_entailment == 1) & (is_longtail == 1), 50.0 * torch.ones_like(gt_entailment), importance_weights)
        
        loss_e = compute_focal_loss(
            logits=fragment_logits_e,
            gt_labels=gt_entailment,
            importance_weights=importance_weights,
            pool_padding_mask=pool_padding_mask,
            focal_gamma=self.config.task_e_focal_gamma,
            focal_alpha=self.config.task_e_focal_alpha,
        )
        
        # Optional LQ entropy reg for Task E
        loss_e_entropy = torch.tensor(0.0, device=self.device)
        if weights['lq_entropy_task_e'] > 0 and ca_raw_scores_per_head is not None:
            loss_e_entropy = compute_lq_entropy_loss(
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                target_ratio=0.5,  # Task E: allow concentration
            )
        
        # ========== 4. Task C: Condensing-Generation (MOVED UP) ==========
        # NOTE: Task C must run BEFORE Task S to extract posterior in same step
        # This enables Bayesian closed-loop: Task C posterior → Task S in same step
        
        # Project Z to LLM dimension (with unified Drop-LQ mask)
        z_prefix = self.head_c(
            z=z,
            lq_drop_mask=lq_drop_mask,  # Use shared mask
            training=True
        )  # [batch, N_lq, d_llm]
        
        # TODO: Replace with actual LLM forward (dual-path teacher forcing)
        # Placeholder: Mock NLL values
        nll_with_evidence = torch.tensor(2.5, device=self.device)
        nll_without_evidence = torch.tensor(3.8, device=self.device)
        llm_attention_to_z = torch.randn(batch_size, 8, 20, self.config.n_queries, device=self.device)  # [B, H, S_a, N]
        answer_start_idx = 10  # Mock
        
        # Extract CA weights for posterior computation
        # ca_raw_scores_per_head: List[[B, H, N, K]] per layer
        # Average over layers and heads → [B, N, K]
        if ca_raw_scores_per_head is not None and len(ca_raw_scores_per_head) > 0:
            ca_scores_stacked = torch.stack(ca_raw_scores_per_head, dim=0)  # [L, B, H, N, K]
            ca_scores_avg = ca_scores_stacked.mean(dim=[0, 2])  # [B, N, K]
            ca_weights = torch.softmax(ca_scores_avg, dim=-1)  # [B, N, K]
        else:
            ca_weights = None
        
        # Build training subset U FIRST (needed for Task C)
        # Use ranking logits from a preliminary forward (or use gt_scores heuristic)
        # For now, use gt_scores to build subset (Task S will refine later)
        train_subset_mask_preliminary = build_train_subset_mask(
            ranking_logits=gt_scores,  # Use teacher scores as preliminary ranking
            gt_scores=gt_scores,
            pool_padding_mask=pool_padding_mask,
            rho_top=self.config.task_s_rho_top,
            l_prime=self.config.task_s_l_prime,
        )
        
        # Get subset indices from train_subset_mask
        subset_indices_list = []
        for b in range(batch_size):
            indices = torch.nonzero(train_subset_mask_preliminary[b], as_tuple=False).squeeze(-1)
            if len(indices) == 0:
                indices = torch.zeros(1, dtype=torch.long, device=self.device)
            subset_indices_list.append(indices)
        
        # Pad to same length (for batching)
        max_subset_size = max(len(idx) for idx in subset_indices_list)
        subset_indices = torch.zeros(batch_size, max_subset_size, dtype=torch.long, device=self.device)
        for b, indices in enumerate(subset_indices_list):
            subset_indices[b, :len(indices)] = indices
        
        # Compute condensing loss
        loss_c_dict = compute_condensing_loss(
            nll_with_evidence=nll_with_evidence,
            nll_without_evidence=nll_without_evidence,
            llm_attention_weights=llm_attention_to_z,
            ca_weights=ca_weights,
            subset_indices=subset_indices,
            answer_start_idx=answer_start_idx,
            softplus_beta=self.config.task_c_softplus_beta,
            margin_mode=self.config.task_c_margin_mode,
            margin_fixed=self.config.task_c_margin_fixed,
            margin_adaptive_ratio=self.config.task_c_margin_adaptive_ratio,
            margin_min=self.config.task_c_margin_min,
            margin_max=self.config.task_c_margin_max,
        )
        loss_c = loss_c_dict['loss_c']
        posterior_q_psi_U = loss_c_dict['posterior_q_psi_U']  # [batch, |U|], detached
        
        # Optional LQ entropy reg for Task C
        loss_c_entropy = torch.tensor(0.0, device=self.device)
        if weights['lq_entropy_task_c'] > 0 and ca_raw_scores_per_head is not None:
            loss_c_entropy = compute_lq_entropy_loss(
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                target_ratio=0.7,
            )
        
        # ========== 5. Expand Posterior to Full K (Same-Step Integration) ==========
        # Scatter posterior_q_psi_U [batch, |U|] → posterior_scores_expanded [batch, K_max]
        posterior_scores_expanded = torch.zeros(batch_size, K_max, device=self.device)
        if weights['w_C'] > 0 and posterior_q_psi_U is not None:
            # Task C active: use extracted posterior
            for b in range(batch_size):
                subset_idx = train_subset_mask_preliminary[b].nonzero(as_tuple=False).squeeze(-1)
                if len(subset_idx) > 0:
                    # Copy posterior values to full tensor
                    posterior_scores_expanded[b, subset_idx] = posterior_q_psi_U[b, :len(subset_idx)]
            posterior_for_task_s = posterior_scores_expanded
        else:
            # Warm-up phase (w_C=0): no posterior available
            posterior_for_task_s = None
        
        # ========== 6. Task S: Fragment Ranking (MOVED DOWN) ==========
        # Now Task S uses same-step posterior from Task C
        head_s_out = self.head_s(
            z=z,
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=pool_padding_mask,
            training=True,
            lq_drop_mask=lq_drop_mask,  # Use shared mask
        )
        ranking_logits = head_s_out['ranking_logits']  # [batch, K]
        
        # Build refined training subset U (using actual ranking logits)
        train_subset_mask = build_train_subset_mask(
            ranking_logits=ranking_logits.detach(),
            gt_scores=gt_scores,
            pool_padding_mask=pool_padding_mask,
            rho_top=self.config.task_s_rho_top,
            l_prime=self.config.task_s_l_prime,
        )
        
        # Compute ranking loss with same-step posterior
        loss_s_dict = compute_ranking_loss(
            ranking_logits=ranking_logits,
            gt_scores=gt_scores,
            posterior_scores=posterior_for_task_s,  # ✅ Same-step posterior from Task C
            pool_padding_mask=pool_padding_mask,
            train_subset_mask=train_subset_mask,
            lambda_teach=weights['lambda_teach'],
            lambda_post=weights['lambda_post'],
            lambda_entropy=weights['lambda_entropy'],
            tau_pred=1.0,
            tau_gt=1.0,
            alpha_gt=0.7,
        )
        loss_s = loss_s_dict['loss']
        
        # Optional LQ entropy reg for Task S
        loss_s_entropy = torch.tensor(0.0, device=self.device)
        if weights['lq_entropy_task_s'] > 0 and ca_raw_scores_per_head is not None:
            loss_s_entropy = compute_lq_entropy_loss(
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                target_ratio=0.7,  # Task S: conservative diversity
            )
        
        # ========== 7. Weighted Loss Combination ==========
        loss_total = (
            weights['w_E'] * (loss_e + weights['lq_entropy_task_e'] * loss_e_entropy) +
            weights['w_S'] * (loss_s + weights['lq_entropy_task_s'] * loss_s_entropy) +
            weights['w_C'] * (loss_c + weights['lq_entropy_task_c'] * loss_c_entropy)
        )
        
        # ========== 7. Backward + Optimizer Step ==========
        self.optimizer.zero_grad()
        loss_total.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.qformer.parameters()) +
            list(self.head_e.parameters()) +
            list(self.head_s.parameters()) +
            list(self.head_c.parameters()),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # ========== 8. Update Step Counter ==========
        self.global_step += 1
        
        # ========== 9. Return Metrics ==========
        return {
            'loss_total': loss_total.item(),
            'loss_e': loss_e.item(),
            'loss_s': loss_s.item(),
            'loss_c': loss_c.item(),
            'loss_s_teach': loss_s_dict['loss_teach'].item(),
            'loss_s_post': loss_s_dict['loss_post'],
            'loss_s_entropy': loss_s_dict['loss_entropy'].item(),
            'nll_gain': loss_c_dict['nll_gain'].item(),
            'margin': loss_c_dict['margin'].item(),
            'w_E': weights['w_E'],
            'w_S': weights['w_S'],
            'w_C': weights['w_C'],
            'lambda_teach': weights['lambda_teach'],
            'lambda_post': weights['lambda_post'],
            'phase': weights['phase'],
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.qformer.train()
        self.head_e.train()
        self.head_s.train()
        self.head_c.train()
        
        epoch_metrics = {}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)
            
            # Accumulate
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss_total']:.4f}",
                'phase': metrics['phase'],
                'step': self.global_step,
            })
            
            if self.global_step >= self.config.max_steps:
                break
        
        # Average
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)
        
        return epoch_metrics


def main():
    """Main entry point for testing."""
    print("="*80)
    print("Joint Training (E+S+C) - Placeholder Test")
    print("="*80)
    
    config = JointTrainingConfig(
        n_queries=16,
        hidden_dim=768,
        num_layers=6,
        max_steps=1000,
        batch_size=4,
    )
    
    trainer = JointTrainer(config)
    
    print("\n✅ Trainer initialized successfully!")
    print("\nTODO:")
    print("  - Integrate retriever adapter (replace mock encoding)")
    print("  - Integrate LLM adapter (replace mock NLL/attention)")
    print("  - Load real datasets")
    print("  - Add evaluation loop")
    print("  - Add checkpoint saving/loading")
    print("  - Add logging (tensorboard/wandb)")
    print("="*80)


if __name__ == "__main__":
    main()
