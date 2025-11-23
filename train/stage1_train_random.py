"""
Stage-1 Training: Joint TASK E + TASK S (Random-Init Q-Former).

This script trains both TASK E (entailment classification) and TASK S (fragment ranking)
using a randomly initialized Q-Former without any pre-trained weights. This serves as a
baseline to compare with XLM-RoBERTa-based Q-Former.

Key Differences from stage1_train.py:
======================================
- Uses DRQFormer (random init) instead of XLMRobertaDRQFormer (pre-trained)
- Same token-level query embeddings interface as XLM-RoBERTa version
- Always uses pre-computed embeddings (no online tokenization)
- Faster training (no XLM-R backbone)
- Unified Drop-LQ for multi-task regularization

Architecture:
=============
- Input: Pre-computed token-level query embeddings [T, 768] + evidence embeddings [K, 768]
- Q-Former: Randomly initialized Transformer (32 LQs, 6 layers, 8 heads)
- Output: Knowledge-infused representations [32, 768] → EntailmentHead + FragmentRankingHead

Data Format:
============
Each sample in ms_xlm_embeddings.pkl contains:
- query_embedding: dict with 'token_emb_768' [seq_len, 768]
- evidence_embeddings: [K, 768] pre-computed evidence embeddings
- evidence_labels: [K] binary labels (0/1) for Task E
- evidence_ranking: list of tuples [(idx, score), ...] for Task S
"""

import sys
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.qformer_random_init import DRQFormer
from src.models.heads import EntailmentHead, FragmentRankingHead
from src.losses import compute_focal_loss
from train.schedule import get_lr_schedule


@dataclass
class Stage1RandomConfig:
    """Configuration for Stage-1 training with random-init Q-Former."""
    # Data
    train_data_path: str = r"D:\LLMs\DR-QFormer\DR-QFormer\smoking_64.pkl"
    val_split: float = 0.1
    shuffle_data: bool = True
    
    # Model architecture (Random-Init Q-Former)
    n_queries: int = 32
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # Task E hyperparameters
    task_e_tau: float = 0.3
    task_e_focal_gamma: float = 2.0
    task_e_focal_alpha: float = 0.85
    task_e_w_pos: float = 1.255
    task_e_w_longtail: float = 50.0
    
    # Task S hyperparameters
    task_s_tau_head: float = 1.0
    task_s_tau_lq: float = 1.0
    task_s_rho_top: float = 5.0
    task_s_l_prime: int = 15
    teacher_tau: float = 0.5
    ranking_loss_type: str = "listnet"
    
    # Multi-task loss weights
    w_task_e: float = 0.5
    w_task_s: float = 0.5
    
    # Curriculum learning for Task S
    lambda_teach_start: float = 1.0
    lambda_teach_end: float = 1.0
    lambda_post_start: float = 0.0
    lambda_post_end: float = 0.0
    
    # Unified Drop-LQ
    p_drop_lq_unified: float = 0.0
    
    # Training hyperparameters
    batch_size: int = 8
    num_epochs: int = 10
    max_steps: int = 500000
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 5000
    save_dir: str = "./checkpoints/stage1_random"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class Stage1RandomDataset(Dataset):
    """
    Dataset for Stage-1 training with random-init Q-Former.
    
    Returns both Task E labels and Task S teacher scores.
    """
    
    def __init__(self, data_dict: Dict, sample_ids: List[str]):
        self.data_dict = data_dict
        self.sample_ids = sample_ids
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single sample with token-level query embeddings.
        
        Returns:
            dict with keys:
                - query: str
                - query_token_embeddings: [seq_len, 768] FloatTensor
                - query_attention_mask: [seq_len] LongTensor
                - evidence_embeddings: [K, 768] FloatTensor
                - evidence_labels: [K] FloatTensor (Task E)
                - teacher_scores: [K] FloatTensor (Task S)
                - sample_id: str
        """
        sample_id = self.sample_ids[idx]
        sample = self.data_dict[sample_id]
        
        # Extract query token embeddings
        query_emb = sample['query_embedding']
        token_emb = query_emb['token_emb_768'].squeeze(0)  # [seq_len, 768]
        attention_mask = query_emb['attention_mask'].squeeze(0)  # [seq_len]
        
        # Evidence embeddings and labels
        evidence_embeddings = sample['evidence_embeddings']  # [K, 768]
        evidence_labels = sample['evidence_labels']  # [K]
        
        # Evidence ranking (Task S teacher signals)
        evidence_ranking = sample['evidence_ranking']  # List[(idx, score)]
        
        # Convert ranking to teacher scores
        K = evidence_embeddings.shape[0]
        teacher_scores = np.zeros(K, dtype=np.float32)
        for rank_pos, ranking_item in enumerate(evidence_ranking):
            if isinstance(ranking_item, (tuple, list)) and len(ranking_item) >= 2:
                frag_idx, rerank_score = ranking_item[0], ranking_item[1]
            elif isinstance(ranking_item, (tuple, list)) and len(ranking_item) == 1:
                frag_idx = ranking_item[0]
                rerank_score = 1.0 - (rank_pos / max(len(evidence_ranking), 1))
            else:
                frag_idx = ranking_item
                rerank_score = 1.0 - (rank_pos / max(len(evidence_ranking), 1))
            
            if isinstance(frag_idx, (np.ndarray, np.integer)):
                frag_idx = int(frag_idx)
            elif not isinstance(frag_idx, int):
                try:
                    frag_idx = int(frag_idx)
                except (ValueError, TypeError):
                    continue
            
            if 0 <= frag_idx < K:
                teacher_scores[frag_idx] = float(rerank_score)
        
        # Convert to tensors
        result = {
            'query': sample['query'],
            'query_token_embeddings': torch.from_numpy(token_emb).float() if isinstance(token_emb, np.ndarray) else token_emb.float(),
            'query_attention_mask': torch.from_numpy(attention_mask).long() if isinstance(attention_mask, np.ndarray) else attention_mask.long(),
            'evidence_embeddings': torch.from_numpy(evidence_embeddings).float(),
            'evidence_labels': torch.from_numpy(evidence_labels).float(),
            'teacher_scores': torch.from_numpy(teacher_scores).float(),
            'sample_id': sample_id,
        }
        
        return result


def collate_stage1_random_batch(batch: List[Dict]) -> Dict:
    """
    Collate function with dynamic padding for both query tokens and evidence.
    
    Args:
        batch: List of samples from Stage1RandomDataset
    
    Returns:
        Collated batch dict
    """
    batch_size = len(batch)
    max_K = max(sample['evidence_embeddings'].shape[0] for sample in batch)
    max_T = max(sample['query_token_embeddings'].shape[0] for sample in batch)
    
    # Initialize padded tensors
    query_token_embeddings = torch.zeros(batch_size, max_T, 768, dtype=torch.float32)
    query_attention_mask = torch.zeros(batch_size, max_T, dtype=torch.long)
    evidence_embeddings = torch.zeros(batch_size, max_K, 768, dtype=torch.float32)
    evidence_labels = torch.zeros(batch_size, max_K, dtype=torch.float32)
    teacher_scores = torch.zeros(batch_size, max_K, dtype=torch.float32)
    pool_padding_mask = torch.zeros(batch_size, max_K, dtype=torch.bool)
    
    queries = []
    sample_ids = []
    
    for b, sample in enumerate(batch):
        # Query tokens
        T_curr = sample['query_token_embeddings'].shape[0]
        query_token_embeddings[b, :T_curr] = sample['query_token_embeddings']
        query_attention_mask[b, :T_curr] = sample['query_attention_mask']
        
        # Evidence
        K_curr = sample['evidence_embeddings'].shape[0]
        evidence_embeddings[b, :K_curr] = sample['evidence_embeddings']
        evidence_labels[b, :K_curr] = sample['evidence_labels']
        teacher_scores[b, :K_curr] = sample['teacher_scores']
        pool_padding_mask[b, :K_curr] = True
        
        queries.append(sample['query'])
        sample_ids.append(sample['sample_id'])
    
    return {
        'queries': queries,
        'query_token_embeddings': query_token_embeddings,  # [batch, max_T, 768]
        'query_attention_mask': query_attention_mask,  # [batch, max_T]
        'evidence_embeddings': evidence_embeddings,  # [batch, K, 768]
        'evidence_labels': evidence_labels,  # [batch, K]
        'teacher_scores': teacher_scores,  # [batch, K]
        'pool_padding_mask': pool_padding_mask,  # [batch, K]
        'sample_ids': sample_ids,
    }


def compute_ranking_loss_random(
    ranking_logits: torch.Tensor,
    teacher_scores: torch.Tensor,
    pool_padding_mask: torch.Tensor,
    loss_type: str = "listnet",
    teacher_tau: float = 0.5,
) -> torch.Tensor:
    """
    Compute ranking loss using teacher scores.
    
    Args:
        ranking_logits: [batch, K] predicted ranking scores
        teacher_scores: [batch, K] teacher scores from reranker
        pool_padding_mask: [batch, K] bool mask (True=valid, False=padding)
        loss_type: "listnet" or "listmle"
        teacher_tau: Temperature for teacher distribution
    
    Returns:
        loss: Scalar ranking loss
    """
    # Check for valid samples (at least one non-padding fragment)
    valid_mask = pool_padding_mask.sum(dim=-1) > 0  # [batch]
    if not valid_mask.any():
        return torch.tensor(0.0, device=ranking_logits.device, requires_grad=True)
    
    # Mask out padding
    ranking_logits = ranking_logits.masked_fill(~pool_padding_mask, -1e9)
    teacher_scores = teacher_scores.masked_fill(~pool_padding_mask, -1e9)
    
    if loss_type == "listnet":
        # ListNet: KL divergence between distributions
        log_pred_dist = F.log_softmax(ranking_logits, dim=-1)  # [batch, K]
        teacher_dist = F.softmax(teacher_scores / teacher_tau, dim=-1)  # [batch, K]
        
        # Check for NaN in teacher distribution
        if torch.isnan(teacher_dist).any():
            valid_counts = pool_padding_mask.sum(dim=-1, keepdim=True).float()  # [batch, 1]
            teacher_dist = pool_padding_mask.float() / valid_counts.clamp(min=1.0)
        
        # KL(teacher || pred)
        loss = F.kl_div(
            log_pred_dist,
            teacher_dist,
            reduction='none',
            log_target=False
        )  # [batch, K]
        
        loss = loss.sum(dim=-1)  # [batch]
        loss = (loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
        
    elif loss_type == "listmle":
        # ListMLE: Use MSE between normalized scores
        loss = F.mse_loss(
            ranking_logits[pool_padding_mask],
            teacher_scores[pool_padding_mask] / teacher_tau,
            reduction='mean'
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    # Check for NaN
    if torch.isnan(loss):
        print("⚠️  Warning: NaN ranking loss detected, returning 0.0")
        return torch.tensor(0.0, device=ranking_logits.device, requires_grad=True)
    
    return loss


class Stage1RandomTrainer:
    """
    Trainer for Stage-1 with Random-Init Q-Former (Joint Task E + Task S).
    """
    
    def __init__(self, config: Stage1RandomConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.global_step = 0
        
        # Training history
        self.train_history = {
            'epoch': [],
            'total_loss': [],
            'task_e_loss': [],
            'task_s_loss': [],
        }
        self.val_history = {
            'epoch': [],
            'val_loss': [],
            'val_task_e_loss': [],
            'val_task_s_loss': [],
        }
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Initialize models
        print("="*80)
        print("Initializing Stage-1 Trainer (Random-Init Q-Former)")
        print("="*80)
        
        # Random-Init Q-Former
        print("\n🎲 Initializing Random Q-Former (no pre-trained weights)...")
        self.qformer = DRQFormer(
            n_queries=config.n_queries,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        ).to(self.device)
        
        # Task E: EntailmentHead
        self.head_e = EntailmentHead(
            hidden_dim=config.hidden_dim,
            num_fragments=20,  # Max K
            tau=config.task_e_tau,
            p_drop_lq=0.0,
            focal_gamma=config.task_e_focal_gamma,
            focal_alpha=config.task_e_focal_alpha,
        ).to(self.device)
        
        # Task S: FragmentRankingHead
        self.head_s = FragmentRankingHead(
            hidden_dim=config.hidden_dim,
            num_fragments=20,  # Max K
            tau_head=config.task_s_tau_head,
            tau_lq=config.task_s_tau_lq,
            rho_top=config.task_s_rho_top,
            l_prime=config.task_s_l_prime,
            p_drop_lq=0.0,
        ).to(self.device)
        
        # Optimizer
        trainable_params = (
            list(self.qformer.parameters()) +
            list(self.head_e.parameters()) +
            list(self.head_s.parameters())
        )
        self.optimizer = AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # LR scheduler
        num_warmup_steps = int(config.max_steps * config.warmup_ratio)
        self.lr_scheduler = get_lr_schedule(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=config.max_steps,
        )
        
        # Checkpoint directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n✅ Models initialized on {self.device}")
        print(f"   Q-Former (Random): {sum(p.numel() for p in self.qformer.parameters()):,} params")
        print(f"   EntailmentHead: {sum(p.numel() for p in self.head_e.parameters()):,} params")
        print(f"   FragmentRankingHead: {sum(p.numel() for p in self.head_s.parameters()):,} params")
        print(f"   Total trainable: {sum(p.numel() for p in trainable_params):,} params")
        print("="*80)
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step (joint Task E + Task S).
        
        Args:
            batch: Collated batch from collate_stage1_random_batch
        
        Returns:
            metrics: Dict of loss components
        """
        # Move to device
        query_token_embeddings = batch['query_token_embeddings'].to(self.device)
        query_attention_mask = batch['query_attention_mask'].to(self.device)
        evidence_embeddings = batch['evidence_embeddings'].to(self.device)
        evidence_labels = batch['evidence_labels'].to(self.device)
        teacher_scores = batch['teacher_scores'].to(self.device)
        pool_padding_mask = batch['pool_padding_mask'].to(self.device)
        
        batch_size = query_token_embeddings.shape[0]
        
        # Generate unified Drop-LQ mask
        lq_drop_mask = None
        if self.qformer.training and self.config.p_drop_lq_unified > 0:
            n_queries = self.config.n_queries
            lq_drop_mask = torch.rand(batch_size, n_queries, 1, device=self.device) > self.config.p_drop_lq_unified
            # Ensure at least 1 LQ kept per sample
            all_dropped = (lq_drop_mask.sum(dim=1, keepdim=True) == 0)
            if all_dropped.any():
                for b in range(batch_size):
                    if all_dropped[b, 0, 0]:
                        random_idx = torch.randint(0, n_queries, (1,), device=self.device)
                        lq_drop_mask[b, random_idx, 0] = True
        
        # Random Q-Former forward
        Z, all_aux = self.qformer(
            query_embeds=query_token_embeddings,
            p_embeds=evidence_embeddings,
            pool_padding_mask=pool_padding_mask,
            lq_drop_mask=lq_drop_mask,
        )
        # Z: [batch, N_lq, hidden_dim]
        
        # Extract CA raw scores from all layers
        ca_raw_scores_per_head = all_aux.get('ca_raw_scores_per_head', [])
        
        # Task E: Fragment Entailment
        head_e_out = self.head_e(
            z=Z,
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=pool_padding_mask,
            lq_drop_mask=lq_drop_mask,
            training=True,
        )
        logits_e = head_e_out['fragment_logits']  # [batch, K]
        
        # Build importance weights
        importance_weights = torch.ones_like(evidence_labels)
        importance_weights = torch.where(
            evidence_labels == 1,
            self.config.task_e_w_pos * torch.ones_like(evidence_labels),
            importance_weights
        )
        
        # Task E loss
        loss_e = compute_focal_loss(
            logits=logits_e,
            gt_labels=evidence_labels,
            importance_weights=importance_weights,
            pool_padding_mask=pool_padding_mask,
            focal_gamma=self.config.task_e_focal_gamma,
            focal_alpha=self.config.task_e_focal_alpha,
        )
        
        # Task S: Fragment Ranking
        head_s_out = self.head_s(
            z=Z,
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=pool_padding_mask,
            lq_drop_mask=lq_drop_mask,
            training=True,
        )
        ranking_logits = head_s_out['ranking_logits']  # [batch, K]
        
        # Task S loss
        loss_s = compute_ranking_loss_random(
            ranking_logits=ranking_logits,
            teacher_scores=teacher_scores,
            pool_padding_mask=pool_padding_mask,
            loss_type=self.config.ranking_loss_type,
            teacher_tau=self.config.teacher_tau,
        )
        
        # Total loss
        total_loss = (
            self.config.w_task_e * loss_e +
            self.config.w_task_s * loss_s
        )
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.qformer.parameters()) + 
            list(self.head_e.parameters()) + 
            list(self.head_s.parameters()),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        self.global_step += 1
        
        return {
            'total_loss': total_loss.item(),
            'task_e_loss': loss_e.item(),
            'task_s_loss': loss_s.item(),
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.qformer.train()
        self.head_e.train()
        self.head_s.train()
        
        epoch_metrics = {}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)
            
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            num_batches += 1
            
            pbar.set_postfix({
                'total': f"{metrics['total_loss']:.4f}",
                'E': f"{metrics['task_e_loss']:.4f}",
                'S': f"{metrics['task_s_loss']:.4f}",
                'step': self.global_step,
            })
            
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")
            
            if self.global_step >= self.config.max_steps:
                break
        
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)
        
        self.train_history['epoch'].append(epoch)
        self.train_history['total_loss'].append(epoch_metrics['total_loss'])
        self.train_history['task_e_loss'].append(epoch_metrics['task_e_loss'])
        self.train_history['task_s_loss'].append(epoch_metrics['task_s_loss'])
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.qformer.eval()
        self.head_e.eval()
        self.head_s.eval()
        
        total_loss = 0.0
        total_loss_e = 0.0
        total_loss_s = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            query_token_embeddings = batch['query_token_embeddings'].to(self.device)
            query_attention_mask = batch['query_attention_mask'].to(self.device)
            evidence_embeddings = batch['evidence_embeddings'].to(self.device)
            evidence_labels = batch['evidence_labels'].to(self.device)
            teacher_scores = batch['teacher_scores'].to(self.device)
            pool_padding_mask = batch['pool_padding_mask'].to(self.device)
            
            Z, all_aux = self.qformer(
                query_embeds=query_token_embeddings,
                p_embeds=evidence_embeddings,
                pool_padding_mask=pool_padding_mask,
            )
            
            ca_raw_scores_per_head = all_aux.get('ca_raw_scores_per_head', [])
            
            # Task E
            head_e_out = self.head_e(
                z=Z,
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                training=False,
            )
            logits_e = head_e_out['fragment_logits']
            
            # Build importance weights
            importance_weights = torch.ones_like(evidence_labels)
            importance_weights = torch.where(
                evidence_labels == 1,
                self.config.task_e_w_pos * torch.ones_like(evidence_labels),
                importance_weights
            )
            
            loss_e = compute_focal_loss(
                logits=logits_e,
                gt_labels=evidence_labels,
                importance_weights=importance_weights,
                pool_padding_mask=pool_padding_mask,
                focal_gamma=self.config.task_e_focal_gamma,
                focal_alpha=self.config.task_e_focal_alpha,
            )
            
            # Task S
            head_s_out = self.head_s(
                z=Z,
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                training=False,
            )
            ranking_logits = head_s_out['ranking_logits']
            
            loss_s = compute_ranking_loss_random(
                ranking_logits=ranking_logits,
                teacher_scores=teacher_scores,
                pool_padding_mask=pool_padding_mask,
                loss_type=self.config.ranking_loss_type,
                teacher_tau=self.config.teacher_tau,
            )
            
            total_loss += (self.config.w_task_e * loss_e + self.config.w_task_s * loss_s).item()
            total_loss_e += loss_e.item()
            total_loss_s += loss_s.item()
            num_batches += 1
        
        return {
            'val_loss': total_loss / max(num_batches, 1),
            'val_task_e_loss': total_loss_e / max(num_batches, 1),
            'val_task_s_loss': total_loss_s / max(num_batches, 1),
        }
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint_path = self.save_dir / filename
        torch.save({
            'global_step': self.global_step,
            'qformer_state_dict': self.qformer.state_dict(),
            'head_e_state_dict': self.head_e.state_dict(),
            'head_s_state_dict': self.head_s.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.qformer.load_state_dict(checkpoint['qformer_state_dict'])
        self.head_e.load_state_dict(checkpoint['head_e_state_dict'])
        self.head_s.load_state_dict(checkpoint['head_s_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        print(f"📂 Checkpoint loaded: {checkpoint_path}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        if not self.train_history['epoch']:
            print("⚠️  No training history to plot")
            return
        
        if save_path is None:
            save_path = self.save_dir / "training_curves.png"
        else:
            save_path = Path(save_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Stage-1 Training (Random Q-Former)', fontsize=16, fontweight='bold')
        
        epochs = self.train_history['epoch']
        
        # Total loss
        axes[0].plot(epochs, self.train_history['total_loss'], 'b-', label='Train Total', linewidth=2)
        if self.val_history['epoch']:
            axes[0].plot(self.val_history['epoch'], self.val_history['val_loss'], 
                        'r--', label='Val Total', linewidth=2, marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Task E loss
        axes[1].plot(epochs, self.train_history['task_e_loss'], 'g-', label='Train E', linewidth=2)
        if self.val_history['epoch']:
            axes[1].plot(self.val_history['epoch'], self.val_history['val_task_e_loss'], 
                        'orange', linestyle='--', label='Val E', linewidth=2, marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Task E Loss (Entailment)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Task S loss
        axes[2].plot(epochs, self.train_history['task_s_loss'], 'm-', label='Train S', linewidth=2)
        if self.val_history['epoch']:
            axes[2].plot(self.val_history['epoch'], self.val_history['val_task_s_loss'], 
                        'cyan', linestyle='--', label='Val S', linewidth=2, marker='o')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Task S Loss (Ranking)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Training curves saved to: {save_path}")


def load_and_split_data(
    data_path: str,
    val_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[Tuple, Tuple]:
    """Load pickle data and split into train/val."""
    print(f"\n📂 Loading data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    print(f"✅ Loaded {len(data_dict)} samples")
    
    sample_ids = list(data_dict.keys())
    
    if shuffle:
        random.seed(seed)
        random.shuffle(sample_ids)
    
    split_idx = int(len(sample_ids) * (1 - val_split))
    train_ids = sample_ids[:split_idx]
    val_ids = sample_ids[split_idx:]
    
    print(f"📊 Split: {len(train_ids)} train, {len(val_ids)} val")
    
    return (data_dict, train_ids), (data_dict, val_ids)


def main():
    """Main training loop."""
    config = Stage1RandomConfig()
    
    print("="*80)
    print("Stage-1 Training: Random-Init Q-Former (Task E + Task S)")
    print("="*80)
    print("\nConfiguration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Load data
    train_data, val_data = load_and_split_data(
        config.train_data_path,
        val_split=config.val_split,
        shuffle=config.shuffle_data,
        seed=config.seed,
    )
    
    # Initialize trainer
    trainer = Stage1RandomTrainer(config)
    
    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = Stage1RandomDataset(train_data[0], train_data[1])
    val_dataset = Stage1RandomDataset(val_data[0], val_data[1])
    print(f"✅ Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_stage1_random_batch,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_stage1_random_batch,
        num_workers=0,
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training (Random-Init Q-Former)")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*80}")
        
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"\n📈 Train Metrics:")
        for key, value in train_metrics.items():
            print(f"   {key}: {value:.4f}")
        
        if (epoch + 1) % max(1, config.num_epochs // 5) == 0:
            val_metrics = trainer.evaluate(val_loader)
            print(f"\n📊 Validation Metrics:")
            for key, value in val_metrics.items():
                print(f"   {key}: {value:.4f}")
            
            trainer.val_history['epoch'].append(epoch)
            trainer.val_history['val_loss'].append(val_metrics['val_loss'])
            trainer.val_history['val_task_e_loss'].append(val_metrics['val_task_e_loss'])
            trainer.val_history['val_task_s_loss'].append(val_metrics['val_task_s_loss'])
            
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                trainer.save_checkpoint("best.pt")
                print(f"✨ New best model! Val loss: {best_val_loss:.4f}")
        
        if trainer.global_step >= config.max_steps:
            print(f"\n🛑 Reached max_steps ({config.max_steps}), stopping training")
            break
    
    print("\n" + "="*80)
    print("✅ Training completed!")
    print(f"💾 Checkpoints saved in: {config.save_dir}")
    print("="*80)
    
    print("\n📈 Generating training curves...")
    trainer.plot_training_curves()
    print("✅ Training curves generated!")


if __name__ == "__main__":
    main()
