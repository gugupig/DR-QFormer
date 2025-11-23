"""
TASK E Only Training: Fragment-level Entailment Tagging (Random-Init Q-Former).

This script trains TASK E (entailment classification) using a randomly initialized Q-Former
without any pre-trained weights. This serves as a baseline to compare with XLM-RoBERTa-based Q-Former.

Key Differences from task_e_only.py:
=====================================
- Uses DRQFormer (random init) instead of XLMRobertaDRQFormer (pre-trained)
- Same token-level query embeddings interface as XLM-RoBERTa version
- Always uses pre-computed embeddings (no online tokenization)
- Faster training (no XLM-R backbone)

Architecture:
=============
- Input: Pre-computed token-level query embeddings [T, 768] + evidence embeddings [K, 768]
- Q-Former: Randomly initialized Transformer (32 LQs, 6 layers, 8 heads)
- Output: Knowledge-infused representations [32, 768] → EntailmentHead

Data Format:
============
Each sample in ms_xlm_embeddings.pkl contains:
- query_embedding: dict with 'token_emb_768' [seq_len, 768]
- evidence_embeddings: [K, 768] pre-computed evidence embeddings
- evidence_labels: [K] binary labels (0/1)
"""

import sys
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.qformer_random_init import DRQFormer
from src.models.heads import EntailmentHead
from src.losses import compute_focal_loss
from train.schedule import get_lr_schedule


@dataclass
class TaskERandomConfig:
    """Configuration for TASK E with random-init Q-Former."""
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
    task_e_tau: float = 0.5
    task_e_focal_gamma: float = 1.5
    task_e_focal_alpha: float = 0.85
    task_e_w_pos: float = 1.255
    task_e_w_longtail: float = 100.0
    
    # Unified Drop-LQ
    p_drop_lq_unified: float = 0.0
    
    # Training hyperparameters
    batch_size: int = 16
    num_epochs: int = 10
    max_steps: int = 500000
    lr: float = 1e-5
    weight_decay: float = 0.001
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 5000
    save_dir: str = "./checkpoints/task_e_random"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class RandomQFormerDataset(Dataset):
    """
    Dataset for Random-Init Q-Former training.
    
    Simplified compared to SmokingDataset:
    - Always uses pre-computed embeddings
    - Returns pooled query embedding [768] (mean pooling over tokens)
    - No online tokenization needed
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
                - query_token_embeddings: [seq_len, 768] FloatTensor (token-level)
                - query_attention_mask: [seq_len] LongTensor
                - evidence_embeddings: [K, 768] FloatTensor
                - evidence_labels: [K] FloatTensor
                - sample_id: str
        """
        sample_id = self.sample_ids[idx]
        sample = self.data_dict[sample_id]
        
        # Extract query token embeddings (keep token-level info)
        query_emb = sample['query_embedding']
        token_emb = query_emb['token_emb_768'].squeeze(0)  # [seq_len, 768]
        attention_mask = query_emb['attention_mask'].squeeze(0)  # [seq_len]
        
        # Evidence embeddings and labels
        evidence_embeddings = sample['evidence_embeddings']  # [K, 768]
        evidence_labels = sample['evidence_labels']  # [K]
        
        # Convert to tensors
        result = {
            'query': sample['query'],
            'query_token_embeddings': torch.from_numpy(token_emb).float() if isinstance(token_emb, np.ndarray) else token_emb.float(),
            'query_attention_mask': torch.from_numpy(attention_mask).long() if isinstance(attention_mask, np.ndarray) else attention_mask.long(),
            'evidence_embeddings': torch.from_numpy(evidence_embeddings).float(),
            'evidence_labels': torch.from_numpy(evidence_labels).float(),
            'sample_id': sample_id,
        }
        
        return result


def collate_random_qformer_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for Random Q-Former with dynamic padding for both query tokens and evidence.
    
    Args:
        batch: List of samples from RandomQFormerDataset
    
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
        pool_padding_mask[b, :K_curr] = True
        
        queries.append(sample['query'])
        sample_ids.append(sample['sample_id'])
    
    return {
        'queries': queries,
        'query_token_embeddings': query_token_embeddings,  # [batch, max_T, 768]
        'query_attention_mask': query_attention_mask,  # [batch, max_T]
        'evidence_embeddings': evidence_embeddings,  # [batch, K, 768]
        'evidence_labels': evidence_labels,  # [batch, K]
        'pool_padding_mask': pool_padding_mask,  # [batch, K]
        'sample_ids': sample_ids,
    }


class TaskERandomTrainer:
    """
    Trainer for TASK E with Random-Init Q-Former.
    """
    
    def __init__(self, config: TaskERandomConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.global_step = 0
        
        # Training history
        self.train_history = {
            'epoch': [],
            'loss': [],
        }
        self.val_history = {
            'epoch': [],
            'val_loss': [],
        }
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Initialize models
        print("="*80)
        print("Initializing TASK E Trainer (Random-Init Q-Former)")
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
        
        # Optimizer
        trainable_params = (
            list(self.qformer.parameters()) +
            list(self.head_e.parameters())
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
        Single training step.
        
        Args:
            batch: Collated batch from collate_random_qformer_batch
        
        Returns:
            metrics: Dict of loss components
        """
        # Move to device
        query_token_embeddings = batch['query_token_embeddings'].to(self.device)  # [batch, T, 768]
        query_attention_mask = batch['query_attention_mask'].to(self.device)  # [batch, T]
        evidence_embeddings = batch['evidence_embeddings'].to(self.device)
        evidence_labels = batch['evidence_labels'].to(self.device)
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
        
        # Random Q-Former forward with token-level query embeddings
        # query_token_embeddings: [batch, T, 768] - same as XLM-RoBERTa version
        Z, all_aux = self.qformer(
            query_embeds=query_token_embeddings,
            p_embeds=evidence_embeddings,
            pool_padding_mask=pool_padding_mask,
            lq_drop_mask=lq_drop_mask,
        )
        # Z: [batch, N_lq, hidden_dim]
        
        # Extract CA raw scores from all layers
        ca_raw_scores_per_head = all_aux.get('ca_raw_scores_per_head', [])
        
        # Task E: Entailment Tagging
        head_e_out = self.head_e(
            z=Z,
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=pool_padding_mask,
            lq_drop_mask=lq_drop_mask,
            training=True,
        )
        fragment_logits_e = head_e_out['fragment_logits']  # [batch, K]
        
        # Build importance weights
        importance_weights = torch.ones_like(evidence_labels)
        importance_weights = torch.where(
            evidence_labels == 1,
            self.config.task_e_w_pos * torch.ones_like(evidence_labels),
            importance_weights
        )
        
        # Compute focal loss
        loss = compute_focal_loss(
            logits=fragment_logits_e,
            gt_labels=evidence_labels,
            importance_weights=importance_weights,
            pool_padding_mask=pool_padding_mask,
            focal_gamma=self.config.task_e_focal_gamma,
            focal_alpha=self.config.task_e_focal_alpha,
        )
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.qformer.parameters()) + list(self.head_e.parameters()),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        self.global_step += 1
        
        return {'loss': loss.item()}
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.qformer.train()
        self.head_e.train()
        
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
                'loss': f"{metrics['loss']:.4f}",
                'step': self.global_step,
            })
            
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")
            
            if self.global_step >= self.config.max_steps:
                break
        
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)
        
        self.train_history['epoch'].append(epoch)
        self.train_history['loss'].append(epoch_metrics['loss'])
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.qformer.eval()
        self.head_e.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            query_token_embeddings = batch['query_token_embeddings'].to(self.device)
            query_attention_mask = batch['query_attention_mask'].to(self.device)
            evidence_embeddings = batch['evidence_embeddings'].to(self.device)
            evidence_labels = batch['evidence_labels'].to(self.device)
            pool_padding_mask = batch['pool_padding_mask'].to(self.device)
            
            Z, all_aux = self.qformer(
                query_embeds=query_token_embeddings,
                p_embeds=evidence_embeddings,
                pool_padding_mask=pool_padding_mask,
            )
            
            ca_raw_scores_per_head = all_aux.get('ca_raw_scores_per_head', [])
            
            head_e_out = self.head_e(
                z=Z,
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                training=False,
            )
            fragment_logits_e = head_e_out['fragment_logits']
            
            importance_weights = torch.where(
                evidence_labels == 1,
                self.config.task_e_w_pos * torch.ones_like(evidence_labels),
                torch.ones_like(evidence_labels)
            )
            
            loss = compute_focal_loss(
                logits=fragment_logits_e,
                gt_labels=evidence_labels,
                importance_weights=importance_weights,
                pool_padding_mask=pool_padding_mask,
                focal_gamma=self.config.task_e_focal_gamma,
                focal_alpha=self.config.task_e_focal_alpha,
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'val_loss': total_loss / max(num_batches, 1)}
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint_path = self.save_dir / filename
        torch.save({
            'global_step': self.global_step,
            'qformer_state_dict': self.qformer.state_dict(),
            'head_e_state_dict': self.head_e.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.qformer.load_state_dict(checkpoint['qformer_state_dict'])
        self.head_e.load_state_dict(checkpoint['head_e_state_dict'])
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
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('TASK E Training (Random Q-Former)', fontsize=16, fontweight='bold')
        
        epochs = self.train_history['epoch']
        ax.plot(epochs, self.train_history['loss'], 'b-', label='Train Loss', linewidth=2)
        
        if self.val_history['epoch']:
            ax.plot(self.val_history['epoch'], self.val_history['val_loss'], 
                   'r--', label='Val Loss', linewidth=2, marker='o')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Entailment Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
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
    config = TaskERandomConfig()
    
    print("="*80)
    print("TASK E Training: Random-Init Q-Former (Baseline)")
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
    trainer = TaskERandomTrainer(config)
    
    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = RandomQFormerDataset(train_data[0], train_data[1])
    val_dataset = RandomQFormerDataset(val_data[0], val_data[1])
    print(f"✅ Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_random_qformer_batch,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_random_qformer_batch,
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
