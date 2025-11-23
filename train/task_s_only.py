"""
TASK S Only Training: Fragment Ranking with Teacher Signals.

This script trains only TASK S (fragment ranking) without TASK E (entailment).
Uses teacher signals from reranking scores in the PKL file.

Key Features:
=============
- ONLY trains FragmentRankingHead (TASK S)
- NO EntailmentHead (TASK E)
- Uses reranking scores as teacher signals
- Simpler loss computation (only ranking loss)
- Supports pre-computed embeddings

Data Format:
============
Each sample in the PKL file contains:
- query: str (query text)
- query_embedding: dict with keys ['input_ids', 'attention_mask', 'token_emb_768']
  - input_ids: [1, seq_len] token IDs
  - attention_mask: [1, seq_len] attention mask
  - token_emb_768: [1, seq_len, 768] pre-computed token embeddings
- answer: str (answer text)
- evidence_text: list of strings (fragment texts)
- evidence_embeddings: ndarray [K, 768] pre-computed evidence embeddings
- evidence_ranking: list of tuples [(idx, score), ...] reranking scores (teacher signals)

Training Strategy:
==================
- Use reranking scores as soft labels for ranking loss
- Temperature scaling for knowledge distillation
- Support both ListNet and ListMLE losses
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModel

from src.models.qformer_xlm import XLMRobertaDRQFormer
from src.models.heads import FragmentRankingHead
from train.schedule import get_lr_schedule


@dataclass
class TaskSConfig:
    """Configuration for TASK S only training."""
    # Data
    train_data_path: str = r"D:\LLMs\DR-QFormer\DR-QFormer\ms_xlm_embeddings.pkl"
    val_split: float = 0.1  # Validation split ratio
    shuffle_data: bool = True
    
    # Embedding options
    use_precomputed_embeddings: bool = True  # Use pre-computed token embeddings from PKL
    
    # Model architecture
    xlm_model_name: str = "xlm-roberta-base"  # hidden_dim and num_heads auto-detected from model
    n_queries: int = 32
    use_ca_layers: Optional[List[int]] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10])
    freeze_xlmr: bool = False
    
    # Task S hyperparameters
    task_s_tau: float = 1.0  # Temperature for ranking scores
    teacher_tau: float = 0.5  # Temperature for teacher scores (lower = sharper distribution)
                               # Since reranker outputs are already softmax-normalized probabilities,
                               # using tau < 1.0 helps create a more peaked distribution for better learning signal
    ranking_loss_type: str = "listnet"  # "listnet" or "listmle"
    
    # Unified Drop-LQ
    p_drop_lq_unified: float = 0.0
    
    # Training hyperparameters
    batch_size: int = 16
    num_epochs: int = 10
    max_steps: Optional[int] = None  # Auto-calculate if None
    lr: float = 5e-5
    weight_decay: float = 0.001
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    save_dir: str = "./checkpoints/task_s_only"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def compute_ranking_loss(
    logits: torch.Tensor,
    teacher_scores: torch.Tensor,
    pool_padding_mask: torch.Tensor,
    tau: float = 1.0,
    teacher_tau: float = 0.5,
    loss_type: str = "listnet"
) -> torch.Tensor:
    """
    Compute ranking loss using teacher signals (reranking scores).
    
    Args:
        logits: Model predictions [B, K]
        teacher_scores: Teacher reranking scores [B, K]
        pool_padding_mask: Valid fragments mask [B, K]
        tau: Temperature for student logits
        teacher_tau: Temperature for teacher scores (knowledge distillation)
        loss_type: "listnet" or "listmle"
    
    Returns:
        loss: Scalar ranking loss
    """
    batch_size, K = logits.shape
    
    # Mask out padding positions (set to very negative value)
    masked_logits = logits.clone()
    masked_teacher = teacher_scores.clone()
    masked_logits[~pool_padding_mask] = -1e9
    masked_teacher[~pool_padding_mask] = -1e9
    
    if loss_type == "listnet":
        # ListNet: KL divergence between probability distributions
        # Student distribution
        student_probs = F.softmax(masked_logits / tau, dim=-1)  # [B, K]
        
        # Teacher distribution
        teacher_probs = F.softmax(masked_teacher / teacher_tau, dim=-1)  # [B, K]
        
        # KL divergence: KL(teacher || student)
        # Note: Use log_softmax for numerical stability
        log_student_probs = F.log_softmax(masked_logits / tau, dim=-1)
        
        # KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        kl_div = F.kl_div(
            log_student_probs,
            teacher_probs,
            reduction='none'
        )  # [B, K]
        
        # Mask and average
        kl_div = kl_div * pool_padding_mask.float()
        num_valid = pool_padding_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        loss = (kl_div.sum(dim=-1) / num_valid.squeeze(-1)).mean()
        
    elif loss_type == "listmle":
        # ListMLE: Maximum likelihood estimation for ranking
        # Compute log probability of teacher ranking order
        
        # Sort by teacher scores (descending)
        sorted_teacher_scores, sorted_indices = torch.sort(
            masked_teacher, dim=-1, descending=True
        )  # [B, K]
        
        # Gather student logits in teacher's ranking order
        batch_indices = torch.arange(batch_size, device=logits.device).unsqueeze(-1).expand(-1, K)
        sorted_logits = masked_logits[batch_indices, sorted_indices]  # [B, K]
        
        # Compute log likelihood
        # P(ranking) = product_{i=1}^{K} exp(s_i) / sum_{j=i}^{K} exp(s_j)
        # log P = sum_{i=1}^{K} [s_i - log(sum_{j=i}^{K} exp(s_j))]
        
        # Compute cumulative logsumexp from right to left
        # logsumexp_{j>=i} = logsumexp(s_i, s_{i+1}, ..., s_K)
        cumsum_logits = torch.logcumsumexp(sorted_logits.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        
        # Log likelihood for each position
        log_likelihood = sorted_logits - cumsum_logits  # [B, K]
        
        # Mask and average
        sorted_mask = pool_padding_mask[batch_indices, sorted_indices]
        log_likelihood = log_likelihood * sorted_mask.float()
        num_valid = sorted_mask.sum(dim=-1).float().clamp(min=1.0)
        loss = -(log_likelihood.sum(dim=-1) / num_valid).mean()
        
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss


class RankingDataset(Dataset):
    """
    Dataset for TASK S training with teacher signals.
    
    Loads pre-computed embeddings and reranking scores from PKL.
    """
    
    def __init__(
        self, 
        data_dict: Dict, 
        sample_ids: List[str], 
        tokenizer: Optional[AutoTokenizer] = None,
        xlm_model: Optional[AutoModel] = None, 
        device: str = 'cpu',
        max_query_len: int = 512, 
        use_precomputed_embeddings: bool = True
    ):
        """
        Args:
            data_dict: Dictionary loaded from pickle file
            sample_ids: List of sample IDs to use (for train/val split)
            tokenizer: XLM-RoBERTa tokenizer (only needed if use_precomputed_embeddings=False)
            xlm_model: XLM-RoBERTa model for generating embeddings (only needed if use_precomputed_embeddings=False)
            device: Device for XLM model inference
            max_query_len: Maximum query sequence length
            use_precomputed_embeddings: If True, use pre-computed embeddings from PKL
        """
        self.data_dict = data_dict
        self.sample_ids = sample_ids
        self.tokenizer = tokenizer
        self.xlm_model = xlm_model
        self.device = device
        self.max_query_len = max_query_len
        self.use_precomputed_embeddings = use_precomputed_embeddings
        
        if not use_precomputed_embeddings and (tokenizer is None or xlm_model is None):
            raise ValueError("tokenizer and xlm_model must be provided when use_precomputed_embeddings=False")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single sample with all required fields.
        
        Returns:
            dict with keys:
                - query: str
                - query_input_ids: [seq_len] LongTensor
                - query_attention_mask: [seq_len] LongTensor
                - query_token_embeddings: [seq_len, 768] FloatTensor (if precomputed)
                - evidence_embeddings: [K, 768] FloatTensor
                - teacher_scores: [K] FloatTensor (reranking scores)
                - sample_id: str
        """
        sample_id = self.sample_ids[idx]
        sample = self.data_dict[sample_id]
        
        # Query encoding
        if self.use_precomputed_embeddings:
            query_emb = sample['query_embedding']
            query_input_ids = query_emb['input_ids'].squeeze(0)
            query_attention_mask = query_emb['attention_mask'].squeeze(0)
            query_token_embeddings = query_emb['token_emb_768'].squeeze(0)
        else:
            query_text = sample['query']
            query_encoded = self.tokenizer(
                query_text,
                padding=False,
                truncation=True,
                max_length=self.max_query_len,
                return_tensors='pt',
            )
            query_input_ids = query_encoded['input_ids'].squeeze(0)
            query_attention_mask = query_encoded['attention_mask'].squeeze(0)
            
            with torch.no_grad():
                self.xlm_model.eval()
                encoded_input = {
                    'input_ids': query_encoded['input_ids'].to(self.device),
                    'attention_mask': query_encoded['attention_mask'].to(self.device)
                }
                output = self.xlm_model(**encoded_input)
                query_token_embeddings = output.last_hidden_state.squeeze(0).cpu()
        
        # Evidence embeddings
        if self.use_precomputed_embeddings:
            evidence_embeddings = sample['evidence_embeddings']  # [K, 768]
        else:
            evidence_texts = sample['evidence_text']
            K = len(evidence_texts)
            evidence_embeddings = np.zeros((K, 768), dtype=np.float32)
            
            with torch.no_grad():
                self.xlm_model.eval()
                for i, text in enumerate(evidence_texts):
                    if text:
                        encoded = self.tokenizer(
                            text,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt'
                        )
                        encoded_input = {
                            'input_ids': encoded['input_ids'].to(self.device),
                            'attention_mask': encoded['attention_mask'].to(self.device)
                        }
                        output = self.xlm_model(**encoded_input)
                        cls_embedding = output.last_hidden_state[:, 0, :]
                        evidence_embeddings[i] = cls_embedding.squeeze(0).cpu().numpy()
        
        # Extract teacher scores from evidence_ranking
        # evidence_ranking format: List[(idx, score), ...] or List[(text, score), ...]
        evidence_ranking = sample.get('evidence_ranking', [])
        K = len(evidence_embeddings)
        teacher_scores = np.zeros(K, dtype=np.float32)
        
        # Parse ranking format
        if evidence_ranking:
            for item in evidence_ranking:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    idx_or_text, score = item[0], item[1]
                    
                    # Determine if idx_or_text is an index or text
                    if isinstance(idx_or_text, (int, np.integer)):
                        # Direct index
                        idx = int(idx_or_text)
                        if 0 <= idx < K:
                            teacher_scores[idx] = float(score)
                    elif isinstance(idx_or_text, str):
                        # Text-based, need to match with evidence_text
                        evidence_texts = sample.get('evidence_text', [])
                        try:
                            idx = evidence_texts.index(idx_or_text)
                            if 0 <= idx < K:
                                teacher_scores[idx] = float(score)
                        except (ValueError, AttributeError):
                            # Text not found, skip
                            pass
            
            # NOTE: Teacher scores from reranker are already list-wise softmax normalized
            # (sum to 1.0, represent probability distribution)
            # No need for additional z-score normalization as it would destroy the distribution
            # 
            # If the distribution is too flat (entropy too high), consider:
            # 1. Using lower teacher_tau in loss computation (e.g., 0.5 instead of 1.0)
            # 2. Using raw scores (normalize=False in reranker) instead of probabilities
            #teacher_scores = np.log(np.clip(teacher_scores, 1e-8, 1.0))  # Keep teacher_scores as-is (already normalized probabilities)
            pass
        
        # Convert to tensors
        result = {
            'query': sample['query'],
            'query_input_ids': torch.from_numpy(query_input_ids) if isinstance(query_input_ids, np.ndarray) else query_input_ids.long(),
            'query_attention_mask': torch.from_numpy(query_attention_mask) if isinstance(query_attention_mask, np.ndarray) else query_attention_mask.long(),
            'evidence_embeddings': torch.from_numpy(evidence_embeddings).float(),
            'teacher_scores': torch.from_numpy(teacher_scores).float(),
            'sample_id': sample_id,
        }
        
        if self.use_precomputed_embeddings:
            result['query_token_embeddings'] = torch.from_numpy(query_token_embeddings) if isinstance(query_token_embeddings, np.ndarray) else query_token_embeddings.float()
        
        return result


def collate_task_s_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for TASK S batches with dynamic K padding.
    
    Args:
        batch: List of samples from RankingDataset
    
    Returns:
        Collated batch dict with padded tensors
    """
    batch_size = len(batch)
    
    # Find max dimensions
    max_seq_len = max(sample['query_input_ids'].shape[0] for sample in batch)
    max_K = max(sample['evidence_embeddings'].shape[0] for sample in batch)
    
    use_precomputed = 'query_token_embeddings' in batch[0]
    
    # Initialize padded tensors
    query_input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    query_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    if use_precomputed:
        query_token_embeddings = torch.zeros(batch_size, max_seq_len, 768, dtype=torch.float32)
    evidence_embeddings = torch.zeros(batch_size, max_K, 768, dtype=torch.float32)
    teacher_scores = torch.zeros(batch_size, max_K, dtype=torch.float32)
    pool_padding_mask = torch.zeros(batch_size, max_K, dtype=torch.bool)
    
    queries = []
    sample_ids = []
    
    # Fill tensors
    for b, sample in enumerate(batch):
        seq_len = sample['query_input_ids'].shape[0]
        query_input_ids[b, :seq_len] = sample['query_input_ids']
        query_attention_mask[b, :seq_len] = sample['query_attention_mask']
        if use_precomputed:
            query_token_embeddings[b, :seq_len] = sample['query_token_embeddings']
        
        K_curr = sample['evidence_embeddings'].shape[0]
        evidence_embeddings[b, :K_curr] = sample['evidence_embeddings']
        teacher_scores[b, :K_curr] = sample['teacher_scores']
        pool_padding_mask[b, :K_curr] = True
        
        queries.append(sample['query'])
        sample_ids.append(sample['sample_id'])
    
    result = {
        'queries': queries,
        'query_input_ids': query_input_ids,
        'query_attention_mask': query_attention_mask,
        'evidence_embeddings': evidence_embeddings,
        'teacher_scores': teacher_scores,
        'pool_padding_mask': pool_padding_mask,
        'sample_ids': sample_ids,
    }
    
    if use_precomputed:
        result['query_token_embeddings'] = query_token_embeddings
    
    return result


class TaskSTrainer:
    """
    Trainer for TASK S only: Fragment Ranking with Teacher Signals.
    """
    
    def __init__(self, config: TaskSConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.global_step = 0
        
        # Initialize tokenizer/model (only if needed)
        if config.use_precomputed_embeddings:
            print("⚡ Using pre-computed token embeddings from PKL (bypassing XLM-R tokenizer/embeddings)")
            self.tokenizer = None
            self.xlm_embedding_model = None
        else:
            print(f"Loading tokenizer and model for embedding generation: {config.xlm_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(config.xlm_model_name)
            self.xlm_embedding_model = AutoModel.from_pretrained(config.xlm_model_name).to(self.device)
            self.xlm_embedding_model.eval()
            print(f"✅ XLM-RoBERTa model loaded for embedding generation")
        
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
        print("Initializing TASK S Only Trainer (Fragment Ranking)")
        print("="*80)
        
        # Q-Former (BLIP-2 Style)
        # Note: hidden_dim and num_heads are automatically obtained from XLM-R config
        # Cross-attention layers are automatically initialized with pre-trained weights (BLIP-2 paradigm)
        self.qformer = XLMRobertaDRQFormer(
            xlm_model_name=config.xlm_model_name,
            n_queries=config.n_queries,
            dropout=0.1,
            use_ca_layers=config.use_ca_layers,
            freeze_xlmr=config.freeze_xlmr,
            bypass_embeddings=config.use_precomputed_embeddings,
        ).to(self.device)
        
        # Task S: FragmentRankingHead
        self.head_s = FragmentRankingHead(
            hidden_dim=self.qformer.hidden_dim,  # Get from Q-Former
            num_fragments=20,  # Max K (will handle dynamic K via padding mask)
            p_drop_lq=0.0,  # Use unified Drop-LQ
        ).to(self.device)
        
        # Optimizer
        trainable_params = (
            list(self.qformer.parameters()) +
            list(self.head_s.parameters())
        )
        self.optimizer = AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # LR scheduler
        if config.max_steps is None:
            # Auto-calculate
            print("⚠️  max_steps is None, will calculate after loading data")
            self.lr_scheduler = None
        else:
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
        print(f"   Q-Former: {sum(p.numel() for p in self.qformer.parameters()):,} params")
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
    
    def initialize_scheduler(self, num_training_steps: int):
        """Initialize LR scheduler after knowing dataset size."""
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        self.lr_scheduler = get_lr_schedule(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(f"✅ LR scheduler initialized: {num_warmup_steps} warmup steps, {num_training_steps} total steps")
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step (forward + backward).
        
        Args:
            batch: Collated batch from collate_task_s_batch
        
        Returns:
            metrics: Dict of loss components and metrics
        """
        # Move batch to device
        query_input_ids = batch['query_input_ids'].to(self.device)
        query_attention_mask = batch['query_attention_mask'].to(self.device)
        evidence_embeddings = batch['evidence_embeddings'].to(self.device)
        teacher_scores = batch['teacher_scores'].to(self.device)
        pool_padding_mask = batch['pool_padding_mask'].to(self.device)
        
        batch_size = query_input_ids.shape[0]
        
        # Generate unified Drop-LQ mask (training only)
        lq_drop_mask = None
        if self.qformer.training and self.config.p_drop_lq_unified > 0:
            lq_drop_mask = torch.rand(batch_size, self.config.n_queries, 1, device=self.device) > self.config.p_drop_lq_unified
            # Ensure at least 1 LQ kept per sample
            all_dropped = (lq_drop_mask.sum(dim=1, keepdim=True) == 0)
            if all_dropped.any():
                for b in range(batch_size):
                    if all_dropped[b, 0, 0]:
                        random_idx = torch.randint(0, self.config.n_queries, (1,), device=self.device)
                        lq_drop_mask[b, random_idx, 0] = True
        
        # Q-Former forward
        query_token_emb = batch.get('query_token_embeddings', None)
        if query_token_emb is not None:
            query_token_emb = query_token_emb.to(self.device)
        
        Z, all_aux = self.qformer(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            evidence_emb=evidence_embeddings,
            evidence_mask=pool_padding_mask,
            precomputed_query_emb=query_token_emb,
        )
        
        # Extract CA raw scores
        ca_raw_scores_per_head = [
            aux.get('ca_raw_scores_per_head') 
            for aux in all_aux 
            if aux and 'ca_raw_scores_per_head' in aux
        ]
        
        # ========== TASK S: Fragment Ranking ==========
        head_s_out = self.head_s(
            z=Z,
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=pool_padding_mask,
            lq_drop_mask=lq_drop_mask,
            training=True,
        )
        fragment_logits_s = head_s_out['ranking_logits']  # [batch, K]
        
        # Compute ranking loss with teacher signals
        loss = compute_ranking_loss(
            logits=fragment_logits_s,
            teacher_scores=teacher_scores,
            pool_padding_mask=pool_padding_mask,
            tau=self.config.task_s_tau,
            teacher_tau=self.config.teacher_tau,
            loss_type=self.config.ranking_loss_type,
        )
        
        # Debug: Check for suspicious values
        if self.global_step % 100 == 0:
            print(f"\n[Step {self.global_step}] Loss Debug:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Teacher scores - min: {teacher_scores.min().item():.4f}, max: {teacher_scores.max().item():.4f}, mean: {teacher_scores.mean().item():.4f}")
            print(f"  Model logits - min: {fragment_logits_s.min().item():.4f}, max: {fragment_logits_s.max().item():.4f}, mean: {fragment_logits_s.mean().item():.4f}")
            print(f"  Teacher std: {teacher_scores.std().item():.4f}, Model std: {fragment_logits_s.std().item():.4f}")
            # Check if all teacher scores are the same (or too similar)
            unique_teachers = torch.unique(teacher_scores[pool_padding_mask])
            print(f"  Unique teacher scores per batch: {len(unique_teachers)}")
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.qformer.parameters()) + list(self.head_s.parameters()),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.qformer.train()
        self.head_s.train()
        
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
                'loss': f"{metrics['loss']:.4f}",
                'step': self.global_step,
            })
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")
            
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        # Average
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)
        
        # Record training history
        self.train_history['epoch'].append(epoch)
        self.train_history['loss'].append(epoch_metrics['loss'])
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.qformer.eval()
        self.head_s.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            query_input_ids = batch['query_input_ids'].to(self.device)
            query_attention_mask = batch['query_attention_mask'].to(self.device)
            evidence_embeddings = batch['evidence_embeddings'].to(self.device)
            teacher_scores = batch['teacher_scores'].to(self.device)
            pool_padding_mask = batch['pool_padding_mask'].to(self.device)
            
            # Q-Former forward
            query_token_emb = batch.get('query_token_embeddings', None)
            if query_token_emb is not None:
                query_token_emb = query_token_emb.to(self.device)
            
            Z, all_aux = self.qformer(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                evidence_emb=evidence_embeddings,
                evidence_mask=pool_padding_mask,
                precomputed_query_emb=query_token_emb,
            )
            
            # Extract CA scores
            ca_raw_scores_per_head = [
                aux.get('ca_raw_scores_per_head')
                for aux in all_aux
                if aux and 'ca_raw_scores_per_head' in aux
            ]
            
            # Task S
            head_s_out = self.head_s(
                z=Z,
                ca_raw_scores_per_head=ca_raw_scores_per_head,
                pool_padding_mask=pool_padding_mask,
                training=False,
            )
            fragment_logits_s = head_s_out['ranking_logits']
            
            loss = compute_ranking_loss(
                logits=fragment_logits_s,
                teacher_scores=teacher_scores,
                pool_padding_mask=pool_padding_mask,
                tau=self.config.task_s_tau,
                teacher_tau=self.config.teacher_tau,
                loss_type=self.config.ranking_loss_type,
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        val_metrics = {
            'val_loss': total_loss / max(num_batches, 1),
        }
        
        return val_metrics
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint_path = self.save_dir / filename
        torch.save({
            'global_step': self.global_step,
            'qformer_state_dict': self.qformer.state_dict(),
            'head_s_state_dict': self.head_s.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.qformer.load_state_dict(checkpoint['qformer_state_dict'])
        self.head_s.load_state_dict(checkpoint['head_s_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        print(f"📂 Checkpoint loaded: {checkpoint_path}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training and validation curves."""
        if not self.train_history['epoch']:
            print("⚠️  No training history to plot")
            return
        
        if save_path is None:
            save_path = self.save_dir / "training_curves.png"
        else:
            save_path = Path(save_path)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('TASK S Training Metrics', fontsize=16, fontweight='bold')
        
        epochs = self.train_history['epoch']
        
        # Plot: Loss
        ax.plot(epochs, self.train_history['loss'], 'b-', label='Train Loss', linewidth=2)
        if self.val_history['epoch']:
            ax.plot(self.val_history['epoch'], self.val_history['val_loss'], 
                   'r--', label='Val Loss', linewidth=2, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ranking Loss')
        ax.set_title('TASK S (Fragment Ranking) Loss')
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
    """
    Load pickle data and split into train/val.
    
    Returns:
        (train_data, val_data) where each is (data_dict, sample_ids)
    """
    print(f"\n📂 Loading data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    print(f"✅ Loaded {len(data_dict)} samples")
    
    # Get sample IDs
    sample_ids = list(data_dict.keys())
    
    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(sample_ids)
    
    # Split
    split_idx = int(len(sample_ids) * (1 - val_split))
    train_ids = sample_ids[:split_idx]
    val_ids = sample_ids[split_idx:]
    
    print(f"📊 Split: {len(train_ids)} train, {len(val_ids)} val")
    
    return (data_dict, train_ids), (data_dict, val_ids)


def main():
    """Main training loop."""
    config = TaskSConfig()
    
    print("="*80)
    print("TASK S Only Training: Fragment Ranking with Teacher Signals")
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
    trainer = TaskSTrainer(config)
    
    # Create datasets
    if config.use_precomputed_embeddings:
        print("\n📦 Creating datasets with pre-computed embeddings...")
    else:
        print("\n📦 Creating datasets with XLM-RoBERTa tokenizer...")
    
    train_dataset = RankingDataset(
        train_data[0], train_data[1], 
        tokenizer=trainer.tokenizer,
        xlm_model=trainer.xlm_embedding_model,
        device=config.device,
        max_query_len=512,
        use_precomputed_embeddings=config.use_precomputed_embeddings
    )
    val_dataset = RankingDataset(
        val_data[0], val_data[1], 
        tokenizer=trainer.tokenizer,
        xlm_model=trainer.xlm_embedding_model,
        device=config.device,
        max_query_len=512,
        use_precomputed_embeddings=config.use_precomputed_embeddings
    )
    print(f"✅ Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_task_s_batch,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_task_s_batch,
        num_workers=0,
    )
    
    # Initialize scheduler if needed
    if trainer.lr_scheduler is None:
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config.num_epochs
        trainer.initialize_scheduler(total_steps)
        config.max_steps = total_steps
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training (TASK S Only - Fragment Ranking)")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"\n📈 Train Metrics:")
        for key, value in train_metrics.items():
            print(f"   {key}: {value:.4f}")
        
        # Evaluate
        if (epoch + 1) % max(1, config.num_epochs // 5) == 0:
            val_metrics = trainer.evaluate(val_loader)
            print(f"\n📊 Validation Metrics:")
            for key, value in val_metrics.items():
                print(f"   {key}: {value:.4f}")
            
            # Record validation history
            trainer.val_history['epoch'].append(epoch)
            trainer.val_history['val_loss'].append(val_metrics['val_loss'])
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                trainer.save_checkpoint("best.pt")
                print(f"✨ New best model! Val loss: {best_val_loss:.4f}")
        
        # Early stopping
        if config.max_steps and trainer.global_step >= config.max_steps:
            print(f"\n🛑 Reached max_steps ({config.max_steps}), stopping training")
            break
    
    print("\n" + "="*80)
    print("✅ Training completed!")
    print(f"💾 Checkpoints saved in: {config.save_dir}")
    print("="*80)
    
    # Plot training curves
    print("\n📈 Generating training curves...")
    trainer.plot_training_curves()
    print("✅ Training curves generated!")


if __name__ == "__main__":
    main()
