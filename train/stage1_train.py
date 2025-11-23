"""
Stage-1 Training: Joint TASK E + TASK S (MVP - No LLM, No TASK C).

This script implements the minimal viable product for DR-QFormer training:
- TASK E: Fragment-level entailment tagging
- TASK S: Fragment-level ranking/sorting
- No TASK C: No LLM integration in Stage-1
- Small evidence pool: K_max = 11 (all samples used for training, no subset masking)
- Batch processing: Efficient training with variable K per sample

Key Features:
=============
1. Uses existing smoking_train_with_NoE.pkl format
2. Handles pre-computed embeddings (query + evidence)
3. XLM-RoBERTa-based Q-Former (multilingual support)
4. Unified Drop-LQ for multi-task training
5. Dynamic K padding (supports variable evidence pool sizes)
6. Curriculum learning for TASK S (teacher → posterior transition)

Data Format:
============
Each sample in smoking_train_with_NoE.pkl contains:
- query: str (query text, length=71)
- query_embedding: dict with keys ['input_ids', 'attention_mask', 'token_emb_768']
  - input_ids: [1, seq_len] token IDs
  - attention_mask: [1, seq_len] attention mask
  - token_emb_768: [1, seq_len, 768] pre-computed token embeddings
- answer: str (answer text)
- evidence_labels: ndarray [11] binary labels (0/1)
- evidence_text: list of 10 strings (fragment texts)
- evidence_embeddings: ndarray [11, 768] pre-computed evidence embeddings
- evidence_ranking: list of 11 tuples (fragment_idx, score)

Note: Evidence pool size is 11 (K_max=11), but only 10 fragments have text.
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
from src.models.heads import EntailmentHead, FragmentRankingHead
from src.losses import (
    compute_focal_loss,
    compute_ranking_loss,
    get_curriculum_weights,
)
from train.schedule import get_lr_schedule


@dataclass
class Stage1Config:
    """Configuration for Stage-1 training."""
    # Data
    train_data_path: str = r"D:\LLMs\DR-QFormer\DR-QFormer\ms_xlm_embeddings.pkl"
    val_split: float = 0.1  # Validation split ratio
    shuffle_data: bool = True
    
    # Embedding options
    use_precomputed_embeddings: bool = True  # Use pre-computed token embeddings from PKL (bypasses XLM-R tokenizer/embeddings)
    
    # Model architecture
    xlm_model_name: str = "xlm-roberta-base"
    n_queries: int = 32
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    use_ca_layers: Optional[List[int]] = None  # None = all layers
    freeze_xlmr: bool = False
    
    # Task E hyperparameters
    task_e_tau: float = 0.5
    task_e_focal_gamma: float = 2.0
    task_e_focal_alpha: float = 0.25
    task_e_w_pos: float = 10.0  # Positive class weight
    task_e_w_longtail: float = 50.0  # Longtail class weight
    
    # Task S hyperparameters
    task_s_tau_head: float = 0.1
    task_s_tau_lq: float = 0.2
    task_s_rho_top: float = 0.2  # 20% for small K=10
    task_s_l_prime: int = 3  # Student hard negatives
    task_s_lambda_teach_start: float = 1.0
    task_s_lambda_teach_end: float = 0.2
    task_s_lambda_post_start: float = 0.0  # No posterior in Stage-1
    task_s_lambda_post_end: float = 0.0
    task_s_lambda_entropy: float = 0.01
    
    # Multi-task loss weights
    w_task_e: float = 1.0
    w_task_s: float = 1.0
    
    # Unified Drop-LQ
    p_drop_lq_unified: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 8
    num_epochs: int = 10
    max_steps: int = 50000
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    save_dir: str = "./checkpoints/stage1"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class SmokingDataset(Dataset):
    """
    Dataset for smoking_train_with_NoE.pkl format.
    
    Handles pre-computed embeddings and dynamic K per sample.
    Supports two modes:
    1. use_precomputed_embeddings=True: Use pre-computed token embeddings from PKL
    2. use_precomputed_embeddings=False: Re-encode with XLM-RoBERTa tokenizer
    """
    
    def __init__(self, data_dict: Dict, sample_ids: List[str], tokenizer: Optional[AutoTokenizer] = None, 
                 xlm_model: Optional[AutoModel] = None, device: str = 'cuda',
                 max_query_len: int = 512, use_precomputed_embeddings: bool = True):
        """
        Args:
            data_dict: Dictionary loaded from pickle file
            sample_ids: List of sample IDs to use (for train/val split)u
            tokenizer: XLM-RoBERTa tokenizer (only needed if use_precomputed_embeddings=False)
            xlm_model: XLM-RoBERTa model for generating embeddings (only needed if use_precomputed_embeddings=False)
            device: Device for XLM model inference
            max_query_len: Maximum query sequence length
            use_precomputed_embeddings: If True, use pre-computed embeddings from PKL; 
                                       If False, re-encode with tokenizer and generate embeddings with XLM model
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
                - answer: str
                - query_input_ids: [seq_len] LongTensor
                - query_attention_mask: [seq_len] LongTensor
                - evidence_embeddings: [K, 768] FloatTensor (K may vary)
                - evidence_labels: [K] FloatTensor (binary 0/1)
                - evidence_scores: [K] FloatTensor (ranking scores)
                - sample_id: str
        """
        sample_id = self.sample_ids[idx]
        sample = self.data_dict[sample_id]
        
        # Query encoding: two modes
        if self.use_precomputed_embeddings:
            # Mode 1: Use pre-computed token embeddings from PKL (Qwen3-Embedding)
            # This bypasses XLM-R tokenizer/embeddings entirely
            query_emb = sample['query_embedding']
            query_input_ids = query_emb['input_ids'].squeeze(0)  # [seq_len] - Qwen3 token IDs (for shape only, not used for embedding)
            query_attention_mask = query_emb['attention_mask'].squeeze(0)  # [seq_len] - Valid token mask (used by Q-Former)
            query_token_embeddings = query_emb['token_emb_768'].squeeze(0)  # [seq_len, 768] - Pre-computed embeddings (used by Q-Former)
        else:
            # Mode 2: Generate embeddings with XLM-RoBERTa model
            # This uses XLM-R tokenizer + model to generate fresh token-level embeddings
            query_text = sample['query']
            query_encoded = self.tokenizer(
                query_text,
                padding=False,  # Will pad in collate_fn
                truncation=True,
                max_length=self.max_query_len,
                return_tensors='pt',
            )
            query_input_ids = query_encoded['input_ids'].squeeze(0)  # [seq_len]
            query_attention_mask = query_encoded['attention_mask'].squeeze(0)  # [seq_len]
            
            # Generate token-level embeddings using XLM-RoBERTa model
            with torch.no_grad():
                self.xlm_model.eval()
                encoded_input = {
                    'input_ids': query_encoded['input_ids'].to(self.device),
                    'attention_mask': query_encoded['attention_mask'].to(self.device)
                }
                output = self.xlm_model(**encoded_input)
                # output.last_hidden_state: [1, seq_len, 768]
                query_token_embeddings = output.last_hidden_state.squeeze(0).cpu()  # [seq_len, 768]
        
        # Extract evidence
        if self.use_precomputed_embeddings:
            # Use pre-computed evidence embeddings from PKL
            evidence_embeddings = sample['evidence_embeddings']  # [K, 768]
        else:
            # ⚠️ Evidence embeddings still need to be generated separately
            # They come from a frozen retriever model, not from Q-Former
            # This is consistent with the original DR-QFormer design
            raise ValueError(
                "use_precomputed_embeddings=False is not fully supported yet for evidence.\n"
                "Evidence embeddings must be pre-computed and stored in PKL.\n"
                "Only query embeddings can be generated on-the-fly by Q-Former's embedding layer."
            )
        
        evidence_labels = sample['evidence_labels']  # [K]
        evidence_ranking = sample['evidence_ranking']  # List[(idx, score)] - reranker output
        
        # Convert ranking to per-fragment scores
        # evidence_ranking format: [(idx_0, score_0), (idx_1, score_1), ...]
        # where idx is the fragment index, score is the reranker confidence
        K = len(evidence_labels)
        evidence_scores = np.zeros(K, dtype=np.float32)
        
        for rank_pos, ranking_item in enumerate(evidence_ranking):
            # Parse ranking item: should be (idx, score) tuple
            if isinstance(ranking_item, (tuple, list)) and len(ranking_item) >= 2:
                frag_idx, rerank_score = ranking_item[0], ranking_item[1]
            elif isinstance(ranking_item, (tuple, list)) and len(ranking_item) == 1:
                # Fallback: only index provided
                frag_idx = ranking_item[0]
                rerank_score = 1.0 - (rank_pos / max(len(evidence_ranking), 1))
            else:
                # Fallback: plain index
                frag_idx = ranking_item
                rerank_score = 1.0 - (rank_pos / max(len(evidence_ranking), 1))
            
            # Convert to int
            if isinstance(frag_idx, (np.ndarray, np.integer)):
                frag_idx = int(frag_idx)
            elif not isinstance(frag_idx, int):
                try:
                    frag_idx = int(frag_idx)
                except (ValueError, TypeError):
                    continue  # Skip invalid indices
            
            # Validate bounds and assign score
            if 0 <= frag_idx < K:
                # Use reranker score directly (already normalized by rerank_evidences)
                evidence_scores[frag_idx] = float(rerank_score)
        
        # Convert to tensors
        result = {
            'query': sample['query'],
            'answer': sample['answer'],
            'query_input_ids': torch.from_numpy(query_input_ids) if isinstance(query_input_ids, np.ndarray) else query_input_ids.long(),
            'query_attention_mask': torch.from_numpy(query_attention_mask) if isinstance(query_attention_mask, np.ndarray) else query_attention_mask.long(),
            'evidence_embeddings': torch.from_numpy(evidence_embeddings).float(),
            'evidence_labels': torch.from_numpy(evidence_labels).float(),
            'evidence_scores': torch.from_numpy(evidence_scores).float(),
            'sample_id': sample_id,
        }
        
        # Add pre-computed token embeddings only when bypassing Q-Former's embedding layer
        if self.use_precomputed_embeddings and query_token_embeddings is not None:
            result['query_token_embeddings'] = torch.from_numpy(query_token_embeddings) if isinstance(query_token_embeddings, np.ndarray) else query_token_embeddings.float()
        # Otherwise, query_token_embeddings is None, and Q-Former will use its own embedding layer
        
        return result


def collate_stage1_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for Stage-1 batches with dynamic K padding.
    
    Args:
        batch: List of samples from SmokingDataset
    
    Returns:
        Collated batch dict with padded tensors
    """
    batch_size = len(batch)
    
    # Find max dimensions in this batch
    max_seq_len = max(sample['query_input_ids'].shape[0] for sample in batch)
    max_K = max(sample['evidence_embeddings'].shape[0] for sample in batch)
    
    # Check if using pre-computed embeddings
    use_precomputed = 'query_token_embeddings' in batch[0]
    
    # Initialize padded tensors
    query_input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    query_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    if use_precomputed:
        query_token_embeddings = torch.zeros(batch_size, max_seq_len, 768, dtype=torch.float32)
    evidence_embeddings = torch.zeros(batch_size, max_K, 768, dtype=torch.float32)
    evidence_labels = torch.zeros(batch_size, max_K, dtype=torch.float32)
    evidence_scores = torch.zeros(batch_size, max_K, dtype=torch.float32)
    pool_padding_mask = torch.zeros(batch_size, max_K, dtype=torch.bool)
    
    # Lists for non-tensor data
    queries = []
    answers = []
    sample_ids = []
    
    # Fill tensors
    for b, sample in enumerate(batch):
        # Query tokens
        seq_len = sample['query_input_ids'].shape[0]
        query_input_ids[b, :seq_len] = sample['query_input_ids']
        query_attention_mask[b, :seq_len] = sample['query_attention_mask']
        if use_precomputed:
            query_token_embeddings[b, :seq_len] = sample['query_token_embeddings']
        
        # Evidence
        K_curr = sample['evidence_embeddings'].shape[0]
        evidence_embeddings[b, :K_curr] = sample['evidence_embeddings']
        evidence_labels[b, :K_curr] = sample['evidence_labels']
        evidence_scores[b, :K_curr] = sample['evidence_scores']
        pool_padding_mask[b, :K_curr] = True
        
        # Strings
        queries.append(sample['query'])
        answers.append(sample['answer'])
        sample_ids.append(sample['sample_id'])
    
    result = {
        'queries': queries,
        'answers': answers,
        'query_input_ids': query_input_ids,
        'query_attention_mask': query_attention_mask,
        'evidence_embeddings': evidence_embeddings,
        'evidence_labels': evidence_labels,
        'evidence_scores': evidence_scores,
        'pool_padding_mask': pool_padding_mask,
        'sample_ids': sample_ids,
    }
    
    if use_precomputed:
        result['query_token_embeddings'] = query_token_embeddings
    
    return result


class Stage1Trainer:
    """
    Trainer for Stage-1: Joint TASK E + TASK S without LLM.
    """
    
    def __init__(self, config: Stage1Config):
        self.config = config
        self.device = torch.device(config.device)
        self.global_step = 0
        
        # Initialize XLM-RoBERTa tokenizer (always needed for token IDs)
        # Note: We no longer load a separate XLM-R model for embeddings
        #       Instead, we use Q-Former's built-in trainable embedding layer (BLIP-2 style)
        if config.use_precomputed_embeddings:
            print("⚡ Using pre-computed token embeddings from PKL (bypassing Q-Former's embedding layer)")
            self.tokenizer = None
        else:
            print(f"Loading tokenizer for Q-Former input: {config.xlm_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(config.xlm_model_name)
            print(f"✅ Tokenizer loaded. Q-Former will use its own trainable embedding layer (BLIP-2 paradigm)")
        
        # Training history for plotting
        self.train_history = {
            'epoch': [],
            'loss_total': [],
            'loss_e': [],
            'loss_s': [],
            'loss_s_teach': [],
            'loss_s_entropy': [],
        }
        self.val_history = {
            'epoch': [],
            'val_loss': [],
            'val_loss_e': [],
            'val_loss_s': [],
        }
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Initialize models
        print("="*80)
        print("Initializing Stage-1 Trainer (TASK E + TASK S)")
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
        
        # Task E: EntailmentHead
        self.head_e = EntailmentHead(
            hidden_dim=self.qformer.hidden_dim,  # Get from Q-Former
            num_fragments=11,  # Dynamic K
            tau=config.task_e_tau,
            p_drop_lq=0.0,  # Use unified Drop-LQ
            focal_gamma=config.task_e_focal_gamma,
            focal_alpha=config.task_e_focal_alpha,
        ).to(self.device)
        
        # Task S: FragmentRankingHead
        self.head_s = FragmentRankingHead(
            hidden_dim=self.qformer.hidden_dim,  # Get from Q-Former
            num_fragments=11,  # Dynamic K
            tau_head=config.task_s_tau_head,
            tau_lq=config.task_s_tau_lq,
            rho_top=config.task_s_rho_top,
            l_prime=config.task_s_l_prime,
            p_drop_lq=0.0,  # Use unified Drop-LQ
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
        print(f"   Q-Former: {sum(p.numel() for p in self.qformer.parameters()):,} params")
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
        Single training step (forward + backward).
        
        Args:
            batch: Collated batch from collate_stage1_batch
        
        Returns:
            metrics: Dict of loss components and metrics
        """
        # Move batch to device
        query_input_ids = batch['query_input_ids'].to(self.device)
        query_attention_mask = batch['query_attention_mask'].to(self.device)
        evidence_embeddings = batch['evidence_embeddings'].to(self.device)
        evidence_labels = batch['evidence_labels'].to(self.device)
        evidence_scores = batch['evidence_scores'].to(self.device)
        pool_padding_mask = batch['pool_padding_mask'].to(self.device)
        
        batch_size = query_input_ids.shape[0]
        K_max = evidence_embeddings.shape[1]
        
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
        
        # Q-Former forward (one pass for both tasks)
        # If using pre-computed embeddings, pass them directly
        query_token_emb = batch.get('query_token_embeddings', None)
        if query_token_emb is not None:
            query_token_emb = query_token_emb.to(self.device)
        
        Z, all_aux = self.qformer(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            evidence_emb=evidence_embeddings,
            evidence_mask=pool_padding_mask,
            precomputed_query_emb=query_token_emb,  # New parameter
        )
        # Z: [batch, N_lq, hidden_dim]
        
        # Extract CA raw scores from all layers
        ca_raw_scores_per_head = [
            aux.get('ca_raw_scores_per_head') 
            for aux in all_aux 
            if aux and 'ca_raw_scores_per_head' in aux
        ]
        
        # ========== TASK E: Entailment Tagging ==========
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
        # Note: No longtail flags in smoking dataset, so skip w_longtail
        
        # Compute focal loss
        loss_e = compute_focal_loss(
            logits=fragment_logits_e,
            gt_labels=evidence_labels,
            importance_weights=importance_weights,
            pool_padding_mask=pool_padding_mask,
            focal_gamma=self.config.task_e_focal_gamma,
            focal_alpha=self.config.task_e_focal_alpha,
        )
        
        # ========== TASK S: Fragment Ranking ==========
        head_s_out = self.head_s(
            z=Z,
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=pool_padding_mask,
            lq_drop_mask=lq_drop_mask,
            training=True,
        )
        ranking_logits = head_s_out['ranking_logits']  # [batch, K]
        
        # Get curriculum weights
        curriculum = get_curriculum_weights(
            current_step=self.global_step,
            total_steps=self.config.max_steps,
            lambda_teach_start=self.config.task_s_lambda_teach_start,
            lambda_teach_end=self.config.task_s_lambda_teach_end,
            lambda_post_start=self.config.task_s_lambda_post_start,
            lambda_post_end=self.config.task_s_lambda_post_end,
        )
        
        # Stage-1: No subset mask (K_max=11, use all fragments)
        train_subset_mask = None
        
        # Compute ranking loss (no posterior in Stage-1)
        loss_s_dict = compute_ranking_loss(
            ranking_logits=ranking_logits,
            gt_scores=evidence_scores,
            posterior_scores=None,  # No LLM in Stage-1
            pool_padding_mask=pool_padding_mask,
            train_subset_mask=train_subset_mask,
            lambda_teach=curriculum['lambda_teach'],
            lambda_post=curriculum['lambda_post'],
            lambda_entropy=self.config.task_s_lambda_entropy,
            tau_pred=1.0,
            tau_gt=1.0,
            alpha_gt=0.7,
        )
        loss_s = loss_s_dict['loss']
        
        # ========== Combined Loss ==========
        loss_total = (
            self.config.w_task_e * loss_e +
            self.config.w_task_s * loss_s
        )
        
        # Backward
        self.optimizer.zero_grad()
        loss_total.backward()
        
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
        
        # Return metrics
        return {
            'loss_total': loss_total.item(),
            'loss_e': loss_e.item(),
            'loss_s': loss_s.item(),
            'loss_s_teach': loss_s_dict['loss_teach'].item(),
            'loss_s_entropy': loss_s_dict['loss_entropy'].item(),
            'lambda_teach': curriculum['lambda_teach'],
            'lambda_post': curriculum['lambda_post'],
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
            
            # Accumulate
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss_total']:.4f}",
                'loss_e': f"{metrics['loss_e']:.4f}",
                'loss_s': f"{metrics['loss_s']:.4f}",
                'step': self.global_step,
            })
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")
            
            if self.global_step >= self.config.max_steps:
                break
        
        # Average
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)
        
        # Record training history
        self.train_history['epoch'].append(epoch)
        for key in ['loss_total', 'loss_e', 'loss_s', 'loss_s_teach', 'loss_s_entropy']:
            if key in epoch_metrics:
                self.train_history[key].append(epoch_metrics[key])
        
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
            # Move to device
            query_input_ids = batch['query_input_ids'].to(self.device)
            query_attention_mask = batch['query_attention_mask'].to(self.device)
            evidence_embeddings = batch['evidence_embeddings'].to(self.device)
            evidence_labels = batch['evidence_labels'].to(self.device)
            evidence_scores = batch['evidence_scores'].to(self.device)
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
            
            # Task E
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
            
            loss_e = compute_focal_loss(
                logits=fragment_logits_e,
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
            
            loss_s_dict = compute_ranking_loss(
                ranking_logits=ranking_logits,
                gt_scores=evidence_scores,
                posterior_scores=None,
                pool_padding_mask=pool_padding_mask,
                train_subset_mask=None,
                lambda_teach=1.0,
                lambda_post=0.0,
                lambda_entropy=0.0,  # No regularization in eval
            )
            loss_s = loss_s_dict['loss']
            
            loss_total = (
                self.config.w_task_e * loss_e +
                self.config.w_task_s * loss_s
            )
            
            total_loss += loss_total.item()
            total_loss_e += loss_e.item()
            total_loss_s += loss_s.item()
            num_batches += 1
        
        val_metrics = {
            'val_loss': total_loss / max(num_batches, 1),
            'val_loss_e': total_loss_e / max(num_batches, 1),
            'val_loss_s': total_loss_s / max(num_batches, 1),
        }
        
        return val_metrics
    
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
        """Plot training and validation curves."""
        if not self.train_history['epoch']:
            print("⚠️  No training history to plot")
            return
        
        if save_path is None:
            save_path = self.save_dir / "training_curves.png"
        else:
            save_path = Path(save_path)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stage-1 Training Metrics', fontsize=16, fontweight='bold')
        
        epochs = self.train_history['epoch']
        
        # Plot 1: Total Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.train_history['loss_total'], 'b-', label='Train Total Loss', linewidth=2)
        if self.val_history['epoch']:
            ax.plot(self.val_history['epoch'], self.val_history['val_loss'], 
                   'r--', label='Val Total Loss', linewidth=2, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Task E Loss
        ax = axes[0, 1]
        ax.plot(epochs, self.train_history['loss_e'], 'g-', label='Train Task E Loss', linewidth=2)
        if self.val_history['epoch']:
            ax.plot(self.val_history['epoch'], self.val_history['val_loss_e'], 
                   'orange', linestyle='--', label='Val Task E Loss', linewidth=2, marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Task E (Entailment) Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Task S Loss
        ax = axes[1, 0]
        ax.plot(epochs, self.train_history['loss_s'], 'm-', label='Train Task S Loss', linewidth=2)
        if self.val_history['epoch']:
            ax.plot(self.val_history['epoch'], self.val_history['val_loss_s'], 
                   'cyan', linestyle='--', label='Val Task S Loss', linewidth=2, marker='^')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Task S (Ranking) Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Task S Components
        ax = axes[1, 1]
        ax.plot(epochs, self.train_history['loss_s_teach'], 'purple', 
               label='Task S Teacher Loss', linewidth=2, alpha=0.7)
        ax.plot(epochs, self.train_history['loss_s_entropy'], 'brown', 
               label='Task S Entropy Loss', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Task S Loss Components')
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
) -> Tuple[SmokingDataset, SmokingDataset]:
    """
    Load pickle data and split into train/val.
    
    Args:
        data_path: Path to smoking_train_with_NoE.pkl
        val_split: Validation split ratio
        shuffle: Whether to shuffle sample IDs
        seed: Random seed
    
    Returns:
        (train_dataset, val_dataset)
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
    
    # Return data tuples (tokenizer will be provided when creating Dataset objects)
    return (data_dict, train_ids), (data_dict, val_ids)


def main():
    """Main training loop."""
    config = Stage1Config()
    
    print("="*80)
    print("Stage-1 Training: TASK E + TASK S (No LLM)")
    print("="*80)
    print("\nConfiguration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Load data (returns tuples of (data_dict, sample_ids))
    train_data, val_data = load_and_split_data(
        config.train_data_path,
        val_split=config.val_split,
        shuffle=config.shuffle_data,
        seed=config.seed,
    )
    
    # Initialize trainer first to get tokenizer
    trainer = Stage1Trainer(config)
    
    # Create Dataset objects
    if config.use_precomputed_embeddings:
        print("\n📦 Creating datasets with pre-computed embeddings...")
    else:
        print("\n📦 Creating datasets with XLM-RoBERTa tokenizer...")
    
    train_dataset = SmokingDataset(
        train_data[0], train_data[1], 
        tokenizer=trainer.tokenizer,
        xlm_model=None,  # No longer needed - Q-Former has its own embeddings
        device=config.device,
        max_query_len=512,
        use_precomputed_embeddings=config.use_precomputed_embeddings
    )
    val_dataset = SmokingDataset(
        val_data[0], val_data[1], 
        tokenizer=trainer.tokenizer,
        xlm_model=None,  # No longer needed - Q-Former has its own embeddings
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
        collate_fn=collate_stage1_batch,
        num_workers=0,  # Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_stage1_batch,
        num_workers=0,
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
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
            for key in ['val_loss', 'val_loss_e', 'val_loss_s']:
                if key in val_metrics:
                    trainer.val_history[key].append(val_metrics[key])
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                trainer.save_checkpoint("best.pt")
                print(f"✨ New best model! Val loss: {best_val_loss:.4f}")
        
        # Early stopping
        if trainer.global_step >= config.max_steps:
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
