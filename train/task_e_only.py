"""
TASK E Only Training: Fragment-level Entailment Tagging.

This script trains only TASK E (entailment classification) without TASK S (ranking).
All other components remain the same as Stage-1 training:
- Uses existing smoking_train_with_NoE.pkl format
- Handles pre-computed embeddings (query + evidence)
- XLM-RoBERTa-based Q-Former (multilingual support)
- Unified Drop-LQ for regularization
- Dynamic K padding (supports variable evidence pool sizes)

Key Differences from Stage-1:
==============================
- ONLY trains EntailmentHead (TASK E)
- NO FragmentRankingHead (TASK S)
- Simpler loss computation (only focal loss)
- No curriculum learning needed

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

Note: Evidence pool size is 11 (K_max=11), but only 10 fragments have text.
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

from transformers import AutoTokenizer, AutoModel

from src.models.qformer_xlm import XLMRobertaDRQFormer
from src.models.heads import EntailmentHead
from src.losses import compute_focal_loss
from train.schedule import get_lr_schedule


@dataclass
class TaskEConfig:
    """Configuration for TASK E only training."""
    # Data
    train_data_path: str = r"D:\LLMs\DR-QFormer\DR-QFormer\ms_xlm_embeddings.pkl"
    val_split: float = 0.1  # Validation split ratio
    shuffle_data: bool = True
    
    # Embedding options
    use_precomputed_embeddings: bool = True  # Use pre-computed token embeddings from PKL (bypasses XLM-R tokenizer/embeddings)
    
    # Model architecture
    xlm_model_name: str = "xlm-roberta-base"  # hidden_dim and num_heads auto-detected from model
    n_queries: int = 32
    use_ca_layers: Optional[List[int]] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10])  # None = all layers, use list for alternating layers
    freeze_xlmr: bool = False
    
    # Task E hyperparameters
    task_e_tau: float = 0.5 # Temperature for scaling logits
    task_e_focal_gamma: float = 1.5
    task_e_focal_alpha: float = 0.85
    task_e_w_pos: float = 1.255  # Positive class weight
    task_e_w_longtail: float = 100.0  # Longtail class weight (if applicable)
    
    # Unified Drop-LQ
    p_drop_lq_unified: float = 0.0
    
    # Training hyperparameters
    batch_size: int = 16
    num_epochs: int = 15
    max_steps: int = 500000 # Set to None to use num_epochs instead
    lr: float = 5e-05
    weight_decay: float = 0.001
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 5000
    save_dir: str = "./checkpoints/task_e_only"
    
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
                 max_query_len: int = 512, use_precomputed_embeddings: bool = False):
        """
        Args:
            data_dict: Dictionary loaded from pickle file
            sample_ids: List of sample IDs to use (for train/val split)
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
            # Mode 2: BLIP-2 Style - Only tokenize, do NOT pre-compute embeddings
            # Query text will be embedded inside Q-Former (like BLIP-2's text encoding)
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
            
            # NOTE: No pre-computation of token embeddings (BLIP-2 paradigm)
            # Q-Former will embed input_ids internally using its own embedding layer
        
        # Extract evidence (BLIP-2 Style: corresponds to Vision Encoder output)
        if self.use_precomputed_embeddings:
            # Use pre-computed evidence embeddings from PKL
            evidence_embeddings = sample['evidence_embeddings']  # [K, 768]
        else:
            # BLIP-2 Style: Use separate frozen encoder for evidence (like Vision Encoder for images)
            # Generate sentence-level embeddings using XLM-RoBERTa as the "evidence encoder"
            evidence_texts = sample['evidence_text']  # List of strings
            K = len(sample['evidence_labels'])
            evidence_embeddings = np.zeros((K, 768), dtype=np.float32)
            
            with torch.no_grad():
                self.xlm_model.eval()
                for i, text in enumerate(evidence_texts):
                    if text:  # Skip empty texts
                        # Tokenize evidence text
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
                        # Use [CLS] token embedding as sentence representation
                        # output.last_hidden_state: [1, seq_len, 768]
                        cls_embedding = output.last_hidden_state[:, 0, :]  # [1, 768]
                        evidence_embeddings[i] = cls_embedding.squeeze(0).cpu().numpy()
        
        evidence_labels = sample['evidence_labels']  # [K]
        
        # Convert to tensors
        result = {
            'query': sample['query'],
            'answer': sample['answer'],
            'query_input_ids': torch.from_numpy(query_input_ids) if isinstance(query_input_ids, np.ndarray) else query_input_ids.long(),
            'query_attention_mask': torch.from_numpy(query_attention_mask) if isinstance(query_attention_mask, np.ndarray) else query_attention_mask.long(),
            'evidence_embeddings': torch.from_numpy(evidence_embeddings).float(),
            'evidence_labels': torch.from_numpy(evidence_labels).float(),
            'sample_id': sample_id,
        }
        
        # Add pre-computed token embeddings ONLY if using precomputed mode
        # (In BLIP-2 mode, we don't pre-compute query embeddings)
        if self.use_precomputed_embeddings:
            result['query_token_embeddings'] = torch.from_numpy(query_token_embeddings) if isinstance(query_token_embeddings, np.ndarray) else query_token_embeddings.float()
        
        return result


def collate_task_e_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for TASK E batches with dynamic K padding.
    
    BLIP-2 Paradigm:
    - Query: input_ids will be embedded inside Q-Former (like BLIP-2 text encoding)
    - Evidence: pre-computed sentence embeddings (like BLIP-2 vision encoder output)
    
    Args:
        batch: List of samples from SmokingDataset
    
    Returns:
        Collated batch dict with padded tensors
    """
    batch_size = len(batch)
    
    # Find max dimensions in this batch
    max_seq_len = max(sample['query_input_ids'].shape[0] for sample in batch)
    max_K = max(sample['evidence_embeddings'].shape[0] for sample in batch)
    
    # Check if using pre-computed query embeddings (only in precomputed mode, not BLIP-2 mode)
    use_precomputed = 'query_token_embeddings' in batch[0]
    
    # Initialize padded tensors
    query_input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    query_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    if use_precomputed:
        query_token_embeddings = torch.zeros(batch_size, max_seq_len, 768, dtype=torch.float32)
    evidence_embeddings = torch.zeros(batch_size, max_K, 768, dtype=torch.float32)
    evidence_labels = torch.zeros(batch_size, max_K, dtype=torch.float32)
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
        'pool_padding_mask': pool_padding_mask,
        'sample_ids': sample_ids,
    }
    
    if use_precomputed:
        result['query_token_embeddings'] = query_token_embeddings
    
    return result


class TaskETrainer:
    """
    Trainer for TASK E only: Fragment-level Entailment Tagging.
    """
    
    def __init__(self, config: TaskEConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.global_step = 0
        
        # Initialize XLM-RoBERTa tokenizer and model for evidence encoding (BLIP-2 style)
        if config.use_precomputed_embeddings:
            print("⚡ Using pre-computed embeddings from PKL (both query and evidence)")
            self.tokenizer = None
            self.xlm_embedding_model = None
        else:
            print(f"🔧 BLIP-2 Mode: Loading tokenizer and evidence encoder")
            print(f"   - Tokenizer: {config.xlm_model_name} (for query tokenization)")
            print(f"   - Evidence Encoder: {config.xlm_model_name} (like Vision Encoder in BLIP-2)")
            self.tokenizer = AutoTokenizer.from_pretrained(config.xlm_model_name)
            # Evidence encoder: separate frozen model (like BLIP-2's frozen Vision Encoder)
            self.xlm_embedding_model = AutoModel.from_pretrained(config.xlm_model_name).to(self.device)
            self.xlm_embedding_model.eval()  # Frozen encoder for evidence
            for param in self.xlm_embedding_model.parameters():
                param.requires_grad = False  # Freeze evidence encoder (BLIP-2 paradigm)
            print(f"✅ Evidence encoder loaded and frozen (BLIP-2 style)")
        
        # Training history for plotting
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
        print("Initializing TASK E Only Trainer (Entailment Classification)")
        print("="*80)
        
        # Q-Former (BLIP-2 Style)
        # Note: hidden_dim and num_heads are automatically obtained from XLM-R config
        # Cross-attention layers are automatically initialized with pre-trained weights (BLIP-2 paradigm)
        # 
        # BLIP-2 Paradigm:
        # - Query (text): Embedded inside Q-Former using its own embedding layer
        # - Evidence: Pre-computed by separate frozen encoder (like Vision Encoder)
        self.qformer = XLMRobertaDRQFormer(
            xlm_model_name=config.xlm_model_name,
            n_queries=config.n_queries,
            dropout=0.1,
            use_ca_layers=config.use_ca_layers,
            freeze_xlmr=config.freeze_xlmr,
            bypass_embeddings=config.use_precomputed_embeddings,  # False in BLIP-2 mode
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
        print(f"   Q-Former: {sum(p.numel() for p in self.qformer.parameters()):,} params (hidden_dim={self.qformer.hidden_dim}, num_heads={self.qformer.num_heads})")
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
        Single training step (forward + backward).
        
        Args:
            batch: Collated batch from collate_task_e_batch
        
        Returns:
            metrics: Dict of loss components and metrics
        """
        # Move batch to device
        query_input_ids = batch['query_input_ids'].to(self.device)
        query_attention_mask = batch['query_attention_mask'].to(self.device)
        evidence_embeddings = batch['evidence_embeddings'].to(self.device)
        evidence_labels = batch['evidence_labels'].to(self.device)
        pool_padding_mask = batch['pool_padding_mask'].to(self.device)
        
        batch_size = query_input_ids.shape[0]
        
        # Generate unified Drop-LQ mask (training only)
        lq_drop_mask = None
        if self.qformer.training and self.config.p_drop_lq_unified > 0:
            n_queries = self.qformer.n_queries
            lq_drop_mask = torch.rand(batch_size, n_queries, 1, device=self.device) > self.config.p_drop_lq_unified
            # Ensure at least 1 LQ kept per sample
            all_dropped = (lq_drop_mask.sum(dim=1, keepdim=True) == 0)
            if all_dropped.any():
                for b in range(batch_size):
                    if all_dropped[b, 0, 0]:
                        random_idx = torch.randint(0, n_queries, (1,), device=self.device)
                        lq_drop_mask[b, random_idx, 0] = True
        
        # Q-Former forward (BLIP-2 Style)
        # - Query: input_ids embedded inside Q-Former (like BLIP-2 text encoding)
        # - Evidence: pre-computed embeddings from frozen encoder (like BLIP-2 vision encoder)
        query_token_emb = batch.get('query_token_embeddings', None)
        if query_token_emb is not None:
            query_token_emb = query_token_emb.to(self.device)
        
        Z, all_aux = self.qformer(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            evidence_emb=evidence_embeddings,
            evidence_mask=pool_padding_mask,
            precomputed_query_emb=query_token_emb,  # None in BLIP-2 mode, only used in precomputed mode
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
        
        # --- Gradient Check (Debug) ---
        if self.global_step == 0 or self.global_step % 10 == 0:
            print(f"\n[Step {self.global_step}] Gradient Flow Check:")
            
            # 1. Check Q-Former Gradients
            qformer_grads = [p for p in self.qformer.parameters() if p.grad is not None]
            if not qformer_grads:
                print("  ❌ Q-Former: NO GRADIENTS! (Check connection between Head and Q-Former)")
            else:
                grad_norm = torch.sqrt(torch.stack([torch.norm(p.grad)**2 for p in qformer_grads]).sum())
                print(f"  ✅ Q-Former: {len(qformer_grads)} params with grad (Norm: {grad_norm:.4f})")
            
            # 2. Check Head E Gradients
            head_e_params = list(self.head_e.parameters())
            if not head_e_params:
                 print("  ℹ️  Head E: No trainable parameters (Functional Head) - This is expected.")
            else:
                head_e_grads = [p for p in head_e_params if p.grad is not None]
                if not head_e_grads:
                    print("  ❌ Head E: NO GRADIENTS!")
                else:
                    grad_norm = torch.sqrt(torch.stack([torch.norm(p.grad)**2 for p in head_e_grads]).sum())
                    print(f"  ✅ Head E: {len(head_e_grads)} params with grad (Norm: {grad_norm:.4f})")

            # 3. Check Frozen XLM-R (if configured)
            if self.config.freeze_xlmr:
                xlm_grads = []
                for name, p in self.qformer.named_parameters():
                    if ('xlm' in name or 'roberta' in name) and p.grad is not None:
                        xlm_grads.append(name)
                
                if xlm_grads:
                     print(f"  ⚠️ WARNING: freeze_xlmr=True but found gradients in: {xlm_grads[:5]}...")
                else:
                     print("  ✅ XLM-R is frozen (no gradients found)")
        # ------------------------------
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.qformer.parameters()) + list(self.head_e.parameters()),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        self.global_step += 1
        
        # Return metrics
        return {
            'loss': loss.item(),
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.qformer.train()
        self.head_e.train()
        
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
            
            if self.global_step >= self.config.max_steps:
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
        self.head_e.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            query_input_ids = batch['query_input_ids'].to(self.device)
            query_attention_mask = batch['query_attention_mask'].to(self.device)
            evidence_embeddings = batch['evidence_embeddings'].to(self.device)
            evidence_labels = batch['evidence_labels'].to(self.device)
            pool_padding_mask = batch['pool_padding_mask'].to(self.device)
            
            # Q-Former forward (BLIP-2 Style)
            query_token_emb = batch.get('query_token_embeddings', None)
            if query_token_emb is not None:
                query_token_emb = query_token_emb.to(self.device)
            
            Z, all_aux = self.qformer(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                evidence_emb=evidence_embeddings,
                evidence_mask=pool_padding_mask,
                precomputed_query_emb=query_token_emb,  # None in BLIP-2 mode
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
        """Plot training and validation curves."""
        if not self.train_history['epoch']:
            print("⚠️  No training history to plot")
            return
        
        if save_path is None:
            save_path = self.save_dir / "training_curves.png"
        else:
            save_path = Path(save_path)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('TASK E Training Metrics', fontsize=16, fontweight='bold')
        
        epochs = self.train_history['epoch']
        
        # Plot: Loss
        ax.plot(epochs, self.train_history['loss'], 'b-', label='Train Loss', linewidth=2)
        if self.val_history['epoch']:
            ax.plot(self.val_history['epoch'], self.val_history['val_loss'], 
                   'r--', label='Val Loss', linewidth=2, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('TASK E (Entailment) Loss')
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
    
    Args:
        data_path: Path to smoking_train_with_NoE.pkl
        val_split: Validation split ratio
        shuffle: Whether to shuffle sample IDs
        seed: Random seed
    
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
    
    # Return data tuples (tokenizer will be provided when creating Dataset objects)
    return (data_dict, train_ids), (data_dict, val_ids)


def main():
    """Main training loop."""
    config = TaskEConfig()
    
    print("="*80)
    print("TASK E Only Training: Fragment-level Entailment Tagging")
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
    trainer = TaskETrainer(config)
    
    # Create Dataset objects
    if config.use_precomputed_embeddings:
        print("\n📦 Creating datasets with pre-computed embeddings...")
    else:
        print("\n📦 Creating datasets with XLM-RoBERTa tokenizer...")
    
    train_dataset = SmokingDataset(
        train_data[0], train_data[1], 
        tokenizer=trainer.tokenizer,
        xlm_model=trainer.xlm_embedding_model,
        device=config.device,
        max_query_len=512,
        use_precomputed_embeddings=config.use_precomputed_embeddings
    )
    val_dataset = SmokingDataset(
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
        collate_fn=collate_task_e_batch,
        num_workers=0,  # Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_task_e_batch,
        num_workers=0,
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training (TASK E Only)")
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
