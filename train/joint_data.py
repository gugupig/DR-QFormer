"""
Joint Training Data Loader for Multi-Task Learning (E+S+C).

Unified data loading for all three tasks:
- Task E: Binary entailment labels per fragment
- Task S: Soft reranker scores per fragment + optional posterior
- Task C: Query-answer pairs + fragment pool + subset indices

Design:
- Dynamic K: Supports variable fragment pool sizes per sample
- Padding mask: Handles different K values within a batch
- Shared Q-Former forward: One pass produces all task inputs
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
except ImportError:
    print("Warning: PyTorch not available")
    torch = None


@dataclass
class JointTrainingSample:
    """
    Unified training sample for all tasks.
    
    Attributes:
        query: Query text string
        answer: Answer text string (for Primal QA mode)
        fragments: List of retrieved fragment texts
        
        # Task E labels
        gt_entailment: Binary labels [K] (1=entails, 0=does not entail)
        is_longtail: Binary flags [K] (1=rare example, 0=common)
        
        # Task S labels
        gt_scores: Teacher reranker scores [K] (e.g., BM25, DPR scores)
        posterior_scores: Optional LLM posterior scores [K] from Task C
        
        # Task C labels
        # (Query-answer pair is the supervision, evaluated via NLL reduction)
        
        # Metadata
        sample_id: Unique sample identifier
    """
    query: str
    answer: str
    fragments: List[str]
    
    gt_entailment: np.ndarray  # [K]
    is_longtail: np.ndarray  # [K]
    gt_scores: np.ndarray  # [K]
    posterior_scores: Optional[np.ndarray] = None  # [K]
    
    sample_id: Optional[str] = None


class JointTrainingDataset(Dataset):
    """
    Dataset for joint training of Tasks E, S, and C.
    
    Data Format (JSON or JSONL):
    ----------------------------
    {
        "sample_id": "sample_001",
        "query": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "fragments": [
            "Paris is the capital city of France.",
            "France is a country in Western Europe.",
            "Lyon is the second-largest city in France.",
            ...
        ],
        "gt_entailment": [1, 0, 0, ...],  # Task E: Binary labels
        "is_longtail": [0, 0, 1, ...],    # Task E: Longtail flags
        "gt_scores": [0.9, 0.3, 0.1, ...],  # Task S: Reranker scores
        "posterior_scores": [0.8, 0.15, 0.05, ...]  # Task S: Optional posterior (null initially)
    }
    
    Args:
        data_path: Path to JSON/JSONL file
        max_fragments: Maximum K per sample (truncate if larger)
        min_fragments: Minimum K per sample (skip if smaller)
    """
    
    def __init__(
        self,
        data_path: str,
        max_fragments: int = 5000,
        min_fragments: int = 10,
    ):
        self.data_path = Path(data_path)
        self.max_fragments = max_fragments
        self.min_fragments = min_fragments
        
        # Load data
        self.samples = self._load_data()
        
        print(f"Loaded {len(self.samples)} samples from {self.data_path}")
    
    def _load_data(self) -> List[JointTrainingSample]:
        """Load data from JSON/JSONL file."""
        samples = []
        
        if not self.data_path.exists():
            print(f"Warning: Data file not found: {self.data_path}")
            return samples
        
        # Determine file format
        if self.data_path.suffix == '.jsonl':
            # Line-delimited JSON
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        sample = self._parse_sample(data)
                        if sample is not None:
                            samples.append(sample)
        else:
            # Single JSON array
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                for data in data_list:
                    sample = self._parse_sample(data)
                    if sample is not None:
                        samples.append(sample)
        
        return samples
    
    def _parse_sample(self, data: Dict[str, Any]) -> Optional[JointTrainingSample]:
        """Parse a single sample from dictionary."""
        # Extract fields
        query = data.get('query', '')
        answer = data.get('answer', '')
        fragments = data.get('fragments', [])
        
        # Skip if too few or too many fragments
        K = len(fragments)
        if K < self.min_fragments:
            return None
        if K > self.max_fragments:
            # Truncate
            fragments = fragments[:self.max_fragments]
            K = self.max_fragments
        
        # Task E labels
        gt_entailment = np.array(data.get('gt_entailment', [0] * K), dtype=np.float32)[:K]
        is_longtail = np.array(data.get('is_longtail', [0] * K), dtype=np.float32)[:K]
        
        # Task S labels
        gt_scores = np.array(data.get('gt_scores', np.random.randn(K)), dtype=np.float32)[:K]
        
        # Normalize gt_scores to [0, 1] if needed
        if gt_scores.max() > 1.0 or gt_scores.min() < 0.0:
            gt_scores = (gt_scores - gt_scores.min()) / (gt_scores.max() - gt_scores.min() + 1e-8)
        
        # Optional posterior scores (initially None, filled by Task C during training)
        posterior_scores = data.get('posterior_scores', None)
        if posterior_scores is not None:
            posterior_scores = np.array(posterior_scores, dtype=np.float32)[:K]
        
        # Sample ID
        sample_id = data.get('sample_id', f"sample_{np.random.randint(0, 1000000)}")
        
        return JointTrainingSample(
            query=query,
            answer=answer,
            fragments=fragments,
            gt_entailment=gt_entailment,
            is_longtail=is_longtail,
            gt_scores=gt_scores,
            posterior_scores=posterior_scores,
            sample_id=sample_id,
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> JointTrainingSample:
        return self.samples[idx]


def collate_joint_batch(batch: List[JointTrainingSample]) -> Dict[str, Any]:
    """
    Collate function for joint training batches.
    
    Handles variable K per sample by:
    1. Finding max K in batch (K_max)
    2. Padding all samples to K_max
    3. Creating pool_padding_mask (True=valid, False=padding)
    
    Args:
        batch: List of JointTrainingSample
    
    Returns:
        Collated batch dictionary with keys:
            - queries: List[str] [batch]
            - answers: List[str] [batch]
            - fragments: List[List[str]] [batch, K_max]
            - gt_entailment: Tensor [batch, K_max]
            - is_longtail: Tensor [batch, K_max]
            - gt_scores: Tensor [batch, K_max]
            - posterior_scores: Tensor [batch, K_max] or None
            - pool_padding_mask: BoolTensor [batch, K_max]
            - sample_ids: List[str] [batch]
    """
    if torch is None:
        raise ImportError("PyTorch is required for collate_joint_batch")
    
    batch_size = len(batch)
    
    # Find max K in this batch
    K_max = max(len(sample.fragments) for sample in batch)
    
    # Initialize tensors
    gt_entailment_padded = torch.zeros(batch_size, K_max, dtype=torch.float32)
    is_longtail_padded = torch.zeros(batch_size, K_max, dtype=torch.float32)
    gt_scores_padded = torch.zeros(batch_size, K_max, dtype=torch.float32)
    posterior_scores_padded = torch.zeros(batch_size, K_max, dtype=torch.float32)
    pool_padding_mask = torch.zeros(batch_size, K_max, dtype=torch.bool)
    
    # Lists for text data
    queries = []
    answers = []
    fragments_padded = []
    sample_ids = []
    has_posterior = False
    
    # Pad each sample
    for b, sample in enumerate(batch):
        K_curr = len(sample.fragments)
        
        # Text data
        queries.append(sample.query)
        answers.append(sample.answer)
        sample_ids.append(sample.sample_id)
        
        # Pad fragments
        fragments = sample.fragments.copy()
        if K_curr < K_max:
            fragments.extend(["<PAD>"] * (K_max - K_curr))
        fragments_padded.append(fragments)
        
        # Pad labels
        gt_entailment_padded[b, :K_curr] = torch.from_numpy(sample.gt_entailment)
        is_longtail_padded[b, :K_curr] = torch.from_numpy(sample.is_longtail)
        gt_scores_padded[b, :K_curr] = torch.from_numpy(sample.gt_scores)
        
        if sample.posterior_scores is not None:
            posterior_scores_padded[b, :K_curr] = torch.from_numpy(sample.posterior_scores)
            has_posterior = True
        
        # Set mask (True for valid fragments)
        pool_padding_mask[b, :K_curr] = True
    
    # Return batch
    return {
        'queries': queries,
        'answers': answers,
        'fragments': fragments_padded,
        'gt_entailment': gt_entailment_padded,
        'is_longtail': is_longtail_padded,
        'gt_scores': gt_scores_padded,
        'posterior_scores': posterior_scores_padded if has_posterior else None,
        'pool_padding_mask': pool_padding_mask,
        'sample_ids': sample_ids,
    }


def create_joint_dataloaders(
    train_data_path: str,
    dev_data_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    max_fragments: int = 5000,
    min_fragments: int = 10,
) -> tuple:
    """
    Create train and dev dataloaders for joint training.
    
    Args:
        train_data_path: Path to training data
        dev_data_path: Path to dev data
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        max_fragments: Maximum fragments per sample
        min_fragments: Minimum fragments per sample
    
    Returns:
        (train_loader, dev_loader)
    """
    if torch is None:
        raise ImportError("PyTorch is required")
    
    # Create datasets
    train_dataset = JointTrainingDataset(
        data_path=train_data_path,
        max_fragments=max_fragments,
        min_fragments=min_fragments,
    )
    
    dev_dataset = JointTrainingDataset(
        data_path=dev_data_path,
        max_fragments=max_fragments,
        min_fragments=min_fragments,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_joint_batch,
        pin_memory=True,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_joint_batch,
        pin_memory=True,
    )
    
    return train_loader, dev_loader


# ===================================================================
# Placeholder: Retriever Integration
# ===================================================================
# TODO: Replace with actual retriever adapter
def encode_texts_placeholder(texts: List[str], device='cuda') -> torch.Tensor:
    """
    Placeholder for text encoding.
    
    In production, this should call:
    - RetrieverAdapter.encode_queries(queries)
    - RetrieverAdapter.encode_passages(fragments)
    
    Args:
        texts: List of text strings
        device: Device to place tensors
    
    Returns:
        embeddings: [len(texts), d_ret] tensor
    """
    if torch is None:
        raise ImportError("PyTorch is required")
    
    # Mock embeddings (768D)
    return torch.randn(len(texts), 768, device=device)


# ===================================================================
# Placeholder: LLM Integration
# ===================================================================
# TODO: Replace with actual FrozenLLM adapter
def tokenize_texts_placeholder(texts: List[str], max_length: int = 128) -> torch.Tensor:
    """
    Placeholder for text tokenization.
    
    In production, this should call:
    - FrozenLLM.tokenizer(texts, max_length=max_length, truncation=True, padding=True)
    
    Args:
        texts: List of text strings
        max_length: Maximum sequence length
    
    Returns:
        input_ids: [len(texts), max_length] tensor
    """
    if torch is None:
        raise ImportError("PyTorch is required")
    
    # Mock token IDs (random integers in vocab range)
    return torch.randint(0, 32000, (len(texts), max_length))


# ===================================================================
# Example Usage
# ===================================================================
if __name__ == "__main__":
    # Create dummy dataset for testing
    print("Creating dummy dataset...")
    
    dummy_data = [
        {
            "sample_id": f"sample_{i:03d}",
            "query": f"What is the capital of country {i}?",
            "answer": f"The capital is city {i}.",
            "fragments": [f"Fragment {j} about country {i}" for j in range(np.random.randint(50, 200))],
            "gt_entailment": np.random.randint(0, 2, size=np.random.randint(50, 200)).tolist(),
            "is_longtail": np.random.randint(0, 2, size=np.random.randint(50, 200)).tolist(),
            "gt_scores": np.random.randn(np.random.randint(50, 200)).tolist(),
        }
        for i in range(100)
    ]
    
    # Save dummy data
    dummy_path = Path("data/dummy_joint_train.json")
    dummy_path.parent.mkdir(exist_ok=True)
    with open(dummy_path, 'w') as f:
        json.dump(dummy_data, f)
    
    print(f"Saved dummy data to {dummy_path}")
    
    # Create dataset and dataloader
    if torch is not None:
        dataset = JointTrainingDataset(
            data_path=str(dummy_path),
            max_fragments=1000,
            min_fragments=10,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_joint_batch,
        )
        
        print(f"\nDataset size: {len(dataset)}")
        print("\nTesting batch collation...")
        
        for batch_idx, batch in enumerate(loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  queries: {len(batch['queries'])} samples")
            print(f"  fragments: {len(batch['fragments'])} x {len(batch['fragments'][0])} (padded)")
            print(f"  gt_entailment: {batch['gt_entailment'].shape}")
            print(f"  gt_scores: {batch['gt_scores'].shape}")
            print(f"  pool_padding_mask: {batch['pool_padding_mask'].shape}")
            print(f"  Valid fragments per sample: {batch['pool_padding_mask'].sum(dim=1).tolist()}")
            
            if batch_idx >= 2:
                break
        
        print("\n✅ Data loading test passed!")
