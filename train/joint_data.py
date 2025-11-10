"""
Joint Training Data Module for DR-QFormer

Provides data loading, batching, and collation for joint E/S/C training.
Handles variable-length fragments with pool_padding_mask.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


@dataclass
class JointBatch:
    """
    Batch for joint training of Task E, S, and C.
    
    All tasks share the same Q-Former forward pass, so they receive:
    - Same query/answer embeddings
    - Same fragment pool embeddings
    - Same pool_padding_mask
    """
    # Common inputs (shared across E/S/C)
    query_embeds: torch.Tensor          # [batch, 1, d] - single query embedding per sample
    answer_embeds: torch.Tensor         # [batch, 1, d] - single answer embedding per sample
    fragment_embeds: torch.Tensor       # [batch, K_max, d] - padded fragment pool
    pool_padding_mask: torch.Tensor     # [batch, K_max] - True for valid fragments
    
    # Task E labels
    entailment_labels: torch.Tensor     # [batch, K_max] - {0: neg, 1: pos}
    ranking_scores: torch.Tensor        # [batch, K_max] - teacher scores (e.g., BM25)
    question_tokens: torch.Tensor       # [batch, L_q] - tokenized question
    answer_tokens: torch.Tensor         # [batch, L_a] - tokenized answer
    
    # Optional fields
    entailment_weights: Optional[torch.Tensor] = None  # [batch, K_max] - importance weights
    ranking_temperature: Optional[float] = None  # For adaptive temperature in ListNet
    
    # Optional: Dual mode data (for QG task)
    dual_mode: bool = False
    dual_query_embeds: Optional[torch.Tensor] = None
    dual_answer_embeds: Optional[torch.Tensor] = None
    dual_entailment_labels: Optional[torch.Tensor] = None
    dual_ranking_scores: Optional[torch.Tensor] = None
    dual_question_tokens: Optional[torch.Tensor] = None
    dual_answer_tokens: Optional[torch.Tensor] = None
    
    # Metadata
    batch_size: int = 0
    k_valid: Optional[List[int]] = None  # Valid fragments per sample (for debugging)


class JointDataset(Dataset):
    """
    Dataset for joint training.
    
    Each sample contains:
    - Query and answer (for Q-Former conditioning)
    - Fragment pool with variable length K
    - Labels for E (entailment), S (ranking), C (generation)
    
    Placeholders for actual data loading:
    - Replace with your retriever (e.g., BM25, dense retriever)
    - Replace with your LLM tokenizer
    - Replace with your embedding model (for query/answer/fragments)
    """
    
    def __init__(
        self,
        data_path: str,
        max_fragments: int = 100,
        dual_mode: bool = False,
        # PLACEHOLDER: Add your retriever/tokenizer/embedder here
    ):
        """
        Args:
            data_path: Path to dataset (e.g., JSON/CSV with Q/A pairs)
            max_fragments: Maximum K per sample (pad shorter pools)
            dual_mode: Whether to include QG (dual) data
        """
        self.data_path = data_path
        self.max_fragments = max_fragments
        self.dual_mode = dual_mode
        
        # PLACEHOLDER: Load dataset
        # Example: self.data = load_qa_dataset(data_path)
        self.data = self._load_placeholder_data()
        
        # PLACEHOLDER: Initialize retriever
        # Example: self.retriever = BM25Retriever(corpus_path)
        self.retriever = None
        
        # PLACEHOLDER: Initialize embedding models
        # Example: self.embedder = SentenceTransformer(model_name)
        self.embedder = None
        
        # PLACEHOLDER: Initialize LLM tokenizer
        # Example: self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer = None
    
    def _load_placeholder_data(self) -> List[Dict[str, Any]]:
        """
        PLACEHOLDER: Load your actual dataset here.
        
        Expected format:
        [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "corpus_id": "corpus_123",  # For retrieval
                # Optional: Pre-retrieved fragments
                "fragments": ["Paris is the capital...", "France is a country..."],
                "entailment_labels": [1, 0],  # For Task E
                "ranking_scores": [0.95, 0.45],  # For Task S
            },
            ...
        ]
        """
        print(f"[JointDataset] PLACEHOLDER: Loading data from {self.data_path}")
        print("[JointDataset] TODO: Replace with actual data loading logic")
        
        # Dummy data for structure
        return [
            {
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "fragments": [f"Fragment {i}-{j}" for j in range(np.random.randint(10, 50))],
                "entailment_labels": np.random.randint(0, 2, size=np.random.randint(10, 50)).tolist(),
                "ranking_scores": np.random.rand(np.random.randint(10, 50)).tolist(),
            }
            for i in range(100)
        ]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a single sample with all necessary data for E/S/C.
        
        PLACEHOLDER: Replace embeddings with actual model outputs.
        """
        sample = self.data[idx]
        
        # PLACEHOLDER: Retrieve fragments if not pre-retrieved
        if "fragments" not in sample and self.retriever is not None:
            # fragments = self.retriever.retrieve(sample["question"], top_k=self.max_fragments)
            fragments = [f"Retrieved fragment {i}" for i in range(20)]
        else:
            fragments = sample.get("fragments", [])
        
        k_actual = len(fragments)
        
        # PLACEHOLDER: Embed query, answer, fragments
        # query_embed = self.embedder.encode(sample["question"])  # [d]
        # answer_embed = self.embedder.encode(sample["answer"])   # [d]
        # fragment_embeds = self.embedder.encode(fragments)       # [K, d]
        
        # Dummy embeddings (replace with actual)
        d = 128
        query_embed = np.random.randn(d).astype(np.float32)
        answer_embed = np.random.randn(d).astype(np.float32)
        fragment_embeds = np.random.randn(k_actual, d).astype(np.float32)
        
        # Task E labels (entailment)
        entailment_labels = sample.get("entailment_labels", [0] * k_actual)
        entailment_weights = sample.get("entailment_weights", [1.0] * k_actual)
        
        # Task S labels (ranking)
        ranking_scores = sample.get("ranking_scores", np.random.rand(k_actual).tolist())
        
        # Task C labels (tokenized Q/A)
        # PLACEHOLDER: Tokenize with LLM tokenizer
        # question_tokens = self.tokenizer.encode(sample["question"], return_tensors="pt")[0]
        # answer_tokens = self.tokenizer.encode(sample["answer"], return_tensors="pt")[0]
        
        # Dummy tokens (replace with actual)
        question_tokens = np.random.randint(0, 32000, size=10)
        answer_tokens = np.random.randint(0, 32000, size=15)
        
        return {
            "query_embed": query_embed,
            "answer_embed": answer_embed,
            "fragment_embeds": fragment_embeds,
            "k_actual": k_actual,
            "entailment_labels": entailment_labels,
            "entailment_weights": entailment_weights,
            "ranking_scores": ranking_scores,
            "question_tokens": question_tokens,
            "answer_tokens": answer_tokens,
            # Metadata
            "question_text": sample["question"],
            "answer_text": sample["answer"],
        }


def collate_joint_batch(batch: List[Dict[str, Any]]) -> JointBatch:
    """
    Collate function for DataLoader.
    
    Handles variable-length K by padding to max K in batch and creating pool_padding_mask.
    """
    batch_size = len(batch)
    k_max = max(item["k_actual"] for item in batch)
    d = batch[0]["query_embed"].shape[0]
    
    # Initialize tensors
    query_embeds = torch.zeros(batch_size, 1, d)
    answer_embeds = torch.zeros(batch_size, 1, d)
    fragment_embeds = torch.zeros(batch_size, k_max, d)
    pool_padding_mask = torch.zeros(batch_size, k_max, dtype=torch.bool)
    
    entailment_labels = torch.zeros(batch_size, k_max, dtype=torch.long)
    entailment_weights = torch.ones(batch_size, k_max)
    ranking_scores = torch.zeros(batch_size, k_max)
    
    # Track valid K per sample
    k_valid = []
    
    # Fill tensors
    for i, item in enumerate(batch):
        k = item["k_actual"]
        k_valid.append(k)
        
        # Embeddings
        query_embeds[i, 0] = torch.from_numpy(item["query_embed"])
        answer_embeds[i, 0] = torch.from_numpy(item["answer_embed"])
        fragment_embeds[i, :k] = torch.from_numpy(item["fragment_embeds"])
        pool_padding_mask[i, :k] = True
        
        # Task E
        entailment_labels[i, :k] = torch.tensor(item["entailment_labels"][:k], dtype=torch.long)
        entailment_weights[i, :k] = torch.tensor(item["entailment_weights"][:k])
        
        # Task S
        ranking_scores[i, :k] = torch.tensor(item["ranking_scores"][:k])
    
    # Task C: Pad question/answer tokens
    max_q_len = max(len(item["question_tokens"]) for item in batch)
    max_a_len = max(len(item["answer_tokens"]) for item in batch)
    
    question_tokens = torch.zeros(batch_size, max_q_len, dtype=torch.long)
    answer_tokens = torch.zeros(batch_size, max_a_len, dtype=torch.long)
    
    for i, item in enumerate(batch):
        q_len = len(item["question_tokens"])
        a_len = len(item["answer_tokens"])
        question_tokens[i, :q_len] = torch.from_numpy(item["question_tokens"])
        answer_tokens[i, :a_len] = torch.from_numpy(item["answer_tokens"])
    
    return JointBatch(
        query_embeds=query_embeds,
        answer_embeds=answer_embeds,
        fragment_embeds=fragment_embeds,
        pool_padding_mask=pool_padding_mask,
        entailment_labels=entailment_labels,
        entailment_weights=entailment_weights,
        ranking_scores=ranking_scores,
        ranking_temperature=None,  # Computed adaptively in trainer
        question_tokens=question_tokens,
        answer_tokens=answer_tokens,
        batch_size=batch_size,
        k_valid=k_valid,
    )


def create_joint_dataloader(
    data_path: str,
    batch_size: int = 8,
    max_fragments: int = 100,
    num_workers: int = 4,
    shuffle: bool = True,
    dual_mode: bool = False,
) -> DataLoader:
    """
    Create DataLoader for joint training.
    
    Args:
        data_path: Path to dataset
        batch_size: Batch size
        max_fragments: Max K per sample
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        dual_mode: Whether to include dual (QG) data
    
    Returns:
        DataLoader with collate_joint_batch
    """
    dataset = JointDataset(
        data_path=data_path,
        max_fragments=max_fragments,
        dual_mode=dual_mode,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_joint_batch,
        pin_memory=True,
    )
    
    return dataloader


# ============================================================================
# Utility functions for data inspection
# ============================================================================

def inspect_batch(batch: JointBatch) -> None:
    """Print batch statistics for debugging."""
    print("=" * 80)
    print("Joint Batch Inspection")
    print("=" * 80)
    print(f"Batch size: {batch.batch_size}")
    print(f"Valid fragments per sample: {batch.k_valid}")
    print(f"K_max (padded): {batch.fragment_embeds.shape[1]}")
    print()
    
    print("Shapes:")
    print(f"  query_embeds: {batch.query_embeds.shape}")
    print(f"  answer_embeds: {batch.answer_embeds.shape}")
    print(f"  fragment_embeds: {batch.fragment_embeds.shape}")
    print(f"  pool_padding_mask: {batch.pool_padding_mask.shape}")
    print()
    
    print("Task E (Entailment):")
    print(f"  Labels: {batch.entailment_labels.shape}")
    if batch.entailment_weights is not None:
        print(f"  Weights: {batch.entailment_weights.shape}")
    print(f"  Positive rate: {(batch.entailment_labels == 1).float().mean():.3f}")
    print()
    
    print("Task S (Ranking):")
    print(f"  Scores: {batch.ranking_scores.shape}")
    print(f"  Score range: [{batch.ranking_scores.min():.3f}, {batch.ranking_scores.max():.3f}]")
    print()
    
    print("Task C (Generation):")
    print(f"  Question tokens: {batch.question_tokens.shape}")
    print(f"  Answer tokens: {batch.answer_tokens.shape}")
    print()


if __name__ == "__main__":
    # Test data loading
    print("Testing JointDataset and collate_joint_batch...")
    
    # Create dummy dataloader
    dataloader = create_joint_dataloader(
        data_path="dummy_path.json",
        batch_size=4,
        max_fragments=50,
        num_workers=0,
        shuffle=False,
    )
    
    # Inspect first batch
    batch = next(iter(dataloader))
    inspect_batch(batch)
    
    print("\n✅ Data module test complete!")
    print("⚠️  Remember to replace placeholders with actual:")
    print("   - Retriever (e.g., BM25, dense retriever)")
    print("   - Embedding model (for query/answer/fragments)")
    print("   - LLM tokenizer (for Task C)")
    print("   - Dataset loading logic")
