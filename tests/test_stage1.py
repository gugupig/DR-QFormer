"""
Test script for Stage-1 training components.

This script verifies that all components work correctly before running full training.
"""

import sys
from pathlib import Path
import pickle
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.stage1_train import (
    Stage1Config,
    SmokingDataset,
    collate_stage1_batch,
    Stage1Trainer,
)


def create_mock_data_dict(num_samples=10):
    """Create mock data matching smoking_train_with_NoE.pkl format."""
    data_dict = {}
    
    for i in range(num_samples):
        sample_id = f"sample_{i:03d}"
        
        # Mock query embedding
        seq_len = np.random.randint(20, 50)
        query_emb = {
            'input_ids': torch.randint(0, 30000, (1, seq_len)),
            'attention_mask': torch.ones(1, seq_len),
            'token_emb_768': torch.randn(1, seq_len, 768),
        }
        
        # Mock evidence (K=11 to match real data)
        K = 11
        evidence_labels = np.random.randint(0, 2, size=K).astype(np.float32)
        evidence_embeddings = np.random.randn(K, 768).astype(np.float32)
        evidence_text = [f"Fragment {j} for sample {i}" for j in range(10)]
        
        # Mock ranking (higher rank = better)
        evidence_ranking = [(j, 1.0 - j/K) for j in range(K)]
        
        data_dict[sample_id] = {
            'query': f"What is the answer to question {i}?",
            'answer': f"Answer {i}",
            'query_embedding': query_emb,
            'evidence_labels': evidence_labels,
            'evidence_text': evidence_text,
            'evidence_embeddings': evidence_embeddings,
            'evidence_ranking': evidence_ranking,
        }
    
    return data_dict


def test_dataset():
    """Test SmokingDataset class."""
    print("="*80)
    print("Test 1: SmokingDataset")
    print("="*80)
    
    # Create mock data
    data_dict = create_mock_data_dict(num_samples=10)
    sample_ids = list(data_dict.keys())
    
    # Create dataset
    dataset = SmokingDataset(data_dict, sample_ids)
    
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Test __getitem__
    sample = dataset[0]
    print(f"\n📦 Sample 0 structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} ({value.dtype})")
        else:
            print(f"   {key}: {type(value).__name__}")
    
    print("\n✅ Test 1 passed!\n")


def test_collate():
    """Test collate_stage1_batch function."""
    print("="*80)
    print("Test 2: collate_stage1_batch")
    print("="*80)
    
    # Create mock data with variable K
    data_dict = {}
    for i in range(5):
        K = np.random.randint(8, 12)  # Variable K
        seq_len = np.random.randint(20, 50)
        
        data_dict[f"sample_{i}"] = {
            'query': f"Query {i}",
            'answer': f"Answer {i}",
            'query_embedding': {
                'input_ids': torch.randint(0, 30000, (1, seq_len)),
                'attention_mask': torch.ones(1, seq_len),
                'token_emb_768': torch.randn(1, seq_len, 768),
            },
            'evidence_labels': np.random.randint(0, 2, size=K).astype(np.float32),
            'evidence_text': [f"Frag {j}" for j in range(K-1)],
            'evidence_embeddings': np.random.randn(K, 768).astype(np.float32),
            'evidence_ranking': [(j, 1.0 - j/K) for j in range(K)],
        }
    
    dataset = SmokingDataset(data_dict, list(data_dict.keys()))
    
    # Create batch
    batch = [dataset[i] for i in range(3)]
    collated = collate_stage1_batch(batch)
    
    print(f"\n📦 Collated batch structure:")
    for key, value in collated.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} ({value.dtype})")
        elif isinstance(value, list):
            print(f"   {key}: list[{len(value)}]")
    
    # Verify padding mask
    print(f"\n🔍 Padding mask check:")
    for b in range(collated['pool_padding_mask'].shape[0]):
        num_valid = collated['pool_padding_mask'][b].sum().item()
        print(f"   Sample {b}: {num_valid} valid fragments")
    
    print("\n✅ Test 2 passed!\n")


def test_model_forward():
    """Test model forward pass."""
    print("="*80)
    print("Test 3: Model Forward Pass")
    print("="*80)
    
    config = Stage1Config(
        n_queries=16,
        num_layers=2,  # Reduce for speed
        batch_size=2,
        device='cpu',  # Use CPU for testing
    )
    
    # Create mock batch
    batch_size = 2
    seq_len = 30
    K = 10
    
    batch = {
        'queries': ['Query 1', 'Query 2'],
        'answers': ['Answer 1', 'Answer 2'],
        'query_input_ids': torch.randint(0, 30000, (batch_size, seq_len)),
        'query_attention_mask': torch.ones(batch_size, seq_len),
        'evidence_embeddings': torch.randn(batch_size, K, 768),
        'evidence_labels': torch.randint(0, 2, (batch_size, K)).float(),
        'evidence_scores': torch.rand(batch_size, K),
        'pool_padding_mask': torch.ones(batch_size, K, dtype=torch.bool),
        'sample_ids': ['s1', 's2'],
    }
    
    print(f"\n🚀 Initializing trainer (CPU)...")
    trainer = Stage1Trainer(config)
    
    print(f"\n🔄 Running forward pass...")
    trainer.qformer.eval()
    trainer.head_e.eval()
    trainer.head_s.eval()
    
    with torch.no_grad():
        # Q-Former forward
        Z, all_aux = trainer.qformer(
            input_ids=batch['query_input_ids'],
            attention_mask=batch['query_attention_mask'],
            evidence_emb=batch['evidence_embeddings'],
            evidence_mask=batch['pool_padding_mask'],
        )
        
        print(f"   Q-Former output Z: {Z.shape}")
        print(f"   Number of aux dicts: {len(all_aux)}")
        
        # Extract CA scores
        ca_raw_scores_per_head = [
            aux.get('ca_raw_scores_per_head')
            for aux in all_aux
            if aux and 'ca_raw_scores_per_head' in aux
        ]
        print(f"   CA scores from {len(ca_raw_scores_per_head)} layers")
        
        # Task E
        head_e_out = trainer.head_e(
            z=Z,
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=batch['pool_padding_mask'],
            training=False,
        )
        print(f"   Task E logits: {head_e_out['fragment_logits'].shape}")
        
        # Task S
        head_s_out = trainer.head_s(
            z=Z,
            ca_raw_scores_per_head=ca_raw_scores_per_head,
            pool_padding_mask=batch['pool_padding_mask'],
            training=False,
        )
        print(f"   Task S logits: {head_s_out['ranking_logits'].shape}")
    
    print("\n✅ Test 3 passed!\n")


def test_train_step():
    """Test single training step."""
    print("="*80)
    print("Test 4: Training Step")
    print("="*80)
    
    config = Stage1Config(
        n_queries=16,
        num_layers=2,
        batch_size=2,
        device='cpu',
        max_steps=100,
    )
    
    # Create mock batch
    batch_size = 2
    seq_len = 30
    K = 10
    
    batch = {
        'queries': ['Query 1', 'Query 2'],
        'answers': ['Answer 1', 'Answer 2'],
        'query_input_ids': torch.randint(0, 30000, (batch_size, seq_len)),
        'query_attention_mask': torch.ones(batch_size, seq_len),
        'evidence_embeddings': torch.randn(batch_size, K, 768),
        'evidence_labels': torch.randint(0, 2, (batch_size, K)).float(),
        'evidence_scores': torch.rand(batch_size, K),
        'pool_padding_mask': torch.ones(batch_size, K, dtype=torch.bool),
        'sample_ids': ['s1', 's2'],
    }
    
    print(f"\n🚀 Initializing trainer...")
    trainer = Stage1Trainer(config)
    
    print(f"\n🔄 Running training step...")
    trainer.qformer.train()
    trainer.head_e.train()
    trainer.head_s.train()
    
    metrics = trainer.train_step(batch)
    
    print(f"\n📊 Training metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\n✅ Global step: {trainer.global_step}")
    print("\n✅ Test 4 passed!\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Stage-1 Component Tests")
    print("="*80 + "\n")
    
    try:
        test_dataset()
        test_collate()
        test_model_forward()
        test_train_step()
        
        print("="*80)
        print("✅ All tests passed!")
        print("="*80)
        print("\nYou can now run the full training with:")
        print("  python train/stage1_train.py")
        print("\nOr test with real data:")
        print("  python train/stage1_train.py --train_data_path smoking_train_with_NoE.pkl")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ Test failed!")
        print("="*80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
