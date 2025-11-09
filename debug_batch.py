"""Debug script to check DataLoader batch format."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader, Dataset

class DummyRankingDataset(Dataset):
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        K = 100
        return {
            "queries": f"What is the capital of country {idx}?",
            "fragments": [f"Fragment {i} about country {idx}" for i in range(K)],
            "gt_scores": torch.randn(K).softmax(dim=0).numpy(),
            "answers": f"Capital {idx}",
        }

dataset = DummyRankingDataset(size=5)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch_idx, batch in enumerate(loader):
    print(f"\nBatch {batch_idx}:")
    print(f"Keys: {batch.keys()}")
    print(f"queries type: {type(batch['queries'])}")
    print(f"fragments type: {type(batch['fragments'])}")
    if isinstance(batch['queries'], (list, tuple)):
        print(f"queries length: {len(batch['queries'])}")
        print(f"fragments length: {len(batch['fragments'])}")
        print(f"First query: {batch['queries'][0]}")
        print(f"First fragments count: {len(batch['fragments'][0])}")
    break
