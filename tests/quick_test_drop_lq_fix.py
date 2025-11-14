"""Quick test to verify unified Drop-LQ fix."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from train.task_joint import JointTrainer, JointTrainingConfig

config = JointTrainingConfig()
config.n_queries = 8
config.p_drop_lq_unified = 0.3
config.device = "cpu"

trainer = JointTrainer(config)
trainer.qformer.train()

batch = {
    'queries': ['What?'] * 2,
    'answers': ['AI'] * 2,
    'fragments': [[f'F{j}' for j in range(5)] for _ in range(2)],
    'gt_entailment': torch.randint(0, 2, (2, 5)),
    'gt_scores': torch.rand(2, 5),
    'is_longtail': torch.zeros(2, 5, dtype=torch.long),
    'posterior_scores': torch.rand(2, 5),
    'pool_padding_mask': torch.ones(2, 5, dtype=torch.bool),
}

# Capture mask passed to Q-Former
captured_mask = []
orig_forward = trainer.qformer.forward
def mock(*args, **kwargs):
    captured_mask.append(kwargs.get('lq_drop_mask'))
    return orig_forward(*args, **kwargs)
trainer.qformer.forward = mock

# Run
metrics = trainer.train_step(batch)

# Check
mask = captured_mask[0]
assert mask is not None, "❌ FAILED: No mask generated!"
assert mask.shape == (2, 8, 1), f"❌ Wrong shape: {mask.shape}"
print(f"✅ Unified Drop-LQ WORKING!")
print(f"   Mask shape: {mask.shape}, kept: {mask.sum().item()}/{2*8}")
