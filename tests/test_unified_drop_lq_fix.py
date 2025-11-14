"""
Test to verify unified Drop-LQ is actually enabled in joint training.

This test addresses the issue where:
- Q-Former accepts lq_drop_mask parameter
- All three task heads support lq_drop_mask parameter
- But the joint trainer was NOT generating or passing the unified mask
- Result: lq_drop_mask was always None, unified Drop-LQ was never active

The fix:
1. Add p_drop_lq_unified config parameter
2. Generate unified mask in train_step() before Q-Former forward
3. Pass mask to Q-Former and all three task heads
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from train.task_joint import JointTrainer, JointTrainingConfig
from train.schedule import ScheduleConfig


def test_unified_drop_lq_generation():
    """Test that unified Drop-LQ mask is generated during training."""
    print("\n" + "="*80)
    print("TEST 1: Unified Drop-LQ Mask Generation")
    print("="*80)
    
    # Setup config with unified Drop-LQ enabled
    config = JointTrainingConfig()
    config.n_queries = 8
    config.p_drop_lq_unified = 0.3  # 30% drop rate (easier to observe)
    config.max_steps = 100
    config.device = "cpu"
    
    trainer = JointTrainer(config)
    trainer.qformer.train()  # Enable training mode
    
    # Create mock batch
    batch = {
        'queries': ['What is AI?'] * 4,
        'answers': ['Artificial Intelligence'] * 4,
        'fragments': [[f'Fragment {j}' for j in range(10)] for _ in range(4)],
        'gt_entailment': torch.randint(0, 2, (4, 10)),
        'gt_scores': torch.rand(4, 10),
        'is_longtail': torch.randint(0, 2, (4, 10)),
        'posterior_scores': torch.rand(4, 10),
        'pool_padding_mask': torch.ones(4, 10, dtype=torch.bool),
    }
    
    # Mock forward to capture lq_drop_mask
    original_qformer_forward = trainer.qformer.forward
    captured_mask = [None]
    
    def mock_forward(*args, **kwargs):
        captured_mask[0] = kwargs.get('lq_drop_mask', None)
        return original_qformer_forward(*args, **kwargs)
    
    trainer.qformer.forward = mock_forward
    
    # Run training step
    metrics = trainer.train_step(batch)
    
    # Verify mask was generated and passed
    assert captured_mask[0] is not None, "❌ FAILED: lq_drop_mask is None (not generated)"
    
    mask = captured_mask[0]
    assert mask.shape == (4, 8, 1), f"❌ FAILED: Wrong shape {mask.shape}, expected (4, 8, 1)"
    assert mask.dtype == torch.bool, f"❌ FAILED: Wrong dtype {mask.dtype}, expected bool"
    
    # Check that some LQs are dropped (with 30% rate, should have drops)
    num_kept = mask.sum().item()
    num_total = 4 * 8
    drop_rate_actual = 1.0 - (num_kept / num_total)
    
    print(f"\n✅ Unified Drop-LQ mask generated:")
    print(f"   - Shape: {mask.shape}")
    print(f"   - Dtype: {mask.dtype}")
    print(f"   - LQs kept: {num_kept} / {num_total} ({100*(1-drop_rate_actual):.1f}%)")
    print(f"   - Actual drop rate: {100*drop_rate_actual:.1f}% (target: 30%)")
    
    # Check drop rate is reasonable (20-40% range for 30% target)
    assert 0.15 < drop_rate_actual < 0.45, \
        f"❌ FAILED: Drop rate {100*drop_rate_actual:.1f}% too far from 30% target"
    
    print("\n🎉 TEST 1 PASSED!")


def test_safety_mechanism():
    """Test that at least 1 LQ is kept per sample (safety mechanism)."""
    print("\n" + "="*80)
    print("TEST 2: Safety Mechanism (Extreme Drop Rate)")
    print("="*80)
    
    # Setup config with extreme drop rate
    config = JointTrainingConfig()
    config.n_queries = 8
    config.p_drop_lq_unified = 0.95  # 95% drop rate (should still keep ≥1 per sample)
    config.max_steps = 100
    config.device = "cpu"
    
    trainer = JointTrainer(config)
    trainer.qformer.train()
    
    # Create mock batch
    batch = {
        'queries': ['Test query'] * 4,
        'answers': ['Test answer'] * 4,
        'fragments': [[f'Frag {j}' for j in range(10)] for _ in range(4)],
        'gt_entailment': torch.randint(0, 2, (4, 10)),
        'gt_scores': torch.rand(4, 10),
        'is_longtail': torch.zeros(4, 10, dtype=torch.long),
        'posterior_scores': torch.rand(4, 10),
        'pool_padding_mask': torch.ones(4, 10, dtype=torch.bool),
    }
    
    # Capture mask
    captured_mask = [None]
    original_qformer_forward = trainer.qformer.forward
    
    def mock_forward(*args, **kwargs):
        captured_mask[0] = kwargs.get('lq_drop_mask', None)
        return original_qformer_forward(*args, **kwargs)
    
    trainer.qformer.forward = mock_forward
    
    # Run training step
    metrics = trainer.train_step(batch)
    
    mask = captured_mask[0]
    assert mask is not None, "❌ FAILED: No mask generated"
    
    # Check that EVERY sample has at least 1 LQ kept
    lqs_kept_per_sample = mask.sum(dim=1).squeeze(-1)  # [batch]
    
    print(f"\n✅ Safety mechanism working:")
    print(f"   - Drop rate: 95%")
    print(f"   - LQs kept per sample: {lqs_kept_per_sample.tolist()}")
    
    for b in range(4):
        assert lqs_kept_per_sample[b] >= 1, \
            f"❌ FAILED: Sample {b} has 0 LQs kept (safety failed)"
    
    print(f"   - All samples have ≥1 LQ kept ✓")
    print("\n🎉 TEST 2 PASSED!")


def test_disable_unified_drop_lq():
    """Test that unified Drop-LQ can be disabled (p_drop_lq_unified=0.0)."""
    print("\n" + "="*80)
    print("TEST 3: Unified Drop-LQ Disabled")
    print("="*80)
    
    # Setup config with unified Drop-LQ disabled
    config = JointTrainingConfig()
    config.n_queries = 8
    config.p_drop_lq_unified = 0.0  # Disabled
    config.max_steps = 100
    config.device = "cpu"
    
    trainer = JointTrainer(config)
    trainer.qformer.train()
    
    # Create mock batch
    batch = {
        'query': ['Test'] * 2,
        'answer': ['Test'] * 2,
        'fragments': [[f'F{j}' for j in range(5)] for _ in range(2)],
        'gt_entailment': torch.randint(0, 2, (2, 5)),
        'gt_scores': torch.rand(2, 5),
        'is_longtail': torch.zeros(2, 5, dtype=torch.long),
        'posterior_scores': torch.rand(2, 5),
        'pool_padding_mask': torch.ones(2, 5, dtype=torch.bool),
    }
    
    # Capture mask
    captured_mask = [None]
    original_qformer_forward = trainer.qformer.forward
    
    def mock_forward(*args, **kwargs):
        captured_mask[0] = kwargs.get('lq_drop_mask', None)
        return original_qformer_forward(*args, **kwargs)
    
    trainer.qformer.forward = mock_forward
    
    # Run training step
    metrics = trainer.train_step(batch)
    
    # Verify mask is None when disabled
    assert captured_mask[0] is None, \
        f"❌ FAILED: lq_drop_mask should be None when p_drop_lq_unified=0.0, got {captured_mask[0]}"
    
    print(f"\n✅ Unified Drop-LQ correctly disabled:")
    print(f"   - p_drop_lq_unified = 0.0")
    print(f"   - lq_drop_mask = None ✓")
    print("\n🎉 TEST 3 PASSED!")


def test_eval_mode_disables_drop_lq():
    """Test that Drop-LQ is disabled in eval mode."""
    print("\n" + "="*80)
    print("TEST 4: Drop-LQ Disabled in Eval Mode")
    print("="*80)
    
    # Setup config
    config = JointTrainingConfig()
    config.n_queries = 8
    config.p_drop_lq_unified = 0.3  # 30% drop rate
    config.max_steps = 100
    config.device = "cpu"
    
    trainer = JointTrainer(config)
    trainer.qformer.eval()  # Set to eval mode
    
    # Create mock batch
    batch = {
        'query': ['Test'] * 2,
        'answer': ['Test'] * 2,
        'fragments': [[f'F{j}' for j in range(5)] for _ in range(2)],
        'gt_entailment': torch.randint(0, 2, (2, 5)),
        'gt_scores': torch.rand(2, 5),
        'is_longtail': torch.zeros(2, 5, dtype=torch.long),
        'posterior_scores': torch.rand(2, 5),
        'pool_padding_mask': torch.ones(2, 5, dtype=torch.bool),
    }
    
    # Capture mask
    captured_mask = [None]
    original_qformer_forward = trainer.qformer.forward
    
    def mock_forward(*args, **kwargs):
        captured_mask[0] = kwargs.get('lq_drop_mask', None)
        return original_qformer_forward(*args, **kwargs)
    
    trainer.qformer.forward = mock_forward
    
    # Run training step (but Q-Former is in eval mode)
    with torch.no_grad():
        metrics = trainer.train_step(batch)
    
    # Verify mask is None in eval mode
    assert captured_mask[0] is None, \
        f"❌ FAILED: lq_drop_mask should be None in eval mode, got {captured_mask[0]}"
    
    print(f"\n✅ Drop-LQ correctly disabled in eval mode:")
    print(f"   - Q-Former training = False")
    print(f"   - lq_drop_mask = None ✓")
    print("\n🎉 TEST 4 PASSED!")


def test_all_heads_receive_mask():
    """Test that all three task heads receive the unified mask."""
    print("\n" + "="*80)
    print("TEST 5: All Task Heads Receive Unified Mask")
    print("="*80)
    
    # Setup config
    config = JointTrainingConfig()
    config.n_queries = 8
    config.p_drop_lq_unified = 0.2  # 20% drop
    config.max_steps = 100
    config.device = "cpu"
    
    trainer = JointTrainer(config)
    trainer.qformer.train()
    
    # Create mock batch
    batch = {
        'query': ['Q'] * 2,
        'answer': ['A'] * 2,
        'fragments': [[f'F{j}' for j in range(5)] for _ in range(2)],
        'gt_entailment': torch.randint(0, 2, (2, 5)),
        'gt_scores': torch.rand(2, 5),
        'is_longtail': torch.zeros(2, 5, dtype=torch.long),
        'posterior_scores': torch.rand(2, 5),
        'pool_padding_mask': torch.ones(2, 5, dtype=torch.bool),
    }
    
    # Capture masks passed to each head
    captured_masks = {'E': None, 'S': None, 'C': None}
    
    original_head_e_forward = trainer.head_e.forward
    original_head_s_forward = trainer.head_s.forward
    original_head_c_forward = trainer.head_c.forward
    
    def mock_head_e_forward(*args, **kwargs):
        captured_masks['E'] = kwargs.get('lq_drop_mask', None)
        return original_head_e_forward(*args, **kwargs)
    
    def mock_head_s_forward(*args, **kwargs):
        captured_masks['S'] = kwargs.get('lq_drop_mask', None)
        return original_head_s_forward(*args, **kwargs)
    
    def mock_head_c_forward(*args, **kwargs):
        captured_masks['C'] = kwargs.get('lq_drop_mask', None)
        return original_head_c_forward(*args, **kwargs)
    
    trainer.head_e.forward = mock_head_e_forward
    trainer.head_s.forward = mock_head_s_forward
    trainer.head_c.forward = mock_head_c_forward
    
    # Run training step
    metrics = trainer.train_step(batch)
    
    # Verify all heads received the same mask
    mask_e = captured_masks['E']
    mask_s = captured_masks['S']
    mask_c = captured_masks['C']
    
    assert mask_e is not None, "❌ FAILED: Task E did not receive mask"
    assert mask_s is not None, "❌ FAILED: Task S did not receive mask"
    assert mask_c is not None, "❌ FAILED: Task C did not receive mask"
    
    # Check that all masks are identical (unified)
    assert torch.equal(mask_e, mask_s), "❌ FAILED: Task E and S received different masks"
    assert torch.equal(mask_e, mask_c), "❌ FAILED: Task E and C received different masks"
    
    print(f"\n✅ All task heads received unified mask:")
    print(f"   - Task E mask: shape {mask_e.shape}, {mask_e.sum().item()} LQs kept")
    print(f"   - Task S mask: shape {mask_s.shape}, {mask_s.sum().item()} LQs kept")
    print(f"   - Task C mask: shape {mask_c.shape}, {mask_c.sum().item()} LQs kept")
    print(f"   - All masks identical ✓")
    print("\n🎉 TEST 5 PASSED!")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("UNIFIED DROP-LQ FIX VERIFICATION TESTS")
    print("="*80)
    
    try:
        test_unified_drop_lq_generation()
        test_safety_mechanism()
        test_disable_unified_drop_lq()
        test_eval_mode_disables_drop_lq()
        test_all_heads_receive_mask()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*80)
        print("\nUnified Drop-LQ is now correctly implemented:")
        print("  ✅ Mask generation enabled in JointTrainer")
        print("  ✅ Mask passed to Q-Former forward")
        print("  ✅ Mask passed to all three task heads (E, S, C)")
        print("  ✅ Safety mechanism working (≥1 LQ kept)")
        print("  ✅ Can be disabled (p_drop_lq_unified=0.0)")
        print("  ✅ Automatically disabled in eval mode")
        print("  ✅ All heads receive identical mask (unified)")
        print("\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
