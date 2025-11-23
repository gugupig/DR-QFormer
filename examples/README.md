# Examples Directory

This directory contains example scripts demonstrating how to use DR-QFormer components.

## Available Examples

### Stage-2 Posterior Extraction Example

**File**: `stage2_posterior_extraction_example.py`

**Purpose**: Demonstrates full integration of MACS×LQ-CA posterior extraction in Stage-2 joint training (Task E + S + C).

**What It Shows**:
- Complete training step with posterior feedback
- Curriculum learning schedule (λ_teacher → λ_post transition)
- End-to-end MACS posterior extraction
- Integration with Task S loss

**Run**:
```bash
python examples/stage2_posterior_extraction_example.py
```

**Output**:
```
================================================================================
Stage-2 Training Example: Joint E+S+C with MACS Posterior Feedback
================================================================================

Running 10 training steps with curriculum learning...
- Warmup: steps 0-3 (teacher only)
- Transition: steps 3-8 (teacher→posterior)
- Steady: steps 8+ (posterior dominant)

Step   0 | λ_t=1.00 λ_p=0.00 | L_total=... | L_E=... | L_S=... | L_C=... | Posterior=✗
Step   1 | λ_t=1.00 λ_p=0.00 | L_total=... | L_E=... | L_S=... | L_C=... | Posterior=✗
...
Step   8 | λ_t=0.20 λ_p=0.80 | L_total=... | L_E=... | L_S=... | L_C=... | Posterior=✓
Step   9 | λ_t=0.20 λ_p=0.80 | L_total=... | L_E=... | L_S=... | L_C=... | Posterior=✓
```

**Key Components**:

1. **`stage2_training_step_with_posterior()`**
   - Complete training step function
   - Q-Former forward (all three tasks)
   - LLM teacher forcing (Task C)
   - MACS posterior extraction
   - Task S loss with posterior feedback

2. **`curriculum_schedule()`**
   - Implements λ_teacher and λ_post annealing
   - Three phases: Warmup → Transition → Steady

3. **`example_stage2_training_loop()`**
   - Full training loop with dummy data
   - Demonstrates curriculum learning in action
   - Shows loss progression

**Note**: Currently uses placeholder LLM (returns dummy outputs). Once real Qwen LLM is integrated in `src/adapters/llm.py`, this example will produce real posterior distributions.

---

## Usage Tips

### Modify for Real Data

Replace dummy batch with real dataset:

```python
from src.data import load_msmarco_batch

batch = load_msmarco_batch(
    batch_size=16,
    split='train',
)
```

### Adjust Curriculum Schedule

Customize transition speed:

```python
schedule = curriculum_schedule(
    current_step=step,
    warmup_steps=2000,      # Longer warmup
    transition_steps=8000,  # Slower transition
)
```

### Monitor Posterior Quality

Add posterior validation metrics:

```python
# Check JS divergence between prior and posterior
js_div = loss_s_dict['loss_post']
print(f"Prior-Posterior JS Divergence: {js_div:.4f}")

# Check posterior entropy
entropy = -(evidence_posterior * evidence_posterior.log()).sum(dim=-1)
print(f"Posterior Entropy: {entropy.mean():.3f}")
```

---

## Next Steps

1. **Integrate Real LLM**: Replace `FrozenLLM` placeholder with actual Qwen model
2. **Add Real Data**: Use MS-MARCO or other RAG datasets
3. **Create Full Training Script**: Move from example to production training in `train/stage2_joint.py`
4. **Add Validation Loop**: Evaluate posterior quality and retrieval metrics
5. **Scale Up**: Test with G=64, 128, 256 evidence pools

---

## Related Documentation

- **MACS Guide**: `documents/MACS_POSTERIOR_EXTRACTION_GUIDE.md`
- **Implementation Summary**: `documents/MACS_IMPLEMENTATION_SUMMARY.md`
- **Visual Flow**: `documents/MACS_VISUAL_FLOW.md`
- **Core Implementation**: `src/utils/macs.py`
- **Tests**: `tests/test_macs_posterior.py`

---

## Questions?

See the main README or documentation in `documents/` directory.
