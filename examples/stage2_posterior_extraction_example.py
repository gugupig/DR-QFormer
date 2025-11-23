"""
Example: Using MACS posterior extraction in Stage-2 training (Task E + S + C).

This script demonstrates how to integrate MACS×LQ-CA posterior feedback into
the Task S loss during joint training.

Stage-2 Training Loop:
======================
1. Q-Former forward (Tasks E, S, C)
2. Task E loss: Focal loss on fragment entailment
3. Task S loss: ListNet + JS divergence (teacher + posterior)
4. Task C loss: Contrastive NLL with LLM
5. LLM attention → MACS posterior extraction → Task S feedback

Key Innovation:
===============
Posterior feedback loop: Task C's LLM attention reveals which evidence the LLM
actually uses, and this posterior signal refines Task S's prior predictions.

Usage:
======
    python examples/stage2_posterior_extraction_example.py
"""

import torch
from torch import Tensor
from typing import Dict, Optional

# Import DR-QFormer components
from src.models.qformer import QFormer
from src.models.heads import EntailmentHead, FragmentRankingHead, CondenseHead
from src.adapters.llm import FrozenLLM
from src.losses import compute_entailment_loss, compute_ranking_loss, compute_condensing_loss
from src.utils.macs import extract_posterior_from_llm_outputs


def stage2_training_step_with_posterior(
    qformer: QFormer,
    entailment_head: EntailmentHead,
    ranking_head: FragmentRankingHead,
    condense_head: CondenseHead,
    frozen_llm: FrozenLLM,
    batch: Dict[str, Tensor],
    lambda_teacher: float = 0.5,
    lambda_post: float = 0.5,
    use_posterior_feedback: bool = True,
) -> Dict[str, Tensor]:
    """
    Single training step for Stage-2 joint training with posterior feedback.
    
    Args:
        qformer: DR-QFormer model
        entailment_head: Task E head
        ranking_head: Task S head
        condense_head: Task C head
        frozen_llm: Frozen LLM for generation
        batch: Training batch with keys:
            - 'query_embeddings': [B, T_q, 768]
            - 'evidence_embeddings': [B, G, 768]
            - 'entailment_labels': [B, G]
            - 'teacher_scores': [B, G]
            - 'query_input_ids': [B, S_q]
            - 'answer_input_ids': [B, S_a]
            - 'padding_mask': [B, G]
            - 'subset_mask': [B, G]  # Dynamic subset U
        lambda_teacher: Weight for teacher supervision in Task S
        lambda_post: Weight for posterior feedback in Task S
        use_posterior_feedback: Whether to compute and use posterior
    
    Returns:
        loss_dict: Dictionary with all losses and metrics
    """
    batch_size = batch['query_embeddings'].shape[0]
    device = batch['query_embeddings'].device
    
    # =========================================================================
    # Step 1: Q-Former Forward (Primal Mode)
    # =========================================================================
    qformer_outputs = qformer(
        query_embeddings=batch['query_embeddings'],
        evidence_pool_embeddings=batch['evidence_embeddings'],
        mode='primal',  # QA direction
    )
    
    # Outputs:
    # - 'lqs_after': [B, N, 768] - Final LQ representations
    # - 'ca_raw_scores_per_head': List[Tuple] - Per-layer, per-head CA scores
    # - 'ca_raw_scores_avg': List[Tensor] - Per-layer averaged CA scores
    
    # =========================================================================
    # Step 2: Task E - Entailment Prediction
    # =========================================================================
    entailment_outputs = entailment_head(
        ca_raw_scores_per_head=qformer_outputs['ca_raw_scores_per_head'],
        pool_padding_mask=batch['padding_mask'],
        training=True,  # Enable Drop-LQ
    )
    
    loss_e = compute_entailment_loss(
        logits=entailment_outputs['logits'],
        labels=batch['entailment_labels'],
        padding_mask=batch['padding_mask'],
        importance_weights=batch.get('importance_weights', None),
    )
    
    # =========================================================================
    # Step 3: Task S - Ranking with Posterior Feedback
    # =========================================================================
    ranking_outputs = ranking_head(
        ca_raw_scores_per_head=qformer_outputs['ca_raw_scores_per_head'],
        pool_padding_mask=batch['padding_mask'],
        training=False,  # No stochasticity in ranking
    )
    
    # Initialize posterior as None (will be computed from Task C if enabled)
    posterior_scores = None
    
    # =========================================================================
    # Step 4: Task C - Condensing Generation (if posterior feedback enabled)
    # =========================================================================
    if use_posterior_feedback:
        # Project LQs to LLM dimension
        z_prefix = condense_head(qformer_outputs['lqs_after'])  # [B, N, d_llm]
        
        # Dual-path teacher forcing
        llm_outputs = frozen_llm.teacher_forcing_dual_path(
            z_prefix=z_prefix,
            query_input_ids=batch['query_input_ids'],
            answer_input_ids=batch['answer_input_ids'],
            capture_attention=True,  # Need attention for posterior
        )
        
        # Compute Task C loss
        loss_c = compute_condensing_loss(
            nll_with_evidence=llm_outputs['nll_with_evidence'],
            nll_without_evidence=llm_outputs['nll_without_evidence'],
            softplus_beta=10.0,
            margin_mode='adaptive',
            margin_adaptive_ratio=0.5,
        )
        
        # =====================================================================
        # Step 5: MACS Posterior Extraction (SA part: LLM attention → LQ importance)
        # =====================================================================
        
        # Get Q-Former CA weights (LQs → evidence)
        # Use the averaged CA scores from last layer as proxy for attention weights
        ca_weights = qformer_outputs['ca_raw_scores_avg'][-1]  # [B, N, G]
        ca_weights = torch.softmax(ca_weights, dim=-1)  # Normalize to attention weights
        
        # Extract evidence posterior using MACS×LQ-CA
        posterior_scores = extract_posterior_from_llm_outputs(
            llm_outputs=llm_outputs,
            qformer_ca_weights=ca_weights,
            subset_indices=batch['subset_mask'].nonzero(as_tuple=True)[1] if batch.get('subset_mask') is not None else None,
            num_lqs=qformer_outputs['lqs_after'].shape[1],
            alpha=0.8,  # MACS smoothing
            temperature=1.0,  # Posterior softmax temperature
        )  # [B, |U|] or [B, G]
        
        # Detach posterior - treat as observed teacher signal
        posterior_scores = posterior_scores.detach()
    
    else:
        # No posterior feedback - only Task E + S without Task C
        loss_c = torch.tensor(0.0, device=device)
    
    # =========================================================================
    # Step 6: Compute Task S Loss with Posterior Feedback
    # =========================================================================
    loss_s_dict = compute_ranking_loss(
        ranking_logits=ranking_outputs['logits'],
        gt_scores=batch['teacher_scores'],
        posterior_scores=posterior_scores,  # None if not using posterior
        mask=batch['padding_mask'],
        subset_mask=batch.get('subset_mask', None),
        lambda_teacher=lambda_teacher,
        lambda_post=lambda_post,
        tau=0.1,
        alpha_gt=0.7,
        lambda_ent_reg=0.001,
    )
    
    # =========================================================================
    # Step 7: Total Loss
    # =========================================================================
    loss_total = (
        1.0 * loss_e +
        1.0 * loss_s_dict['loss'] +
        1.0 * loss_c
    )
    
    # =========================================================================
    # Return Comprehensive Loss Dictionary
    # =========================================================================
    return {
        'loss': loss_total,
        'loss_e': loss_e,
        'loss_s': loss_s_dict['loss'],
        'loss_s_teacher': loss_s_dict['loss_teacher'],
        'loss_s_post': loss_s_dict['loss_post'],
        'loss_c': loss_c,
        'posterior_available': posterior_scores is not None,
    }


def curriculum_schedule(
    current_step: int,
    warmup_steps: int = 1000,
    transition_steps: int = 5000,
) -> Dict[str, float]:
    """
    Curriculum learning schedule for λ_teacher and λ_post.
    
    Schedule:
    ---------
    1. Warmup (0 - warmup_steps):
       - λ_teacher = 1.0 (pure teacher supervision)
       - λ_post = 0.0 (no posterior)
    
    2. Transition (warmup_steps - transition_steps):
       - λ_teacher: 1.0 → 0.2 (linear decay)
       - λ_post: 0.0 → 0.8 (linear increase)
    
    3. Steady (transition_steps - end):
       - λ_teacher = 0.2 (minimal teacher guidance)
       - λ_post = 0.8 (strong posterior alignment)
    
    Args:
        current_step: Current training step
        warmup_steps: Steps to use pure teacher supervision
        transition_steps: Steps to complete transition to posterior
    
    Returns:
        schedule_dict: {'lambda_teacher': float, 'lambda_post': float}
    """
    if current_step < warmup_steps:
        # Phase 1: Pure teacher
        return {'lambda_teacher': 1.0, 'lambda_post': 0.0}
    
    elif current_step < transition_steps:
        # Phase 2: Gradual transition
        progress = (current_step - warmup_steps) / (transition_steps - warmup_steps)
        lambda_teacher = 1.0 - 0.8 * progress  # 1.0 → 0.2
        lambda_post = 0.8 * progress            # 0.0 → 0.8
        return {'lambda_teacher': lambda_teacher, 'lambda_post': lambda_post}
    
    else:
        # Phase 3: Posterior-dominant
        return {'lambda_teacher': 0.2, 'lambda_post': 0.8}


# =============================================================================
# Example Training Loop
# =============================================================================

def example_stage2_training_loop():
    """
    Complete example of Stage-2 training with MACS posterior extraction.
    """
    print("=" * 80)
    print("Stage-2 Training Example: Joint E+S+C with MACS Posterior Feedback")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Initialize Models
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    qformer = QFormer(
        hidden_dim=768,
        num_heads=8,
        num_layers=6,
        n_queries=32,
    ).to(device)
    
    entailment_head = EntailmentHead(
        n_queries=32,
        num_layers=6,
        num_heads=8,
        tau=0.5,
        drop_lq_prob=0.2,
    ).to(device)
    
    ranking_head = FragmentRankingHead(
        n_queries=32,
        num_layers=6,
        num_heads=8,
        tau_head=0.1,
        tau_lq=0.2,
    ).to(device)
    
    condense_head = CondenseHead(
        hidden_dim=768,
        llm_hidden_dim=4096,
    ).to(device)
    
    frozen_llm = FrozenLLM(
        model_name="Qwen/Qwen-7B",
        device=device,
        freeze=True,
    )
    
    # -------------------------------------------------------------------------
    # Create Dummy Batch
    # -------------------------------------------------------------------------
    batch_size = 4
    num_evidence = 64
    
    batch = {
        'query_embeddings': torch.randn(batch_size, 10, 768, device=device),
        'evidence_embeddings': torch.randn(batch_size, num_evidence, 768, device=device),
        'entailment_labels': torch.randint(0, 2, (batch_size, num_evidence), device=device),
        'teacher_scores': torch.randn(batch_size, num_evidence, device=device),
        'query_input_ids': torch.randint(0, 50000, (batch_size, 20), device=device),
        'answer_input_ids': torch.randint(0, 50000, (batch_size, 30), device=device),
        'padding_mask': torch.ones(batch_size, num_evidence, dtype=torch.bool, device=device),
        'subset_mask': torch.rand(batch_size, num_evidence, device=device) > 0.7,  # ~30% subset
    }
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    num_steps = 10
    warmup_steps = 3
    transition_steps = 8
    
    print(f"\nRunning {num_steps} training steps with curriculum learning...")
    print(f"- Warmup: steps 0-{warmup_steps} (teacher only)")
    print(f"- Transition: steps {warmup_steps}-{transition_steps} (teacher→posterior)")
    print(f"- Steady: steps {transition_steps}+ (posterior dominant)")
    print()
    
    for step in range(num_steps):
        # Get curriculum schedule
        schedule = curriculum_schedule(step, warmup_steps, transition_steps)
        lambda_teacher = schedule['lambda_teacher']
        lambda_post = schedule['lambda_post']
        
        # Forward pass
        loss_dict = stage2_training_step_with_posterior(
            qformer=qformer,
            entailment_head=entailment_head,
            ranking_head=ranking_head,
            condense_head=condense_head,
            frozen_llm=frozen_llm,
            batch=batch,
            lambda_teacher=lambda_teacher,
            lambda_post=lambda_post,
            use_posterior_feedback=(lambda_post > 0),  # Enable when λ_post > 0
        )
        
        # Log results
        print(f"Step {step:3d} | "
              f"λ_t={lambda_teacher:.2f} λ_p={lambda_post:.2f} | "
              f"L_total={loss_dict['loss']:.4f} | "
              f"L_E={loss_dict['loss_e']:.4f} | "
              f"L_S={loss_dict['loss_s']:.4f} "
              f"(teacher={loss_dict['loss_s_teacher']:.4f}, post={loss_dict['loss_s_post']:.4f}) | "
              f"L_C={loss_dict['loss_c']:.4f} | "
              f"Posterior={'✓' if loss_dict['posterior_available'] else '✗'}")
    
    print("\n" + "=" * 80)
    print("✓ Stage-2 training example complete!")
    print("=" * 80)
    print("\nKey Observations:")
    print("1. λ_teacher decreases from 1.0 → 0.2 over curriculum")
    print("2. λ_post increases from 0.0 → 0.8 over curriculum")
    print("3. L_S_post starts at 0.0 and grows as posterior feedback engages")
    print("4. Task C only computed when λ_post > 0 (efficiency)")
    print("\nNext Steps:")
    print("- Integrate real Qwen LLM (replace FrozenLLM placeholder)")
    print("- Add optimizer and backward pass")
    print("- Add validation loop and checkpointing")
    print("- Scale up to G=128, 256 with dynamic subset U")


if __name__ == '__main__':
    example_stage2_training_loop()
