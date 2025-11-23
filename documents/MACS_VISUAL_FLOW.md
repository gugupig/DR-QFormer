# MACS×LQ-CA Posterior Extraction - Visual Flow

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        Stage-2 Training with Posterior Feedback                ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1: Q-Former Forward (Task E + S + C Heads)                             │
└─────────────────────────────────────────────────────────────────────────────┘
    
    Query Embeddings [B, T_q, 768]        Evidence Pool [B, G, 768]
            │                                      │
            └──────────────┬───────────────────────┘
                           ↓
                    ┌──────────────┐
                    │   Q-Former   │  (SA + CA + FFN)
                    │  (Trainable) │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
     LQs_after [B, 32, 768]    CA raw scores (all layers)
              │                         │
              │                    ┌────┴─────────┐
              │                    ↓              ↓
              │            Task E Head      Task S Head
              │               │                  │
              │           E logits          S logits
              │               │                  │
              │               ↓                  ↓
              │          L_E (Focal)      L_S (ListNet + JS)
              │
              ↓
    ┌─────────────────┐
    │ Condense Head   │  (768 → 4096 projection)
    └────────┬────────┘
             ↓
    Z_prefix [B, 32, 4096]


┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 2: Task C - LLM Teacher Forcing (Dual-Path)                            │
└─────────────────────────────────────────────────────────────────────────────┘

    Z_prefix [B, 32, 4096]    Query IDs [B, S_q]    Answer IDs [B, S_a]
         │                           │                      │
         └───────────────┬───────────┴──────────────────────┘
                         ↓
           ┌─────────────────────────────────┐
           │  Frozen Qwen LLM (eval mode)    │
           │  teacher_forcing_dual_path()    │
           └─────────┬───────────────────────┘
                     │
        ┌────────────┴────────────┐
        ↓                         ↓
   Path A (with Z)          Path B (without Z)
   NLL_with [scalar]        NLL_without [scalar]
        │                         │
        └──────────┬──────────────┘
                   ↓
              L_C = Softplus(margin - ΔNLL)
                   │
                   ↓
           Attentions [Tuple]
           └─> [B, H, S, S] × num_layers


┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 3: MACS Posterior Extraction (SA Part: Answer → LQ Importance)         │
└─────────────────────────────────────────────────────────────────────────────┘

    Attentions [Tuple of [B, H, S, S]]
         │
         ↓
    ╔════════════════════════════════════════════════════════════════╗
    ║  compute_macs_to_lqs()                                         ║
    ║  ────────────────────────────────────────────────────────────  ║
    ║  1. Slice to first num_lqs positions:                          ║
    ║     target_attn = attentions[..., :32]                         ║
    ║                                                                 ║
    ║  2. Max-pool over heads:                                       ║
    ║     layer_max_attn = max(target_attn, dim=heads)               ║
    ║                                                                 ║
    ║  3. Cumulative product with exponential smoothing:             ║
    ║     joint_att = ones([B, S, 32])                               ║
    ║     for layer in layers:                                       ║
    ║         smoothed = alpha * att[layer] + (1-alpha) * 1.0        ║
    ║         joint_att *= smoothed                                  ║
    ║                                                                 ║
    ║  4. Z-score normalization:                                     ║
    ║     joint_att = (joint_att - mean) / (std + eps)               ║
    ╚════════════════════════════════════════════════════════════════╝
         │
         ↓
    MACS Map [B, S, 32]  (all tokens → all LQs)
         │
         ↓
    ╔════════════════════════════════════════════════════════════════╗
    ║  extract_answer_lq_posterior()                                 ║
    ║  ────────────────────────────────────────────────────────────  ║
    ║  1. Slice to answer span:                                      ║
    ║     answer_to_lqs = macs_map[:, answer_start:answer_end, :]    ║
    ║                                                                 ║
    ║  2. Aggregate over answer tokens:                              ║
    ║     lq_posterior = mean(answer_to_lqs, dim=1)                  ║
    ╚════════════════════════════════════════════════════════════════╝
         │
         ↓
    LQ Posterior [B, 32]  (which LQs LLM used for answer)


┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 4: Evidence Posterior (CA Part: LQ → Evidence via Q-Former CA)         │
└─────────────────────────────────────────────────────────────────────────────┘

    LQ Posterior [B, 32]           CA Weights [B, 32, G]
         │                                │
         │            (from Q-Former CA scores)
         │                                │
         └────────────┬───────────────────┘
                      ↓
         ╔════════════════════════════════════════════════════╗
         ║  compute_evidence_posterior()                      ║
         ║  ───────────────────────────────────────────────   ║
         ║  Formula: p(e_k | q, a) = Σ_j p(LQ_j | a) × p(e_k | LQ_j)  ║
         ║                                                     ║
         ║  1. Weighted sum over LQs:                         ║
         ║     evidence_logits = einsum('bn,bnk->bk',         ║
         ║                               lq_posterior,         ║
         ║                               ca_weights)           ║
         ║                                                     ║
         ║  2. Softmax to get distribution:                   ║
         ║     evidence_posterior = softmax(evidence_logits)  ║
         ╚════════════════════════════════════════════════════╝
                      │
                      ↓
         Evidence Posterior [B, G]  (what LLM actually used)
                      │
                      ↓ .detach()  (treat as teacher signal)
         Evidence Posterior [B, G]  (detached, no grad)


┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 5: Task S Loss with Posterior Feedback                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    Task S Logits [B, G]      Teacher Scores [B, G]     Evidence Posterior [B, G]
         │                           │                           │
         │                           │                           │
         └────────────┬──────────────┴───────────────────────────┘
                      ↓
         ╔═══════════════════════════════════════════════════════════════════╗
         ║  compute_ranking_loss()                                           ║
         ║  ────────────────────────────────────────────────────────────────  ║
         ║  Component 1: Teacher Supervision (ListNet)                       ║
         ║    p_teacher = softmax(teacher_scores / tau)                      ║
         ║    p_student = softmax(student_logits / tau)                      ║
         ║    L_teacher = CE(p_teacher, p_student)                           ║
         ║                                                                    ║
         ║  Component 2: Posterior Alignment (JS Divergence)                 ║
         ║    p_posterior = softmax(evidence_posterior / tau)                ║
         ║    L_post = JS(p_student || p_posterior)                          ║
         ║           = 0.5 * KL(p_student || p_mid) +                        ║
         ║             0.5 * KL(p_posterior || p_mid)                        ║
         ║                                                                    ║
         ║  Total Loss:                                                      ║
         ║    L_S = λ_teacher * L_teacher + λ_post * L_post                  ║
         ║                                                                    ║
         ║  Curriculum Learning:                                             ║
         ║    Early:  λ_teacher=1.0, λ_post=0.0  (teacher only)              ║
         ║    Mid:    λ_teacher→0.2, λ_post→0.8  (transition)                ║
         ║    Late:   λ_teacher=0.2, λ_post=0.8  (posterior dominant)        ║
         ╚═══════════════════════════════════════════════════════════════════╝
                      │
                      ↓
               L_S (scalar loss)


┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 6: Total Loss and Backward                                             │
└─────────────────────────────────────────────────────────────────────────────┘

    L_E (Focal)      L_S (ListNet+JS)      L_C (Contrastive NLL)
         │                  │                        │
         └──────────────────┴────────────────────────┘
                            ↓
                   L_total = w_E * L_E + w_S * L_S + w_C * L_C
                            │
                            ↓ .backward()
                   ┌────────────────────┐
                   │  Gradient Flow     │
                   │  (Q-Former + Heads)│
                   └────────────────────┘
                            │
                            ↓
                   Optimizer.step()


╔═══════════════════════════════════════════════════════════════════════════════╗
║                            Key Innovation: Posterior Feedback Loop             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

    Task S predicts:   π(e|q)  ←─────┐
         │                            │  JS divergence
         │                            │  (align prior with posterior)
         ↓                            │
    LLM reveals:      q(e|q,a) ───────┘
                      (via MACS×LQ-CA)

    Interpretation:
    ──────────────
    - π(e|q):  Q-Former's PRIOR belief about evidence importance (before seeing LLM)
    - q(e|q,a): POSTERIOR "ground truth" from LLM's actual evidence usage
    - Goal: Minimize divergence → Q-Former learns what LLM truly needs


╔═══════════════════════════════════════════════════════════════════════════════╗
║                      Curriculum Learning Schedule                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

    Step:     0 ─────────────── 1000 ─────────────── 5000 ─────────────── End
              │                    │                     │                    │
    λ_teacher: 1.0 ════════════════╗                    ╚═════════════════ 0.2
                                   ║                   
                                   ║ Transition       
                                   ║                   
    λ_post:    0.0 ════════════════╝                    ╔═════════════════ 0.8
              │                    │                     │                    │
    Phase:    Warmup              Transition           Steady
              (Teacher only)       (Teacher→Posterior)  (Posterior dominant)

    Rationale:
    ──────────
    - Early: Learn basic ranking from strong teacher (Qwen-Reranker)
    - Mid:   Gradually trust LLM's observed usage patterns
    - Late:  Fully aligned with LLM's true needs (core innovation)


╔═══════════════════════════════════════════════════════════════════════════════╗
║                            Expected Impact                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

    Metric                Before Stage-2       After Stage-2 (Late)
    ──────────────────    ──────────────       ────────────────────
    NDCG@10               Baseline (teacher)   +5-10% (posterior-aligned)
    Answer NLL            N/A                  Monotonic decrease
    Prior-Posterior       N/A                  JS divergence → 0.05-0.1
    
    Why Improvement?
    ────────────────
    Task S learns not just "what reranker likes" but "what LLM actually uses"
    → Directly addresses retriever-LLM objective mismatch
    → Evidence selection optimized for downstream generation quality
```
