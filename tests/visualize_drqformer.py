"""
Visualize DR-QFormer architecture and attention flow.
"""

import sys
sys.path.insert(0, 'd:/LLMs/DR-QFormer/DR-QFormer')

import torch
from dr_qformer.models.qformer import DRQFormer


def visualize_architecture():
    """Print detailed architecture breakdown."""
    print("=" * 80)
    print("DR-QFormer Architecture Visualization")
    print("=" * 80)
    
    model = DRQFormer(n_queries=32, hidden_dim=768, num_layers=6, num_heads=8)
    
    print("\n📦 Model Components:")
    print("-" * 80)
    
    total_params = 0
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        print(f"  {name:50s} {str(param.shape):30s} {params:>12,} params")
    
    print("-" * 80)
    print(f"  {'TOTAL':50s} {'':30s} {total_params:>12,} params")
    print(f"  {'':50s} {'':30s} {total_params * 4 / 1024 / 1024:>11,.2f} MB (FP32)")
    
    print("\n" + "=" * 80)


def visualize_forward_flow():
    """Visualize the forward pass data flow."""
    print("\n" + "=" * 80)
    print("Forward Pass Data Flow")
    print("=" * 80)
    
    batch_size = 2
    n_queries = 32
    hidden_dim = 768
    k_fragments = 10
    
    model = DRQFormer(n_queries=n_queries, hidden_dim=hidden_dim, num_layers=3)
    
    query_embeds = torch.randn(batch_size, 1, hidden_dim)
    p_embeds = torch.randn(batch_size, k_fragments, hidden_dim)
    
    print(f"\n📥 INPUT:")
    print(f"  query_embeds:  {str(query_embeds.shape):30s}  (from frozen retriever)")
    print(f"  p_embeds:      {str(p_embeds.shape):30s}  (k fragments from retriever)")
    
    print(f"\n🔄 PROCESSING:")
    print(f"  ┌─ Stage 1: Self-Attention (SA)")
    print(f"  │  • Expand LQs:              [1, {n_queries}, {hidden_dim}] → [{batch_size}, {n_queries}, {hidden_dim}]")
    print(f"  │  • Concat with q_embed:     [{batch_size}, {n_queries}, {hidden_dim}] + [{batch_size}, 1, {hidden_dim}] → [{batch_size}, {n_queries+1}, {hidden_dim}]")
    print(f"  │  • Self-attention mask:     ({n_queries+1}) x ({n_queries+1}) all-ones (bidirectional)")
    print(f"  │  • Output (LQs_aware):      [{batch_size}, {n_queries}, {hidden_dim}]")
    print(f"  │")
    print(f"  ├─ Stage 2: Cross-Attention (CA)")
    print(f"  │  • Query:                   LQs_aware [{batch_size}, {n_queries}, {hidden_dim}]")
    print(f"  │  • Key/Value:               p_embeds [{batch_size}, {k_fragments}, {hidden_dim}]")
    print(f"  │  • Cross-attention mask:    {n_queries} x {k_fragments} all-ones")
    print(f"  │  • Output (Z):              [{batch_size}, {n_queries}, {hidden_dim}]")
    print(f"  │")
    print(f"  └─ Stage 3: Feed-Forward (FFN)")
    print(f"     • Input:                   [{batch_size}, {n_queries}, {hidden_dim}]")
    print(f"     • FFN:                     {hidden_dim} → {hidden_dim*4} → {hidden_dim}")
    print(f"     • Output (Z_final):        [{batch_size}, {n_queries}, {hidden_dim}]")
    
    z, aux = model(query_embeds=query_embeds, p_embeds=p_embeds)
    
    print(f"\n📤 OUTPUT:")
    print(f"  z (to task heads/LLM):  {str(z.shape):30s}  (knowledge-infused representations)")
    print(f"  aux['layer_outputs']:   {len(aux['layer_outputs'])} layers")
    print(f"  aux['z_raw']:           {str(aux['z_raw'].shape):30s}  (full sequence with q_embed)")
    
    print("\n" + "=" * 80)


def visualize_dual_training():
    """Visualize dual training modes."""
    print("\n" + "=" * 80)
    print("Dual Training Modes")
    print("=" * 80)
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            PRIMAL MODE (QA)                                  ║
║                         Query → Answer Prediction                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

  User Query: "What is the capital of France?"
       ↓
  [Frozen Retriever] → Retrieve k=100 fragments
       ↓
  q_embed [1, d]  +  P_embeds [k, d]
       ↓
  ┌──────────────────┐
  │   DR-QFormer     │  ← Trainable parameters
  │  (Cross-Attn)    │
  └──────────────────┘
       ↓
  Z_qa [N, d]  (N=32 knowledge vectors)
       ↓
  ┌────────────────┬────────────────┬────────────────┐
  │   Task E       │   Task S       │   Task C       │
  │  (Entailment)  │  (Sorting)     │  (Generation)  │
  ├────────────────┼────────────────┼────────────────┤
  │ k logits       │ CA attn        │ Answer via     │
  │ [0,1,0,...]    │ weights        │ Frozen LLM     │
  │ ↓              │ ↓              │ ↓              │
  │ BCE Loss       │ KL Loss        │ Contrastive    │
  │ vs gt_k        │ vs soft_target │ Generation     │
  └────────────────┴────────────────┴────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║                             DUAL MODE (QG)                                   ║
║                         Answer → Query Generation                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Ground Truth Answer: "Paris"
       ↓
  [Frozen Retriever] → Same k=100 fragments (from original query)
       ↓
  a_embed [1, d]  +  P_embeds [k, d]
       ↓
  ┌──────────────────┐
  │   DR-QFormer     │  ← SAME parameters (implicit dual constraint)
  │  (Cross-Attn)    │
  └──────────────────┘
       ↓
  Z_qg [N, d]  (N=32 knowledge vectors)
       ↓
  ┌────────────────┬────────────────┬────────────────┐
  │   Task E       │   Task S       │   Task C       │
  │  (Entailment)  │  (Sorting)     │  (Generation)  │
  ├────────────────┼────────────────┼────────────────┤
  │ k logits       │ CA attn        │ Query via      │
  │ [0,1,0,...]    │ weights        │ Frozen LLM     │
  │ ↓              │ ↓              │ ↓              │
  │ BCE Loss       │ KL Loss        │ Reward Loss    │
  │ vs gt_k (SAME!)│ vs soft_target │ (EM/ROUGE)     │
  └────────────────┴────────────────┴────────────────┘

═══════════════════════════════════════════════════════════════════════════════
  🔄 IMPLICIT DUAL CONSTRAINT: Same Q-Former parameters updated by BOTH modes
     Forces bidirectional understanding: Query ↔ Pool ↔ Answer
═══════════════════════════════════════════════════════════════════════════════
""")


def visualize_comparison():
    """Compare BLIP-2 vs DR-QFormer."""
    print("\n" + "=" * 80)
    print("BLIP-2 vs DR-QFormer Comparison")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BLIP-2 Q-Former                                  │
│                     (Visual-Language Pre-training)                           │
└─────────────────────────────────────────────────────────────────────────────┘

  Image → [Vision Encoder] → 256 patches [batch, 256, d]
                ↓
         ┌─────────────┐
         │  Q-Former   │  Stage 1: Vision-language alignment (NO LLM)
         │   (32 LQs)  │  • ITC: Image-text contrastive
         └─────────────┘  • ITM: Image-text matching
                ↓         • ITG: Image-grounded text generation
         32 queries
                ↓
         ┌─────────────┐
         │   LLM       │  Stage 2: Generative learning (WITH LLM)
         │  (Frozen)   │  • ITG: Image-to-text generation
         └─────────────┘

  🔑 Key: OFFLINE, Query-agnostic (image summarization BEFORE question)

┌─────────────────────────────────────────────────────────────────────────────┐
│                            DR-QFormer                                        │
│                     (Retrieval-Augmented Generation)                         │
└─────────────────────────────────────────────────────────────────────────────┘

  Query → [Retriever] → k=100 fragments [batch, k, d]
    +         (Frozen)
  q_embed [batch, 1, d]
    │
    ├─────────┐
    │         ↓
    │  ┌─────────────┐
    │  │ DR-QFormer  │  Single-stage: All tasks with frozen LLM
    │  │   (32 LQs)  │  • Task E: Entailment (fragment filtering)
    │  └─────────────┘  • Task S: Sorting (attention supervision)
    │         ↓         • Task C: Generation (contrastive/reward)
    │    32 queries
    │         ↓
    │  ┌─────────────┐
    └─>│   LLM       │  Always frozen, used from start
       │  (Frozen)   │
       └─────────────┘

  🔑 Key: ONLINE, Query-sensitive (dynamic processing AFTER question)

═══════════════════════════════════════════════════════════════════════════════
                              KEY DIFFERENCES
═══════════════════════════════════════════════════════════════════════════════
  BLIP-2                          │  DR-QFormer
──────────────────────────────────┼──────────────────────────────────────────
  Offline (image → summary)       │  Online (query → knowledge extraction)
  High SNR (all patches relevant) │  Low SNR (1-2% fragments relevant)
  Compressor role                 │  Filter + Sorter + Reasoner role
  Two-stage training              │  Single-stage training
  Query-agnostic                  │  Query-sensitive
  Vision → Language               │  Text → Text (RAG)
═══════════════════════════════════════════════════════════════════════════════
""")


def main():
    """Run all visualizations."""
    visualize_architecture()
    visualize_forward_flow()
    visualize_dual_training()
    visualize_comparison()
    
    print("\n" + "=" * 80)
    print("✅ Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
