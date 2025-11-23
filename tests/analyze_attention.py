"""
Visualize and analyze attention weights from DR-QFormer.

This script demonstrates how to:
1. Extract SA/CA attention weights from each layer
2. Analyze which LQs attend to which fragments
3. Visualize attention patterns
4. Export attention weights for further analysis
"""

import sys
sys.path.insert(0, 'd:/LLMs/DR-QFormer/DR-QFormer')

import torch
import numpy as np
from dr_qformer.models.qformer import DRQFormer


def analyze_attention_patterns(model, query_embeds, p_embeds, mode="Primal"):
    """
    Analyze attention patterns in DR-QFormer.
    
    Args:
        model: DRQFormer instance
        query_embeds: Query embeddings [batch, 1, d]
        p_embeds: Fragment embeddings [batch, k, d]
        mode: "Primal" or "Dual"
    
    Returns:
        Analysis results dict with attention weights and statistics
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Attention Patterns - {mode} Mode")
    print(f"{'='*80}")
    
    # Forward pass
    with torch.no_grad():
        if mode == "Primal":
            z, aux = model(query_embeds=query_embeds, p_embeds=p_embeds)
        else:
            z, aux = model(answer_embeds=query_embeds, p_embeds=p_embeds)
    
    batch_size = query_embeds.size(0)
    n_queries = model.n_queries
    num_layers = model.num_layers
    k_fragments = p_embeds.size(1)
    
    print(f"\nModel Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learnable Queries (LQs): {n_queries}")
    print(f"  - Layers: {num_layers}")
    print(f"  - Attention heads: {model.num_heads}")
    print(f"  - Fragments: {k_fragments}")
    
    # Extract attention weights
    sa_weights = aux['sa_attn_weights']  # List of [batch, num_heads, N+1, N+1]
    ca_weights = aux['ca_attn_weights']  # List of [batch, num_heads, N, k]
    
    results = {
        'sa_weights': sa_weights,
        'ca_weights': ca_weights,
        'z': z,
        'mode': mode
    }
    
    # === Analyze Self-Attention (SA) Patterns ===
    print(f"\n{'='*80}")
    print("Self-Attention (SA) Analysis")
    print(f"{'='*80}")
    
    for layer_idx in [0, num_layers // 2, num_layers - 1]:  # First, middle, last
        sa = sa_weights[layer_idx]  # [batch, num_heads, N+1, N+1]
        sa_avg = sa.mean(dim=1)  # Average across heads: [batch, N+1, N+1]
        
        print(f"\nLayer {layer_idx}:")
        
        # LQs attending to query/answer embedding (last token)
        lqs_to_qa = sa_avg[:, :n_queries, n_queries]  # [batch, N]
        print(f"  LQs → Query/Answer embedding:")
        print(f"    - Max attention: {lqs_to_qa.max().item():.4f}")
        print(f"    - Mean attention: {lqs_to_qa.mean().item():.4f}")
        print(f"    - Top 3 attending LQs: {lqs_to_qa[0].topk(3).indices.tolist()}")
        
        # Query/answer embedding attending to LQs
        qa_to_lqs = sa_avg[:, n_queries, :n_queries]  # [batch, N]
        print(f"  Query/Answer embedding → LQs:")
        print(f"    - Max attention: {qa_to_lqs.max().item():.4f}")
        print(f"    - Mean attention: {qa_to_lqs.mean().item():.4f}")
        print(f"    - Top 3 attended LQs: {qa_to_lqs[0].topk(3).indices.tolist()}")
        
        # LQs attending to each other
        lqs_to_lqs = sa_avg[:, :n_queries, :n_queries]  # [batch, N, N]
        lqs_self_attn = torch.diagonal(lqs_to_lqs[0], dim1=0, dim2=1)
        print(f"  LQs self-attention (diagonal):")
        print(f"    - Mean self-attention: {lqs_self_attn.mean().item():.4f}")
    
    # === Analyze Cross-Attention (CA) Patterns ===
    print(f"\n{'='*80}")
    print("Cross-Attention (CA) Analysis")
    print(f"{'='*80}")
    
    for layer_idx in [0, num_layers // 2, num_layers - 1]:  # First, middle, last
        ca = ca_weights[layer_idx]  # [batch, num_heads, N, k]
        ca_avg = ca.mean(dim=1)  # Average across heads: [batch, N, k]
        
        print(f"\nLayer {layer_idx}:")
        
        # Which fragments receive most attention?
        frag_attention = ca_avg[0].mean(dim=0)  # Average across LQs: [k]
        top_frags = frag_attention.topk(3)
        print(f"  Top 3 attended fragments (averaged across LQs):")
        for i, (frag_idx, attn) in enumerate(zip(top_frags.indices, top_frags.values)):
            print(f"    {i+1}. Fragment {frag_idx.item():2d}: {attn.item():.4f}")
        
        # Which LQs are most selective?
        lq_selectivity = ca_avg[0].max(dim=1).values  # Max attention per LQ: [N]
        top_selective_lqs = lq_selectivity.topk(3)
        print(f"  Top 3 most selective LQs (highest max attention):")
        for i, (lq_idx, sel) in enumerate(zip(top_selective_lqs.indices, top_selective_lqs.values)):
            print(f"    {i+1}. LQ {lq_idx.item():2d}: {sel.item():.4f}")
        
        # Attention diversity (entropy)
        ca_flat = ca_avg[0]  # [N, k]
        entropy = -(ca_flat * torch.log(ca_flat + 1e-10)).sum(dim=1)  # [N]
        print(f"  Attention diversity (entropy):")
        print(f"    - Mean entropy: {entropy.mean().item():.4f}")
        print(f"    - Max entropy (most diverse): {entropy.max().item():.4f}")
        print(f"    - Min entropy (most focused): {entropy.min().item():.4f}")
    
    # === Per-Head Analysis ===
    print(f"\n{'='*80}")
    print("Per-Head Attention Analysis")
    print(f"{'='*80}")
    
    last_ca = ca_weights[-1]  # [batch, num_heads, N, k]
    print(f"\nLast layer CA weights: {last_ca.shape}")
    
    for head_idx in range(min(3, model.num_heads)):  # Show first 3 heads
        head_ca = last_ca[0, head_idx]  # [N, k]
        
        # Find most attended fragment for this head
        max_per_lq = head_ca.max(dim=1)
        most_attended_frag = max_per_lq.values.argmax()
        max_attention = max_per_lq.values[most_attended_frag].item()
        
        print(f"\nHead {head_idx}:")
        print(f"  - Most attended fragment by any LQ: {most_attended_frag.item()}")
        print(f"  - Max attention: {max_attention:.4f}")
        print(f"  - Mean attention: {head_ca.mean().item():.4f}")
    
    # === LQ-Fragment Mapping ===
    print(f"\n{'='*80}")
    print("LQ-Fragment Attention Mapping (Last Layer)")
    print(f"{'='*80}")
    
    last_ca_avg = ca_weights[-1].mean(dim=1)[0]  # [N, k]
    
    print(f"\nTop fragment for each LQ (first 10 LQs):")
    for lq_idx in range(min(10, n_queries)):
        frag_attns = last_ca_avg[lq_idx]  # [k]
        top_frag = frag_attns.argmax().item()
        top_attn = frag_attns[top_frag].item()
        
        # Get top 3 fragments for this LQ
        top3 = frag_attns.topk(3)
        top3_str = ", ".join([f"F{idx.item()}({val.item():.3f})" 
                              for idx, val in zip(top3.indices, top3.values)])
        
        print(f"  LQ {lq_idx:2d}: {top3_str}")
    
    print(f"\nTop LQ for each fragment:")
    for frag_idx in range(k_fragments):
        lq_attns = last_ca_avg[:, frag_idx]  # [N]
        top_lq = lq_attns.argmax().item()
        top_attn = lq_attns[top_lq].item()
        
        # Get top 3 LQs for this fragment
        top3 = lq_attns.topk(3)
        top3_str = ", ".join([f"LQ{idx.item()}({val.item():.3f})" 
                              for idx, val in zip(top3.indices, top3.values)])
        
        print(f"  Fragment {frag_idx:2d}: {top3_str}")
    
    return results


def save_attention_weights(results, save_path="attention_weights.npz"):
    """Save attention weights for further analysis."""
    print(f"\n{'='*80}")
    print(f"Saving Attention Weights to {save_path}")
    print(f"{'='*80}")
    
    # Convert to numpy
    sa_weights_np = [w.cpu().numpy() for w in results['sa_weights']]
    ca_weights_np = [w.cpu().numpy() for w in results['ca_weights']]
    z_np = results['z'].cpu().numpy()
    
    # Save
    np.savez(
        save_path,
        sa_weights=sa_weights_np,
        ca_weights=ca_weights_np,
        z=z_np,
        mode=results['mode']
    )
    
    print(f"✓ Saved attention weights:")
    print(f"  - SA weights: {len(sa_weights_np)} layers")
    print(f"  - CA weights: {len(ca_weights_np)} layers")
    print(f"  - Output embeddings (z): {z_np.shape}")


def compare_primal_dual_attention():
    """Compare attention patterns between Primal and Dual modes."""
    print(f"\n{'='*80}")
    print("Comparing Primal vs Dual Attention Patterns")
    print(f"{'='*80}")
    
    # Initialize model
    model = DRQFormer(
        n_queries=32,
        hidden_dim=768,
        num_layers=6,
        num_heads=8,
        max_fragments=10,
        dropout=0.0
    )
    model.eval()
    
    # Create inputs
    batch_size = 1
    query_embeds = torch.randn(batch_size, 1, 768)
    answer_embeds = torch.randn(batch_size, 1, 768)
    p_embeds = torch.randn(batch_size, 10, 768)
    
    # Analyze Primal mode
    results_primal = analyze_attention_patterns(
        model, query_embeds, p_embeds, mode="Primal"
    )
    
    # Analyze Dual mode
    results_dual = analyze_attention_patterns(
        model, answer_embeds, p_embeds, mode="Dual"
    )
    
    # Compare patterns
    print(f"\n{'='*80}")
    print("Primal vs Dual Comparison (Last Layer CA)")
    print(f"{'='*80}")
    
    ca_primal = results_primal['ca_weights'][-1][0].mean(dim=0)  # [N, k]
    ca_dual = results_dual['ca_weights'][-1][0].mean(dim=0)  # [N, k]
    
    # Compute difference
    ca_diff = (ca_primal - ca_dual).abs()
    
    print(f"\nAttention pattern differences:")
    print(f"  - Mean absolute difference: {ca_diff.mean().item():.4f}")
    print(f"  - Max absolute difference: {ca_diff.max().item():.4f}")
    print(f"  - Correlation: {torch.corrcoef(torch.stack([ca_primal.flatten(), ca_dual.flatten()]))[0, 1].item():.4f}")
    
    # Save results
    save_attention_weights(results_primal, "attention_weights_primal.npz")
    save_attention_weights(results_dual, "attention_weights_dual.npz")
    
    print(f"\n{'='*80}")
    print("✅ Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    compare_primal_dual_attention()
    
    print(f"\n{'='*80}")
    print("📊 Summary of Attention Weight Export Features")
    print(f"{'='*80}")
    print("✅ Exported attention weights:")
    print("   - SA weights: [batch, num_heads, N+1, N+1] per layer")
    print("   - CA weights: [batch, num_heads, N, k] per layer")
    print("   - Per-head weights (not averaged)")
    print("\n✅ Analysis capabilities:")
    print("   - Which LQs attend to query/answer embedding")
    print("   - Which fragments receive most attention")
    print("   - Attention selectivity and diversity")
    print("   - Per-head attention patterns")
    print("   - LQ-to-fragment attention mapping")
    print("\n✅ Use cases:")
    print("   - Understand which LQs specialize in which fragments")
    print("   - Debug attention collapse or over-smoothing")
    print("   - Visualize attention flow through layers")
    print("   - Compare Primal (QA) vs Dual (QG) attention")
    print(f"{'='*80}")
