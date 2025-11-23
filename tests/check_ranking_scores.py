"""
Test script to verify teacher score normalization and loss computation.
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F

def load_and_check_scores(pkl_path: str, num_samples: int = 10):
    """Load PKL and check teacher score distribution."""
    
    print("="*80)
    print("Analyzing Teacher Scores from PKL")
    print("="*80)
    
    with open(pkl_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    sample_ids = list(data_dict.keys())[:num_samples]
    
    all_raw_scores = []
    all_normalized_scores = []
    
    for sample_id in sample_ids:
        sample = data_dict[sample_id]
        evidence_ranking = sample.get('evidence_ranking', [])
        K = len(sample['evidence_embeddings'])
        
        # Extract raw scores
        raw_scores = np.zeros(K, dtype=np.float32)
        for item in evidence_ranking:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                idx, score = int(item[0]), float(item[1])
                if 0 <= idx < K:
                    raw_scores[idx] = score
        
        # Normalize (same logic as in dataset)
        normalized_scores = raw_scores.copy()
        nonzero_mask = normalized_scores > 0
        if nonzero_mask.sum() > 1:
            nonzero_scores = normalized_scores[nonzero_mask]
            mean_score = nonzero_scores.mean()
            std_score = nonzero_scores.std()
            if std_score > 1e-6:
                normalized_scores[nonzero_mask] = (nonzero_scores - mean_score) / std_score
        
        all_raw_scores.append(raw_scores)
        all_normalized_scores.append(normalized_scores)
        
        print(f"\nSample {sample_id}:")
        print(f"  Raw scores: min={raw_scores.min():.4f}, max={raw_scores.max():.4f}, "
              f"std={raw_scores.std():.4f}")
        
        # Handle std formatting correctly
        norm_std = normalized_scores[nonzero_mask].std() if nonzero_mask.sum() > 0 else 0.0
        print(f"  Normalized: min={normalized_scores.min():.4f}, max={normalized_scores.max():.4f}, "
              f"std={float(norm_std):.4f}")
        print(f"  Top-3 raw: {sorted(raw_scores, reverse=True)[:3]}")
        print(f"  Top-3 normalized: {sorted(normalized_scores, reverse=True)[:3]}")
    
    # Aggregate statistics
    all_raw = np.concatenate(all_raw_scores)
    all_norm = np.concatenate(all_normalized_scores)
    
    print("\n" + "="*80)
    print("Aggregate Statistics:")
    print("="*80)
    print(f"Raw scores:")
    print(f"  Range: [{all_raw.min():.4f}, {all_raw.max():.4f}]")
    print(f"  Mean: {all_raw.mean():.4f}, Std: {all_raw.std():.4f}")
    print(f"\nNormalized scores:")
    print(f"  Range: [{all_norm.min():.4f}, {all_norm.max():.4f}]")
    print(f"  Mean: {all_norm.mean():.4f}, Std: {all_norm.std():.4f}")
    
    # Test loss computation
    print("\n" + "="*80)
    print("Testing Loss Computation:")
    print("="*80)
    
    # Create mock data
    batch_size = 4
    # Use actual K from the data (includes NoE)
    K = len(all_normalized_scores[0])
    
    # Use real normalized scores for first sample
    teacher_scores = torch.from_numpy(all_normalized_scores[0]).unsqueeze(0).repeat(batch_size, 1)
    # Random student logits
    student_logits = torch.randn(batch_size, K)
    mask = torch.ones(batch_size, K, dtype=torch.bool)
    
    # Compute ListNet loss
    masked_logits = student_logits.clone()
    masked_teacher = teacher_scores.clone()
    masked_logits[~mask] = -1e9
    masked_teacher[~mask] = -1e9
    
    tau = 1.0
    student_probs = F.softmax(masked_logits / tau, dim=-1)
    teacher_probs = F.softmax(masked_teacher / tau, dim=-1)
    log_student_probs = F.log_softmax(masked_logits / tau, dim=-1)
    
    kl_div = F.kl_div(log_student_probs, teacher_probs, reduction='none')
    loss = kl_div.mean()
    
    print(f"Teacher probs: {teacher_probs[0].numpy()}")
    print(f"Student probs: {student_probs[0].numpy()}")
    print(f"KL divergence: {loss.item():.6f}")
    
    # Check if teacher distribution is too flat
    entropy = -(teacher_probs * torch.log(teacher_probs + 1e-9)).sum(dim=-1).mean()
    max_entropy = np.log(K)
    print(f"\nTeacher distribution entropy: {entropy.item():.4f} (max: {max_entropy:.4f})")
    print(f"Normalized entropy: {(entropy.item() / max_entropy * 100):.1f}% of maximum")
    
    if entropy.item() / max_entropy > 0.95:
        print("⚠️  WARNING: Teacher distribution is very flat (almost uniform)!")
        print("   This means the ranking signal is very weak.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    pkl_path = r"D:\LLMs\DR-QFormer\DR-QFormer\ms_xlm_embeddings.pkl"
    load_and_check_scores(pkl_path, num_samples=10)
