"""
Quick script to check evidence_ranking format and order in the data file.
"""
import pickle
import numpy as np

data_path = r"D:\LLMs\DR-QFormer\DR-QFormer\smoking_train_ms_subset.pkl"

print(f"Loading data from {data_path}...")
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(f"Total samples: {len(data)}")

# Check first sample
sample_id = list(data.keys())[0]
sample = data[sample_id]

print(f"\n=== Sample ID: {sample_id} ===")
print(f"Query: {sample['query'][:100]}...")
print(f"\nEvidence pool size K: {len(sample['evidence_labels'])}")
print(f"Evidence_ranking length: {len(sample['evidence_ranking'])}")

print("\n=== evidence_ranking format ===")
print(f"Type: {type(sample['evidence_ranking'])}")
print(f"First 5 items:")
for i, item in enumerate(sample['evidence_ranking'][:5]):
    print(f"  [{i}] {item} (type: {type(item)})")
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        print(f"       -> idx={item[0]} (type: {type(item[0])}), score={item[1]} (type: {type(item[1])})")

print("\n=== Checking ranking order ===")
# Extract scores from ranking
scores_in_order = []
for item in sample['evidence_ranking']:
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        scores_in_order.append(float(item[1]))
    else:
        scores_in_order.append(None)

print(f"Scores in ranking order: {scores_in_order[:10]}")

# Check if scores are in descending order (higher score = more relevant)
is_descending = all(scores_in_order[i] >= scores_in_order[i+1] 
                    for i in range(len(scores_in_order)-1) 
                    if scores_in_order[i] is not None and scores_in_order[i+1] is not None)
print(f"Are scores in descending order? {is_descending}")

print("\n=== Reconstructed evidence_scores array ===")
# Simulate what training code does
K = len(sample['evidence_labels'])
evidence_scores = np.zeros(K, dtype=np.float32)

for rank_pos, ranking_item in enumerate(sample['evidence_ranking']):
    if isinstance(ranking_item, (tuple, list)) and len(ranking_item) >= 2:
        frag_idx, rerank_score = ranking_item[0], ranking_item[1]
        
        if isinstance(frag_idx, (np.ndarray, np.integer)):
            frag_idx = int(frag_idx)
        elif not isinstance(frag_idx, int):
            try:
                frag_idx = int(frag_idx)
            except (ValueError, TypeError):
                continue
        
        if 0 <= frag_idx < K:
            evidence_scores[frag_idx] = float(rerank_score)

print(f"evidence_scores shape: {evidence_scores.shape}")
print(f"evidence_scores: {evidence_scores}")
print(f"Max score: {evidence_scores.max():.4f}")
print(f"Min score: {evidence_scores.min():.4f}")
print(f"Non-zero scores: {(evidence_scores > 0).sum()} / {K}")

print("\n=== Cross-check with labels ===")
evidence_labels = sample['evidence_labels']
print(f"Positive labels (sum): {evidence_labels.sum()}")
print(f"evidence_labels: {evidence_labels}")

# Check if top-scored fragments have positive labels
top_k = 3
top_indices = np.argsort(evidence_scores)[::-1][:top_k]
print(f"\nTop {top_k} scored fragments:")
for i, idx in enumerate(top_indices):
    print(f"  Rank {i+1}: Fragment {idx}, score={evidence_scores[idx]:.4f}, label={evidence_labels[idx]}")

print("\n✅ Inspection complete!")
