# Evidence Ranking Format Fix

## 🔴 问题描述

数据格式不一致导致训练代码无法正确解析ranking信息。

### 原始问题

**Reranker输出格式** (`rerank_evidences`函数):
```python
evidence_ranking = [
    ("Beijing is the capital...", 0.95),  # (text, score)
    ("Shanghai is the largest...", 0.82),
    ("Paris is the capital...", 0.23),
    ("<NO_EVIDENCE>", 0.01)
]
```

**训练代码期望格式**:
```python
evidence_ranking = [
    (0, 0.95),  # (idx, score)
    (1, 0.82),
    (2, 0.23)
]
```

**冲突**: 训练代码尝试将text当作index使用，导致类型错误。

## ✅ 解决方案

### 1. 修复数据格式（Notebook）

在`dataset_smoking.ipynb`中添加转换函数：

```python
def fix_evidence_ranking_format(data_dict: dict) -> dict:
    """
    修复 evidence_ranking 格式：从 List[(text, score)] 转换为 List[(idx, score)]
    """
    fixed_data = {}
    
    for key, item in data_dict.items():
        fixed_item = item.copy()
        
        evidence_text = item['evidence_text']  # List[str]
        evidence_ranking = item['evidence_ranking']  # List[(text, score)]
        
        # 创建文本到索引的映射
        text_to_idx = {text: idx for idx, text in enumerate(evidence_text)}
        
        # 转换为 (idx, score) 格式
        idx_score_pairs = []
        for text, score in evidence_ranking:
            if text == "<NO_EVIDENCE>":
                continue  # 跳过NoE标记
            
            if text in text_to_idx:
                idx = text_to_idx[text]
                idx_score_pairs.append((idx, score))
        
        fixed_item['evidence_ranking'] = idx_score_pairs
        fixed_data[key] = fixed_item
    
    return fixed_data

# 应用修复
fixed_ms_subset = fix_evidence_ranking_format(ms_subset)

# 保存
with open("smoking_train_ms_subset.pkl", "wb") as f:
    pickle.dump(fixed_ms_subset, f)
```

### 2. 增强训练代码健壮性

修改`train/stage1_train.py`中的解析逻辑：

```python
# 更健壮的ranking解析
for rank_pos, ranking_item in enumerate(evidence_ranking):
    # Parse ranking item: should be (idx, score) tuple
    if isinstance(ranking_item, (tuple, list)) and len(ranking_item) >= 2:
        frag_idx, rerank_score = ranking_item[0], ranking_item[1]
    elif isinstance(ranking_item, (tuple, list)) and len(ranking_item) == 1:
        # Fallback: only index provided
        frag_idx = ranking_item[0]
        rerank_score = 1.0 - (rank_pos / max(len(evidence_ranking), 1))
    else:
        # Fallback: plain index
        frag_idx = ranking_item
        rerank_score = 1.0 - (rank_pos / max(len(evidence_ranking), 1))
    
    # Convert to int with error handling
    if isinstance(frag_idx, (np.ndarray, np.integer)):
        frag_idx = int(frag_idx)
    elif not isinstance(frag_idx, int):
        try:
            frag_idx = int(frag_idx)
        except (ValueError, TypeError):
            continue  # Skip invalid indices
    
    # Validate and assign
    if 0 <= frag_idx < K:
        evidence_scores[frag_idx] = float(rerank_score)
```

## 📊 修复前后对比

### 修复前

```python
# 数据格式
sample['evidence_ranking'] = [
    ("Beijing is capital...", 0.95),
    ("Shanghai is largest...", 0.82),
    # ...
]

# 训练代码尝试：
frag_idx = "Beijing is capital..."  # ❌ 类型错误！
evidence_scores[frag_idx] = 0.95     # ❌ IndexError!
```

### 修复后

```python
# 数据格式
sample['evidence_ranking'] = [
    (0, 0.95),  # fragment 0 的 reranker score
    (1, 0.82),  # fragment 1 的 reranker score
    # ...
]

# 训练代码：
frag_idx = 0                         # ✅ 正确的索引
evidence_scores[frag_idx] = 0.95     # ✅ 正确赋值
```

## 🔍 数据验证

### 验证检查项

```python
# 1. 检查格式
assert isinstance(evidence_ranking, list)
assert all(isinstance(item, tuple) for item in evidence_ranking)
assert all(len(item) == 2 for item in evidence_ranking)

# 2. 检查索引范围
K = len(evidence_text)
for idx, score in evidence_ranking:
    assert isinstance(idx, int)
    assert 0 <= idx < K
    assert isinstance(score, (int, float))
    assert 0.0 <= score <= 1.0

# 3. 检查无重复
indices = [idx for idx, _ in evidence_ranking]
assert len(indices) == len(set(indices))  # No duplicates
```

## 📝 正确的数据结构

### 完整样本格式

```python
sample = {
    'query': str,                              # Query text
    'query_embedding': {
        'input_ids': Tensor[1, T],            # Token IDs
        'attention_mask': Tensor[1, T],       # Attention mask
        'token_emb_768': Tensor[1, T, 768]   # Pre-computed embeddings
    },
    'answer': str,                            # Ground truth answer
    
    'evidence_text': List[str],               # K fragments (text)
    'evidence_embeddings': ndarray[K, 768],   # K fragments (embeddings)
    'evidence_labels': ndarray[K],            # K binary labels (0/1)
    
    'evidence_ranking': List[Tuple[int, float]],  # ✅ (idx, score) pairs
    # Example:
    # [
    #   (2, 0.95),  # Fragment 2 最相关 (reranker confidence 0.95)
    #   (0, 0.87),  # Fragment 0 次相关
    #   (5, 0.65),  # Fragment 5
    #   ...
    # ]
}
```

### Ranking解释

`evidence_ranking`列表中的元素：
- **索引** (`idx`): Fragment在`evidence_text`中的位置 (0-based)
- **分数** (`score`): Reranker的相关性置信度 (0.0 ~ 1.0)
- **顺序**: 按相关性从高到低排序

**关键点**:
1. 使用索引而非文本（内存高效，查找快速）
2. 保留reranker原始分数（保持质量信息）
3. 排除`<NO_EVIDENCE>`标记（仅用于reranker内部）

## 🚀 使用步骤

### 步骤1: 修复现有数据

在notebook中运行：

```python
# 加载数据
with open("smoking_train_ms_subset.pkl", "rb") as f:
    ms_subset = pickle.load(f)

# 应用修复
fixed_ms_subset = fix_evidence_ranking_format(ms_subset)

# 验证
sample = fixed_ms_subset[list(fixed_ms_subset.keys())[0]]
print(f"evidence_ranking[0]: {sample['evidence_ranking'][0]}")
# 输出: (0, 0.9523)  ✅ (idx, score) 格式

# 保存
with open("smoking_train_ms_subset.pkl", "wb") as f:
    pickle.dump(fixed_ms_subset, f)
```

### 步骤2: 运行训练

```bash
python train/stage1_train.py
```

训练代码现在可以正确解析ranking数据。

## 🐛 常见问题

### Q1: 为什么不直接用reranker返回(idx, score)?

**A**: Reranker函数(`rerank_evidences`)的设计目标是返回人类可读的(text, score)对，便于调试和验证。数据保存时需要转换为(idx, score)以提高存储和计算效率。

### Q2: 丢失了<NO_EVIDENCE>信息吗？

**A**: `<NO_EVIDENCE>`是reranker的内部标记，表示"所有evidence都不相关"的概率。在训练中：
- 如果所有fragments的labels都是0，模型会学习到"无相关evidence"
- 不需要显式的NoE标记

### Q3: Score的含义是什么？

**A**: Score是reranker对"该fragment与query相关"的置信度：
- **高分** (0.9+): 高度相关
- **中分** (0.5~0.9): 部分相关
- **低分** (<0.5): 不太相关

训练时，这些分数用于ListNet loss的teacher signal。

## 📚 相关文件

- `dataset_smoking.ipynb` - 数据预处理和格式修复
- `train/stage1_train.py` - 训练脚本（已增强解析逻辑）
- `src/losses.py` - ListNet loss使用ranking scores

## ✅ 验证清单

修复完成后，确认：

- [ ] `evidence_ranking`中所有元素都是`(int, float)`格式
- [ ] 所有索引都在`0 <= idx < K`范围内
- [ ] 没有重复的索引
- [ ] Scores在`0.0 ~ 1.0`范围内
- [ ] 训练代码可以正常加载数据
- [ ] 训练可以正常启动（无IndexError）

如果所有检查都通过，修复成功！🎉
