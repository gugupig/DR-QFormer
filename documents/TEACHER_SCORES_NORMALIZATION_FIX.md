# Task S Teacher Scores 归一化问题分析

## 🔍 问题发现

Teacher scores 来自 pair-wise reranker 的输出，reranker 代码中已经做了 **list-wise softmax 归一化**：

```python
# In rerank_evidences(normalize=True)
raw_scores = torch.tensor(_pair_scores(pair_texts))  # P("yes") for each pair
probs = F.softmax(raw_scores, dim=0)  # 已经归一化，和为 1.0
# 返回的 scores 已经是概率分布
```

## ❌ 原来的错误做法

在 `task_s_only.py` 的 `RankingDataset.__getitem__` 中，对已经归一化的概率分布又做了 z-score 标准化：

```python
# 错误：对概率分布做 z-score
teacher_scores = (scores - mean) / std  
# 破坏了概率分布的性质（和不再为1）
```

**为什么这是错误的：**
1. Reranker 输出已经是概率分布（sum = 1.0）
2. Z-score 标准化会破坏这个性质
3. ListNet loss 需要概率分布（用于计算KL散度）

## ✅ 正确的做法

### 方案1：直接使用（已实施）

```python
# 不做任何二次标准化
teacher_scores = reranker_output  # 已经是概率分布
```

### 方案2：如果分布太平坦，调整 temperature

```python
# 在 loss 计算时降低 teacher_tau
teacher_tau: float = 0.5  # < 1.0 会让分布更 sharp

# In compute_ranking_loss
teacher_probs = F.softmax(teacher_scores / teacher_tau, dim=-1)
# 低温度 → 更 peaked 的分布 → 更强的学习信号
```

## 📊 分析：为什么分布看起来很平坦？

从测试数据看到的分数范围 0.04-0.16，这些**不是原始的 P("yes")**，而是：

1. **Raw P("yes")** → 可能范围更大（0.01-0.9）
2. **Softmax 归一化** → 变成概率分布（0.04-0.16）
3. **和为 1.0** → 这是正常的！

即使原始分数差异很大，softmax 后也会"压缩"到一个范围内。关键是**相对顺序**保持不变。

## 🎯 如何判断分布是否有效？

### 不要只看绝对值，要看：

1. **熵 (Entropy)**：
   ```python
   H = -(probs * log(probs)).sum()
   # H < 0.8 * log(K) → 分布有区分度
   # H ≈ log(K) → 接近均匀分布（太平坦）
   ```

2. **Top-1 vs Top-K 概率比**：
   ```python
   top1_prob / mean(rest_probs) > 2.0  # 说明有明显的头部
   ```

3. **KL 散度的绝对值**：
   ```python
   # 如果 KL loss 在 0.001-0.01 范围 → 可能太平坦
   # 如果 KL loss 在 0.1-0.5 范围 → 正常
   ```

## 🔧 修复总结

### 已实施的更改：

1. **移除二次标准化**：
   - 删除了 z-score 标准化代码
   - 直接使用 reranker 输出的概率分布

2. **降低 teacher_tau**：
   - 从 1.0 → 0.5
   - 在 loss 计算时增强分布的 sharpness

3. **添加注释说明**：
   - 解释为什么不需要二次标准化
   - 提示如何调整 temperature

### 预期效果：

```python
# Before (错误的二次标准化)
Reranker probs: [0.15, 0.14, ..., 0.05]  # sum=1.0, 概率分布
Z-score:        [1.2, 0.8, ..., -1.5]     # 破坏了概率性质 ❌
Softmax again:  [0.35, 0.25, ..., 0.02]   # 二次 softmax，混乱 ❌

# After (正确的做法)
Reranker probs: [0.15, 0.14, ..., 0.05]  # sum=1.0
Teacher tau=0.5: [0.22, 0.19, ..., 0.03] # 更 sharp 的分布 ✅
KL loss: 合理的学习信号 ✅
```

## 📝 关键教训

**在做数据预处理时，要清楚每一步的语义：**

- ✅ Raw scores → 需要归一化
- ✅ Logits → 需要 softmax
- ❌ Probabilities → 不要再归一化！

**Reranker 输出的概率分布是有意义的**，不要轻易破坏它！

---

## 参考

- ListNet: Learning to Rank with Soft Labels
- Knowledge Distillation with Temperature Scaling
- Softmax Temperature: Controlling Distribution Sharpness
