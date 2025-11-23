# Random Q-Former vs XLM-RoBERTa Q-Former: 输入形状不匹配问题

## 问题发现 🔍

训练时发现 **Random Q-Former 和 XLM-RoBERTa Q-Former 的损失值完全相同**（都在 0.2 左右），这非常不合理。经过仔细检查，发现了以下关键问题：

## 根本原因 ⚠️

### 1. **输入形状不匹配**

#### XLM-RoBERTa Q-Former（正确的实现）
```python
# task_e_only.py
query_token_embeddings: [batch, T, 768]  # T 个 token 的 embeddings
attention_mask: [batch, T]

# Q-Former forward
Z, aux = qformer(
    input_ids=input_ids,  # [batch, T]
    attention_mask=attention_mask,  # [batch, T]
    evidence_emb=evidence_embeddings,  # [batch, K, 768]
    evidence_mask=evidence_mask,  # [batch, K]
    precomputed_query_emb=query_token_embeddings,  # [batch, T, 768] ✅ 多 token
)

# 内部处理：
# 1. LQs: [batch, N, 768]
# 2. Concat: [LQs, query_tokens] → [batch, N+T, 768]  ✅
# 3. Self-Attention on [batch, N+T, 768]
# 4. Extract LQs: [batch, N, 768]
# 5. Cross-Attention with evidence
```

#### Random Q-Former（错误的实现 - 已修复前）
```python
# task_e_only_random.py (BUG)
query_embedding: [batch, 768]  # Mean pooled! ❌ 丢失了 token-level 信息

# Q-Former forward
Z, aux = qformer(
    query_embeds=query_embedding.unsqueeze(1),  # [batch, 1, 768] ❌ 只有一个 token!
    p_embeds=evidence_embeddings,  # [batch, K, 768]
    pool_padding_mask=pool_padding_mask,  # [batch, K]
)

# 内部处理：
# 1. LQs: [batch, N, 768]
# 2. Concat: [LQs, single_token] → [batch, N+1, 768]  ❌ 只有 1 个 token!
# 3. Self-Attention on [batch, N+1, 768]  ❌ 信息量远小于 [batch, N+T, 768]
# 4. Extract LQs: [batch, N, 768]
# 5. Cross-Attention with evidence
```

### 2. **信息量差异巨大**

| 模型 | Query 输入形状 | 序列长度 | 信息量 |
|------|----------------|----------|--------|
| XLM-RoBERTa Q-Former | `[batch, T, 768]` | T ≈ 32 tokens | **完整 query 序列** ✅ |
| Random Q-Former (旧) | `[batch, 1, 768]` | 1 token | **Mean pooled（信息严重损失）** ❌ |

**这解释了为什么两者损失相同**：
- Random Q-Former 输入信息量太少（只有 mean pooled 的单个向量）
- XLM-RoBERTa Q-Former 输入完整 token 序列（包含位置、词序信息）
- 但由于 Random Q-Former 没有预训练权重，即使用了完整信息也学不好
- 所以表现都差不多 → 损失都在 0.2 左右

## 修复方案 ✅

### 1. 修改 `RandomQFormerDataset`

**Before:**
```python
# Mean pooling（❌ 错误）
mask_expanded = attention_mask.unsqueeze(-1).float()
sum_embeddings = (token_emb * mask_expanded).sum(dim=0)  # [768]
sum_mask = mask_expanded.sum(dim=0).clamp(min=1e-9)
query_pooled = sum_embeddings / sum_mask  # [768]

return {
    'query_embedding': query_pooled,  # [768] ❌
}
```

**After:**
```python
# 保留完整 token embeddings（✅ 正确）
return {
    'query_token_embeddings': token_emb,  # [T, 768] ✅
    'query_attention_mask': attention_mask,  # [T] ✅
}
```

### 2. 修改 `collate_random_qformer_batch`

**Before:**
```python
query_embeddings = torch.zeros(batch_size, 768)  # ❌ 单向量
query_embeddings[b] = sample['query_embedding']
```

**After:**
```python
# 动态 padding 处理多 token
max_T = max(sample['query_token_embeddings'].shape[0] for sample in batch)
query_token_embeddings = torch.zeros(batch_size, max_T, 768)  # ✅
query_attention_mask = torch.zeros(batch_size, max_T)  # ✅

for b, sample in enumerate(batch):
    T_curr = sample['query_token_embeddings'].shape[0]
    query_token_embeddings[b, :T_curr] = sample['query_token_embeddings']
    query_attention_mask[b, :T_curr] = sample['query_attention_mask']
```

### 3. 修改 `train_step`

**Before:**
```python
query_embeddings = batch['query_embeddings'].to(device)  # [batch, 768]
query_embeds = query_embeddings.unsqueeze(1)  # [batch, 1, 768] ❌

Z, aux = qformer(
    query_embeds=query_embeds,  # ❌ 只有 1 个 token
    p_embeds=evidence_embeddings,
    pool_padding_mask=pool_padding_mask,
)
```

**After:**
```python
query_token_embeddings = batch['query_token_embeddings'].to(device)  # [batch, T, 768] ✅
query_attention_mask = batch['query_attention_mask'].to(device)  # [batch, T] ✅

Z, aux = qformer(
    query_embeds=query_token_embeddings,  # ✅ 完整 token 序列
    p_embeds=evidence_embeddings,
    pool_padding_mask=pool_padding_mask,
)
```

### 4. 修改 `qformer_random_init.py`

**Before:**
```python
# Stage 1 Input: Concatenate [LQs, q/a_embed]
# [batch, N, d] + [batch, 1, d] → [batch, N+1, d]  ❌ 假设只有 1 个 token
x = torch.cat([lqs, cond_embed], dim=1)
```

**After:**
```python
# Stage 1 Input: Concatenate [LQs, tokens]
# [batch, N, d] + [batch, T, d] → [batch, N+T, d]  ✅ 支持多 token
num_tokens = cond_embed.size(1)  # T (can be 1 or more)
x = torch.cat([lqs, cond_embed], dim=1)
```

## 现在两个 Q-Former 的一致性 ✅

| 特性 | XLM-RoBERTa Q-Former | Random Q-Former |
|------|----------------------|-----------------|
| Query 输入 | `[batch, T, 768]` ✅ | `[batch, T, 768]` ✅ |
| Evidence 输入 | `[batch, K, 768]` ✅ | `[batch, K, 768]` ✅ |
| LQs + Tokens | `[batch, N+T, 768]` ✅ | `[batch, N+T, 768]` ✅ |
| Self-Attention | LQs ↔ Query Tokens ✅ | LQs ↔ Query Tokens ✅ |
| Cross-Attention | LQs → Evidence ✅ | LQs → Evidence ✅ |
| Output | `[batch, N, 768]` ✅ | `[batch, N, 768]` ✅ |

## 预期结果 📊

修复后，两个模型应该有显著差异：

### Random-Init Q-Former（预期）
- **Loss**: 开始高（0.6-0.7），缓慢下降
- **收敛速度**: 慢（~40k steps）
- **最终 F1**: ~0.65-0.70
- **原因**: 完全随机初始化，需要从头学习

### XLM-RoBERTa Q-Former（预期）
- **Loss**: 开始较低（0.3-0.4），快速下降
- **收敛速度**: 快（~20k steps）
- **最终 F1**: ~0.75-0.80
- **原因**: Pre-trained weights 提供语义理解能力

### 如果修复后损失还是一样
可能的原因：
1. **数据问题**: `evidence_labels` 全是 0 或全是 1
2. **Head 问题**: `EntailmentHead` 没有正确使用 Q-Former 输出
3. **梯度问题**: Q-Former 参数没有更新（被 freeze 了）
4. **Loss 问题**: Focal loss 计算有误

## 验证步骤 🧪

### 1. 检查输入形状
```python
print(f"Query tokens: {query_token_embeddings.shape}")  # 应该是 [batch, T, 768]
print(f"Evidence: {evidence_embeddings.shape}")  # [batch, K, 768]
```

### 2. 检查标签分布
```python
print(f"Labels: {evidence_labels}")
print(f"Positive ratio: {evidence_labels.mean()}")  # 应该在 0.1-0.3 之间
```

### 3. 检查梯度
```python
for name, param in qformer.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm()}")
```

### 4. 检查 Q-Former 输出
```python
print(f"Q-Former output Z: {Z.shape}")  # [batch, N_lq, 768]
print(f"Z mean: {Z.mean()}, std: {Z.std()}")
print(f"Z has NaN: {Z.isnan().any()}")
```

## 总结 📝

这次发现的问题揭示了一个重要的教训：
- **输入形状不匹配会导致模型学不到正确的信息**
- **Mean pooling 会丢失 token-level 的序列信息**（词序、位置等）
- **对比实验必须保证输入信息量一致**，否则对比无意义

修复后，Random Q-Former 应该能接收到和 XLM-RoBERTa Q-Former 相同的输入信息量，从而可以公平对比两者的性能差异。
