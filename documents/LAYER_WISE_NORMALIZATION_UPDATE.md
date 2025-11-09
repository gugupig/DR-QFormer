# 逐层标准化更新说明

## 概述

根据 Spec v1.1 规格要求，将 EntailmentHead 的聚合顺序从"先跨层平均，再标准化"调整为"逐层标准化后再聚合"。

## 修改内容

### 修改文件
- `dr_qformer/models/heads.py` - EntailmentHead.forward()
- `test_task_e.py` - 更新测试以使用新接口

### 旧流程（已弃用）
```
1. 跨层平均聚合 → [B, H, N, K]
2. Mask padding
3. LayerNorm（每头沿 K 维）
4. 头均值 → [B, N, K]
5. Drop-LQ
6. LSE(τ) → [B, K]
```

**问题**：先聚合再标准化，无法保留每层的特征差异。

### 新流程（Spec v1.1 要求）
```
对于每一层 (layer_idx in 12):
  1. Mask padding（设置为 -1e4）
  2. LayerNorm（每头沿 K 维标准化）
  3. 头均值 → [B, N, K]

4. 跨层平均聚合 → [B, N, K]
5. Drop-LQ（训练时）
6. LSE(τ) → [B, K]
```

**优势**：每层独立标准化后再聚合，保留层间特征差异，符合规格要求。

## 实现细节

### EntailmentHead.forward() 核心代码

```python
# Step 1: Process each layer independently
normalized_layers = []

for layer_idx, ca_raw in enumerate(ca_raw_scores_per_head):
    # ca_raw: [batch, num_heads, N, k]
    
    # Step 1a: Apply pool_padding_mask BEFORE LayerNorm
    if pool_padding_mask is not None:
        mask_expanded = pool_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, k]
        ca_raw_masked = ca_raw.masked_fill(~mask_expanded, -1e4)
    else:
        ca_raw_masked = ca_raw
    
    # Step 1b: Manual normalization per head on last dimension (k fragments)
    mean = ca_raw_masked.mean(dim=-1, keepdim=True)  # [batch, num_heads, N, 1]
    var = ca_raw_masked.var(dim=-1, keepdim=True, unbiased=False)
    norm_scores = (ca_raw_masked - mean) / torch.sqrt(var + self.eps)  # [batch, num_heads, N, k]
    
    # Step 1c: Average over heads → [batch, N, k]
    layer_scores_avg = norm_scores.mean(dim=1)
    normalized_layers.append(layer_scores_avg)

# Step 2: Aggregate across layers (mean)
ca_scores_stacked = torch.stack(normalized_layers, dim=0)  # [num_layers, batch, N, k]
ca_scores_avg = ca_scores_stacked.mean(dim=0)  # [batch, N, k]

# Step 3: Drop-LQ regularization (training only)
if training and self.p_drop_lq > 0:
    ca_scores_dropped = self._apply_drop_lq(ca_scores_avg)
else:
    ca_scores_dropped = ca_scores_avg

# Step 4: LogSumExp aggregation over N LQs → [batch, k]
fragment_logits = self._logsumexp_aggregate(ca_scores_dropped)
```

### 数学表示

**旧方法（已弃用）**：
```
scores_agg = mean_over_layers(ca_raw)           # [B, H, N, K]
scores_norm = LayerNorm(Mask(scores_agg))       # [B, H, N, K]
scores_head_avg = mean_over_heads(scores_norm)  # [B, N, K]
```

**新方法（Spec v1.1）**：
```
For each layer i:
  scores_i_norm = LayerNorm(Mask(ca_raw[i]))    # [B, H, N, K]
  scores_i_avg = mean_over_heads(scores_i_norm) # [B, N, K]

scores_layer_avg = mean_over_layers([scores_1_avg, ..., scores_12_avg])  # [B, N, K]
```

## 关键特性

1. **逐层标准化**：每层独立进行 LayerNorm，保留层间差异
2. **Mask 优先**：在标准化前应用 padding mask（设置为 -1e4）
3. **逐头标准化**：沿片段维度 K 对每个注意力头标准化
4. **动态 K 支持**：手动归一化（mean/std）支持批内可变 K
5. **层间聚合**：标准化后的层级分数进行平均聚合

## 验证结果

### test_task_e.py
```
✅ Test 1: EntailmentHead Shape Test - PASSED
✅ Test 2: Drop-LQ Safety Protection Test - PASSED
✅ Test 3: Focal Loss Computation Test - PASSED
✅ Test 4: Focal Loss with Importance Weights - PASSED
✅ Test 5: End-to-End Forward + Backward Pass - PASSED
```

### test_spec_v11_strict.py
```
✅ 目标 1: 使用 pre-softmax CA 原始打分（QKᵀ/√d）
✅ 目标 2: Dual 训练（Primal + Dual 两次前向，共享参数）
✅ 目标 3: Padding 全链路传播（跨注意力、Head、损失）
✅ 目标 4: Drop-LQ 训练开启，评估关闭（带防全丢保护）
✅ 目标 5: 动态 K_pool 支持（不依赖固定 K）
✅ 目标 6: 动态 importance_weights 构造
✅ 目标 7: 调试输出返回（detach 后）
✅ 目标 8: EntailmentHead 接受 hidden_dim（已忽略）
```

## 向后兼容性

- **接口变化**：无变化，仍使用 `ca_raw_scores_per_head`（列表）
- **输出格式**：无变化，仍返回字典 `{'fragment_logits': ..., 'ca_raw_scores_avg': ..., 'ca_raw_scores_per_head': ...}`
- **训练代码**：无需修改，`train/task_e.py` 保持不变

## 性能影响

- **计算量**：略微增加（逐层循环处理）
- **内存占用**：增加中间结果存储（12 层 × [B, N, K]）
- **训练效果**：理论上应提升（保留层间特征差异）

## 后续工作

1. 在真实数据集上训练并比较性能
2. 分析逐层标准化对不同数据集的影响
3. 考虑可选的层加权聚合（而非简单平均）

---

**日期**：2024-11-08  
**版本**：v1.1  
**状态**：✅ 已完成并验证
