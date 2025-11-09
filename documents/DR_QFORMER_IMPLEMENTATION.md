# DR-QFormer Implementation Guide

## 概述

DR-QFormer (Differentiable RAG Q-Former) 是一个借鉴 BLIP-2 架构、专为 RAG 系统设计的参数高效中间件。本实现完全遵循设计文档中的三阶段交叉注意力架构。

## 核心架构

### 三层系统
```
冻结检索器 (Retriever) → DR-QFormer (可训练) → 冻结LLM
```

### DR-QFormer 内部结构 (三阶段)

#### Stage 1: Self-Attention (SA)
- **输入**: `Concat([LQs, q_embed/a_embed])` - 形状 `[batch, N+1, d]`
- **作用**: LQs与query/answer embedding深度融合
- **掩码**: `(N+1) x (N+1)` 全连通 (双向注意力)
- **输出**: 查询/答案感知的LQs `[batch, N, d]`

#### Stage 2: Cross-Attention (CA)
- **Query**: LQs_aware `[batch, N, d]`
- **Key/Value**: P_embeds `[batch, k, d]` (来自冻结检索器)
- **掩码**: `N x k` 全连通 (每个LQ关注所有片段)
- **输出**: 知识注入的表示 `Z [batch, N, d]`

#### Stage 3: Feed-Forward Network (FFN)
- **架构**: `Linear(d → 4d) → GELU → Dropout → Linear(4d → d)`
- **输出**: 最终表示 `Z_final [batch, N, d]` → 送入任务头或冻结LLM

## 实现细节

### 1. 模型初始化

```python
from dr_qformer.models.qformer import DRQFormer

model = DRQFormer(
    n_queries=32,        # N个可学习查询token (LQs)
    hidden_dim=768,      # 隐藏维度
    num_layers=6,        # Transformer层数
    num_heads=8,         # 注意力头数
    max_fragments=10,    # 最大片段数k (用于文档)
    dropout=0.1          # Dropout率
)
```

**参数量**: ~57M (可训练), 约占完整系统的1-2%

### 2. 训练模式

#### Primal Mode (QA: Query → Answer)
```python
# 输入: 查询embedding + 片段embeddings
query_embeds = retriever.encode(query_text)      # [batch, 1, d]
p_embeds = retriever.encode(fragments)            # [batch, k, d]

# Q-Former推理
z_qa, aux = model(
    query_embeds=query_embeds,
    p_embeds=p_embeds
)  # z_qa: [batch, N, d]

# 送入任务头
entailment_logits = entailment_head(z_qa)         # Task E
sorting_weights = sorting_head(z_qa, aux)         # Task S
answer = llm.generate(query_text, z_qa)           # Task C
```

#### Dual Mode (QG: Answer → Query)
```python
# 输入: 答案embedding + 片段embeddings
answer_embeds = retriever.encode(answer_text)    # [batch, 1, d]
p_embeds = retriever.encode(fragments)            # [batch, k, d]

# Q-Former推理
z_qg, aux = model(
    answer_embeds=answer_embeds,
    p_embeds=p_embeds
)  # z_qg: [batch, N, d]

# 送入任务头
entailment_logits = entailment_head(z_qg)         # Task E (Dual)
sorting_weights = sorting_head(z_qg, aux)         # Task S (Dual)
question = llm.generate(answer_text, z_qg)        # Task C (Dual)
```

### 3. 注意力掩码

#### 默认行为 (全连通)
```python
# 不提供掩码 = 全连通注意力
z, aux = model(query_embeds=q_embed, p_embeds=p_embeds)
```

#### 自定义掩码 (处理填充)
```python
import torch

# 创建padding mask (True = 忽略该位置)
batch_size, k_fragments = 4, 10
ca_mask = torch.zeros(n_queries, k_fragments, dtype=torch.bool)
ca_mask[:, 8:] = True  # 忽略最后2个片段 (padding)

z, aux = model(
    query_embeds=q_embed,
    p_embeds=p_embeds,
    ca_mask=ca_mask
)
```

### 4. 辅助输出

```python
z, aux = model(query_embeds=q_embed, p_embeds=p_embeds)

# aux包含:
aux['layer_outputs']      # List[Tensor] - 每层的输出 (长度=num_layers)
aux['sa_attn_weights']    # 自注意力权重 (可用于可视化)
aux['ca_attn_weights']    # 交叉注意力权重 (可用于Task S)
aux['z_raw']              # 完整序列 [batch, N+1, d] (包含q/a_embed)
```

## 与BLIP-2的关键区别

| 特性 | BLIP-2 Q-Former | DR-QFormer |
|------|----------------|------------|
| **工作模式** | 离线、查询无关 | 在线、查询敏感 |
| **输入** | 图像patches (高信噪比) | 文本片段 (极低信噪比) |
| **Stage 1** | 总结图像 | 融合Query/Answer |
| **Stage 2** | 压缩视觉信息 | 从片段中提取知识 |
| **核心任务** | ITC, ITM, ITG | Task E, S, C |
| **训练策略** | 两阶段预训练 | 单阶段融合训练 |

## 三个训练任务

### Task E: 蕴含标注 (Entailment)
- **目的**: 片段级过滤，识别可回答片段
- **输出**: k个logits (每个片段一个)
- **损失**: BCE Loss vs 黄金标签
- **对偶**: QA和QG共享同一个gt_k标签

### Task S: 排序监督 (Sorting)
- **目的**: 学习片段重要性排序
- **输出**: CA层的attention weights
- **损失**: KL散度 vs soft target分布
- **对偶**: QA和QG各自的attention weights

### Task C: 精炼生成 (Condensing)
- **目的**: 确保Z向量对冻结LLM有用
- **输出**: 送入LLM的Z表示
- **损失**: 对比生成 (Primal) / 奖励优化 (Dual)
- **关键**: 不可微指标 (EM, ROUGE)

## 隐式对偶约束

DR-QFormer通过**参数共享**实现对偶学习：
- 同一组参数被Primal (QA)和Dual (QG)的梯度同时更新
- 无需显式的对偶约束损失 (如Tang et al. 2017的Ldual)
- 强制模型学习Query ↔ Pool ↔ Answer的双向逻辑关系

## 使用示例

### 完整训练流程示例

```python
import torch
from dr_qformer.models.qformer import DRQFormer

# 1. 初始化
model = DRQFormer(n_queries=32, hidden_dim=768, num_layers=6)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 2. Primal Mode (QA)
query_embeds = frozen_retriever.encode(query_text)
p_embeds = frozen_retriever.encode(retrieved_fragments)

z_qa, aux_qa = model(query_embeds=query_embeds, p_embeds=p_embeds)

# Task E: 蕴含损失
loss_e_qa = bce_loss(entailment_head(z_qa), gt_labels)

# Task S: 排序损失
loss_s_qa = kl_div(aux_qa['ca_attn_weights'], soft_targets)

# Task C: 生成损失
answer_pred = frozen_llm.generate(query_text, z_qa)
loss_c_qa = contrastive_generation_loss(answer_pred, gt_answer)

# 3. Dual Mode (QG)
answer_embeds = frozen_retriever.encode(answer_text)
z_qg, aux_qg = model(answer_embeds=answer_embeds, p_embeds=p_embeds)

loss_e_qg = bce_loss(entailment_head(z_qg), gt_labels)  # 同一个gt
loss_s_qg = kl_div(aux_qg['ca_attn_weights'], soft_targets)
loss_c_qg = reward_loss(frozen_llm.generate(answer_text, z_qg), gt_query)

# 4. 总损失
total_loss = (loss_e_qa + loss_e_qg) + (loss_s_qa + loss_s_qg) + (loss_c_qa + loss_c_qg)
total_loss.backward()
optimizer.step()
```

## 参数效率对比

| 配置 | N | d | L | 参数量 | 内存(FP32) |
|------|---|---|---|--------|-----------|
| 小型 | 16 | 512 | 4 | ~15M | ~58 MB |
| 中型 | 32 | 768 | 6 | ~57M | ~216 MB |
| 大型 | 64 | 1024 | 12 | ~227M | ~866 MB |

**推荐配置**: 中型 (N=32, d=768, L=6)
- 平衡性能和效率
- 适配大多数检索器和LLM (hidden_dim=768)
- 训练时间和内存占用合理

## 下一步

1. **实现任务头** (Task E, S, C):
   - `dr_qformer/models/heads.py`
   - EntailmentHead, SortingHead, CondenseHead

2. **实现训练循环**:
   - `train/task_e.py`, `train/task_s.py`, `train/task_c.py`
   - 对偶训练逻辑

3. **集成冻结模型**:
   - `dr_qformer/adapters/retriever.py` (Contriever, DPR, E5)
   - `dr_qformer/adapters/llm.py` (LLaMA, Mistral, Phi)

4. **数据准备**:
   - 实现 `dr_qformer/data/interfaces.py`
   - 生成gt_k标签, soft_targets等

## 测试

运行测试验证实现:
```bash
python simple_test_qformer.py
```

预期输出:
```
✅ All tests passed!
📊 DR-QFormer Architecture Summary:
   - Total Parameters: 56,736,769
   - Memory (FP32): ~216.43 MB
🎯 Key Features:
   ✓ Online, query-sensitive processing
   ✓ Cross-attention to fragment embeddings
   ✓ Dual training (QA ↔ QG)
```

## 参考

- **BLIP-2**: Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training"
- **参考实现**: `blip2_impl_examples/vqa-qformer-comparison-master/`
- **设计文档**: 本项目的DR-RAG和DR-QFormer介绍

## 关键设计决策

1. **Pre-LayerNorm**: 在attention/FFN之前进行归一化 (更稳定)
2. **残差连接**: 所有子层使用skip connections
3. **参数初始化**: 遵循BLIP-2的初始化策略 (std=0.02)
4. **Dropout**: 所有子层使用相同的dropout率
5. **Temperature**: 为Task E保留温度参数 (但在Q-Former中不使用)

---

**实现状态**: ✅ 核心Q-Former架构完成  
**下一步**: 实现任务头和训练逻辑
