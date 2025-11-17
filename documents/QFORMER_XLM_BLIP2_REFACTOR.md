# XLM-RoBERTa DR-QFormer: BLIP-2 风格重构总结

## 修改日期
2025-11-17

## 修改目标
将 `qformer_xlm.py` 中的查询处理方式从"pooled 句向量"升级为"token-level"的 BLIP-2 风格，让查询文本以 token 序列的形式真正参与 self-attention。

---

## 核心修改

### 1. `QueryEvidenceCrossAttention` 类重构

#### 修改前
```python
def forward(
    self,
    lqs_aware: Tensor,
    qa_embed: Optional[Tensor],      # ← 单独的句向量
    context: Tensor,
    context_mask: Optional[Tensor] = None,
    ca_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict]:
    # ...
    # 返回: [LQs, qa_embed] 拼接的张量 [B, N+1, d]
    updated_x = torch.cat([lqs_aware, qa_embed], dim=1)
    return updated_x, aux
```

#### 修改后
```python
def forward(
    self,
    lqs_aware: Tensor,               # ← 已经包含 query 信息（来自 SA）
    context: Tensor,
    context_mask: Optional[Tensor] = None,
    ca_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict]:
    # ...
    # 返回: 仅更新后的 LQs [B, N_q, d]
    updated_lqs = lqs_aware
    return updated_lqs, aux
```

**关键变化：**
- ✅ 移除 `qa_embed` 参数
- ✅ 不再拼接额外的 query/answer embedding
- ✅ 直接返回更新后的 LQs（维度从 `[B, N+1, d]` 改为 `[B, N_q, d]`）
- ✅ 保留所有 CA 诊断接口（`ca_attn_weights`, `ca_raw_scores_per_head`, `ca_raw_scores_avg`）

---

### 2. `XLMRobertaDRQFormer.forward` 方法重构

#### 修改前
```python
def forward(
    self,
    input_ids: Tensor,
    attention_mask: Tensor,
    evidence_emb: Tensor,
    evidence_mask: Optional[Tensor] = None,
    query_embeds: Optional[Tensor] = None,    # ← pooled query embedding
    answer_embeds: Optional[Tensor] = None,   # ← pooled answer embedding
) -> Tuple[Tensor, List[Dict]]:
    # ...
    # 构造 qa_embed
    if query_embeds is not None:
        qa_embed = query_embeds
    elif answer_embeds is not None:
        qa_embed = answer_embeds
    else:
        qa_embed = hidden[:, self.n_queries:self.n_queries+1, :]
    
    # 调用 CA 时传入 qa_embed
    updated, aux = self.cross_layers[i](
        lqs_aware=lq_states,
        qa_embed=qa_embed,              # ← 传入单独的句向量
        context=evidence_emb,
        context_mask=evidence_mask,
    )
    lq_states = updated[:, :self.n_queries, :]  # 提取前 N 个
```

#### 修改后
```python
def forward(
    self,
    input_ids: Tensor,              # ← query token 序列
    attention_mask: Tensor,
    evidence_emb: Tensor,
    evidence_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, List[Dict]]:
    # ...
    # 1. 获取 query token embeddings
    token_emb = self.embeddings(input_ids)  # [B, T, d]
    
    # 2. 拼接 [LQs, query_tokens]
    hidden = torch.cat([lqs, token_emb], dim=1)  # [B, N_q+T, d]
    
    # 3. 构造 mask（LQs 全为 1，query tokens 使用 attention_mask）
    lq_mask = torch.ones(batch_size, self.n_queries, ...)
    extended_mask = torch.cat([lq_mask, attention_mask], dim=1)
    
    # 4. 逐层处理
    for i, xlmr_layer in enumerate(self.encoder_layers):
        # (a) XLM-R SA + FFN：LQs 和 query tokens 双向交互
        hidden = xlmr_layer(hidden, attention_mask=extended_mask_4d)[0]
        
        # (b) 分离 LQs 和 query tokens
        lq_states = hidden[:, :self.n_queries, :]    # [B, N_q, d] ← 已包含 query 信息
        tok_states = hidden[:, self.n_queries:, :]   # [B, T, d]
        
        # (c) CA：query-aware LQs attend to evidence
        if self.cross_layers[i] is not None:
            updated_lqs, aux = self.cross_layers[i](
                lqs_aware=lq_states,        # ← 已包含 query 信息
                context=evidence_emb,
                context_mask=evidence_mask,
            )
            lq_states = updated_lqs         # [B, N_q, d]
        
        # (d) 拼接回去
        hidden = torch.cat([lq_states, tok_states], dim=1)
    
    # 5. 最终只提取 LQs
    Z = hidden[:, :self.n_queries, :]
    Z = self.final_ln(Z)
    return Z, all_aux
```

**关键变化：**
- ✅ 移除 `query_embeds` 和 `answer_embeds` 参数
- ✅ Query 以 token 序列（`input_ids`）形式输入
- ✅ LQs 和 query tokens 在每一层都进行双向 self-attention
- ✅ LQs 通过 SA 自动变得"query-aware"
- ✅ CA 输入的 `lqs_aware` 已经包含 query 信息
- ✅ 不再需要单独的 `qa_embed` 变量

---

## BLIP-2 风格的完整数据流

### 输入
```python
input_ids:       [B, T]      # Query token 序列（例如："What is the capital of France?"）
attention_mask:  [B, T]      # 1=有效 token，0=padding
evidence_emb:    [B, K, 768] # 检索到的证据片段嵌入
evidence_mask:   [B, K]      # True=有效片段，False=padding
```

### 处理流程

#### 第 0 层（Embedding）
```
LQs:          [B, 32, 768]   ← 可学习参数（随机初始化）
Query tokens: [B, T, 768]    ← XLM-R embeddings(input_ids)

拼接: [B, 32+T, 768]
```

#### 第 1-12 层（XLM-R Encoder）

**每一层包含 3 个阶段：**

1. **Self-Attention (SA)**
   ```
   输入: [LQs, query_tokens]  [B, 32+T, 768]
   
   Attention 模式:
   - LQs[i] 可以 attend to 所有 LQs[j] 和所有 query_tokens[k]
   - query_tokens[i] 可以 attend to 所有 LQs[j] 和所有 query_tokens[k]
   
   结果: LQs 逐渐融合 query 信息，变得 "query-aware"
   ```

2. **Cross-Attention (CA)** (仅在指定层，如第 6 层和第 12 层)
   ```
   输入 Query: LQs[0:32]         [B, 32, 768]  ← 已包含 query 信息
   输入 Key/Value: evidence_emb  [B, K, 768]
   
   LQs[i] attend to 所有 evidence[j]
   
   结果: LQs 融合 evidence 信息
   ```

3. **拼接回去**
   ```
   [updated_LQs, query_tokens] -> [B, 32+T, 768]
   ```

#### 输出
```
Z = LQs[0:32]  [B, 32, 768]
- 包含 query 语义（来自 SA）
- 包含 evidence 信息（来自 CA）
- 可以送入下游任务头
```

---

## 测试验证

创建了 `test_qformer_xlm_blip2.py` 脚本，包含 6 个测试：

1. ✅ **模型初始化测试**
   - 验证模型加载成功
   - 统计可训练参数

2. ✅ **前向传播测试（无 padding）**
   - 输入全有效的 token 和 evidence
   - 验证输出形状正确：`[B, 32, 768]`
   - 验证 aux 字典数量：12 个（每层一个）

3. ✅ **CA 层验证**
   - 验证 CA 仅在指定层（5, 11）应用
   - 验证 CA 输出包含 3 个诊断字段：
     - `ca_attn_weights`: [B, 12, 32, K]
     - `ca_raw_scores_per_head`: [B, 12, 32, K]
     - `ca_raw_scores_avg`: [B, 32, K]
   - 验证注意力权重和为 1

4. ✅ **Padding 处理测试**
   - 验证对 query tokens 的 padding 处理
   - 验证对 evidence 的 padding 处理
   - 验证 CA 不会 attend 到 padded evidence

5. ✅ **Query-Awareness 验证**
   - 用两个不同的 query 输入
   - 验证输出的 LQs 确实不同
   - 证明 LQs 包含了 query 信息

6. ✅ **梯度流测试**
   - 验证梯度可以从 LQs 通过 CA 流向 evidence
   - 验证 learnable query tokens 可以接收梯度

---

## 与原始 DR-QFormer 的对比

| 特性 | 原始 DRQFormer | XLMRobertaDRQFormer (BLIP-2 风格) |
|------|----------------|-----------------------------------|
| Query 输入方式 | Pooled embedding `[B, 1, d]` | Token 序列 `input_ids [B, T]` |
| SA 参与方式 | `[LQs, query_embed]` 在 SA 中交互 | `[LQs, query_tokens]` 双向 SA |
| Query 信息融合 | 依赖单个句向量 | Token-level 多层交互 |
| CA 输入 | `lqs_aware` + `qa_embed` | 仅 `lqs_aware`（已包含 query 信息）|
| CA 输出 | `[LQs, qa_embed]` [B, N+1, d] | 仅 `LQs` [B, N_q, d] |
| 多语言支持 | ❌ | ✅ XLM-R (100+ 语言) |
| 预训练初始化 | ❌ | ✅ XLM-R 预训练权重 |

---

## 文件修改清单

1. **`src/models/qformer_xlm.py`**
   - ✅ 更新模块文档字符串（添加 BLIP-2 说明和使用示例）
   - ✅ 重构 `QueryEvidenceCrossAttention.forward`（移除 `qa_embed`）
   - ✅ 更新 `XLMRobertaDRQFormer` 文档字符串
   - ✅ 重构 `XLMRobertaDRQFormer.forward`（移除 `query_embeds`/`answer_embeds`）
   - ✅ 简化 CA 调用逻辑
   - ✅ 更新测试脚本（移除 `query_embeds`）

2. **`test_qformer_xlm_blip2.py`** (新文件)
   - ✅ 创建完整的测试套件
   - ✅ 6 个测试覆盖所有关键功能
   - ✅ 包含 padding、query-awareness、梯度流验证

---

## 使用示例

```python
from transformers import AutoTokenizer
from src.models.qformer_xlm import XLMRobertaDRQFormer

# 1. 初始化模型
model = XLMRobertaDRQFormer(
    xlm_model_name="xlm-roberta-base",
    n_queries=32,
    use_ca_layers=[5, 11],  # CA 仅在第 6 和第 12 层
    freeze_xlmr=False,       # 全参数训练
)

# 2. 准备输入
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Query 文本（支持 100+ 种语言）
query_text = "法国的首都是什么？"
tokens = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True)

# Evidence embeddings（从检索器获取）
evidence_emb = retriever.encode(retrieved_passages)  # [B, K, 768]
evidence_mask = torch.ones(B, K, dtype=torch.bool)

# 3. 前向传播
Z, all_aux = model(
    input_ids=tokens['input_ids'],
    attention_mask=tokens['attention_mask'],
    evidence_emb=evidence_emb,
    evidence_mask=evidence_mask,
)

# Z: [B, 32, 768] - query-aware、evidence-fused LQ 表征
# 送入下游任务头进行预测
predictions = task_head(Z)
```

---

## 优势总结

### 1. 更接近 BLIP-2 的成功范式
- Token-level query 处理 vs. pooled embedding
- 多层双向 SA 实现细粒度交互
- LQs 自动变得 query-aware

### 2. 更强的表达能力
- 保留 query 的 token-level 细节
- 不依赖单一句向量的质量
- 支持长 query（受 XLM-R 512 token 限制）

### 3. 多语言支持
- XLM-R 预训练权重覆盖 100+ 语言
- 无需额外的跨语言对齐

### 4. 代码简化
- 移除 `qa_embed` 的传递逻辑
- CA 接口更简洁（仅 3 个必需参数）
- 更容易理解和维护

---

## 注意事项

1. **Tokenizer 选择**：必须使用 `xlm-roberta-base` 对应的 tokenizer
2. **序列长度**：XLM-R 最大支持 512 tokens（包括 LQs + query tokens）
3. **内存占用**：比原始版本略高（因为 query tokens 参与所有层）
4. **兼容性**：不兼容原始 `DRQFormer` 的 checkpoint（接口不同）

---

## 后续工作建议

1. ✅ 在真实数据集上验证性能（FEVER、HotpotQA、MS MARCO）
2. ✅ 与原始 DR-QFormer 进行对比实验
3. ✅ 探索不同的 CA 层配置（例如：只在最后一层）
4. ✅ 尝试冻结 XLM-R 进行参数高效训练
5. ✅ 支持 Dual 模式（answer tokens 作为输入）

---

**修改完成时间**: 2025-11-17  
**修改人员**: GitHub Copilot (Claude Sonnet 4.5)
