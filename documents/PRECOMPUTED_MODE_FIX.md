# Precomputed Embeddings Mode Fix

## 问题描述

### 发现的严重 Bug

在 `use_precomputed_embeddings=True` 模式下，存在一个严重的 bug 导致模型使用了错误的 embeddings：

**问题流程**：
1. PKL 文件中的 `query_embedding['input_ids']` 包含 **Qwen3 token IDs**（vocab size ~152K）
2. 这些 Qwen3 token IDs 被传递给 Q-Former 的 `input_ids` 参数
3. Q-Former 初始化时 **没有设置 `bypass_embeddings=True`**
4. 因此 Q-Former 的 XLM-RoBERTa embedding 层会将 Qwen3 token IDs 当作 XLM-R token IDs 来查找
5. **结果**：产生完全错误的 embeddings，导致训练失败

**为什么这是个严重问题**：
- Qwen3 tokenizer 的 vocab size 约为 152K
- XLM-RoBERTa tokenizer 的 vocab size 约为 250K
- 两者的 token ID 映射完全不同
- 使用 Qwen3 ID 查找 XLM-R embedding table 会得到随机、无意义的向量

### 正确的行为

**Precomputed Mode** (`use_precomputed_embeddings=True`):
- `input_ids`：来自 PKL（Qwen3 token IDs）- **仅用于形状信息，不用于 embedding 查找**
- `attention_mask`：来自 PKL（对应 Qwen3 序列）- **正确使用**
- `query_token_embeddings`：来自 PKL（Qwen3 model 生成）- **正确使用**
- Q-Former 应该设置 `bypass_embeddings=True` 以跳过 XLM-R embedding 层

**Dynamic Mode** (`use_precomputed_embeddings=False`):
- `input_ids`：来自 XLM-R tokenizer - **用于 embedding 查找**
- `attention_mask`：来自 XLM-R tokenizer - **正确使用**
- `query_token_embeddings`：由 XLM-R model 动态生成 - **正确使用**
- Q-Former 应该设置 `bypass_embeddings=False` 以使用 XLM-R embedding 层

## 修复方案

### 代码修改

#### 1. Q-Former 初始化（train/stage1_train.py）

**修改前**：
```python
self.qformer = XLMRobertaDRQFormer(
    xlm_model_name=config.xlm_model_name,
    n_queries=config.n_queries,
    hidden_dim=config.hidden_dim,
    num_heads=config.num_heads,
    dropout=0.1,
    use_ca_layers=config.use_ca_layers,
    freeze_xlmr=config.freeze_xlmr,
    # ❌ 缺少 bypass_embeddings 参数！
).to(self.device)
```

**修改后**：
```python
self.qformer = XLMRobertaDRQFormer(
    xlm_model_name=config.xlm_model_name,
    n_queries=config.n_queries,
    hidden_dim=config.hidden_dim,
    num_heads=config.num_heads,
    dropout=0.1,
    use_ca_layers=config.use_ca_layers,
    freeze_xlmr=config.freeze_xlmr,
    bypass_embeddings=config.use_precomputed_embeddings,  # ✅ 添加这行！
).to(self.device)
```

#### 2. 注释改进（train/stage1_train.py）

**修改前**：
```python
query_input_ids = query_emb['input_ids'].squeeze(0)  # [seq_len] - Qwen3 token IDs (won't be used)
query_attention_mask = query_emb['attention_mask'].squeeze(0)  # [seq_len]
query_token_embeddings = query_emb['token_emb_768'].squeeze(0)  # [seq_len, 768] - Pre-computed embeddings
```

**修改后**：
```python
query_input_ids = query_emb['input_ids'].squeeze(0)  # [seq_len] - Qwen3 token IDs (for shape only, not used for embedding)
query_attention_mask = query_emb['attention_mask'].squeeze(0)  # [seq_len] - Valid token mask (used by Q-Former)
query_token_embeddings = query_emb['token_emb_768'].squeeze(0)  # [seq_len, 768] - Pre-computed embeddings (used by Q-Former)
```

### Q-Former 内部逻辑

Q-Former 的 `forward` 方法中已经有正确的逻辑（src/models/qformer_xlm.py）：

```python
# Step 1: Get token embeddings
if self.bypass_embeddings and precomputed_query_emb is not None:
    # ✅ Precomputed mode: 使用预计算的 embeddings
    token_emb = precomputed_query_emb  # [B, T, d]
else:
    # ✅ Dynamic mode: 使用 XLM-R embeddings
    token_emb = self.embeddings(input_ids)  # [B, T, d]
```

关键是确保 `self.bypass_embeddings` 在初始化时正确设置！

## 验证

### 测试脚本

运行 `verify_precomputed_fix.py` 验证修复：

```bash
python verify_precomputed_fix.py
```

测试内容：
1. ✅ 验证 Q-Former 正确初始化 `bypass_embeddings`
2. ✅ 验证 precomputed mode 不使用 input_ids 进行 embedding 查找
3. ✅ 验证 attention_mask 正确使用
4. ✅ 验证 precomputed_query_emb 正确使用

### 预期输出

```
✅ Q-Former bypass_embeddings: True
✅ Q-Former (dynamic mode) bypass_embeddings: False
✅ Forward pass successful with precomputed embeddings
✅ Attention mask correctly controls which tokens are valid
✅ VERIFICATION PASSED: Precomputed Mode is Correct
```

## 影响分析

### 修复前的问题

**症状**：
- 训练无法收敛
- Loss 不下降或上升
- 模型输出随机

**根本原因**：
- Q-Former 接收了 Qwen3 token IDs
- 但使用 XLM-R embedding table 查找
- 得到的 embeddings 与原始语义完全无关
- 相当于给模型喂随机噪声

### 修复后的改进

**Precomputed Mode** (默认，推荐):
- ✅ 正确使用 PKL 中的 Qwen3 embeddings
- ✅ 避免 tokenizer 不匹配问题
- ✅ 更快（无需重新生成 embeddings）
- ✅ 与数据预处理阶段一致

**Dynamic Mode** (实验性):
- ✅ 使用 XLM-R tokenizer + model 生成一致的 embeddings
- ✅ 适合测试不同的 embedding 模型
- ⚠️  较慢（需要实时计算）

## 使用建议

### 推荐配置

**训练阶段**（推荐使用 precomputed）：
```python
config = Stage1Config(
    use_precomputed_embeddings=True,  # 使用预计算的 Qwen3 embeddings
    train_data_path="smoking_train_ms_subset.pkl",
)
```

**实验阶段**（测试不同 embeddings）：
```python
config = Stage1Config(
    use_precomputed_embeddings=False,  # 动态生成 XLM-R embeddings
    xlm_model_name="xlm-roberta-base",
)
```

### 重要提醒

⚠️  **两种模式不可混用**：
- Precomputed mode 使用 Qwen3 embeddings
- Dynamic mode 使用 XLM-R embeddings
- 混用会导致 embedding 空间不一致

⚠️  **Checkpoint 兼容性**：
- 用 precomputed mode 训练的模型不应用 dynamic mode 继续训练（反之亦然）
- 切换模式需要从头训练

## 技术细节

### PKL 数据结构

```python
sample['query_embedding'] = {
    'input_ids': [1, seq_len],        # Qwen3 token IDs
    'attention_mask': [1, seq_len],   # Qwen3 attention mask
    'token_emb_768': [1, seq_len, 768]  # Qwen3 embeddings
}
```

### Q-Former Forward 流程

```
Precomputed Mode (bypass_embeddings=True):
  input_ids (Qwen3) ──┐
  attention_mask ─────┼──> Q-Former
  precomputed_emb ────┘

Dynamic Mode (bypass_embeddings=False):
  input_ids (XLM-R) ──┐
  attention_mask ─────┼──> Q-Former
                      └──> embeddings(input_ids)
```

### 参数传递链

```
Stage1Config.use_precomputed_embeddings
    ↓
Stage1Trainer.__init__
    ↓
XLMRobertaDRQFormer.__init__(bypass_embeddings=...)
    ↓
XLMRobertaDRQFormer.forward()
    if self.bypass_embeddings and precomputed_query_emb is not None:
        use precomputed_query_emb  ✅
    else:
        use self.embeddings(input_ids)  ✅
```

## 相关文件

- `train/stage1_train.py`: 主训练脚本（已修复）
- `src/models/qformer_xlm.py`: Q-Former 实现（已支持 bypass）
- `verify_precomputed_fix.py`: 验证脚本
- `check_precomputed_mode.py`: 问题检测脚本

## 总结

这个修复确保了：
1. ✅ **正确性**：Precomputed mode 不再错误地使用 XLM-R embeddings 处理 Qwen3 token IDs
2. ✅ **一致性**：embeddings 来源与 tokenizer 匹配
3. ✅ **灵活性**：两种模式都可以正确工作
4. ✅ **可维护性**：清晰的注释说明每个参数的用途

**关键修改**：一行代码，影响巨大
```python
bypass_embeddings=config.use_precomputed_embeddings  # 🎯 This is the fix!
```
