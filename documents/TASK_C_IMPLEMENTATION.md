# Task C Implementation Summary

## 实现日期
2025-11-10

## 设计规格
基于 **v8.0 最终集成版** - Condensing Generation with Contrastive NLL

---

## 核心设计理念

### 与原始设计的差异
- **原始设计 (Reward Margin)**: 基于 ROUGE/BLEU 奖励的生成式对比
- **v8.0 设计 (Contrastive NLL)**: 基于困惑度降低的纯监督学习

### v8.0 核心特性
1. **纯 Teacher Forcing**: 无需生成采样，直接测量 NLL 降低
2. **双路对比**: LLM(with Z) vs. LLM(without Z)
3. **后验回溯**: 从 LLM 注意力实时提取片段重要性
4. **子集优化**: 仅在训练子集 U 上计算后验（效率优化）

---

## 实现的模块

### 1. `dr_qformer/losses.py` - 核心损失函数

#### `compute_condensing_loss()`
**功能**: 计算对比 NLL 损失并提取后验

**输入参数**:
```python
nll_with_evidence: Tensor       # Path A: LLM NLL with Z prefix
nll_without_evidence: Tensor    # Path B: LLM NLL without Z (detached)
llm_attention_weights: Tensor   # [batch, n_heads, seq_total, N_lq]
ca_weights: Tensor              # [batch, N_lq, K] Q-Former CA weights
subset_indices: Tensor          # [batch, |U|] training subset
answer_start_idx: int           # Token position where answer starts
softplus_beta: float = 10.0     # Softplus sharpness
margin_mode: str = 'adaptive'   # 'fixed' or 'adaptive'
margin_fixed: float = 0.5       # Fixed margin value
margin_adaptive_ratio: float = 0.5  # Adaptive margin ratio κ
margin_min: float = 0.1         # Min margin
margin_max: float = 2.0         # Max margin
```

**输出**:
```python
{
    'loss_c': Tensor,              # Condensing loss (scalar, for backprop)
    'nll_gain': Tensor,            # G = nll_no - nll_with (detached)
    'margin': Tensor,              # Computed margin m (detached)
    'posterior_q_psi_U': Tensor,   # [batch, |U|] fragment posterior (detached)
}
```

**损失公式**:
```
1. NLL Gain: G = nll_without - nll_with
2. Adaptive Margin: m = clip(μ_G + κ·σ_G, m_min, m_max)
3. Softplus Loss: L_C = (1/β) · log(1 + exp(β·(m - G)))
```

**后验提取逻辑**:
```python
# Step 1: Average LLM→Z attention over answer tokens and heads
w_lq = mean(llm_attention[:, :, answer_start:, :])  # [batch, N_lq]

# Step 2: Extract CA weights for subset U
ca_weights_U = ca_weights.gather(subset_indices)  # [batch, N_lq, |U|]

# Step 3: Compute posterior via matrix multiplication
q_ψ_U = softmax(w_lq @ ca_weights_U)  # [batch, |U|]
```

**关键特性**:
- ✅ 自适应 Margin 自动调整难度
- ✅ Softplus 提供平滑梯度
- ✅ 后验提取为 detached（用作伪标签）
- ✅ 支持批量和标量 NLL 输入

---

### 2. `dr_qformer/adapters/llm.py` - LLM 适配器

#### 新增方法: `teacher_forcing_dual_path()`

**功能**: 双路 Teacher Forcing 计算对比 NLL

**输入**:
```python
z_prefix: Tensor              # [batch, N_lq, d_llm] Q-Former output
query_input_ids: Tensor       # [batch, S_q] Query tokens
answer_input_ids: Tensor      # [batch, S_a] Answer tokens
capture_attention: bool       # Whether to capture attention
```

**输出**:
```python
{
    'nll_with_evidence': Tensor,      # Path A NLL (with gradient)
    'nll_without_evidence': Tensor,   # Path B NLL (detached)
    'llm_attention_to_z': Tensor,     # [batch, n_heads, S_a, N_lq]
    'answer_start_idx': int,          # N_lq + S_q
}
```

**实现步骤 (TODO - 占位符)**:

```python
# Step 1: 统一输入准备
input_ids = [dummy_Z(N), Query(S_q), Answer(S_a)]
labels = [-100(N), -100(S_q), Answer(S_a)]
common_embeds = LLM.embed_tokens(input_ids)

# Step 2: 构造 Prefix-LM Mask (Mask A)
# Z sees itself (causal)
# Q sees Z + itself (causal)
# A sees Z + Q + itself (causal)
mask_A = construct_prefix_lm_mask(N, S_q, S_a)

# Step 3: Path A (With Evidence)
embeds_A = common_embeds.clone()
embeds_A[:, :N, :] = z_prefix  # Replace dummy Z
register_attention_hook()
outputs_A = LLM(embeds_A, mask_A, labels)
nll_with = outputs_A.loss

# Step 4: 构造阻断 Mask (Mask B)
mask_B = mask_A.clone()
mask_B[N:, :N] = -inf  # Block Q and A from seeing Z

# Step 5: Path B (Without Evidence)
with torch.no_grad():
    outputs_B = LLM(embeds_A, mask_B, labels)
    nll_without = outputs_B.loss

# Step 6: 提取注意力
llm_attention_to_z = captured_attention[:, :, N+S_q:, :N]
```

**Hook 实现 (TODO)**:
```python
def _register_attention_hook(self):
    def hook_fn(module, input, output):
        self._captured_attention = output[1]  # [batch, n_heads, seq, seq]
    
    last_layer = self.model.model.layers[-1]
    self._attention_hook_handle = last_layer.register_forward_hook(hook_fn)
```

**状态**: 🔶 **占位符实现** - 返回 dummy 值用于测试

---

### 3. `dr_qformer/models/heads.py` - CondenseHead

#### 更新: 完整实现

**架构**:
```python
class CondenseHead(nn.Module):
    def __init__(self, hidden_dim=768, llm_hidden_dim=4096):
        # Projection: hidden_dim → llm_hidden_dim
        self.proj = nn.Linear(hidden_dim, llm_hidden_dim)
        
        # Layer normalization for stable embeddings
        self.norm = nn.LayerNorm(llm_hidden_dim)
    
    def forward(self, z):
        # z: [batch, N_lq, hidden_dim]
        z_prefix = self.proj(z)        # [batch, N_lq, llm_hidden_dim]
        z_prefix = self.norm(z_prefix)
        return z_prefix
```

**参数量** (768→4096):
- Projection: 768 × 4096 + 4096 = 3,149,824
- LayerNorm: 4096 × 2 = 8,192
- **总计**: 3,158,016

**状态**: ✅ **完全实现** - 测试通过

---

### 4. `train/task_c.py` - 训练流程

#### KnowledgeCondenser 类

**组件**:
```python
class KnowledgeCondenser(nn.Module):
    def __init__(self, args):
        self.qformer = DRQFormer(...)         # Trainable
        self.condense_head = CondenseHead(...) # Trainable
        self.frozen_llm = FrozenLLM(...)      # Frozen, eval mode
```

**前向传播**:
```python
def forward(self, q_embeds, p_embeds, query_input_ids, answer_input_ids, 
            subset_indices, pool_padding_mask):
    # 1. Q-Former forward
    z, ca_outputs = self.qformer(q_embeds, p_embeds, pool_padding_mask)
    
    # 2. CondenseHead projection
    z_prefix = self.condense_head(z)  # [batch, N_lq, d_llm]
    
    # 3. Dual-path LLM forward (TODO: placeholder)
    llm_outputs = self.frozen_llm.teacher_forcing_dual_path(
        z_prefix, query_input_ids, answer_input_ids, capture_attention=True
    )
    
    # 4. Compute condensing loss
    loss_dict = compute_condensing_loss(
        nll_with_evidence=llm_outputs['nll_with_evidence'],
        nll_without_evidence=llm_outputs['nll_without_evidence'],
        llm_attention_weights=llm_outputs['llm_attention_to_z'],
        ca_weights=ca_outputs.get('ca_weights'),
        subset_indices=subset_indices,
        answer_start_idx=llm_outputs['answer_start_idx'],
        **args,  # margin params, beta, etc.
    )
    
    return loss_dict
```

#### 训练循环

**关键特性**:
- ✅ 只优化 Q-Former + CondenseHead
- ✅ LLM 始终处于 `eval()` 模式
- ✅ 记录 NLL gain, margin, posterior
- ✅ 使用 dummy dataset 测试流程

**命令行参数**:
```bash
python train/task_c.py \
    --n_queries 32 \
    --hidden_dim 768 \
    --llm_hidden_dim 4096 \
    --llm_model_name "microsoft/phi-2" \
    --softplus_beta 10.0 \
    --margin_mode adaptive \
    --margin_adaptive_ratio 0.5 \
    --lr 1e-4 \
    --batch_size 4 \
    --epochs 10
```

**状态**: ✅ **完全实现** - 使用 dummy LLM 可运行

---

## 测试验证

### `test_task_c.py` - 5 个单元测试

| 测试 | 功能 | 状态 |
|------|------|------|
| Test 1 | 基础损失计算 | ✅ PASSED |
| Test 2 | 自适应 Margin | ✅ PASSED |
| Test 3 | 后验提取 | ✅ PASSED |
| Test 4 | CondenseHead 前向 | ✅ PASSED |
| Test 5 | 端到端流程 | ✅ PASSED |

**测试输出摘要**:
```
Test 1: Basic Condensing Loss
  NLL Gain (G): 1.3000 > Margin (0.5) → Loss ≈ 0
  NLL Gain (G): 0.2000 < Margin (0.5) → Loss = 0.3049 ✓

Test 2: Adaptive Margin
  G = 1.5 → Adaptive margin = 1.5 (clipped to [0.1, 2.0]) ✓

Test 3: Posterior Extraction
  LLM attention [2, 8, 200, 32] + CA weights [2, 32, 50]
  → Posterior [2, 10] (softmax normalized) ✓

Test 4: CondenseHead
  768D → 4096D projection: 3,158,016 params ✓

Test 5: End-to-End
  Gradients flow through CondenseHead ✓
```

---

## 设计决策与权衡

### 1. 为什么用 Contrastive NLL 而非 ROUGE Reward？

**优势**:
- ✅ 直接测量证据效用（困惑度降低）
- ✅ 无需奖励模型或采样（更稳定）
- ✅ Teacher forcing 确保稳定梯度
- ✅ 可微分、端到端训练

**劣势**:
- ❌ 需要 LLM 前向传播（计算成本高）
- ❌ 依赖 LLM 质量（冻结 LLM 必须强大）

### 2. 为什么用 Softplus 而非 Hinge/ReLU？

**原因**:
- Smooth gradients everywhere (无死区)
- 可微分穿过 margin 边界
- β 参数可调节陡度 (β→∞ 逼近 ReLU)

**公式**:
```
Softplus(x) = (1/β) · log(1 + exp(β·x))
```

### 3. 为什么用自适应 Margin？

**原因**:
- 自动适应不同批次的 NLL 尺度
- 防止过拟合到固定 margin
- κ·σ_G 提供动态难度调节

**公式**:
```
m = clip(μ_G + κ·σ_G, m_min, m_max)
```

### 4. 为什么只计算子集 U 的后验？

**原因**:
- 训练子集 U 通常很小 (|U| << K)
- 全 K 后验计算昂贵且不必要
- 后验仅用作伪标签（Task S 集成）

---

## 集成路线图

### ✅ 已完成 (Phase 1: 核心框架)

1. ✅ `compute_condensing_loss()` 完整实现
2. ✅ `CondenseHead` 完整实现
3. ✅ `FrozenLLM.teacher_forcing_dual_path()` 接口定义
4. ✅ `KnowledgeCondenser` 训练流程
5. ✅ 5 个单元测试全部通过
6. ✅ Dummy pipeline 端到端运行

### 🔶 待完成 (Phase 2: LLM 集成)

**优先级 P0 (必需)**:
- [ ] 加载实际 LLM 模型 (LLaMA-2-7B / Mistral-7B / Phi-2)
- [ ] 实现 `embed_tokens()` 访问
- [ ] 实现 Prefix-LM mask 构造
- [ ] 实现 attention hook 注册
- [ ] 测试不同 LLM 架构兼容性

**优先级 P1 (重要)**:
- [ ] 实际 QA 数据集加载 (NQ, SQuAD, etc.)
- [ ] 动态子集 U 构造策略
- [ ] Checkpoint 保存/加载
- [ ] 分布式训练支持

**优先级 P2 (优化)**:
- [ ] KV cache 优化 (减少 LLM 计算)
- [ ] Mixed precision training (FP16)
- [ ] Gradient checkpointing
- [ ] 后验提取向量化优化

---

## LLM 集成指南

### 推荐 LLM 选择

| 模型 | 大小 | 优势 | 劣势 |
|------|------|------|------|
| **Phi-2** | 2.7B | 快速、易部署 | 推理能力弱 |
| **Mistral-7B** | 7B | 性能/效率平衡 | 需要 24GB GPU |
| **LLaMA-2-7B** | 7B | 开源、稳定 | 许可限制 |
| **Qwen-7B** | 7B | 多语言 | 中文偏向 |

### 集成步骤

#### Step 1: 加载 LLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

self.tokenizer = AutoTokenizer.from_pretrained(model_name)
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # FP16 for memory
    device_map="auto",
).eval()

# Freeze
for param in self.model.parameters():
    param.requires_grad = False
```

#### Step 2: 实现 Prefix-LM Mask

```python
def construct_prefix_lm_mask(N_lq, S_q, S_a, device):
    seq_len = N_lq + S_q + S_a
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    # Z sees itself (causal)
    for i in range(N_lq):
        mask[i, :i+1] = 1.0
    
    # Q sees Z + itself (causal)
    for i in range(N_lq, N_lq + S_q):
        mask[i, :N_lq] = 1.0  # See all Z
        mask[i, N_lq:i+1] = 1.0  # Causal in Q
    
    # A sees Z + Q + itself (causal)
    for i in range(N_lq + S_q, seq_len):
        mask[i, :N_lq] = 1.0  # See all Z
        mask[i, N_lq:N_lq+S_q] = 1.0  # See all Q
        mask[i, N_lq+S_q:i+1] = 1.0  # Causal in A
    
    # Convert to attention mask format
    mask = (1.0 - mask) * -1e4
    return mask
```

#### Step 3: 实现 Attention Hook

```python
def _register_attention_hook(self):
    def hook_fn(module, input, output):
        # output structure depends on LLM architecture
        # LLaMA: output = (hidden_states, attention_weights, ...)
        self._captured_attention = output[1]  # [batch, n_heads, seq, seq]
    
    # Hook on last decoder layer
    if hasattr(self.model, 'model'):  # LLaMA structure
        last_layer = self.model.model.layers[-1]
    elif hasattr(self.model, 'transformer'):  # GPT structure
        last_layer = self.model.transformer.h[-1]
    else:
        raise NotImplementedError(f"Unknown LLM architecture: {type(self.model)}")
    
    self._attention_hook_handle = last_layer.register_forward_hook(hook_fn)
```

#### Step 4: 双路前向传播

```python
def teacher_forcing_dual_path(self, z_prefix, query_input_ids, answer_input_ids, capture_attention=True):
    batch_size = z_prefix.shape[0]
    N_lq = z_prefix.shape[1]
    S_q = query_input_ids.shape[1]
    S_a = answer_input_ids.shape[1]
    
    # 1. Create dummy Z tokens
    dummy_z_tokens = torch.zeros(batch_size, N_lq, dtype=torch.long, device=self.device)
    
    # 2. Concatenate input_ids
    input_ids = torch.cat([dummy_z_tokens, query_input_ids, answer_input_ids], dim=1)
    
    # 3. Create labels (only compute loss on answer)
    labels = torch.cat([
        torch.full((batch_size, N_lq), -100, dtype=torch.long, device=self.device),
        torch.full((batch_size, S_q), -100, dtype=torch.long, device=self.device),
        answer_input_ids,
    ], dim=1)
    
    # 4. Embed tokens
    common_embeds = self.model.get_input_embeddings()(input_ids)
    
    # 5. Construct masks
    mask_A = construct_prefix_lm_mask(N_lq, S_q, S_a, self.device)
    mask_B = mask_A.clone()
    mask_B[N_lq:, :N_lq] = -1e4  # Block Q and A from seeing Z
    
    # 6. Path A: With Evidence
    embeds_A = common_embeds.clone()
    embeds_A[:, :N_lq, :] = z_prefix
    
    if capture_attention:
        self._register_attention_hook()
    
    outputs_A = self.model(
        inputs_embeds=embeds_A,
        attention_mask=mask_A,
        labels=labels,
        output_attentions=True,
    )
    nll_with = outputs_A.loss
    
    # Extract attention
    llm_attention_to_z = None
    if capture_attention and self._captured_attention is not None:
        llm_attention_to_z = self._captured_attention[:, :, N_lq+S_q:, :N_lq]
        self._remove_attention_hook()
    
    # 7. Path B: Without Evidence
    with torch.no_grad():
        outputs_B = self.model(
            inputs_embeds=embeds_A,  # Same embeddings
            attention_mask=mask_B,   # Blocked mask
            labels=labels,
        )
        nll_without = outputs_B.loss
    
    return {
        'nll_with_evidence': nll_with,
        'nll_without_evidence': nll_without.detach(),
        'llm_attention_to_z': llm_attention_to_z,
        'answer_start_idx': N_lq + S_q,
    }
```

---

## 超参数调优建议

### 损失函数超参数

| 参数 | 推荐范围 | 默认值 | 说明 |
|------|----------|--------|------|
| `softplus_beta` | 5.0 - 20.0 | 10.0 | 越大越接近 ReLU |
| `margin_mode` | fixed/adaptive | adaptive | 自适应更鲁棒 |
| `margin_fixed` | 0.3 - 1.0 | 0.5 | Fixed 模式值 |
| `margin_adaptive_ratio` | 0.3 - 0.7 | 0.5 | κ 越大难度越高 |
| `margin_min` | 0.05 - 0.2 | 0.1 | 最小 margin |
| `margin_max` | 1.5 - 3.0 | 2.0 | 最大 margin |

### 训练超参数

| 参数 | 推荐范围 | 默认值 | 说明 |
|------|----------|--------|------|
| `lr` | 1e-5 - 5e-4 | 1e-4 | Q-Former 学习率 |
| `batch_size` | 2 - 16 | 4 | 受 GPU 内存限制 |
| `n_queries` | 16 - 64 | 32 | LQ 数量 |
| `hidden_dim` | 768 - 1024 | 768 | Q-Former 维度 |
| `llm_hidden_dim` | 2048 - 5120 | 4096 | LLM 维度 (Mistral/LLaMA: 4096) |

### 调优策略

1. **先固定 Margin 验证**: `margin_mode=fixed`, `margin_fixed=0.5`
2. **观察 NLL Gain 分布**: 根据 μ_G, σ_G 调整 adaptive 参数
3. **Beta 调节陡度**: 初始 10.0，若梯度消失则降低到 5.0
4. **Margin 范围**: 根据 NLL 尺度调整 [margin_min, margin_max]

---

## 性能预估

### 计算成本 (单卡 A100 80GB)

**每步训练时间** (batch_size=4):
- Q-Former forward: ~10ms
- CondenseHead: ~2ms
- LLM Path A (with grad): ~150ms (7B model, FP16)
- LLM Path B (no grad): ~80ms
- Loss computation: ~5ms
- **总计**: ~250ms/step

**训练吞吐量**: ~15 samples/sec (单卡)

### 内存占用

**模型参数** (FP16):
- LLM (frozen): 14GB (7B model)
- Q-Former: ~200MB
- CondenseHead: ~12MB
- **总计**: ~14.3GB

**激活内存** (batch_size=4, seq_len=512):
- LLM 激活: ~8GB
- Q-Former 激活: ~1GB
- **总计**: ~9GB

**总 GPU 内存**: ~25GB (可在 A100 40GB 运行)

### 训练规模预估

**数据集大小**: 100K QA 对
**训练设置**: 10 epochs, batch_size=4
**总步数**: 100K / 4 * 10 = 250K steps
**训练时间**: 250K * 250ms ≈ 17.4 小时 (单卡 A100)

---

## 与其他 Task 的集成

### Task S 集成 (Fragment Ranking)

**后验作为伪标签**:
```python
# In Task C training
loss_dict_c = compute_condensing_loss(...)
posterior_q_psi_U = loss_dict_c['posterior_q_psi_U']  # [batch, |U|]

# Pass to Task S as pseudo-labels
loss_dict_s = compute_ranking_loss(
    ...,
    posterior_scores=posterior_q_psi_U,  # Use as soft labels
)
```

**课程学习**:
- 早期: 依赖 teacher scores (BM25/DPR)
- 中期: 混合 teacher + posterior
- 后期: 主要依赖 posterior (LLM 反馈)

### Task E 集成 (Entailment Tagging)

**后验作为重要性权重**:
```python
# Use posterior as importance weights for focal loss
importance_weights = posterior_q_psi_U  # [batch, |U|]

loss_e = compute_focal_loss(
    ...,
    importance_weights=importance_weights,
)
```

---

## 已知限制与未来工作

### 限制

1. **LLM 计算成本高**: 双路前向需要 2x LLM 计算
2. **内存占用大**: 需要大容量 GPU (≥40GB)
3. **依赖 LLM 质量**: 冻结 LLM 必须强大
4. **后验提取依赖注意力**: 不同 LLM 架构需适配

### 未来改进方向

1. **蒸馏优化**: 用小模型蒸馏大 LLM 的判别能力
2. **知识缓存**: 缓存常见 query-fragment 的 Z prefix
3. **多任务学习**: 联合训练 Task C + Task S + Task E
4. **在线 RL**: 从实际生成质量学习（非离线 NLL）

---

## 验收标准

### ✅ 已完成 (Phase 1)

- [x] 核心损失函数实现完整
- [x] CondenseHead 实现并测试通过
- [x] 训练流程框架搭建
- [x] 5 个单元测试全部通过
- [x] Dummy pipeline 端到端运行
- [x] 详细文档和集成指南

### 🔶 待验收 (Phase 2 - LLM 集成后)

- [ ] 实际 LLM forward 通过
- [ ] 真实 QA 数据集训练成功
- [ ] NLL Gain 趋势正确（正值增加）
- [ ] 后验提取数值合理
- [ ] 与 Task S 集成测试

---

## 结论

Task C (Condensing Generation) 的核心框架已完整实现，基于 v8.0 设计规格：

1. ✅ **核心损失**: Contrastive NLL + Softplus + Adaptive Margin
2. ✅ **后验提取**: 从 LLM 注意力自动获取片段重要性
3. ✅ **训练流程**: KnowledgeCondenser 类 + 完整训练循环
4. ✅ **模块完整**: CondenseHead + FrozenLLM 接口
5. ✅ **测试验证**: 5/5 单元测试通过

**下一步**: 集成实际 LLM (LLaMA/Mistral/Phi) 并在真实 QA 数据上验证。

所有占位符已清晰标注，集成路线图完整，可交付！🚀
