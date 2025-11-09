# 注意力权重分析指南 (Attention Weight Analysis Guide)

## 概述 (Overview)

DR-QFormer现在支持详细的注意力权重导出，可以分析**哪几个LQ关注了哪几段**。所有注意力权重都按照每个头独立保存（`average_attn_weights=False`），便于深度分析。

## 快速开始 (Quick Start)

### 1. 基本使用

```python
from dr_qformer.models.qformer import DRQFormer
import torch

# 初始化模型
model = DRQFormer(
    n_queries=32,       # 32个可学习查询tokens
    hidden_dim=768,     # 隐藏维度
    num_layers=6,       # 6层Transformer
    num_heads=8,        # 8个注意力头
    max_fragments=10,   # 最多10个片段
    dropout=0.0
)

# 准备输入
query_embeds = torch.randn(2, 1, 768)    # [batch, 1, d]
p_embeds = torch.randn(2, 10, 768)       # [batch, k, d]

# Forward pass
z, aux = model(query_embeds=query_embeds, p_embeds=p_embeds)

# 获取注意力权重
sa_weights = aux['sa_attn_weights']  # List of [batch, num_heads, N+1, N+1]
ca_weights = aux['ca_attn_weights']  # List of [batch, num_heads, N, k]

print(f"导出了 {len(sa_weights)} 层SA权重")
print(f"导出了 {len(ca_weights)} 层CA权重")
```

### 2. 分析LQ-片段注意力

```python
# 获取最后一层的CA权重
last_layer_ca = ca_weights[-1]  # [batch, num_heads, N, k]

# 平均跨所有头
ca_avg = last_layer_ca.mean(dim=1)[0]  # [N, k]

# 找出每个LQ最关注的片段
for lq_idx in range(32):
    frag_attentions = ca_avg[lq_idx]  # [k]
    top_frag = frag_attentions.argmax().item()
    top_attn = frag_attentions[top_frag].item()
    print(f"LQ {lq_idx:2d} 最关注片段 {top_frag} (注意力: {top_attn:.4f})")

# 找出每个片段被哪个LQ最关注
for frag_idx in range(10):
    lq_attentions = ca_avg[:, frag_idx]  # [N]
    top_lq = lq_attentions.argmax().item()
    top_attn = lq_attentions[top_lq].item()
    print(f"片段 {frag_idx} 被 LQ {top_lq:2d} 最关注 (注意力: {top_attn:.4f})")
```

### 3. 分析每个头的特化

```python
# 分析不同头的注意力模式
last_layer_ca = ca_weights[-1]  # [batch, num_heads, N, k]

for head_idx in range(8):
    head_ca = last_layer_ca[0, head_idx]  # [N, k]
    
    # 找出这个头最关注的片段
    max_attention = head_ca.max().item()
    max_pos = head_ca.argmax()
    lq_idx = max_pos // 10
    frag_idx = max_pos % 10
    
    print(f"Head {head_idx}: LQ {lq_idx} → Fragment {frag_idx} (attn: {max_attention:.4f})")
```

## 数据结构 (Data Structures)

### Self-Attention (SA) 权重
- **形状**: `[batch, num_heads, N+1, N+1]`
- **维度说明**:
  - 前N个位置: 可学习查询tokens (LQs)
  - 第N+1个位置: query/answer embedding
- **用途**: 分析LQs如何融合query/answer信息

```python
sa_weights = aux['sa_attn_weights']  # List of tensors (one per layer)

# 分析LQs对query/answer的注意力
last_sa = sa_weights[-1][0].mean(dim=0)  # [N+1, N+1]
lqs_to_qa = last_sa[:32, 32]  # LQs attending to query/answer
print(f"LQs对query/answer的平均注意力: {lqs_to_qa.mean():.4f}")
```

### Cross-Attention (CA) 权重
- **形状**: `[batch, num_heads, N, k]`
- **维度说明**:
  - N: 32个可学习查询tokens
  - k: 片段数量（如10个检索片段）
- **用途**: 分析哪些LQs关注哪些片段

```python
ca_weights = aux['ca_attn_weights']  # List of tensors (one per layer)

# 分析CA权重
last_ca = ca_weights[-1][0]  # [num_heads, N, k]

# 每个头的注意力分布
for head in range(8):
    print(f"Head {head} 的注意力熵: {entropy(last_ca[head]):.4f}")
```

## 高级分析 (Advanced Analysis)

### 1. 注意力选择性 (Selectivity)

```python
def compute_selectivity(ca_weights):
    """计算每个LQ的注意力选择性（最大注意力值）"""
    ca_avg = ca_weights.mean(dim=1)[0]  # [N, k]
    selectivity = ca_avg.max(dim=1).values  # [N]
    return selectivity

selectivity = compute_selectivity(ca_weights[-1])
print(f"最选择性的LQ: {selectivity.argmax().item()}")
print(f"平均选择性: {selectivity.mean().item():.4f}")
```

### 2. 注意力多样性 (Diversity/Entropy)

```python
import torch

def compute_entropy(ca_weights):
    """计算每个LQ的注意力熵（分布多样性）"""
    ca_avg = ca_weights.mean(dim=1)[0]  # [N, k]
    entropy = -(ca_avg * torch.log(ca_avg + 1e-10)).sum(dim=1)  # [N]
    return entropy

entropy = compute_entropy(ca_weights[-1])
print(f"最多样的LQ: {entropy.argmax().item()}")
print(f"最集中的LQ: {entropy.argmin().item()}")
print(f"平均熵: {entropy.mean().item():.4f}")
```

### 3. 层间注意力演化

```python
def analyze_layer_evolution(ca_weights):
    """分析注意力如何在层间演化"""
    num_layers = len(ca_weights)
    
    for layer_idx in range(num_layers):
        ca = ca_weights[layer_idx][0].mean(dim=0)  # [N, k]
        max_attn = ca.max().item()
        mean_attn = ca.mean().item()
        print(f"Layer {layer_idx}: max={max_attn:.4f}, mean={mean_attn:.4f}")

analyze_layer_evolution(ca_weights)
```

### 4. Primal vs Dual 对比

```python
# Primal模式
z_primal, aux_primal = model(query_embeds=query_embeds, p_embeds=p_embeds)

# Dual模式
z_dual, aux_dual = model(answer_embeds=answer_embeds, p_embeds=p_embeds)

# 比较注意力模式
ca_primal = aux_primal['ca_attn_weights'][-1][0].mean(dim=0)  # [N, k]
ca_dual = aux_dual['ca_attn_weights'][-1][0].mean(dim=0)      # [N, k]

# 计算差异
diff = (ca_primal - ca_dual).abs()
print(f"Primal vs Dual 平均差异: {diff.mean().item():.4f}")
print(f"Primal vs Dual 最大差异: {diff.max().item():.4f}")

# 计算相关性
correlation = torch.corrcoef(torch.stack([
    ca_primal.flatten(), 
    ca_dual.flatten()
]))[0, 1]
print(f"Primal vs Dual 相关性: {correlation.item():.4f}")
```

## 保存和加载权重 (Save & Load)

### 保存注意力权重

```python
import numpy as np

# Forward pass
z, aux = model(query_embeds=query_embeds, p_embeds=p_embeds)

# 转换为numpy
sa_weights_np = [w.cpu().numpy() for w in aux['sa_attn_weights']]
ca_weights_np = [w.cpu().numpy() for w in aux['ca_attn_weights']]

# 保存
np.savez(
    'attention_weights.npz',
    sa_weights=sa_weights_np,
    ca_weights=ca_weights_np,
    z=z.cpu().numpy()
)
print("✓ 已保存注意力权重")
```

### 加载和分析

```python
# 加载
data = np.load('attention_weights.npz', allow_pickle=True)
sa_weights = data['sa_weights']
ca_weights = data['ca_weights']
z = data['z']

print(f"加载了 {len(sa_weights)} 层SA权重")
print(f"加载了 {len(ca_weights)} 层CA权重")

# 分析
last_ca = torch.from_numpy(ca_weights[-1])
print(f"最后一层CA权重形状: {last_ca.shape}")
```

## 可视化建议 (Visualization Tips)

### 1. 热图可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 获取最后一层CA权重并平均跨头
ca = ca_weights[-1][0].mean(dim=0).cpu().numpy()  # [N, k]

# 绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(ca, cmap='viridis', annot=False)
plt.xlabel('Fragment Index')
plt.ylabel('LQ Index')
plt.title('LQ-to-Fragment Attention (Last Layer)')
plt.savefig('attention_heatmap.png', dpi=300)
```

### 2. 注意力分布可视化

```python
# 每个LQ的注意力分布
for lq_idx in [0, 15, 31]:  # 显示3个LQ
    attentions = ca[lq_idx]
    plt.bar(range(len(attentions)), attentions)
    plt.title(f'LQ {lq_idx} Attention Distribution')
    plt.xlabel('Fragment Index')
    plt.ylabel('Attention Weight')
    plt.savefig(f'lq_{lq_idx}_attention.png', dpi=300)
    plt.close()
```

## 使用脚本 (Utility Scripts)

### 运行测试

```bash
# 测试注意力权重导出功能
python test_attention_weights.py

# 运行详细分析
python analyze_attention.py
```

### 输出示例

```
================================================================================
LQ-Fragment Attention Mapping (Last Layer)
================================================================================

Top fragment for each LQ (first 10 LQs):
  LQ  0: F3(0.143), F6(0.113), F5(0.112)
  LQ  1: F3(0.147), F6(0.110), F5(0.107)
  LQ  2: F3(0.143), F6(0.111), F5(0.109)
  ...

Top LQ for each fragment:
  Fragment  0: LQ7(0.095), LQ1(0.095), LQ28(0.095)
  Fragment  1: LQ22(0.095), LQ6(0.094), LQ0(0.094)
  Fragment  3: LQ27(0.150), LQ21(0.150), LQ20(0.149)
  ...
```

## 常见分析任务 (Common Analysis Tasks)

### ✅ 任务1: 找出专注特定片段的LQ

```python
target_fragment = 3
ca_avg = ca_weights[-1][0].mean(dim=0)  # [N, k]
lq_attentions = ca_avg[:, target_fragment]
top_lqs = lq_attentions.topk(5)

print(f"最关注片段{target_fragment}的前5个LQ:")
for idx, attn in zip(top_lqs.indices, top_lqs.values):
    print(f"  LQ {idx.item()}: {attn.item():.4f}")
```

### ✅ 任务2: 检测注意力坍缩 (Attention Collapse)

```python
ca_avg = ca_weights[-1][0].mean(dim=0)  # [N, k]
attention_std = ca_avg.std(dim=1)  # [N]

if attention_std.mean() < 0.01:
    print("⚠️  检测到注意力坍缩！")
else:
    print("✓ 注意力分布正常")
```

### ✅ 任务3: 分析头的特化程度

```python
ca = ca_weights[-1][0]  # [num_heads, N, k]

for head_idx in range(8):
    head_ca = ca[head_idx]  # [N, k]
    # 计算该头的注意力熵
    entropy = -(head_ca * torch.log(head_ca + 1e-10)).sum(dim=-1).mean()
    print(f"Head {head_idx} 熵: {entropy.item():.4f}")
```

## 总结 (Summary)

- ✅ **SA权重**: `[batch, num_heads, N+1, N+1]` - LQs与query/answer的融合
- ✅ **CA权重**: `[batch, num_heads, N, k]` - LQs对片段的注意力
- ✅ **每层独立保存**: 可追踪注意力演化
- ✅ **每头独立保存**: 可分析头特化
- ✅ **支持Primal/Dual**: 可对比两种模式

## 参考资料 (References)

- `test_attention_weights.py` - 功能验证
- `analyze_attention.py` - 详细分析示例
- `dr_qformer/models/qformer.py` - 核心实现
