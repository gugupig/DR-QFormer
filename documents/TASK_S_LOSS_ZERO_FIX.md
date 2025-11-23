# Task S 训练问题诊断与修复

## 🔴 问题现象

训练 Task S 时，loss 直接降到 0，没有正常的学习过程。

## 🔍 根本原因

### 1. Teacher Score 分布过于平坦

从数据检查发现，`evidence_ranking` 中的分数范围非常窄：

```python
# 典型的 evidence_ranking 数据
[(0, 0.14652860164642334),
 (1, 0.14482149481773376),
 (2, 0.140915185213089),
 ...
 (9, 0.05518317595124245)]

# 分数范围：0.04 ~ 0.16
# 相对差异：最大值 / 最小值 ≈ 4x
```

**问题：**
- 所有分数都在 0.04-0.16 之间，相对差异很小
- Softmax 后的概率分布非常平坦（接近均匀分布）
- KL散度loss会非常小（因为目标分布本身就接近uniform）

### 2. Loss降到0的原因

```python
# 原始分数 softmax 后
teacher_probs ≈ [0.11, 0.11, 0.10, ..., 0.09]  # 接近均匀分布

# 学生模型只需要输出接近均匀的分布即可
student_probs ≈ [0.10, 0.10, 0.10, ..., 0.10]

# KL(teacher || student) ≈ 0  # 两个分布都很平坦
```

这不是真正的"学习到了ranking"，而是"teacher信号太弱"。

## ✅ 解决方案

### 方案1：标准化Teacher Scores（已实施）

对每个样本的teacher scores进行z-score标准化：

```python
# 在 RankingDataset.__getitem__ 中
if nonzero_mask.sum() > 1:
    nonzero_scores = teacher_scores[nonzero_mask]
    mean_score = nonzero_scores.mean()
    std_score = nonzero_scores.std()
    if std_score > 1e-6:
        # 标准化：(x - mean) / std
        teacher_scores[nonzero_mask] = (nonzero_scores - mean_score) / std_score
```

**效果：**
- 原始分数：[0.15, 0.14, 0.14, 0.13, ..., 0.05]
- 标准化后：[1.5, 0.8, 0.7, 0.3, ..., -1.2]
- Softmax后分布更加sharp，提供更强的learning signal

### 方案2：调整Temperature参数

降低 `teacher_tau` 来增强分布的sharpness：

```python
# In TaskSConfig
teacher_tau: float = 0.5  # 降低到 0.5（原来是1.0）

# Softmax with lower temperature
teacher_probs = F.softmax(teacher_scores / 0.5, dim=-1)
# 分布会更加peaked（高分更高，低分更低）
```

### 方案3：使用Rank-based Scoring

将连续分数转换为离散排名：

```python
# 基于排名的分数（更稳定）
sorted_indices = np.argsort(-teacher_scores)  # 降序排名
rank_scores = np.zeros_like(teacher_scores)
for rank, idx in enumerate(sorted_indices):
    rank_scores[idx] = K - rank  # 排名1st = K分，2nd = K-1分，...
```

## 📊 验证方法

运行诊断脚本：

```bash
python check_ranking_scores.py
```

检查项：
1. **Teacher分布熵**：应该 < 0.9 * max_entropy
2. **标准化后的分数范围**：应该有明显的正负值分离
3. **Loss值**：不应该在第一个epoch就接近0

## 🎯 预期改进

### 标准化前
```
Raw teacher scores: [0.15, 0.14, 0.14, ..., 0.05]
Softmax probs:      [0.11, 0.11, 0.10, ..., 0.09]
Entropy:            2.25 / 2.30 (98% of max) ⚠️ 太平坦
KL loss:            ~0.001 ❌
```

### 标准化后
```
Normalized scores:  [1.5, 0.8, 0.7, ..., -1.2]
Softmax probs:      [0.32, 0.16, 0.14, ..., 0.02]
Entropy:            1.80 / 2.30 (78% of max) ✅ 有区分度
KL loss:            ~0.1-0.5 ✅ 有学习信号
```

## 🔧 其他潜在问题

### 问题1：Data Format
确认 `evidence_ranking` 格式正确：
```python
# 应该是：List[(fragment_idx, score), ...]
evidence_ranking = [(0, 0.15), (1, 0.14), ..., (9, 0.05)]
```

代码已经正确处理了这个格式（lines 315-340）。

### 问题2：Model Output Range
`FragmentRankingHead` 使用dual-LSE聚合，输出范围可能很大。
确保loss计算时使用了temperature scaling。

### 问题3：Gradient Flow
Task S head 是无参数的（只做LSE聚合），梯度完全依赖Q-Former。
- ✅ 梯度能传回（通过LSE的微分）
- ⚠️ 但缺少task-specific learning capacity

建议：参考Task E的修复，为Task S也添加learnable projection head。

## 📝 修改记录

1. ✅ 修复KeyError：`'fragment_logits'` → `'ranking_logits'`
2. ✅ 添加teacher score标准化（z-score normalization）
3. ✅ 添加训练时的debug logging（每100步）
4. 📋 创建诊断脚本 `check_ranking_scores.py`

## 🚀 下一步

1. 运行诊断脚本验证修复效果
2. 重新训练Task S，观察loss曲线
3. 考虑为Task S添加learnable head（像Task E一样）
4. 如果问题持续，考虑使用rank-based scoring或调整temperature

---

## 参考

- ListNet论文：Learning to Rank with Soft Labels
- Temperature Scaling in Knowledge Distillation
- Ranking Loss Functions for Deep Learning
