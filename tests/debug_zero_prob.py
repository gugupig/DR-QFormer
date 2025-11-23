"""
诊断 reranker 产生 0.0000 概率的原因
"""
import torch
import torch.nn.functional as F
import numpy as np

# 模拟场景：K 个 evidence，log-odds 范围很大
def simulate_reranker_scenario(K=128, max_logit=10.0, min_logit=-15.0):
    """
    模拟 reranker 在极端情况下的行为
    
    Args:
        K: evidence 数量
        max_logit: 最高 log-odds（相关证据）
        min_logit: 最低 log-odds（无关证据）
    """
    print(f"\n{'='*60}")
    print(f"模拟场景：K={K}, max_logit={max_logit}, min_logit={min_logit}")
    print(f"{'='*60}\n")
    
    # 生成模拟的 log-odds 分布
    # 假设：1 个高分证据，其余都是低分
    raw_scores = torch.tensor([max_logit] + [min_logit] * (K - 1), dtype=torch.float32)
    
    print(f"Raw log-odds 范围: [{raw_scores.min():.2f}, {raw_scores.max():.2f}]")
    print(f"Raw log-odds 前5个: {raw_scores[:5].tolist()}")
    
    # 方案1: 直接 softmax（当前实现）
    noe_margin = 1.0
    noe_score = raw_scores.min() - noe_margin
    all_scores = torch.cat([raw_scores, noe_score.unsqueeze(0)], dim=0)
    
    print(f"\nNoE log-odds: {noe_score.item():.2f}")
    print(f"All scores 范围: [{all_scores.min():.2f}, {all_scores.max():.2f}]")
    
    # Softmax 计算
    probs = F.softmax(all_scores, dim=0)
    
    print(f"\n概率分布统计:")
    print(f"  最大概率: {probs.max().item():.6e}")
    print(f"  最小概率: {probs.min().item():.6e}")
    print(f"  NoE 概率: {probs[-1].item():.6e}")
    print(f"  概率和: {probs.sum().item():.6f}")
    
    # 检查有多少概率 < 1e-4 (显示为 0.0000)
    zero_count = (probs < 1e-4).sum().item()
    print(f"\n概率 < 1e-4 的数量: {zero_count}/{len(probs)}")
    
    # 显示前10个和后10个概率
    print(f"\n前10个概率:")
    for i in range(min(10, len(probs))):
        print(f"  [{i}] {probs[i].item():.6e}")
    
    print(f"\n后10个概率:")
    for i in range(max(0, len(probs)-10), len(probs)):
        print(f"  [{i}] {probs[i].item():.6e}")
    
    return probs


# 测试不同场景
print("\n" + "="*60)
print("诊断：Softmax 在极端 log-odds 下的行为")
print("="*60)

# 场景1: 中等范围（正常情况）
probs1 = simulate_reranker_scenario(K=10, max_logit=5.0, min_logit=-5.0)

# 场景2: 大范围（可能导致数值下溢）
probs2 = simulate_reranker_scenario(K=128, max_logit=10.0, min_logit=-15.0)

# 场景3: 极端范围（必然导致数值下溢）
probs3 = simulate_reranker_scenario(K=128, max_logit=15.0, min_logit=-20.0)

# 解释原因
print("\n" + "="*60)
print("结论分析")
print("="*60)
print("""
出现 0.0000 的原因：

1. **Softmax 的数值特性**:
   当 log-odds 范围很大时（例如 max=10, min=-15），softmax 会将概率质量
   集中在最高分项上，导致低分项的概率趋近于 0。

2. **计算公式**:
   P(i) = exp(logit_i) / sum(exp(logit_j))
   
   当 logit_i = -15, logit_max = 10 时：
   exp(-15) / exp(10) ≈ 3e-11 (远小于 float32 精度)

3. **是否真的是 0？**
   不是！只是 < 1e-4，显示为 0.0000。
   实际值在 1e-6 到 1e-10 之间。

4. **对训练的影响**:
   - KL 散度: log(0) → -inf，会导致梯度爆炸
   - 解决方案: 使用 log_softmax + nll_loss，或添加 epsilon

5. **是否需要修复？**
   取决于：
   - 如果用于 display：没问题，只是显示精度
   - 如果用于 KL loss：需要用 log-space 计算
   - 如果希望避免：可以对 logits 做 temperature scaling
""")

# 测试 temperature scaling 效果
print("\n" + "="*60)
print("解决方案：Temperature Scaling")
print("="*60)

def test_temperature_scaling(raw_scores, temperature=1.0):
    """测试温度缩放对概率分布的影响"""
    scaled_scores = raw_scores / temperature
    probs = F.softmax(scaled_scores, dim=0)
    return probs

# 使用场景2的数据
raw_scores = torch.tensor([10.0] + [-15.0] * 127, dtype=torch.float32)
noe_score = raw_scores.min() - 1.0
all_scores = torch.cat([raw_scores, noe_score.unsqueeze(0)], dim=0)

for temp in [1.0, 2.0, 5.0, 10.0]:
    probs = test_temperature_scaling(all_scores, temperature=temp)
    zero_count = (probs < 1e-4).sum().item()
    print(f"\nTemperature={temp:.1f}:")
    print(f"  Max prob: {probs.max().item():.6f}")
    print(f"  Min prob: {probs.min().item():.6e}")
    print(f"  概率 < 1e-4 的数量: {zero_count}/{len(probs)}")
    print(f"  熵: {-(probs * torch.log(probs + 1e-10)).sum().item():.4f} (max={np.log(len(probs)):.4f})")
