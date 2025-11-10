# 接口修复总结

## 修复的问题

### 1. 损失函数命名不符 ✅

**问题**：
- JointTrainer 调用 `compute_entailment_loss` 和 `compute_contrastive_nll_with_posterior`
- src/losses.py 只提供 `compute_focal_loss` 和 `compute_condensing_loss`

**修复**：
- 在 `src/losses.py` 末尾添加函数别名：
  ```python
  # Alias for backward compatibility
  compute_entailment_loss = compute_focal_loss
  
  # Alias for Task C (matches joint trainer expectations)
  compute_contrastive_nll_with_posterior = compute_condensing_loss
  ```
- 更新 `train/task_joint.py` 导入，直接使用 `compute_focal_loss`
- 修复 `compute_task_e_loss` 调用：
  - 使用正确的参数名 `gt_labels` 而非 `labels`
  - 使用 `focal_gamma`/`focal_alpha` 而非 `gamma`/`alpha`
  - 手动计算 metrics（accuracy, precision, recall）

### 2. 统一 Drop-LQ 掩码缺失 ✅

**问题**：
- `src/utils/masks.py` 中 `generate_lq_drop_mask` 只是占位函数
- 联合训练无法生成统一的 Drop-LQ 掩码

**修复**：
- 在 `src/utils/masks.py` 中完整实现 `generate_lq_drop_mask`：
  ```python
  def generate_lq_drop_mask(
      batch_size: int,
      num_lqs: int,
      drop_rate: float = 0.1,
      device: Optional[object] = None,
  ) -> Optional[object]:
      """Generate unified Drop-LQ mask for all tasks (E/S/C)."""
      if device is None:
          device = torch.device("cpu")
      
      num_drop = int(num_lqs * drop_rate)
      
      if num_drop == 0:
          return torch.zeros(batch_size, num_lqs, dtype=torch.bool, device=device)
      
      lq_drop_mask = torch.zeros(batch_size, num_lqs, dtype=torch.bool, device=device)
      
      for b in range(batch_size):
          drop_indices = torch.randperm(num_lqs, device=device)[:num_drop]
          lq_drop_mask[b, drop_indices] = True
      
      return lq_drop_mask
  ```
- 包含完整文档说明统一掩码的重要性（闭环反馈一致性）

### 3. Q-Former 接口不一致 ✅

**问题**：
- `shared_forward` 假设：
  - `self.model.num_lqs` 属性
  - `fragment_embeds` 参数
  - `return_lm_proj=True` 参数
  - 返回 `(z_lm, aux)` 元组
- DRQFormer 实际：
  - `self.model.n_queries` 属性
  - `p_embeds` 参数
  - 无 `return_lm_proj` 参数
  - 返回 `(z, aux)` 元组

**修复**：
- 修正 `shared_forward` 中的模型调用：
  ```python
  # 使用正确属性名
  N = self.model.n_queries  # 而非 self.model.num_lqs
  
  # 使用正确参数名
  z, aux = self.model(
      query_embeds=cond_embeds if mode == "primal" else None,
      answer_embeds=cond_embeds if mode == "dual" else None,
      p_embeds=batch.fragment_embeds,  # 而非 fragment_embeds
      pool_padding_mask=batch.pool_padding_mask,
      lq_drop_mask=lq_drop_mask,
  )
  
  # 返回正确结构
  return {
      'z': z,  # DRQFormer 直接返回 z，不是 z_lm
      'z_final': aux['z_final'],
      'ca_raw_scores': aux.get('ca_raw_scores_per_head'),
      'ca_weights': aux.get('ca_attn_weights'),  # 而非 ca_weights_per_head
      'sa_weights': aux.get('sa_attn_weights'),  # 而非 sa_weights_per_head
      'lq_drop_mask': lq_drop_mask,
      'pool_padding_mask': batch.pool_padding_mask,
  }
  ```

### 4. 排序损失对接错误 ✅

**问题**：
- `compute_task_s_loss` 期待返回 `(loss, metrics)` 元组
- `compute_ranking_loss` 实际返回字典 `{'loss': ..., 'loss_teach': ..., 'loss_post': ..., 'loss_entropy': ...}`
- 参数名不匹配：
  - 期待：`temperature`, `alpha_gt`, `top_l_dynamic`, `top_lprime`
  - 实际：`tau_pred`, `tau_gt`, `alpha_gt`, `rho_top`, `l_prime`
- 缺少训练子集构建

**修复**：
- 修正 `compute_task_s_loss` 参数签名：
  ```python
  def compute_task_s_loss(
      self,
      forward_out: Dict[str, torch.Tensor],
      batch: JointBatch,
      posterior_scores: Optional[torch.Tensor] = None,
      lambda_teach: float = 1.0,
      lambda_post: float = 0.0,
      lambda_ent: float = 0.01,
      tau_pred: float = 1.0,      # 而非 temperature
      tau_gt: float = 1.0,        # 新增
      alpha_gt: float = 0.7,      # 保持
      rho_top: float = 0.02,      # 而非 top_l_dynamic
      l_prime: int = 16,          # 而非 top_lprime
  ) -> Tuple[torch.Tensor, Dict[str, float]]:
  ```
- 添加训练子集构建：
  ```python
  train_subset_mask = build_train_subset_mask(
      ranking_logits=ranking_logits,
      gt_scores=batch.ranking_scores,
      pool_padding_mask=batch.pool_padding_mask,
      rho_top=rho_top,
      l_prime=l_prime,
  )
  ```
- 正确处理返回值：
  ```python
  loss_dict = compute_ranking_loss(...)
  
  loss = loss_dict['loss']
  metrics = {
      'loss_teach': loss_dict['loss_teach'].item(),
      'loss_post': loss_dict['loss_post'].item() if isinstance(...) else ...,
      'loss_entropy': loss_dict['loss_entropy'].item(),
      'subset_size': train_subset_mask.sum().item() / batch_size,
  }
  
  return loss, metrics
  ```

### 5. 示例入口参数错误 ✅

**问题**：
- `__main__` 中使用 `DRQFormer(d=64, N=4, ...)` 等不存在的参数名

**修复**：
- 使用正确的参数名：
  ```python
  model = DRQFormer(
      n_queries=4,      # 而非 N
      hidden_dim=64,    # 而非 d
      num_layers=2,
      num_heads=2,
      max_fragments=10,
      dropout=0.1,
  )
  task_e_head = EntailmentHead(hidden_dim=64, ...)  # 而非 d
  task_s_head = FragmentRankingHead(hidden_dim=64)  # 而非 d
  ```

### 6. 导入语句修复 ✅

**问题**：
- 导入不存在的函数名

**修复**：
- 更新 `train/task_joint.py` 导入：
  ```python
  from ..losses import (
      compute_focal_loss,          # 直接使用，不需要别名
      compute_ranking_loss,
      build_train_subset_mask,     # 新增
      compute_condensing_loss,     # Task C
  )
  from ..utils.masks import generate_lq_drop_mask
  ```

## Task C 占位问题 ⚠️

**当前状态**：
- `compute_task_c_loss` 仍然是占位实现，返回常数 0
- 无法提取后验 `qψ_U` 反馈给 Task S
- 闭环训练无法实际运行

**待实现**：
1. **双路 Teacher Forcing**：
   - Path-A：`[Z, Q, A]` 全注意力前向，提取 LLM→Z 注意力
   - Path-B：`[Z_dummy, Q, A]` 屏蔽 Q/A→Z 注意力（基线）
2. **对比 NLL 损失**：
   - 计算 Gain G = NLL_B - NLL_A
   - 自适应 margin m = μ_G + κ·σ_G
   - Softplus 损失：L_C = softplus(β(m - G))
3. **后验提取**（子集 U）：
   - 从 LLM→Z 注意力回溯到片段：qψ_U = w_lq @ CA_weights_U
   - Detach 后传递给 Task S
4. **LLM 集成**：
   - 注册 attention hook 捕获 LLM→Z 权重
   - 处理多头平均
   - 正确的 token 位置索引（answer_start_idx）

**参考**：
- 详细实现说明在 `compute_condensing_loss` 函数文档中（`src/losses.py` lines 960-1110）
- 包含完整的公式、参数说明和设计决策

## 验证清单

### 已修复 ✅
- [x] 损失函数命名统一（别名 + 直接调用）
- [x] 统一 Drop-LQ 掩码实现
- [x] Q-Former 接口参数名对齐
- [x] Q-Former 返回值结构对齐
- [x] 排序损失参数名对齐
- [x] 排序损失返回值处理（字典→元组）
- [x] 训练子集掩码构建
- [x] 示例代码参数名修正
- [x] 导入语句更新
- [x] Task E metrics 计算

### 待实现 ⚠️
- [ ] Task C 双路 Teacher Forcing
- [ ] LLM attention hook 注册
- [ ] 后验 qψ_U 提取
- [ ] 完整闭环训练测试

## 使用建议

### 测试当前框架
```bash
# 测试联合训练器结构（Task C 仍为占位）
cd train
python task_joint.py

# 预期输出：
# - Trainer 成功初始化
# - Task E/S 损失正常计算
# - Task C 打印 PLACEHOLDER 警告，返回 0
# - 训练步完成，loss_total 包含 E + S（C = 0）
```

### 下一步实现顺序
1. **LLM 占位替换**（`scripts/train_joint.py`）：
   - 加载实际 LLM（LLaMA, Mistral, Phi）
   - 冻结参数
   - 注册 forward hook

2. **Task C 实现**（`train/task_joint.py`）：
   - 构造 inputs_embeds：[Z_lm, Q_tokens, A_tokens]
   - 实现双路前向
   - 调用 `compute_condensing_loss`
   - 提取 qψ_U（detached）

3. **完整测试**：
   - 准备小批量数据（2-4 样本）
   - 运行 1 个 epoch
   - 验证 loss 下降
   - 检查后验反馈到 Task S

## 相关文件

- `src/losses.py`：所有损失函数实现（已完整）
- `src/utils/masks.py`：掩码生成工具（已完整）
- `src/models/qformer.py`：Q-Former 模型（接口已确认）
- `train/task_joint.py`：联合训练器（E/S 完整，C 占位）
- `train/schedule.py`：动态权重调度（已完整）
- `train/joint_data.py`：数据加载（占位，需实际数据集）
- `scripts/train_joint.py`：训练入口（占位，需 LLM 集成）

## 性能预期

### 当前可运行范围
- ✅ Task E 单独训练（Focal Loss）
- ✅ Task S 单独训练（Teacher 监督）
- ✅ Task E + S 联合训练（无后验反馈）
- ⚠️ Task E + S + C 联合训练（C 为常数 0，无实际作用）

### 完整实现后
- ✅ 三任务真正联合训练
- ✅ 先验→后验动态过渡（warmup → bridge → closedloop）
- ✅ 后验 qψ_U 实时反馈到 Task S
- ✅ 闭环优化（LLM 注意力指导片段排序）

## 总结

所有**接口层面**的问题已修复：
- ✅ 函数命名统一
- ✅ 参数名对齐
- ✅ 返回值格式正确
- ✅ 模型属性访问正确
- ✅ 掩码生成完整

**核心逻辑**问题待解决：
- ⚠️ Task C 双路 Teacher Forcing 完全占位
- ⚠️ 无法提取真实后验（qψ_U = None）
- ⚠️ 闭环反馈链条不完整

现有代码可以运行 Task E + S 训练，但要实现完整的"先验-后验闭环"，必须实现 Task C 的 LLM 集成和后验提取。
