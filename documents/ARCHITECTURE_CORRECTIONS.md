# 架构修正总结 (Architecture Correction Summary)

本次更新根据中文需求文档系统地修正了 DR-QFormer 的架构说明和代码注释。

## 主要修正内容 (Major Corrections)

### 1. 核心架构澄清 (Core Architecture Clarification)

#### 原设计 → 修正后设计
- ❌ 原：泛泛的 "BLIP-2-style Q-Former"
- ✅ 改：**交叉注意力方案 (Cross-Attention Architecture)**，明确三个阶段：
  1. **SA Stage**: LQs与q_embed/a_embed融合 → LQs_aware
  2. **CA Stage**: LQs_aware attend到P_embeds → Z
  3. **FFN Stage**: 最终处理 → Z_final

#### 在线、查询相关 (Online, Query-Relevant)
- ✅ 强调Q-Former接收**在线**查询嵌入，而非预计算
- ✅ 明确q_embed和a_embed是**单个嵌入**（长度1），非完整token序列

### 2. 片段级操作 (Fragment-Level Operations)

#### 所有任务统一操作对象
- ✅ **文本片段/chunks** (fragments)，非完整文档
- ✅ k个片段嵌入 P_embeds [batch, k, d]

### 3. 三任务详细设计 (Three Tasks Detailed Design)

#### Task E: 蕴含-标注 (Entailment Tagging)
```
目的: 片段级过滤器
输出: k个logits (对应k个片段)
来源: CA层注意力得分（max-pooled）
监督: gt_k [batch, k] 二元向量
损失: BCE vs gt_k
```

#### Task S: 排序-监督 (Sorting Supervision)
```
目的: 训练CA层注意力权重分配
输出: k个注意力权重 A_weights (CA层softmax输出)
监督: gt_soft_weights [batch, k] 概率分布
损失: KL Divergence vs gt_soft_weights
```

#### Task C: 精炼-生成 (Condensing-Generation)
```
目的: 学习浓缩，确保Z对LLM有用
输出: N个Z向量 → LLM前缀
损失(Primal): 对比生成 - max(Reward(LLM(Q,Z)) - Reward(LLM(Q,Empty)))
损失(Dual): 奖励损失 - max(Sim(LLM(A,Z)→Q', Q_gold))
```

### 4. 对偶约束 (Dual Constraint)

#### 隐式实现 (Implicit via Parameter Sharing)
- ✅ 同一Q-Former参数被Primal和Dual任务梯度共同更新
- ✅ 区别于Tang et al.的**显式概率一致性损失**
- ✅ 强制学习双向逻辑关系 Query ↔ Pool ↔ Answer

#### 训练模式
```python
# Primal (QA)
E_qa: Query → 预测答案相关片段
S_qa: Query → 学习片段排序
C_qa: Query + Z → 生成答案

# Dual (QG)
E_qg: Answer → 预测查询相关片段
S_qg: Answer → 学习片段排序
C_qg: Answer + Z → 生成查询
```

### 5. 注意力掩码 (Attention Masking)

#### SA掩码 (Self-Attention Mask)
```
形状: (N+1) x (N+1)
内容: 全1矩阵（或无掩码）
语义: 完全双向注意力
```

#### CA掩码 (Cross-Attention Mask)
```
形状: N x k
内容: 全1矩阵（或无掩码）
语义: 每个LQ可attend所有k个片段
```

#### Padding掩码 (必需)
```
形状: [batch, k]
作用: 处理变长P_embeds，忽略填充
集成: 由Transformer库自动处理
```

#### 掩码哲学
- ✅ **允许性掩码** (Permissive Masks): 默认全连接
- ✅ **通过损失学习**: 任务E/S/C训练注意力机制过滤和排序
- ✅ **无硬约束**: 不像因果LM，Q-Former是双向的

## 已更新文件 (Updated Files)

### 文档 (Documentation)
- ✅ `README.md`
  - 新增"动机"部分（传统RAG和现有可微RAG的问题）
  - 重写"架构"部分（三阶段交叉注意力流程）
  - 重写"训练任务"部分（Primal/Dual详细说明）
  - 新增"隐式对偶约束"部分
  - 重写"注意力机制与掩码"部分

### 核心代码注释 (Core Code Comments)
- ✅ `dr_qformer/models/qformer.py`
  - DRQFormer类：三阶段架构详细注释
  - forward方法：Primal/Dual模式说明
  - QFormerLayer类：SA/CA/FFN实现指导

- ✅ `dr_qformer/models/heads.py`
  - EntailmentHead：片段级蕴含标注，CA注意力得分
  - SortingHead：片段级排序，注意力权重输出
  - CondenseHead：精炼生成，LLM前缀嵌入

- ✅ `dr_qformer/losses.py`
  - entailment_loss：BCE，gt_k监督，类别不平衡处理
  - sorting_loss：KL散度，gt_soft_weights监督
  - reward_margin_loss：对比生成，奖励差距最大化

- ✅ `dr_qformer/utils/masks.py`
  - build_sa_mask：(N+1)x(N+1)全1掩码说明
  - build_ca_mask：Nxk全1掩码说明
  - build_padding_mask：变长P_embeds处理（必需）

## 待完成工作 (Remaining Tasks)

### 高优先级 (High Priority)
- [ ] 更新训练脚本注释 (train/task_e.py, task_s.py, task_c.py)
  - 添加Primal/Dual模式详细说明
  - 片段级操作流程
  - 对偶约束实现细节

- [ ] 更新架构图 (assets/diagram.md)
  - 反映三阶段交叉注意力流程
  - 显示SA/CA输入输出
  - 标注片段级操作

- [ ] 更新数据格式文档 (DATA_FORMAT.md)
  - gt_k字段：二元标签 [batch, k]
  - gt_soft_weights字段：概率分布 [batch, k]
  - 生成方法说明

- [ ] 更新配置文件注释 (configs/*.yaml)
  - 添加task_mode: primal/dual
  - 片段级参数说明

### 中优先级 (Medium Priority)
- [ ] 更新QUICKSTART.md
  - 反映Primal/Dual训练模式
  - 添加片段级数据准备示例

- [ ] 更新CONTRIBUTING.md
  - 架构设计原则
  - 对偶约束实现指南

## 关键设计原则 (Key Design Principles)

1. **参数效率** (Parameter Efficiency)
   - 仅训练Q-Former (~40-80M) + 任务头 (~1-10M)
   - 冻结检索器 (~100-400M) 和 LLM (~1-10B)
   - 总可训练参数 ~1-2%

2. **在线查询相关** (Online Query-Relevant)
   - Q-Former在推理时接收查询嵌入
   - SA阶段融合LQs与query/answer context
   - 非预计算，支持任意查询

3. **片段级操作** (Fragment-Level Operations)
   - 所有任务操作文本片段，非完整文档
   - k个片段嵌入作为CA的Key/Value

4. **隐式对偶约束** (Implicit Dual Constraint)
   - 参数共享实现双向训练
   - 无需显式一致性损失
   - 更鲁棒的双向逻辑学习

5. **允许性掩码哲学** (Permissive Masking Philosophy)
   - 默认全连接（全1掩码）
   - 通过任务损失学习过滤/排序
   - 仅padding掩码为必需

## 验证清单 (Validation Checklist)

- [x] 架构描述与中文需求一致
- [x] 三任务设计详细准确
- [x] 对偶约束机制明确
- [x] 注意力掩码规格正确
- [x] 代码注释完整详细
- [x] **核心Q-Former实现完成** ✅
- [x] **功能测试全部通过** ✅
- [x] **详细实现文档完成** ✅
- [ ] 训练脚本注释待补充
- [ ] 架构图待更新
- [ ] 数据格式文档待补充

## 最新实现状态 (Latest Implementation Status)

### ✅ v0.2.1 - 注意力权重导出 (2025-11-02)

#### 新增功能 (New Features)
1. **注意力权重导出** (`dr_qformer/models/qformer.py`)
   - ✅ SA权重导出: `[batch, num_heads, N+1, N+1]` 每层
   - ✅ CA权重导出: `[batch, num_heads, N, k]` 每层
   - ✅ 设置 `need_weights=True, average_attn_weights=False`
   - ✅ 保留每个注意力头的独立权重
   
2. **注意力分析工具**
   - ✅ `test_attention_weights.py` - 验证权重导出功能
   - ✅ `analyze_attention.py` - 详细注意力模式分析
   - ✅ 支持分析"哪几个LQ关注了哪几段"
   - ✅ 支持Primal/Dual模式对比

3. **分析能力**
   - ✅ LQs对query/answer embedding的注意力
   - ✅ 片段接收注意力统计
   - ✅ 注意力选择性和多样性 (熵)
   - ✅ 每个头的独立注意力模式
   - ✅ LQ到片段的注意力映射

### ✅ v0.2.0 - 核心Q-Former实现 (2025-11-01)

#### 已完成 (Completed)
1. **DRQFormer主类** (`dr_qformer/models/qformer.py`)
   - ✅ 56.7M可训练参数
   - ✅ 可学习查询tokens (LQs): 32个参数化向量
   - ✅ 6层Transformer堆叠 (可配置)
   - ✅ 温度参数 (为Task E预留)

2. **QFormerLayer实现**
   - ✅ Stage 1: Self-Attention (SA) - LQs融合query/answer
   - ✅ Stage 2: Cross-Attention (CA) - LQs关注片段
   - ✅ Stage 3: Feed-Forward Network (FFN)
   - ✅ Pre-LayerNorm + 残差连接

3. **训练模式**
   - ✅ Primal Mode (QA): `query_embeds` → `z_qa`
   - ✅ Dual Mode (QG): `answer_embeds` → `z_qg`
   - ✅ 隐式对偶约束 (参数共享)

4. **测试与文档**
   - ✅ 功能测试 (`simple_test_qformer.py`) - 5/5通过
   - ✅ 架构可视化 (`visualize_drqformer.py`)
   - ✅ 实现指南 (`DR_QFORMER_IMPLEMENTATION.md`)
   - ✅ 实现总结 (`IMPLEMENTATION_SUMMARY.md`)

#### 测试结果
```
✅ All tests passed!
📊 DR-QFormer Architecture Summary:
   - Parameters: 56,736,769
   - Memory: ~216 MB (FP32)
   - Tests: 5/5 passing

✅ Attention Weight Export:
   - SA weights: [batch, num_heads, N+1, N+1] per layer
   - CA weights: [batch, num_heads, N, k] per layer
   - Per-head weights preserved (not averaged)
   - Can analyze LQ-to-fragment attention mappings
```

#### 下一步 (Next Steps)
1. 实现任务头 (EntailmentHead, SortingHead, CondenseHead)
2. 集成冻结retriever (Contriever, DPR, E5, BGE)
3. 集成冻结LLM (LLaMA, Mistral, Phi)
4. 实现训练任务 (E, S, C) 的损失函数
5. 数据准备工具 (gt_k, soft_targets生成)

---

**更新时间**: 2025-11-01
**负责人**: AI助手根据用户中文需求文档 + BLIP-2参考实现
**状态**: ✅ 核心Q-Former实现完成并测试通过，任务头和训练循环待开发
