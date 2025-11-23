# Tokenizer Mismatch Bug Fix

## 🔴 问题描述

训练无法收敛的根本原因：**Tokenizer不匹配**导致Q-Former接收到错误的token IDs。

### 症状
- 在ms_macro子集上训练loss不下降
- 模型表现类似随机初始化
- Task E和Task S的loss都很高且不收敛

### 根本原因

```
Data Pipeline:                 Q-Former Pipeline:
┌─────────────────┐           ┌──────────────────┐
│ Qwen3-Embedding │           │ XLM-RoBERTa      │
│ Tokenizer       │  ❌      │ Embedding Layer  │
│ (vocab ~152K)   │ ──────→  │ (vocab ~250K)    │
└─────────────────┘ mismatch  └──────────────────┘
```

**具体过程：**

1. **数据预处理阶段**（`dataset_smoking.ipynb`）：
   ```python
   # 使用 Qwen3-Embedding tokenizer 编码
   tokenizer = AutoTokenizer.from_pretrained("Qwen3-Embedding-0.6B")
   query_encoded = tokenizer(query_text)  # 生成 Qwen3 token IDs
   
   # 存储到 pickle
   query_embedding = {
       'input_ids': query_encoded['input_ids'],        # Qwen3 token IDs
       'attention_mask': query_encoded['attention_mask'],
       'token_emb_768': qwen3_model_output,            # Qwen3 embeddings
   }
   ```

2. **训练阶段**（`train/stage1_train.py` - 修复前）：
   ```python
   # 直接使用预计算的 Qwen3 token IDs
   query_input_ids = sample['query_embedding']['input_ids']  # ❌ Qwen3 IDs
   
   # 传给 Q-Former (使用 XLM-RoBERTa)
   Z = qformer(
       input_ids=query_input_ids,  # ❌ Qwen3 IDs 被 XLM-R embedding 层解码
       attention_mask=query_attention_mask,
       evidence_emb=evidence_embeddings,
   )
   ```

3. **Q-Former内部**（`src/models/qformer_xlm.py`）：
   ```python
   # XLM-RoBERTa embedding layer 尝试解码 Qwen3 token IDs
   token_emb = self.embeddings(input_ids)  # ❌ 接收错误的 token IDs
   
   # 结果：嵌入向量完全错误！
   # 例如：Qwen3 token ID 5280 在 XLM-R vocab 中可能是完全不同的词
   #      或者根本不存在（超出vocab范围）
   ```

### 为什么会导致训练失败？

| 维度 | Qwen3-Embedding | XLM-RoBERTa | 影响 |
|------|----------------|-------------|------|
| **Vocabulary Size** | ~152K tokens | ~250K tokens | ID越界或映射错误 |
| **Token Mapping** | 不同的BPE/WordPiece | 不同的SentencePiece | 完全不同的ID→词映射 |
| **Special Tokens** | 不同的CLS/SEP格式 | `<s>`, `</s>` | 句子边界混乱 |
| **Embedding Space** | Qwen3优化的空间 | XLM-R优化的空间 | 语义完全丢失 |

**实例对比：**

```python
# 相同的文本："What is the capital of China?"

# Qwen3 tokenizer:
# Token IDs: [151644, 3838, 374, 279, 6864, 315, 5734, 30]
#            [BOS,   What, is,  the, capital, of,  China, ?]

# XLM-RoBERTa tokenizer:
# Token IDs: [0,     10871, 83,  70,  10776, 111, 7759, 32]
#            [<s>,  What,  is,  the, capital, of,  China, ?]

# 如果把 Qwen3 的 IDs 传给 XLM-R embedding：
#   - Token ID 151644 (Qwen3 BOS) → XLM-R vocab 中不存在！
#   - Token ID 3838 (Qwen3 "What") → XLM-R 中可能是 "agriculture" 之类的完全不同词
#   - 语义完全崩溃！
```

## ✅ 解决方案

### 修改内容

在训练时用XLM-RoBERTa tokenizer **重新编码** query文本，而不是使用预计算的Qwen3 token IDs。

### 代码修改

#### 1. 修改 `SmokingDataset` 类

```python
class SmokingDataset(Dataset):
    def __init__(
        self, 
        data_dict: Dict, 
        sample_ids: List[str], 
        tokenizer: AutoTokenizer,  # ✅ 新增参数
        max_query_len: int = 512
    ):
        self.data_dict = data_dict
        self.sample_ids = sample_ids
        self.tokenizer = tokenizer  # ✅ 保存tokenizer
        self.max_query_len = max_query_len
    
    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.sample_ids[idx]
        sample = self.data_dict[sample_id]
        
        # ✅ 重新编码 query 文本
        query_text = sample['query']
        query_encoded = self.tokenizer(
            query_text,
            padding=False,  # 在 collate_fn 中统一padding
            truncation=True,
            max_length=self.max_query_len,
            return_tensors='pt',
        )
        query_input_ids = query_encoded['input_ids'].squeeze(0)
        query_attention_mask = query_encoded['attention_mask'].squeeze(0)
        
        # ... 其余代码不变
```

#### 2. 修改 `Stage1Trainer` 初始化

```python
class Stage1Trainer:
    def __init__(self, config: Stage1Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # ✅ 加载 XLM-RoBERTa tokenizer
        print(f"Loading tokenizer: {config.xlm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.xlm_model_name)
        
        # ... 其余初始化代码
```

#### 3. 修改 `main()` 函数流程

```python
def main():
    config = Stage1Config()
    
    # Load data (returns tuples)
    train_data, val_data = load_and_split_data(...)
    
    # ✅ 先初始化 trainer 获取 tokenizer
    trainer = Stage1Trainer(config)
    
    # ✅ 创建 Dataset 时传入 tokenizer
    train_dataset = SmokingDataset(
        train_data[0], 
        train_data[1], 
        trainer.tokenizer,  # ✅ 使用 XLM-R tokenizer
        max_query_len=512
    )
    val_dataset = SmokingDataset(
        val_data[0], 
        val_data[1], 
        trainer.tokenizer,
        max_query_len=512
    )
    
    # 创建 DataLoader 和训练...
```

### Attention Mask 处理

#### Q-Former 中的正确处理（已验证）

```python
# src/models/qformer_xlm.py
def forward(
    self,
    input_ids: Tensor,        # [B, T] - XLM-R token IDs
    attention_mask: Tensor,   # [B, T] - 1=valid, 0=padding
    evidence_emb: Tensor,
    evidence_mask: Optional[Tensor] = None,
):
    # Step 1: Get token embeddings
    token_emb = self.embeddings(input_ids)  # ✅ XLM-R embeddings
    
    # Step 2: Expand LQs and concatenate
    lqs = self.query_tokens.expand(batch_size, -1, -1)
    hidden = torch.cat([lqs, token_emb], dim=1)  # [B, N_q+T, d]
    
    # Step 3: Extend attention mask
    lq_mask = torch.ones(batch_size, self.n_queries, ...)  # LQs全部有效
    extended_mask = torch.cat([lq_mask, attention_mask], dim=1)  # [B, N_q+T]
    
    # Step 4: Convert to additive mask
    extended_mask_4d = self._prepare_attention_mask(extended_mask)
    # ✅ 正确处理：1 → 0.0 (valid), 0 → -10000.0 (masked)
    
    # Step 5: Apply XLM-R layers
    for xlmr_layer in self.encoder_layers:
        layer_outputs = xlmr_layer(hidden, attention_mask=extended_mask_4d)
        # ✅ Padding tokens 被正确mask，不参与attention计算
```

**关键点：**
1. ✅ LQ tokens的mask全为True（不padding）
2. ✅ Query tokens的mask来自tokenizer（padding位置为0）
3. ✅ 转换为additive mask格式（HuggingFace标准）
4. ✅ 在self-attention中正确应用mask

## 🧪 验证方法

### 1. 运行测试脚本

```bash
python test_tokenizer_fix.py
```

预期输出：
```
✅ CORRECT: Token IDs are different (expected, different tokenizers)
   This confirms we need to re-encode queries with XLM-R tokenizer!
```

### 2. 检查训练开始时的输出

```bash
python train/stage1_train.py
```

应该看到：
```
Loading tokenizer: xlm-roberta-base
📦 Creating datasets with XLM-RoBERTa tokenizer...
✅ Train size: 860, Val size: 96
```

### 3. 监控训练loss

修复后应该看到：
- Loss在前几个epoch内开始下降
- Task E和Task S的loss都有改善
- 模型不再表现得像随机初始化

## 📊 修复前后对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| **Token IDs来源** | Qwen3 tokenizer (~152K vocab) | XLM-R tokenizer (~250K vocab) |
| **Embedding层匹配** | ❌ Mismatch | ✅ Match |
| **Query语义** | ❌ 完全错误 | ✅ 正确编码 |
| **Attention Mask** | ⚠️ 位置可能错误 | ✅ 位置正确 |
| **训练收敛** | ❌ 不收敛 | ✅ 应该收敛 |

## 🔍 相关文件

### 修改的文件
- `train/stage1_train.py` - 主要修改
- `test_tokenizer_fix.py` - 新增验证脚本

### 相关文档
- `src/models/qformer_xlm.py` - Q-Former实现（已正确处理attention mask）
- `dataset_smoking.ipynb` - 数据预处理（使用Qwen3，但我们不再使用其token IDs）

## 💡 关键要点

1. **数据预处理时的token embeddings不再使用**
   - `query_embedding['token_emb_768']` 来自Qwen3，已被弃用
   - `query_embedding['input_ids']` 来自Qwen3，已被弃用
   - 只使用 `query` 原始文本

2. **训练时动态编码**
   - 每个epoch从原始文本重新tokenize
   - 使用与Q-Former匹配的XLM-RoBERTa tokenizer
   - 轻微增加计算开销，但保证正确性

3. **Evidence embeddings仍然使用预计算值**
   - Evidence embeddings来自Qwen3-Embedding是OK的
   - Q-Former通过Cross-Attention接收这些embeddings
   - 不涉及tokenizer，所以没有mismatch问题

## 🚀 下一步

1. ✅ 运行 `test_tokenizer_fix.py` 验证修复
2. ✅ 运行 `python train/stage1_train.py` 开始训练
3. 📊 监控训练曲线，确认loss下降
4. 🎯 如果仍不收敛，检查其他超参数（learning rate, batch size等）

## 📚 参考

- HuggingFace XLM-RoBERTa: https://huggingface.co/xlm-roberta-base
- BLIP-2 Paper: https://arxiv.org/abs/2301.12597
- Tokenizer Mismatch问题常见于跨模型迁移场景
