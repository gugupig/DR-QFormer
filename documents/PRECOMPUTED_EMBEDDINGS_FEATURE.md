# Pre-computed Token Embeddings Feature

## 概述

新增功能：直接使用PKL文件中预计算的token embeddings，完全绕过XLM-RoBERTa的tokenizer和embedding层。

## 🎯 动机

### 原问题：Tokenizer Mismatch

数据预处理时使用Qwen3-Embedding tokenizer，但Q-Former使用XLM-RoBERTa，导致：
- Token IDs不匹配
- Embedding空间不同
- 训练无法收敛

### 解决方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **方案1**: 重新用XLM-R tokenize | ✅ Token ID匹配<br>✅ Embedding一致 | ❌ 丢弃高质量Qwen3 embeddings<br>❌ 每个epoch重新tokenize（慢） |
| **方案2**: 直接用预计算embeddings | ✅ 使用高质量Qwen3 embeddings<br>✅ 无tokenization开销<br>✅ 避免mismatch | ❌ 跳过XLM-R embedding层（但可能更好） |

**我们同时支持两种方案**，通过`use_precomputed_embeddings`配置切换。

## 🔧 配置

### Stage1Config 新参数

```python
@dataclass
class Stage1Config:
    # Embedding options
    use_precomputed_embeddings: bool = True  # 是否使用PKL中的预计算embeddings
```

### 两种模式

#### Mode 1: 使用预计算embeddings (推荐)

```python
config = Stage1Config()
config.use_precomputed_embeddings = True  # 默认值
```

**工作流程：**
```
PKL File:
  query_embedding:
    - input_ids: [1, T] (Qwen3 token IDs, 不使用)
    - attention_mask: [1, T] (使用)
    - token_emb_768: [1, T, 768] (使用!) ✅

        ↓

SmokingDataset:
  - 提取 query_embedding['token_emb_768']
  - 提取 attention_mask
  
        ↓

Q-Former (bypass_embeddings=True):
  - 直接使用 token_emb_768 作为 token embeddings
  - 跳过 XLM-R embedding层
  - 继续正常的 self-attention 和 cross-attention
```

#### Mode 2: 使用XLM-R tokenizer (兼容模式)

```python
config = Stage1Config()
config.use_precomputed_embeddings = False
```

**工作流程：**
```
PKL File:
  query: str (原始文本)

        ↓

SmokingDataset + XLM-R Tokenizer:
  - tokenizer(query) → input_ids, attention_mask
  
        ↓

Q-Former (bypass_embeddings=False):
  - 使用 XLM-R embedding层
  - input_ids → embeddings
  - 继续正常的 self-attention 和 cross-attention
```

## 📊 数据格式要求

### PKL文件必须包含

对于`use_precomputed_embeddings=True`:

```python
sample = {
    'query': str,  # 原始query文本
    'query_embedding': {
        'input_ids': Tensor[1, T],         # Qwen3 token IDs (不使用，仅用于attention mask对齐)
        'attention_mask': Tensor[1, T],    # ✅ 使用：padding mask
        'token_emb_768': Tensor[1, T, 768] # ✅ 使用：预计算的token embeddings
    },
    'evidence_embeddings': ndarray[K, 768],  # Evidence fragment embeddings
    'evidence_labels': ndarray[K],
    'evidence_ranking': List[Tuple],
    # ... 其他字段
}
```

## 🔍 实现细节

### 1. SmokingDataset 修改

```python
class SmokingDataset:
    def __init__(
        self, 
        data_dict, 
        sample_ids, 
        tokenizer=None,  # 可选
        use_precomputed_embeddings=True  # 新参数
    ):
        self.use_precomputed_embeddings = use_precomputed_embeddings
        
    def __getitem__(self, idx):
        if self.use_precomputed_embeddings:
            # Mode 1: 提取预计算embeddings
            query_emb = sample['query_embedding']
            query_token_embeddings = query_emb['token_emb_768'].squeeze(0)
            attention_mask = query_emb['attention_mask'].squeeze(0)
            
            return {
                'query_token_embeddings': query_token_embeddings,  # [T, 768]
                'query_attention_mask': attention_mask,  # [T]
                # ...
            }
        else:
            # Mode 2: 重新tokenize
            query_encoded = self.tokenizer(sample['query'], ...)
            return {
                'query_input_ids': query_encoded['input_ids'],
                'query_attention_mask': query_encoded['attention_mask'],
                # ...
            }
```

### 2. Collate Function 修改

```python
def collate_stage1_batch(batch):
    use_precomputed = 'query_token_embeddings' in batch[0]
    
    if use_precomputed:
        # 创建 query_token_embeddings tensor
        query_token_embeddings = torch.zeros(B, max_T, 768)
        for b, sample in enumerate(batch):
            T = sample['query_token_embeddings'].shape[0]
            query_token_embeddings[b, :T] = sample['query_token_embeddings']
        
        return {
            'query_token_embeddings': query_token_embeddings,  # [B, T, 768]
            'query_attention_mask': query_attention_mask,
            # ...
        }
```

### 3. Q-Former 修改

```python
class XLMRobertaDRQFormer:
    def __init__(self, ..., bypass_embeddings=False):
        self.bypass_embeddings = bypass_embeddings
        self.embeddings = self.xlmr.embeddings  # XLM-R embedding层
        
    def forward(
        self,
        input_ids,
        attention_mask,
        evidence_emb,
        evidence_mask=None,
        precomputed_query_emb=None  # 新参数
    ):
        # Step 1: 获取token embeddings
        if self.bypass_embeddings and precomputed_query_emb is not None:
            # 使用预计算embeddings
            token_emb = precomputed_query_emb  # [B, T, 768]
        else:
            # 使用XLM-R embedding层
            token_emb = self.embeddings(input_ids)  # [B, T, 768]
        
        # Step 2-7: 正常的SA+CA流程（与原来相同）
        lqs = self.query_tokens.expand(B, -1, -1)
        hidden = torch.cat([lqs, token_emb], dim=1)
        # ... 继续SA+CA ...
```

### 4. Trainer 修改

```python
class Stage1Trainer:
    def __init__(self, config):
        # 根据配置决定是否加载tokenizer
        if config.use_precomputed_embeddings:
            self.tokenizer = None  # 不需要
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(...)
        
        self.qformer = XLMRobertaDRQFormer(
            ...,
            bypass_embeddings=config.use_precomputed_embeddings
        )
    
    def train_step(self, batch):
        # 提取预计算embeddings（如果有）
        query_token_emb = batch.get('query_token_embeddings', None)
        
        Z, all_aux = self.qformer(
            input_ids=batch['query_input_ids'],
            attention_mask=batch['query_attention_mask'],
            evidence_emb=batch['evidence_embeddings'],
            evidence_mask=batch['pool_padding_mask'],
            precomputed_query_emb=query_token_emb  # 传入预计算embeddings
        )
```

## 🧪 测试

### 运行测试脚本

```bash
python test_precomputed_embeddings.py
```

### 预期输出

```
Test 1: Q-Former with bypass_embeddings=True
✅ Output shape: [1, 32, 768]

Test 2: Q-Former with bypass_embeddings=False
✅ Output shape: [1, 32, 768]

Comparison:
  Z1 (bypass) mean: 0.012345, std: 0.543210
  Z2 (normal) mean: -0.023456, std: 0.654321
  Outputs are different: True (Expected)
```

## 💡 使用建议

### 推荐配置 (Mode 1)

```python
config = Stage1Config()
config.use_precomputed_embeddings = True  # 使用Qwen3预计算embeddings
```

**理由：**
1. ✅ **避免tokenizer mismatch** - 不依赖XLM-R tokenizer
2. ✅ **高质量embeddings** - Qwen3-Embedding是专门训练的embedding模型
3. ✅ **训练效率** - 无需每个epoch重新tokenize
4. ✅ **语义保持** - 保留数据预处理时的完整语义信息

### 何时使用Mode 2

```python
config.use_precomputed_embeddings = False
```

**适用场景：**
- 没有预计算的token embeddings
- 需要从头finetune XLM-R embedding层
- 对比实验：测试不同embedding的影响

## 📈 性能对比

### Embedding质量

| 模型 | Vocab Size | Languages | 训练数据 | Embedding质量 |
|------|-----------|-----------|---------|-------------|
| **Qwen3-Embedding-0.6B** | ~152K | 100+ | 大规模retrieval数据 | ⭐⭐⭐⭐⭐ 专门优化 |
| **XLM-RoBERTa-base** | ~250K | 100 | MLM预训练数据 | ⭐⭐⭐⭐ 通用embeddings |

### 训练速度

| 模式 | Tokenization | Embedding Layer | 相对速度 |
|------|-------------|----------------|---------|
| Mode 1 (预计算) | ❌ 跳过 | ❌ 跳过 | **100%** (baseline) |
| Mode 2 (XLM-R) | ✅ 每个epoch | ✅ 每个batch | ~95% (轻微慢5%) |

## 🔧 故障排除

### 问题1: KeyError: 'query_token_embeddings'

**原因**: PKL文件没有`query_embedding['token_emb_768']`字段

**解决**: 
1. 检查PKL格式
2. 或使用`use_precomputed_embeddings=False`

### 问题2: Shape mismatch

**原因**: token_emb_768的维度不是768

**解决**: 
- 确保预计算时使用了`truncate_dim=768`

### 问题3: 训练仍不收敛

**检查项**:
1. ✅ Evidence embeddings质量
2. ✅ Attention mask正确性
3. ✅ Learning rate设置
4. ✅ Loss权重平衡

## 📚 相关文档

- `documents/TOKENIZER_MISMATCH_FIX.md` - Tokenizer mismatch问题详解
- `dataset_smoking.ipynb` - 数据预处理（生成预计算embeddings）
- `src/models/qformer_xlm.py` - Q-Former实现
- `train/stage1_train.py` - 训练脚本

## 🎯 总结

| 特性 | Mode 1 (预计算) | Mode 2 (XLM-R) |
|------|----------------|---------------|
| **避免tokenizer mismatch** | ✅ 是 | ⚠️ 需重新tokenize |
| **Embedding质量** | ⭐⭐⭐⭐⭐ Qwen3 | ⭐⭐⭐⭐ XLM-R |
| **训练速度** | ✅ 更快 | ⚠️ 稍慢 |
| **内存占用** | ✅ 更低 | ⚠️ 稍高 |
| **灵活性** | ⚠️ 依赖预计算 | ✅ 端到端trainable |

**推荐**: 使用**Mode 1 (use_precomputed_embeddings=True)**，享受Qwen3高质量embeddings！
