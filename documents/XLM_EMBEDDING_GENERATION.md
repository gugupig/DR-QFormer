# XLM-RoBERTa Embedding Generation Feature

## 概述

训练代码现在支持使用 XLM-RoBERTa 模型动态生成 embeddings，而不仅仅依赖预计算的 embeddings。

## 背景

之前的训练代码有两种模式：
1. **Mode 1** (`use_precomputed_embeddings=True`): 使用 PKL 文件中的预计算 embeddings（来自 Qwen3-Embedding）
2. **Mode 2** (`use_precomputed_embeddings=False`): 使用 XLM-RoBERTa tokenizer 重新编码，但仍依赖 Q-Former 的 embedding 层

## 修改内容

### 新特性：动态 Embedding 生成

当 `use_precomputed_embeddings=False` 时，训练代码现在会：

1. **Query (问题文本)**: 
   - 使用 XLM-RoBERTa tokenizer 进行分词
   - 使用 XLM-RoBERTa 模型生成 **token-level** embeddings
   - 输出形状：`[seq_len, 768]`
   - 使用 `output.last_hidden_state` 获取每个 token 的表示

2. **Evidence (证据文本列表)**:
   - 对每个证据句子使用 XLM-RoBERTa tokenizer 进行分词
   - 使用 XLM-RoBERTa 模型生成 **sentence-level** embeddings
   - 使用 **[CLS] token** 的 embedding 作为句子表示
   - 输出形状：`[K, 768]`，其中 K 是证据片段数量

### 代码修改

#### 1. 导入模块
```python
from transformers import AutoTokenizer, AutoModel
```

#### 2. SmokingDataset 修改
```python
def __init__(self, data_dict, sample_ids, tokenizer=None, 
             xlm_model=None, device='cpu', 
             max_query_len=512, use_precomputed_embeddings=True):
    self.xlm_model = xlm_model  # 新增：XLM-RoBERTa 模型
    self.device = device        # 新增：设备
```

#### 3. Query Embedding 生成
```python
# Mode 2: Generate embeddings with XLM-RoBERTa model
query_encoded = self.tokenizer(query_text, ...)
with torch.no_grad():
    self.xlm_model.eval()
    encoded_input = {
        'input_ids': query_encoded['input_ids'].to(self.device),
        'attention_mask': query_encoded['attention_mask'].to(self.device)
    }
    output = self.xlm_model(**encoded_input)
    query_token_embeddings = output.last_hidden_state.squeeze(0).cpu()
```

#### 4. Evidence Embedding 生成
```python
# Generate sentence-level embeddings for evidence texts
for i, text in enumerate(evidence_texts):
    if text:
        encoded = self.tokenizer(text, ...)
        output = self.xlm_model(**encoded_input)
        # Use [CLS] token as sentence representation
        cls_embedding = output.last_hidden_state[:, 0, :]
        evidence_embeddings[i] = cls_embedding.squeeze(0).cpu().numpy()
```

#### 5. Stage1Trainer 修改
```python
def __init__(self, config: Stage1Config):
    if not config.use_precomputed_embeddings:
        self.tokenizer = AutoTokenizer.from_pretrained(config.xlm_model_name)
        self.xlm_embedding_model = AutoModel.from_pretrained(
            config.xlm_model_name
        ).to(self.device)
        self.xlm_embedding_model.eval()
```

## 使用方法

### 使用动态生成的 Embeddings

在 `Stage1Config` 中设置：
```python
config = Stage1Config(
    use_precomputed_embeddings=False,  # 使用动态生成
    xlm_model_name="xlm-roberta-base",
)
```

### 使用预计算的 Embeddings

保持默认设置：
```python
config = Stage1Config(
    use_precomputed_embeddings=True,  # 使用 PKL 中的预计算值（默认）
)
```

## 技术细节

### Query Token-Level Embeddings

- **目的**: 为 Q-Former 提供详细的 token-level 表示
- **方法**: 使用 XLM-RoBERTa 的 `last_hidden_state`
- **形状**: `[seq_len, 768]`
- **用途**: 传递给 Q-Former 的 `precomputed_query_emb` 参数

### Evidence Sentence-Level Embeddings

- **目的**: 为每个证据片段提供固定维度的向量表示
- **方法**: 使用 [CLS] token 的 embedding
- **形状**: `[K, 768]`
- **用途**: 传递给 Q-Former 的 `evidence_emb` 参数

### 性能考虑

1. **推理开销**: 动态生成 embeddings 需要额外的前向传播
2. **内存占用**: 需要加载额外的 XLM-RoBERTa 模型
3. **速度对比**:
   - 预计算模式：快速，直接从 PKL 加载
   - 动态生成模式：较慢，需要实时计算

### 建议

- **训练阶段**: 如果 PKL 文件包含高质量的预计算 embeddings，推荐使用 `use_precomputed_embeddings=True`
- **实验阶段**: 如果需要测试不同的 tokenizer/embedding 模型，使用 `use_precomputed_embeddings=False`
- **生产环境**: 使用预计算 embeddings 以提高效率

## 测试

运行测试脚本验证功能：
```bash
python test_xlm_embedding_generation.py
```

测试内容：
1. 加载 XLM-RoBERTa tokenizer 和模型
2. 生成 query 的 token-level embeddings
3. 生成 evidence 的 sentence-level embeddings
4. 验证与训练流程的兼容性

## 与预计算 Embeddings 的差异

| 特性 | 预计算 (Qwen3) | 动态生成 (XLM-R) |
|------|----------------|------------------|
| Tokenizer | Qwen3-Embedding | XLM-RoBERTa |
| Vocab Size | ~152K | ~250K |
| Token IDs | Qwen3 token IDs | XLM-R token IDs |
| Embedding 来源 | Qwen3 model | XLM-R model |
| 计算时机 | 数据预处理时 | 训练时实时 |
| 速度 | 快 | 较慢 |
| 灵活性 | 固定 | 可更换模型 |

## 注意事项

1. **Tokenizer 一致性**: 确保生成 embeddings 的模型与 Q-Former 使用的 tokenizer 一致
2. **设备管理**: XLM-RoBERTa 模型会被加载到与 Q-Former 相同的设备
3. **Batch Processing**: 当前实现在 Dataset 中逐样本处理，未来可优化为批量处理
4. **内存管理**: 使用 `torch.no_grad()` 和 `.eval()` 模式以节省内存

## 未来改进

1. **批量处理**: 在 collate_fn 中批量生成 embeddings
2. **缓存机制**: 缓存已生成的 embeddings 避免重复计算
3. **多模型支持**: 支持使用不同的预训练模型（BERT, RoBERTa 等）
4. **混合模式**: 部分使用预计算，部分动态生成

## 参考

- XLM-RoBERTa 官方文档: https://huggingface.co/docs/transformers/model_doc/xlm-roberta
- [CLS] token 作为句子表示: https://arxiv.org/abs/1810.04805
