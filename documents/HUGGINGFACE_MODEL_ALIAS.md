# HuggingFace 模型路径与缓存机制说明

## 问题：为什么重新下载了模型？

### 模型别名 vs 完整路径

HuggingFace 支持两种模型路径格式：

1. **别名（旧格式）**: `"xlm-roberta-base"`
2. **完整路径（新格式）**: `"FacebookAI/xlm-roberta-base"`

### 缓存机制

虽然这两个路径指向**完全相同的模型**，但 HuggingFace 使用**不同的缓存键**：

```
~/.cache/huggingface/hub/
├── models--xlm-roberta-base/              ← 别名格式的缓存
└── models--FacebookAI--xlm-roberta-base/  ← 完整路径格式的缓存
```

当你切换格式时，HuggingFace 认为是"不同的模型"并重新下载（尽管内容完全相同）。

### 训练时使用的是哪个？

查看你的训练配置：

```python
# train/task_e_only.py, task_s_only.py, stage1_train.py
xlm_model_name: str = "xlm-roberta-base"  # 别名格式
```

所以你之前训练时加载的是 **别名格式缓存的模型**。

## 修复的问题

### 1. 统一使用别名格式（避免重复下载）

```python
# test_qformer_comparison.py
xlm_model_name="xlm-roberta-base"  # ✅ 使用别名，复用已有缓存
# xlm_model_name="FacebookAI/xlm-roberta-base"  # ❌ 会重新下载
```

### 2. 修复 attention_mask 类型错误

**错误原因**: `_prepare_attention_mask` 假设输入是数值类型（0/1），但 PyTorch 推荐使用 `bool` 类型。

**修复方案**: 在处理前先转换为 float：

```python
def _prepare_attention_mask(self, attention_mask: Tensor) -> Tensor:
    # Convert to float if bool type
    if attention_mask.dtype == torch.bool:
        attention_mask = attention_mask.float()
    
    # ... 后续处理
```

### 3. 测试脚本中的 mask 类型

```python
# 正确的做法（推荐）
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)  # ✅
evidence_mask = torch.ones(batch_size, K, dtype=torch.bool)  # ✅

# 旧的做法（也能工作，但不推荐）
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)  # ⚠️
```

## 常见的 HuggingFace 模型别名

| 别名（旧格式） | 完整路径（新格式） |
|----------------|-------------------|
| `xlm-roberta-base` | `FacebookAI/xlm-roberta-base` |
| `xlm-roberta-large` | `FacebookAI/xlm-roberta-large` |
| `bert-base-uncased` | `google-bert/bert-base-uncased` |
| `roberta-base` | `FacebookAI/roberta-base` |
| `gpt2` | `openai-community/gpt2` |
| `t5-base` | `google-t5/t5-base` |

## 建议

### 对于生产环境
使用**完整路径格式**（更明确，更安全）：
```python
xlm_model_name = "FacebookAI/xlm-roberta-base"
```

### 对于当前项目
保持使用**别名格式**（避免重新下载 1.12GB 模型）：
```python
xlm_model_name = "xlm-roberta-base"
```

## 验证两者完全相同

```python
from transformers import AutoModel
import torch

model1 = AutoModel.from_pretrained("xlm-roberta-base")
model2 = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")

# 检查参数是否相同
for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
    assert torch.equal(p1, p2), f"参数不同: {n1}"

print("✅ 两个模型完全相同！")
```

## 总结

- ✅ **修复 1**: 测试脚本使用别名格式，避免重复下载
- ✅ **修复 2**: `_prepare_attention_mask` 支持 bool 类型输入
- ✅ **修复 3**: 测试脚本使用 `torch.bool` 类型的 mask（推荐做法）
- 📝 训练时一直使用的是正确的模型（别名格式缓存）
