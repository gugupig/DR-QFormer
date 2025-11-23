import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple

# ==========================================
# 1. MACS 核心算法: 所有 query tokens → 所有 LQs
# ==========================================
def compute_macs_map(attentions, num_lqs: int = 32, alpha: float = 0.8, use_zscore: bool = True):
    """
    计算 MACS 风格的 saliency map。
    
    Args:
        attentions: tuple(num_layers) of [B, H, S, S]
        num_lqs: LQs 的数量
        alpha: 平滑系数
        use_zscore: 是否在每一层或最终结果应用 Z-score 标准化 (建议 True)
        
    Returns:
        joint_att: [B, S, num_lqs]
    """
    # 1. 堆叠与切片 [Layers, B, H, S, num_lqs]
    # 为了节省显存，建议不要一次性 stack 所有层，而是循环处理
    # 但为了逻辑清晰，这里先保持 stack
    all_layers = torch.stack(attentions, dim=0)
    target_attn = all_layers[..., :num_lqs]

    # 2. Head 聚合 (Max) -> [Layers, B, S, num_lqs]
    layer_max_attn, _ = target_attn.max(dim=2)

    num_layers = layer_max_attn.shape[0]
    batch_size, seq_len, _ = layer_max_attn.shape[1:]

    # 初始化累积矩阵
    joint_att = torch.ones(batch_size, seq_len, num_lqs, device=layer_max_attn.device)
    bias = torch.ones_like(joint_att)

    for i in range(num_layers):
        current_layer = layer_max_attn[i] # [B, S, num_lqs]
        
        # === 关键步骤: Z-score 标准化 (Optional but Recommended) ===
        # MACS 原文中通常在每一层或最终结果上做。
        # 这里我们不仅做平滑，还要保证数值的相对分布意义。
        # 注意：在 dim=-1 (num_lqs) 维度做标准化，意味着"在这个时间步，哪些 LQ 显著强于其他 LQ"
        
        # 平滑更新
        smoothed = alpha * current_layer + (1 - alpha) * bias
        joint_att = joint_att * smoothed

    # === Final Z-score Normalization ===
    if use_zscore:
        # 在 num_lqs 维度计算均值和标准差
        # 意义：对于某个固定的 token (S)，它的注意力在 32 个 LQ 中是如何分布的？
        # 我们希望找出那些 "显著高于平均关注度" 的 LQ
        mean = joint_att.mean(dim=-1, keepdim=True)
        std = joint_att.std(dim=-1, keepdim=True)
        
        # 加上 eps 防止除零
        joint_att = (joint_att - mean) / (std + 1e-6)

    return joint_att


# ==========================================
# 2. 用 chat template 构造带 Q/A 的序列，并切出 span（修正版）
# ==========================================
def build_chat_ids_and_spans(tokenizer, question: str, answer: str):
    """
    使用 Qwen 的 chat template 构造:
      system + user(question) + assistant(answer)

    用 tokenize=False 得到模板字符串，然后再单独 tokenizer(...)：
      - L_sys:       只有 system 的长度
      - L_sys_user:  system + user 的长度
      - L_full:      system + user + assistant 的长度

    Question tokens 基本落在 [L_sys, L_sys_user)
    Answer  tokens 基本落在 [L_sys_user, L_full)
    """
    system_msg = "You are a helpful assistant."

    msgs_sys = [
        {"role": "system", "content": system_msg},
    ]
    msgs_sys_user = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]
    msgs_full = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    # --- 关键修改：先拿模板字符串，再自己 tokenizer ---
    text_sys = tokenizer.apply_chat_template(
        msgs_sys,
        tokenize=False,
        add_generation_prompt=False,
    )
    ids_sys = tokenizer(text_sys, return_tensors="pt").input_ids  # [1, L_sys]

    text_sys_user = tokenizer.apply_chat_template(
        msgs_sys_user,
        tokenize=False,
        add_generation_prompt=False,
    )
    ids_sys_user = tokenizer(text_sys_user, return_tensors="pt").input_ids  # [1, L_sys_user]

    text_full = tokenizer.apply_chat_template(
        msgs_full,
        tokenize=False,
        add_generation_prompt=False,
    )
    ids_full = tokenizer(text_full, return_tensors="pt").input_ids  # [1, L_full]

    L_sys = ids_sys.size(1)
    L_sys_user = ids_sys_user.size(1)
    L_full = ids_full.size(1)

    question_indices_base = list(range(L_sys, L_sys_user))
    answer_indices_base = list(range(L_sys_user, L_full))

    return ids_full, question_indices_base, answer_indices_base


# ==========================================
# 3. 主实验：用模板 + 随机 LQs 做 MACS 可视化
# ==========================================
def run_sanity_check_with_template():
    print("🚀 Starting MACS Sanity Check (with Qwen chat template)...")

    # 配置
    MODEL_ID = r"E:\drag_datasets\llms\Qwen3-4B-Instruct-2507"  # 换成你实际的模型路径或 HF 名称
    NUM_LQS = 32
    ALPHA = 0.8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 tokenizer & model
    print(f"Loading model: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        attn_implementation="eager",  # 要拿 attentions，强制用 eager
    ).to(DEVICE)
    model.eval()

    # 构造 Q/A 文本
    question_text = "What causes rain?"
    answer_text = "Rain is caused by the condensation of water vapor into droplets."

    # 用 chat template 构造完整对话，并拿到 Question / Answer 的 token span
    chat_input_ids, q_idx_base, a_idx_base = build_chat_ids_and_spans(
        tokenizer, question_text, answer_text
    )  # [1, T]
    chat_input_ids = chat_input_ids.to(DEVICE)
    T = chat_input_ids.size(1)
    print(f"Chat seq length (without LQs): {T}")
    print(f"#question tokens (rough): {len(q_idx_base)}, #answer tokens (rough): {len(a_idx_base)}")

    # 找到 Qwen 的 <|im_start|>assistant 位置，后面用来在可视化时排除
    assistant_start_token = "<|im_start|>assistant"
    try:
        assistant_start_id = tokenizer.convert_tokens_to_ids(assistant_start_token)
        if assistant_start_id == tokenizer.unk_token_id:
            print(f"[warn] '{assistant_start_token}' -> unk_token_id，可能不是单独的 special token。")
            assistant_start_id = None
    except Exception as e:
        print(f"[warn] 无法获取 '{assistant_start_token}' 的 id: {e}")
        assistant_start_id = None

    sink_positions_base: List[int] = []
    if assistant_start_id is not None:
        sink_positions_base = (chat_input_ids[0] == assistant_start_id).nonzero(as_tuple=True)[0].tolist()
        print(f"assistant start positions (base indices): {sink_positions_base}")

    # ==========================
    # 插入随机 LQs: [LQs, chat tokens]
    # ==========================
    hidden_dim = model.config.hidden_size
    lq_embeds = torch.randn(
        1, NUM_LQS, hidden_dim,
        device=DEVICE, dtype=model.dtype
    ) * 0.02

    tok_embeds = model.get_input_embeddings()(chat_input_ids)  # [1, T, hidden_dim]
    inputs_embeds = torch.cat([lq_embeds, tok_embeds], dim=1)  # [1, NUM_LQS + T, hidden_dim]

    attention_mask = torch.ones(
        inputs_embeds.size()[:2],
        dtype=torch.long,
        device=DEVICE
    )

    # 把 base indices shift 到 "[LQs, chat_tokens]" 的坐标系中
    q_indices = [NUM_LQS + i for i in q_idx_base]
    a_indices = [NUM_LQS + i for i in a_idx_base]

    # 从 answer 中剔除 assistant 起始 sink token（如果它落在 answer span 内）
    if assistant_start_id is not None and len(sink_positions_base) > 0:
        sink_shifted = {NUM_LQS + i for i in sink_positions_base}
        a_indices = [i for i in a_indices if i not in sink_shifted]

    qa_indices = sorted(set(q_indices + a_indices))

    print(f"Shifted question index sample: {q_indices[:5]}")
    print(f"Shifted answer   index sample: {a_indices[:5]}")

    # ==========================
    # 前向一次，拿 attentions
    # ==========================
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = outputs.attentions  # tuple[num_layers] of [B, H, S, S]

    # ==========================
    # 计算 MACS 全图: 所有 tokens → 所有 LQs
    # ==========================
    print("Computing MACS saliency map...")
    joint_map = compute_macs_map(
        attentions,
        num_lqs=NUM_LQS,
        alpha=ALPHA,
    )  # [1, S, NUM_LQS];  S = NUM_LQS + T

    # 提取子矩阵
    query_to_lqs = joint_map[0, q_indices, :]      # [#Q,  NUM_LQS]
    answer_to_lqs = joint_map[0, a_indices, :]     # [#A', NUM_LQS]
    qa_to_lqs = joint_map[0, qa_indices, :]

    # ==========================
    # 可视化
    # ==========================
    print("Plotting results...")

    # 准备 token 文本 label（用 base indices decode）
    q_labels = [tokenizer.decode([chat_input_ids[0, i].item()]) for i in q_idx_base]
    a_base_filtered = [i - NUM_LQS for i in a_indices]  # 从 shifted index 还原到 chat_input_ids 坐标
    a_labels = [tokenizer.decode([chat_input_ids[0, i].item()]) for i in a_base_filtered]

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    # Plot 1: Question → LQs heatmap
    sns.heatmap(
        query_to_lqs.float().cpu().numpy(),
        ax=axes[0],
        cmap="viridis",
        cbar=True,
    )
    axes[0].set_title("MACS: Question Tokens → LQs (with chat template)")
    axes[0].set_xlabel("LQ Index (0..{})".format(NUM_LQS - 1))
    axes[0].set_ylabel("Question Tokens")
    axes[0].set_yticks(np.arange(len(q_labels)) + 0.5)
    axes[0].set_yticklabels(q_labels, rotation=0)

    # Plot 2: Answer → LQs heatmap
    sns.heatmap(
        answer_to_lqs.float().cpu().numpy(),
        ax=axes[1],
        cmap="magma",
        cbar=True,
    )
    axes[1].set_title("MACS: Answer Tokens → LQs (with chat template)")
    axes[1].set_xlabel("LQ Index (0..{})".format(NUM_LQS - 1))
    axes[1].set_ylabel("Answer Tokens")
    axes[1].set_yticks(np.arange(len(a_labels)) + 0.5)
    axes[1].set_yticklabels(a_labels, rotation=0)

    # Plot 3: 每个 LQ 的聚合 profile
    q_profile = query_to_lqs.mean(dim=0).float().cpu().numpy()
    a_profile = answer_to_lqs.mean(dim=0).float().cpu().numpy()
    qa_profile = qa_to_lqs.mean(dim=0).float().cpu().numpy()

    x = np.arange(NUM_LQS)
    axes[2].plot(x, q_profile, label="Question mean MACS", marker="o", linestyle="--")
    axes[2].plot(x, a_profile, label="Answer mean MACS", marker="x", linestyle="-")
    axes[2].plot(x, qa_profile, label="Q+A mean MACS", color="black", alpha=0.4)

    axes[2].set_title("Aggregated MACS Score per LQ (with chat template)")
    axes[2].set_xlabel("LQ Index")
    axes[2].set_ylabel("Mean MACS Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("macs_sanity_check_with_template.png")
    print("✅ Done! Saved to macs_sanity_check_with_template.png")


if __name__ == "__main__":
    run_sanity_check_with_template()
