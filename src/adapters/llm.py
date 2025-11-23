"""
Frozen LLM adapter for generation and teacher forcing.

Wraps Qwen or other causal LMs for DR-QFormer Stage-2 training.
Based on MACS_example.py usage patterns.
"""

from typing import List, Optional, Dict, Tuple
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM


class FrozenLLM:
    """
    Frozen large language model adapter for DR-QFormer.
    
    Responsibilities:
    - Load and wrap a pretrained causal LM (Qwen recommended)
    - Keep all parameters frozen (no gradients)
    - Provide teacher_forcing_dual_path() for Task C NLL computation
    - Capture LLM attention weights for MACS posterior extraction
    - Support Qwen chat template for span detection
    
    Args:
        model_name: LLM model name/path (e.g., "Qwen/Qwen-7B" or local path)
        device: Device to load model on ("cuda" or "cpu")
        freeze: Whether to freeze parameters (should always be True)
        max_length: Maximum generation length
        torch_dtype: Model dtype (torch.float16 for GPU, torch.float32 for CPU)
        attn_implementation: "eager" to capture attentions (required for MACS)
    
    Example Usage:
        >>> llm = FrozenLLM(
        ...     model_name="E:/drag_datasets/llms/Qwen3-4B-Instruct-2507",
        ...     device="cuda",
        ...     torch_dtype=torch.float16,
        ... )
        >>> 
        >>> # Teacher forcing for Task C
        >>> outputs = llm.teacher_forcing_dual_path(
        ...     z_prefix=z,  # [batch, 32, 4096]
        ...     query_input_ids=q_ids,  # [batch, S_q]
        ...     answer_input_ids=a_ids,  # [batch, S_a]
        ...     capture_attention=True,
        ... )
        >>> 
        >>> # Use attentions for MACS posterior extraction
        >>> attentions = outputs['attentions']
        >>> nll_gain = outputs['nll_without_evidence'] - outputs['nll_with_evidence']
    """
    
    def __init__(
        self,
        model_name: str = r"E:\drag_datasets\llms\Qwen3-4B-Instruct-2507",
        device: str = "cuda",
        freeze: bool = True,
        max_length: int = 256,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: str = "eager",
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._attention_hook_handle = None
        self._captured_attentions = None
        
        # Auto-detect dtype if not specified
        if torch_dtype is None:
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.torch_dtype = torch_dtype
        
        print(f"[FrozenLLM] Loading model: {model_name}")
        print(f"[FrozenLLM] Device: {device}, Dtype: {torch_dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with attention capture enabled
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,  # "eager" to get attentions
        ).to(device)
        
        # Freeze model
        if freeze:
            self.freeze()
        
        # Set to eval mode (no dropout, batch norm in eval, etc.)
        self.model.eval()
        
        print(f"[FrozenLLM] Model loaded successfully. Frozen: {freeze}")
        print(f"[FrozenLLM] Hidden size: {self.model.config.hidden_size}")
        print(f"[FrozenLLM] Num layers: {self.model.config.num_hidden_layers}")
    
    def freeze(self):
        """Freeze all LLM parameters (no gradients computed)."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"[FrozenLLM] All parameters frozen (requires_grad=False)")
    
    def get_hidden_size(self) -> int:
        """Get LLM hidden dimension."""
        return self.model.config.hidden_size if self.model else 0
    
    def build_chat_ids_and_spans(
        self,
        question: str,
        answer: str,
    ) -> Tuple[Tensor, List[int], List[int]]:
        """
        Build chat-formatted input_ids and extract question/answer spans.
        
        Uses Qwen chat template: system + user(question) + assistant(answer)
        
        Args:
            question: Question text
            answer: Answer text
        
        Returns:
            input_ids: [1, T] chat-formatted token IDs
            question_indices: List of token positions for question
            answer_indices: List of token positions for answer
        
        Example:
            >>> ids, q_idx, a_idx = llm.build_chat_ids_and_spans(
            ...     "What causes rain?",
            ...     "Rain is caused by condensation."
            ... )
            >>> # ids: [1, 45] tensor
            >>> # q_idx: [12, 13, 14, 15]  (question tokens)
            >>> # a_idx: [20, 21, 22, ...]  (answer tokens)
        """
        system_msg = "You are a helpful assistant."
        
        # Step 1: System only
        msgs_sys = [{"role": "system", "content": system_msg}]
        text_sys = self.tokenizer.apply_chat_template(
            msgs_sys, tokenize=False, add_generation_prompt=False
        )
        ids_sys = self.tokenizer(text_sys, return_tensors="pt").input_ids
        
        # Step 2: System + User (question)
        msgs_sys_user = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ]
        text_sys_user = self.tokenizer.apply_chat_template(
            msgs_sys_user, tokenize=False, add_generation_prompt=False
        )
        ids_sys_user = self.tokenizer(text_sys_user, return_tensors="pt").input_ids
        
        # Step 3: System + User + Assistant (answer)
        msgs_full = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        text_full = self.tokenizer.apply_chat_template(
            msgs_full, tokenize=False, add_generation_prompt=False
        )
        ids_full = self.tokenizer(text_full, return_tensors="pt").input_ids
        
        # Extract spans
        L_sys = ids_sys.size(1)
        L_sys_user = ids_sys_user.size(1)
        L_full = ids_full.size(1)
        
        question_indices = list(range(L_sys, L_sys_user))
        answer_indices = list(range(L_sys_user, L_full))
        
        return ids_full, question_indices, answer_indices
    
    def generate_with_prefix(
        self,
        query_text: str,
        z: Optional[Tensor] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text with Q-Former prefix conditioning.
        
        Args:
            query_text: Input query string
            z: Q-Former output [1, N, d] to use as prefix (optional)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        
        Returns:
            generated_text: LLM output string
        
        Workflow:
        1. Tokenize query_text
        2. If z provided, prepend as soft prefix to embeddings
        3. Generate with LLM (frozen, no grad)
        4. Decode and return text
        
        TODO:
        - Implement soft prompt prefix injection
        - Handle embedding dimensions mismatch
        - Support batched generation
        - Add generation config options
        """
        # Placeholder implementation
        output = f"[Generated response for: {query_text}]"
        
        # TODO: Implement actual generation with prefix
        pass
        
        return output
    
    def generate_batch(
        self,
        queries: List[str],
        z_batch: Optional[Tensor] = None,
        **kwargs,
    ) -> List[str]:
        """
        Batch generation with prefix conditioning.
        
        Args:
            queries: List of query strings
            z_batch: Q-Former outputs [batch, N, d]
            **kwargs: Generation config (temperature, top_p, etc.)
        
        Returns:
            outputs: List of generated strings
        
        TODO:
        - Implement batched generation
        - Handle variable-length inputs
        - Optimize with KV caching
        """
        # TODO: Implement batch generation
        outputs = [self.generate_with_prefix(q, None, **kwargs) for q in queries]
        return outputs
    
    def encode_text(self, text: str) -> Optional[Tensor]:
        """
        Encode text to LLM embeddings (for analysis/debugging).
        
        Args:
            text: Input text
        
        Returns:
            embeddings: [seq_len, d] token embeddings
        
        TODO:
        - Tokenize and embed text
        - Return frozen embeddings (detached)
        """
        # TODO: Implement encoding
        pass
        return None
    
    def teacher_forcing_dual_path(
        self,
        z_prefix: Tensor,
        query_input_ids: Tensor,
        answer_input_ids: Tensor,
        capture_attention: bool = True,
    ) -> Dict[str, any]:
        """
        Task C: Dual-path teacher forcing for contrastive NLL computation.
        
        Compares LLM perplexity with vs. without evidence prefix Z, and extracts
        attention patterns for MACS posterior extraction.
        
        Workflow:
        ---------
        1. Prepare unified input: [dummy_Z, query, answer]
        2. Path A (with evidence): Replace dummy_Z with z_prefix, compute NLL
        3. Path B (without evidence): Block attention to Z, compute baseline NLL
        4. Capture attentions for MACS posterior extraction
        
        Args:
            z_prefix: Q-Former output [batch, N_lq, d_llm] as soft prompt prefix
            query_input_ids: Query tokens [batch, S_q]
            answer_input_ids: Answer tokens [batch, S_a]
            capture_attention: Whether to capture full attentions tuple for MACS
        
        Returns:
            Dictionary with keys:
                - 'nll_with_evidence': Scalar loss with Z prefix (Path A, has grad)
                - 'nll_without_evidence': Scalar loss without Z (Path B, detached)
                - 'attentions': Tuple of [batch, heads, seq, seq] tensors for MACS
                - 'answer_start_idx': Index where answer begins (N_lq + S_q)
                - 'answer_end_idx': Index where answer ends (N_lq + S_q + S_a)
        
        Example:
            >>> z = condense_head(qformer_outputs['lqs_after'])  # [4, 32, 4096]
            >>> q_ids = torch.randint(0, 50000, (4, 20))  # [4, 20]
            >>> a_ids = torch.randint(0, 50000, (4, 30))  # [4, 30]
            >>> 
            >>> outputs = llm.teacher_forcing_dual_path(z, q_ids, a_ids)
            >>> nll_gain = outputs['nll_without_evidence'] - outputs['nll_with_evidence']
            >>> attentions = outputs['attentions']  # For MACS posterior
        """
        # =====================================================================
        # Step 1: Prepare Unified Input
        # =====================================================================
        batch_size = z_prefix.shape[0]
        N_lq = z_prefix.shape[1]
        S_q = query_input_ids.shape[1]
        S_a = answer_input_ids.shape[1]
        seq_len = N_lq + S_q + S_a
        
        # Create dummy tokens for Z positions (will be replaced by embeddings)
        dummy_z_tokens = torch.zeros(batch_size, N_lq, dtype=torch.long, device=self.device)
        
        # Concatenate: [dummy_Z, Query, Answer]
        input_ids = torch.cat([dummy_z_tokens, query_input_ids, answer_input_ids], dim=1)
        
        # Labels: Only compute loss on answer tokens
        labels = torch.cat([
            torch.full((batch_size, N_lq), -100, dtype=torch.long, device=self.device),
            torch.full((batch_size, S_q), -100, dtype=torch.long, device=self.device),
            answer_input_ids,
        ], dim=1)
        
        # Get common embeddings from input_ids
        common_embeds = self.model.get_input_embeddings()(input_ids)  # [batch, N+Sq+Sa, d]
        
        # =====================================================================
        # Step 2: Path A - With Evidence (Z prefix active)
        # =====================================================================
        # Replace dummy Z embeddings with actual z_prefix
        embeds_A = common_embeds.clone()
        embeds_A[:, :N_lq, :] = z_prefix
        
        # Create standard causal attention mask (all-ones, transformers handles causal)
        attention_mask_A = torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device)
        
        # Forward pass with evidence (Path A)
        outputs_A = self.model(
            inputs_embeds=embeds_A,
            attention_mask=attention_mask_A,
            labels=labels,
            output_attentions=capture_attention,
            use_cache=False,
            return_dict=True,
        )
        
        nll_with_evidence = outputs_A.loss
        attentions = outputs_A.attentions if capture_attention else None
        
        # =====================================================================
        # Step 3: Path B - Without Evidence (Z blocked)
        # =====================================================================
        # Create attention mask that blocks query/answer from seeing Z
        # Shape: [batch, seq_len] where 0 = blocked, 1 = visible
        # Standard causal masking is applied by the model automatically
        attention_mask_B = torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device)
        
        # Block first N_lq positions (Z) - set to 0 so query/answer can't see them
        # Note: We keep embeds_A (which has z_prefix), but mask prevents attention
        attention_mask_B[:, :N_lq] = 0
        
        # Forward pass without evidence (Path B, no gradients)
        with torch.no_grad():
            outputs_B = self.model(
                inputs_embeds=embeds_A,  # Same embeddings as Path A
                attention_mask=attention_mask_B,  # Z positions masked out
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            nll_without_evidence = outputs_B.loss.detach()
        
        # =====================================================================
        # Step 4: Return Results
        # =====================================================================
        return {
            'nll_with_evidence': nll_with_evidence,
            'nll_without_evidence': nll_without_evidence,
            'attentions': attentions,  # Tuple of [B, H, S, S] for MACS
            'answer_start_idx': N_lq + S_q,
            'answer_end_idx': seq_len,
        }
    
    def generate_with_prefix(
        self,
        query_text: str,
        z: Optional[Tensor] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Generate text with optional Q-Former prefix conditioning.
        
        Args:
            query_text: Input query string
            z: Q-Former output [1, N, d] to use as prefix (optional)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            generated_text: LLM output string
        
        Example:
            >>> response = llm.generate_with_prefix(
            ...     "What causes rain?",
            ...     z=z_prefix,  # [1, 32, 4096]
            ...     temperature=0.7,
            ... )
        """
        # Tokenize query
        inputs = self.tokenizer(query_text, return_tensors="pt").to(self.device)
        
        if z is not None:
            # Prepend z as soft prefix
            input_embeds = self.model.get_input_embeddings()(inputs.input_ids)
            inputs_embeds = torch.cat([z, input_embeds], dim=1)
            
            # Adjust attention mask
            z_mask = torch.ones(z.shape[0], z.shape[1], dtype=torch.long, device=self.device)
            attention_mask = torch.cat([z_mask, inputs.attention_mask], dim=1)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
        else:
            # Standard generation without prefix
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def encode_text(self, text: str) -> Tensor:
        """
        Encode text to LLM embeddings.
        
        Args:
            text: Input text
        
        Returns:
            embeddings: [seq_len, d] token embeddings (detached)
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeds = self.model.get_input_embeddings()(inputs.input_ids)
        return embeds[0].detach()  # [seq_len, d]
    
    def __repr__(self) -> str:
        model_name_short = self.model_name.split('/')[-1] if '/' in self.model_name else self.model_name
        return f"FrozenLLM(model={model_name_short}, device={self.device}, frozen=True)"


# Alias for backward compatibility
QwenLLM = FrozenLLM
