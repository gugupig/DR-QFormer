"""
Frozen LLM adapter for generation.

Wraps any causal LM (e.g., LLaMA, Mistral, Phi) and keeps it frozen.
"""

from typing import List, Optional

try:
    import torch
    from torch import Tensor
except ImportError:
    Tensor = None


class FrozenLLM:
    """
    Frozen large language model adapter.
    
    Responsibilities:
    - Load and wrap a pretrained causal LM
    - Keep all parameters frozen (no gradients)
    - Provide interface: teacher_forcing_dual_path() for Task C NLL computation
    - Capture LLM→Z attention weights for posterior extraction
    - Support prefix conditioning from Q-Former output
    
    Args:
        model_name (str): LLM model name/path
        device (str): Device to load model on
        freeze (bool): Whether to freeze parameters (should always be True)
        max_length (int): Maximum generation length
    
    Example LLMs:
    - meta-llama/Llama-2-7b-hf
    - mistralai/Mistral-7B-v0.1
    - microsoft/phi-2
    - google/flan-t5-base (encoder-decoder)
    
    TODO (Integration Placeholder):
    ================================
    - [ ] Load actual LLM model (AutoModelForCausalLM)
    - [ ] Implement teacher_forcing_dual_path() for Task C
    - [ ] Register attention capture hooks
    - [ ] Construct Prefix-LM attention masks
    - [ ] Support both causal and encoder-decoder LMs
    - [ ] Handle different tokenization schemes
    - [ ] Implement efficient generation (KV caching, etc.)
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = "cuda",
        freeze: bool = True,
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self._attention_hook_handle = None
        self._captured_attention = None
        
        # TODO: Load LLM and tokenizer
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        if freeze:
            self.freeze()
    
    def freeze(self):
        """Freeze all LLM parameters."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
    
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
    ) -> dict:
        """
        Task C: Dual-path teacher forcing for contrastive NLL computation.
        
        Purpose: Compare LLM perplexity with vs. without evidence prefix Z.
                 Extract posterior fragment importance from LLM attention.
        
        Workflow:
        =========
        1. Unified Input Preparation:
           - input_ids = [dummy_Z_tokens(N), query_tokens(S_q), answer_tokens(S_a)]
           - labels = [-100(N), -100(S_q), answer_tokens(S_a)]
           - common_embeds = LLM.embed_tokens(input_ids)
        
        2. Path A (With Evidence):
           - embeds_A = common_embeds, but replace first N positions with z_prefix
           - mask_A = Prefix-LM mask (Z sees self, Q sees Z+self, A sees Z+Q+self)
           - Forward: LLM(embeds_A, mask_A, labels) → nll_with_evidence
           - Hook: Capture attention[answer_tokens → Z positions]
        
        3. Path B (Without Evidence - Baseline):
           - embeds_B = embeds_A (same embeddings)
           - mask_B = Block Q and A from seeing Z (set Z columns to -inf)
           - Forward (no_grad): LLM(embeds_B, mask_B, labels) → nll_without_evidence
        
        Args:
            z_prefix: Q-Former output [batch, N_lq, d_llm] as soft prompt prefix
            query_input_ids: Query tokens [batch, S_q]
            answer_input_ids: Answer tokens [batch, S_a]
            capture_attention: Whether to capture LLM→Z attention (for posterior)
        
        Returns:
            dict with keys:
                - 'nll_with_evidence': Scalar NLL with Z prefix (Path A)
                - 'nll_without_evidence': Scalar NLL without Z (Path B, detached)
                - 'llm_attention_to_z': [batch, n_heads, S_a, N_lq] attention weights
                                       (None if capture_attention=False)
                - 'answer_start_idx': Token index where answer starts (N_lq + S_q)
        
        Implementation Steps (TODO - Placeholder):
        ==========================================
        
        Step 1: Prepare Unified Input
        ------------------------------
        ```python
        batch_size = z_prefix.shape[0]
        N_lq = z_prefix.shape[1]
        S_q = query_input_ids.shape[1]
        S_a = answer_input_ids.shape[1]
        
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
        
        # Common embeddings
        common_embeds = self.model.get_input_embeddings()(input_ids)  # [batch, N+Sq+Sa, d]
        ```
        
        Step 2: Construct Prefix-LM Mask (Mask A)
        ------------------------------------------
        ```python
        seq_len = N_lq + S_q + S_a
        mask_A = torch.zeros(seq_len, seq_len, device=self.device)
        
        # Z sees itself (causal within Z)
        for i in range(N_lq):
            mask_A[i, :i+1] = 1.0
        
        # Query sees Z + itself (causal within Q)
        for i in range(N_lq, N_lq + S_q):
            mask_A[i, :N_lq] = 1.0  # See all Z
            mask_A[i, N_lq:i+1] = 1.0  # See previous Q tokens
        
        # Answer sees Z + Q + itself (causal within A)
        for i in range(N_lq + S_q, seq_len):
            mask_A[i, :N_lq] = 1.0  # See all Z
            mask_A[i, N_lq:N_lq+S_q] = 1.0  # See all Q
            mask_A[i, N_lq+S_q:i+1] = 1.0  # See previous A tokens
        
        # Convert to attention mask format (0 = -inf, 1 = 0)
        mask_A = (1.0 - mask_A) * -10000.0
        ```
        
        Step 3: Path A Forward (With Evidence)
        ---------------------------------------
        ```python
        # Replace dummy Z embeddings with actual Z
        embeds_A = common_embeds.clone()
        embeds_A[:, :N_lq, :] = z_prefix
        
        # Register attention hook if needed
        if capture_attention:
            self._register_attention_hook()
        
        # Forward pass
        outputs_A = self.model(
            inputs_embeds=embeds_A,
            attention_mask=mask_A,
            labels=labels,
            output_attentions=True,
        )
        nll_with_evidence = outputs_A.loss
        
        # Extract captured attention
        llm_attention_to_z = None
        if capture_attention and self._captured_attention is not None:
            # Extract attention[answer_tokens → Z]
            # Shape: [batch, n_heads, S_a, N_lq]
            llm_attention_to_z = self._captured_attention[:, :, N_lq+S_q:, :N_lq]
            self._remove_attention_hook()
        ```
        
        Step 4: Construct Blocked Mask (Mask B)
        ----------------------------------------
        ```python
        mask_B = mask_A.clone()
        # Block Q and A from seeing Z (set Z columns to -inf)
        mask_B[N_lq:, :N_lq] = -10000.0
        ```
        
        Step 5: Path B Forward (Without Evidence)
        ------------------------------------------
        ```python
        with torch.no_grad():
            outputs_B = self.model(
                inputs_embeds=embeds_A,  # Same embeddings
                attention_mask=mask_B,   # Blocked mask
                labels=labels,
            )
            nll_without_evidence = outputs_B.loss
        ```
        
        Step 6: Return Results
        ----------------------
        ```python
        return {
            'nll_with_evidence': nll_with_evidence,
            'nll_without_evidence': nll_without_evidence.detach(),
            'llm_attention_to_z': llm_attention_to_z,
            'answer_start_idx': N_lq + S_q,
        }
        ```
        
        Attention Hook Implementation (TODO):
        =====================================
        ```python
        def _register_attention_hook(self):
            # Register hook on last decoder layer
            def hook_fn(module, input, output):
                # output[1] is attention weights
                self._captured_attention = output[1]  # [batch, n_heads, seq, seq]
            
            last_layer = self.model.model.layers[-1]  # LLaMA structure
            self._attention_hook_handle = last_layer.register_forward_hook(hook_fn)
        
        def _remove_attention_hook(self):
            if self._attention_hook_handle is not None:
                self._attention_hook_handle.remove()
                self._attention_hook_handle = None
            self._captured_attention = None
        ```
        
        TODO Integration Checklist:
        ===========================
        - [ ] Load actual LLM model (e.g., LLaMA, Mistral, Phi)
        - [ ] Implement embed_tokens access
        - [ ] Construct Prefix-LM masks correctly
        - [ ] Register attention hooks on correct layer
        - [ ] Test with different LLM architectures
        - [ ] Verify attention extraction shape
        - [ ] Ensure frozen mode (model.eval(), no_grad for Path B)
        - [ ] Handle edge cases (empty query, long sequences)
        """
        # Placeholder implementation
        print("[TODO] FrozenLLM.teacher_forcing_dual_path() not yet implemented")
        print("       This requires actual LLM integration (transformers library)")
        
        # Return dummy values for testing
        import torch
        batch_size = z_prefix.shape[0] if z_prefix is not None else 1
        N_lq = z_prefix.shape[1] if z_prefix is not None else 32
        
        return {
            'nll_with_evidence': torch.tensor(2.5),
            'nll_without_evidence': torch.tensor(3.8),
            'llm_attention_to_z': None,  # [batch, n_heads, S_a, N_lq]
            'answer_start_idx': N_lq + query_input_ids.shape[1] if query_input_ids is not None else N_lq,
        }
    
    def _register_attention_hook(self):
        """
        Register forward hook to capture LLM attention weights.
        
        TODO:
        - Identify correct layer to hook (usually last decoder layer)
        - Handle different LLM architectures (LLaMA, Mistral, GPT, etc.)
        - Extract attention from correct output position
        """
        pass
    
    def _remove_attention_hook(self):
        """Remove attention capture hook and clear cached attention."""
        if self._attention_hook_handle is not None:
            self._attention_hook_handle.remove()
            self._attention_hook_handle = None
        self._captured_attention = None
    
    def __repr__(self) -> str:
        return f"FrozenLLM(model={self.model_name}, frozen=True)"
