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
    - Provide interface: generate_with_prefix(query, Z) -> output_text
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
    
    TODO:
    - Support both causal and encoder-decoder LMs
    - Handle different tokenization schemes
    - Implement efficient generation (KV caching, etc.)
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
    
    def __repr__(self) -> str:
        return f"FrozenLLM(model={self.model_name}, frozen=True)"
