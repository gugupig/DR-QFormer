"""
Retriever adapter for frozen retrieval models.

Wraps any dense retriever (e.g., Contriever, DPR, E5) and keeps it frozen.
"""

from typing import List, Tuple, Optional

try:
    import torch
    from torch import Tensor
except ImportError:
    Tensor = None


class Retriever:
    """
    Frozen retriever adapter.
    
    Responsibilities:
    - Load and wrap a pretrained dense retriever
    - Keep all parameters frozen (no gradients)
    - Provide interface: get_fragments(query, k) -> (texts, embeddings)
    
    Args:
        model_name (str): Retriever model name/path
        device (str): Device to load model on
        freeze (bool): Whether to freeze parameters (should always be True)
    
    Example retrievers:
    - facebook/contriever
    - facebook/contriever-msmarco
    - facebook/dpr-ctx_encoder-single-nq-base
    - intfloat/e5-base-v2
    
    TODO:
    - Integrate with retrieval corpus (vector DB / FAISS)
    - Handle batched queries
    - Cache embeddings for efficiency
    """
    
    def __init__(
        self,
        model_name: str = "facebook/contriever",
        device: str = "cuda",
        freeze: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        
        # TODO: Load retriever model
        # from transformers import AutoModel, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name).to(device)
        
        if freeze:
            self.freeze()
    
    def freeze(self):
        """Freeze all retriever parameters."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def get_fragments(
        self,
        query: str,
        k: int = 10,
        corpus: Optional[List[str]] = None,
    ) -> Tuple[List[str], Optional[Tensor]]:
        """
        Retrieve top-k fragments for a query.
        
        Args:
            query: Query string
            k: Number of fragments to retrieve
            corpus: Optional corpus to search (uses default if None)
        
        Returns:
            texts: List of k retrieved fragment texts
            p_embeds: Fragment embeddings [k, d] (frozen, detached)
        
        TODO:
        - Encode query with retriever
        - Search corpus/vector DB for top-k matches
        - Return both texts and embeddings
        - Ensure embeddings are detached (no grad)
        """
        # Placeholder implementation
        texts = [f"Fragment {i}" for i in range(k)]
        p_embeds = None  # torch.randn(k, 768).to(self.device) if have torch
        
        # TODO: Implement actual retrieval
        pass
        
        return texts, p_embeds
    
    def encode_fragments(self, texts: List[str]) -> Optional[Tensor]:
        """
        Encode text fragments to embeddings.
        
        Args:
            texts: List of text fragments
        
        Returns:
            embeddings: [batch, d] fragment embeddings (detached)
        
        TODO:
        - Tokenize texts
        - Encode with retriever
        - Pool to single vector per fragment
        - Detach from computation graph
        """
        # TODO: Implement encoding
        pass
        return None
    
    def __repr__(self) -> str:
        return f"Retriever(model={self.model_name}, frozen=True)"
