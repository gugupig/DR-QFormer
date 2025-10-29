"""Checkpoint saving and loading utilities."""

from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None


def save_checkpoint(
    checkpoint_path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        checkpoint_path: Path to save checkpoint
        model_state: Model state dict (only trainable params)
        optimizer_state: Optimizer state dict
        epoch: Current epoch number
        metrics: Evaluation metrics
        config: Hyperparameter config
    
    Checkpoint structure:
    {
        "model_state_dict": {...},
        "optimizer_state_dict": {...},
        "epoch": int,
        "metrics": {...},
        "config": {...},
    }
    
    TODO:
    - Implement checkpoint saving with torch.save
    - Create checkpoint directory if needed
    - Save only Q-Former + heads (not retriever/LLM)
    - Add metadata (timestamp, version, etc.)
    """
    if torch is None:
        print("PyTorch not available, cannot save checkpoint")
        return
    
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model_state,
        "epoch": epoch,
    }
    
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    if config is not None:
        checkpoint["config"] = config
    
    # TODO: Implement actual saving
    # torch.save(checkpoint, checkpoint_path)
    pass
    
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
        device: Device to load tensors on
    
    Returns:
        checkpoint: Full checkpoint dictionary
    
    TODO:
    - Implement checkpoint loading with torch.load
    - Handle device mapping
    - Load state dicts into model/optimizer if provided
    - Validate checkpoint format
    - Handle missing/extra keys gracefully
    """
    if torch is None:
        print("PyTorch not available, cannot load checkpoint")
        return {}
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # TODO: Implement actual loading
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint = {}
    
    # TODO: Load state dicts into model/optimizer
    # if model is not None and "model_state_dict" in checkpoint:
    #     model.load_state_dict(checkpoint["model_state_dict"])
    
    # if optimizer is not None and "optimizer_state_dict" in checkpoint:
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    pass
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint


def get_trainable_state_dict(model: Any) -> Dict[str, Any]:
    """
    Extract only trainable parameters from model.
    
    Args:
        model: PyTorch model
    
    Returns:
        state_dict: State dict with only trainable params
    
    TODO:
    - Filter state dict to include only requires_grad=True params
    - Used to save only Q-Former + heads (not frozen components)
    """
    # TODO: Implement filtering
    # state_dict = {
    #     name: param
    #     for name, param in model.state_dict().items()
    #     if param.requires_grad
    # }
    pass
    return {}
