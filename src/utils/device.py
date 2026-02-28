"""Utility functions for device management and deterministic behavior."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device: Preferred device ('auto', 'cpu', 'cuda', 'mps'). If None, uses 'auto'.
        
    Returns:
        torch.device: The selected device.
    """
    if device is None:
        device = "auto"
    
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_num_workers() -> int:
    """Get optimal number of workers for data loading.
    
    Returns:
        int: Number of workers to use.
    """
    return min(4, os.cpu_count() or 1)


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.
    
    Args:
        seconds: Time duration in seconds.
        
    Returns:
        str: Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists.
    """
    os.makedirs(path, exist_ok=True)


def anonymize_filename(filename: str) -> str:
    """Anonymize filename by removing personal identifiers.
    
    Args:
        filename: Original filename.
        
    Returns:
        str: Anonymized filename.
    """
    # Remove common personal identifiers
    anonymized = filename.lower()
    anonymized = anonymized.replace(" ", "_")
    anonymized = anonymized.replace("-", "_")
    
    # Remove file extension for processing
    name, ext = os.path.splitext(anonymized)
    
    # Generate hash-based anonymized name
    import hashlib
    hash_obj = hashlib.md5(name.encode())
    anonymized_name = f"audio_{hash_obj.hexdigest()[:8]}{ext}"
    
    return anonymized_name
