"""
Miscellaneous utility functions for the deepfake detection project.

Contains small, reusable helpers that do not fit into
dataset, model, or evaluation modules.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Sets random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): random seed
        deterministic (bool): whether to enforce deterministic behavior
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate computation device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    """
    Creates directory if it does not exist.

    Args:
        path (str or Path): directory path
    """
    os.makedirs(path, exist_ok=True)


def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module)

    Returns:
        int: number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_config(config: dict):
    """
    Pretty-prints configuration dictionary.

    Args:
        config (dict): experiment configuration
    """
    print("\n===== Experiment Configuration =====")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("====================================\n")

