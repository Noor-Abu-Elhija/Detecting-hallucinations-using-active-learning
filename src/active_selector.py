# src/active_selector.py

import numpy as np


def select_by_uncertainty(uncertainties: np.ndarray, top_k: int) -> np.ndarray:
    """
    Selects the indices of the top_k most uncertain samples.

    Parameters:
    - uncertainties (np.ndarray): Array of uncertainty scores (one per sample).
    - top_k (int): Number of most uncertain samples to return.

    Returns:
    - np.ndarray: Array of selected indices, sorted by uncertainty (highest first).
    """
    if top_k > len(uncertainties):
        raise ValueError("top_k cannot be greater than number of available samples.")

    return np.argsort(uncertainties)[-top_k:][::-1]  # sorted descending


def select_random(unlabeled_size: int, top_k: int, seed: int = 42) -> np.ndarray:
    """
    Randomly selects indices for comparison with uncertainty-based selection.

    Parameters:
    - unlabeled_size (int): Total number of unlabeled samples.
    - top_k (int): Number of samples to select.
    - seed (int): Random seed for reproducibility.

    Returns:
    - np.ndarray: Array of randomly selected indices.
    """
    if top_k > unlabeled_size:
        raise ValueError("top_k cannot be greater than unlabeled_size.")

    np.random.seed(seed)
    return np.random.choice(unlabeled_size, size=top_k, replace=False)
