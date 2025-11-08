# src/active_selector.py
# This module provides selection utilities for active learning,
# including uncertainty-based, random, and entropy-based selection methods.

import numpy as np
from typing import List


def select_by_uncertainty(uncertainties: np.ndarray, top_k: int) -> np.ndarray:
    """Select indices of the top_k most uncertain samples (highest uncertainty)."""
    if top_k > len(uncertainties):
        raise ValueError("top_k cannot exceed number of available samples.")
    return np.argsort(uncertainties)[-top_k:][::-1]


def select_random(unlabeled_size: int, top_k: int, seed: int = 42) -> np.ndarray:
    """Randomly select sample indices for baseline comparison."""
    if top_k > unlabeled_size:
        raise ValueError("top_k cannot exceed unlabeled_size.")
    np.random.seed(seed)
    return np.random.choice(unlabeled_size, size=top_k, replace=False)


# -------------------------
# Regular-entropy selection
# -------------------------

def compute_regular_entropy_scores(
    token_prob_mats: List[np.ndarray],
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute per-sample Shannon entropy scores from token-level probabilities.
    Each sample's entropy is averaged across decoding steps.
    """
    _, per_seq_avg = batch_regular_entropy(token_prob_mats, normalize=normalize)
    return per_seq_avg.astype(np.float64)


def select_by_regular_entropy(
    token_prob_mats: List[np.ndarray],
    top_k: int,
    normalize: bool = False,
) -> np.ndarray:
    """Select top_k most uncertain samples by average token-level entropy."""
    scores = compute_regular_entropy_scores(token_prob_mats, normalize=normalize)
    return select_by_uncertainty(scores, top_k)
