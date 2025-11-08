# src/feature_extraction.py
# This module provides feature extraction methods for measuring embedding-based uncertainty.
# It includes variance and entropy computations using cosine similarities between embeddings.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_embedding_variance(embedding: np.ndarray, neighbors: np.ndarray) -> float:
    """
    Compute semantic uncertainty as the variance of cosine similarities
    between an embedding and its neighbors.

    Args:
        embedding (np.ndarray): 1D array (D,) representing the target embedding.
        neighbors (np.ndarray): 2D array (N, D) representing neighbor embeddings.

    Returns:
        float: Variance of cosine similarities — higher = more uncertain.
    """
    if neighbors.ndim != 2 or embedding.ndim != 1:
        raise ValueError("Expected 'neighbors' as 2D and 'embedding' as 1D array.")

    similarities = cosine_similarity([embedding], neighbors)[0]
    return float(np.var(similarities))


def compute_embedding_entropy(embedding: np.ndarray, neighbors: np.ndarray) -> float:
    """
    Compute semantic entropy using softmax-normalized cosine similarities.

    Args:
        embedding (np.ndarray): 1D array (D,)
        neighbors (np.ndarray): 2D array (N, D)

    Returns:
        float: Shannon entropy of softmax-normalized similarities.
    """
    similarities = cosine_similarity([embedding], neighbors)[0]
    softmax_sim = np.exp(similarities) / np.sum(np.exp(similarities))
    entropy = -np.sum(softmax_sim * np.log(softmax_sim + 1e-10))
    return float(entropy)
