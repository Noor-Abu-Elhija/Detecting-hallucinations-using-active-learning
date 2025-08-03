# src/feature_extraction.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_embedding_variance(embedding: np.ndarray, neighbors: np.ndarray) -> float:
    """
    Computes semantic uncertainty as the variance of cosine similarities
    between the embedding and its neighbors.

    Parameters:
    - embedding (np.ndarray): 1D array of shape (D,) representing the embedding.
    - neighbors (np.ndarray): 2D array of shape (N, D) for N neighbors.

    Returns:
    - float: Variance of cosine similarities.
    """
    if neighbors.ndim != 2 or embedding.ndim != 1:
        raise ValueError("Expect neighbors to be 2D and embedding to be 1D array.")

    similarities = cosine_similarity([embedding], neighbors)[0]
    return float(np.var(similarities))


def compute_embedding_entropy(embedding: np.ndarray, neighbors: np.ndarray) -> float:
    """
    Computes semantic entropy using softmax-normalized similarities.

    Parameters:
    - embedding (np.ndarray): 1D array of shape (D,)
    - neighbors (np.ndarray): 2D array of shape (N, D)

    Returns:
    - float: Entropy over softmax similarities.
    """
    similarities = cosine_similarity([embedding], neighbors)[0]
    softmax_sim = np.exp(similarities) / np.sum(np.exp(similarities))
    entropy = -np.sum(softmax_sim * np.log(softmax_sim + 1e-10))
    return float(entropy)
