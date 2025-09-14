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




# -------------------------
# Regular-entropy selection
# -------------------------

def compute_regular_entropy_scores(
    token_prob_mats: List[np.ndarray],
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute per-sample regular (token-level) entropy scores.

    Each element in `token_prob_mats` is a matrix of shape (T_i, V),
    containing the next-token probability distribution at each decoding step.

    We compute Shannon entropy H_t = -sum_v p_t(v) log p_t(v) at each step,
    then average over steps for a single scalar per sample.

    Parameters
    ----------
    token_prob_mats : List[np.ndarray]
        List of (T_i, V) probability matrices, one per sample.
    normalize : bool
        If True, divide each step's entropy by log(V) so the average is in [0, 1].

    Returns
    -------
    np.ndarray
        Vector of entropy scores (one per sample), higher = more uncertain.
    """
    # Use the batch helper to compute per-sequence averages efficiently
    _, per_seq_avg = batch_regular_entropy(token_prob_mats, normalize=normalize)
    # per_seq_avg is a NumPy array of shape (N,)
    return per_seq_avg.astype(np.float64)


def select_by_regular_entropy(
    token_prob_mats: List[np.ndarray],
    top_k: int,
    normalize: bool = False,
) -> np.ndarray:
    """
    Select indices of the top_k most-uncertain samples using regular (token-level) entropy.

    Parameters
    ----------
    token_prob_mats : List[np.ndarray]
        List where each element is a (T_i, V) token-prob matrix for one sample.
    top_k : int
        Number of samples to select.
    normalize : bool
        If True, normalize entropy by log(V) → scores in [0, 1].

    Returns
    -------
    np.ndarray
        Indices of the selected samples, highest entropy first.
    """
    scores = compute_regular_entropy_scores(token_prob_mats, normalize=normalize)
    return select_by_uncertainty(scores, top_k)
