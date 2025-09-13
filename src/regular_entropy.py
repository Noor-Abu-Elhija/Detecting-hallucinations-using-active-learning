# src/regular_entropy.py
# Plain (token-level) Shannon entropy utilities.
# Computes the per-step and average entropy of the model's next-token distribution.
# Contrast: "semantic entropy" clusters completions by meaning before computing entropy.
# Here we directly measure uncertainty from token probabilities.

from typing import List, Tuple
import numpy as np

EPS = 1e-12

def sequence_token_entropy(token_probs: np.ndarray, normalize: bool = False) -> Tuple[float, np.ndarray]:
    """
    Compute token-level entropy H_t = -sum_v p_t(v) log p_t(v) for each time step t,
    and return the average across the sequence.
    
    Parameters
    ----------
    token_probs : np.ndarray
        Array of shape (T, V) with next-token probabilities at each decoding step.
    normalize : bool
        If True, divide entropies by log(V) to get values in [0,1].
        
    Returns
    -------
    avg_entropy : float
        Mean entropy across time steps.
    entropies : np.ndarray
        Per-step entropy values, shape (T,).
    """
    if token_probs.ndim != 2:
        raise ValueError(f"token_probs must have shape (T, V), got {token_probs.shape}")
    T, V = token_probs.shape
    p = np.clip(token_probs, EPS, 1.0)
    H_t = -np.sum(p * np.log(p), axis=-1)  # natural log
    if normalize:
        H_t = H_t / np.log(V)
    return float(np.mean(H_t)), H_t


def batch_regular_entropy(batch_token_probs: List[np.ndarray], normalize: bool = False) -> Tuple[float, np.ndarray]:
    """
    Given a list of (T_i, V) matrices (one per completion), compute average entropy per sequence
    and return the mean over sequences plus the vector of per-sequence entropies.
    
    Parameters
    ----------
    batch_token_probs : List[np.ndarray]
        List where each element is (T_i, V) token-prob matrix for one completion.
    normalize : bool
        If True, normalize by log(V).
        
    Returns
    -------
    mean_over_sequences : float
        Mean of sequence-average entropies across the batch.
    per_sequence_avg : np.ndarray
        Array of shape (N,) with the average entropy for each completion.
    """
    per_seq = []
    for mat in batch_token_probs:
        avg, _ = sequence_token_entropy(mat, normalize=normalize)
        per_seq.append(avg)
    per_seq = np.array(per_seq, dtype=np.float32)
    return float(np.mean(per_seq)), per_seq
