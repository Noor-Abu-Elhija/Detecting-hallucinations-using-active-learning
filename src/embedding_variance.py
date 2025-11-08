# src/embedding_variance.py
# This module provides functions to compute embedding variance metrics for uncertainty estimation.
# It includes unweighted, weighted, and k-means–based variance calculations for LLM-generated embeddings.

import numpy as np

try:
    from sklearn.cluster import KMeans
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


def compute_embedding_variance(embeddings: np.ndarray):
    """
    Compute unweighted global variance across embeddings.
    Steps:
      1. Compute centroid = mean(embeddings)
      2. Compute per-sample variance = ||emb[i] - centroid||²
      3. Average over all samples
    Returns:
        centroid, per_answer_var, overall_var
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    centroid = emb.mean(axis=0)
    diffs = emb - centroid
    per_answer_var = np.sum(diffs * diffs, axis=1)
    overall_var = float(np.mean(per_answer_var))
    return centroid.astype(np.float32), per_answer_var.astype(np.float32), overall_var


def compute_embedding_variance_weighted(embeddings: np.ndarray, weights: np.ndarray):
    """
    Compute weighted variance using sample weights (e.g., sequence probabilities).
    Steps:
      1. Normalize non-negative weights to sum to 1.
      2. Weighted centroid = Σ w_i * emb[i]
      3. Weighted variance = Σ w_i * ||emb[i] - centroid||²
    Returns:
        centroid, per_answer_var, overall_var
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0.0, np.inf)
    Z = w.sum()

    if Z <= 0 or not np.isfinite(Z):
        w = np.full_like(w, 1.0 / len(w))
    else:
        w = w / Z

    centroid = (emb * w[:, None]).sum(axis=0)
    diffs = emb - centroid
    per_answer_var = np.sum(diffs * diffs, axis=1)
    overall_var = float(np.dot(w, per_answer_var))
    return centroid.astype(np.float32), per_answer_var.astype(np.float32), overall_var


def compute_kmeans_variance(embeddings: np.ndarray, k: int):
    """
    Cluster embeddings via KMeans and compute within-cluster variances.
    Steps:
      1. Fit KMeans(k) to embeddings.
      2. Compute per-sample squared distance to its centroid.
      3. Report per-cluster and overall variance.
    Returns:
        labels, centroids, per_answer_var, overall_var, cluster_vars
    """
    if not _HAVE_SKLEARN:
        raise RuntimeError("scikit-learn not installed. Please install it or run without --k.")

    emb = np.asarray(embeddings, dtype=np.float32)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(emb)
    cents = km.cluster_centers_.astype(np.float32)

    diffs = emb - cents[labels]
    per_answer_var = np.sum(diffs * diffs, axis=1)
    overall_var = float(np.mean(per_answer_var))

    cluster_vars = {}
    for c in range(k):
        idx = np.where(labels == c)[0]
        v = float(per_answer_var[idx].mean()) if len(idx) else 0.0
        cluster_vars[str(c)] = {"count": int(len(idx)), "variance": v}

    return labels.tolist(), cents, per_answer_var.astype(np.float32), overall_var, cluster_vars
