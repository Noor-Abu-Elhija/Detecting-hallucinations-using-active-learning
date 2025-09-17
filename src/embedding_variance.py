# src/embedding_variance.py
import numpy as np

try:
    from sklearn.cluster import KMeans
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


def compute_embedding_variance(embeddings: np.ndarray):
    """
    Unweighted global variance:
      centroid = mean(embeddings)
      per_answer_var[i] = ||emb[i] - centroid||^2
      overall_var = mean(per_answer_var)
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    centroid = emb.mean(axis=0)
    diffs = emb - centroid
    per_answer_var = np.sum(diffs * diffs, axis=1)  # squared L2 per sample
    overall_var = float(np.mean(per_answer_var))
    return centroid.astype(np.float32), per_answer_var.astype(np.float32), overall_var


def compute_embedding_variance_weighted(embeddings: np.ndarray, weights: np.ndarray):
    """
    Weighted global variance (uses sequence probabilities as weights):
      - weights are clipped to >=0 and normalized to sum to 1
      - centroid = sum_i w_i * emb[i]
      - per_answer_var[i] = ||emb[i] - centroid||^2
      - overall_var = sum_i w_i * per_answer_var[i]
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
    Optional KMeans view:
      - fit KMeans(k)
      - report per-answer squared distance to its cluster centroid
      - report per-cluster and overall mean within-cluster variance
    """
    if not _HAVE_SKLEARN:
        raise RuntimeError("scikit-learn not installed. `pip install scikit-learn` or run with --k 0")

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
