import numpy as np

from src.entailment import EntailmentDeberta
from collections import defaultdict
from scipy.special import logsumexp

# --- 1) Better clustering: exact-match fast path + NLI for near-dupes ---
def canonicalize(s: str) -> str:
    return " ".join(s.strip().lower().split())

def get_semantic_ids(completions, model, strict_entailment=False):
    """
    Groups completions into semantic clusters.
    1) exact-duplicate collapse (fast, robust for short phrases)
    2) NLI-based merging for near-duplicates
    Returns: list[int] cluster_id per completion (same length as completions)
    """
    n = len(completions)
    cluster_ids = [-1] * n

    # Fast path: identical strings share a cluster immediately
    canon_map = defaultdict(list)
    for i, c in enumerate(completions):
        canon_map[canonicalize(c)].append(i)

    next_cluster = 0
    assigned = set()
    for canon, idxs in canon_map.items():
        for idx in idxs:
            cluster_ids[idx] = next_cluster
            assigned.add(idx)
        next_cluster += 1

    # Helper for NLI equivalence
    def are_equivalent(a, b):
        imp1 = model.check_implication(a, b)  # e.g., 2=entailed, 1=neutral, 0=contradiction
        imp2 = model.check_implication(b, a)
        if strict_entailment:
            return (imp1 == 2) and (imp2 == 2)
        # Soft bi-entailment: entail one way and not contradiction the other way
        return (imp1 == 2 and imp2 != 0) or (imp2 == 2 and imp1 != 0)

    # NLI merge pass for items not already grouped by exact match
    for i in range(n):
        if i in assigned:
            continue
        if cluster_ids[i] == -1:
            cluster_ids[i] = next_cluster
            next_cluster += 1
        for j in range(i + 1, n):
            if j in assigned or cluster_ids[j] != -1:
                continue
            if are_equivalent(completions[i], completions[j]):
                cluster_ids[j] = cluster_ids[i]

    return cluster_ids

# --- 2) Proper per-cluster mass in log-space (optionally averaged by size) ---
def cluster_log_masses(cluster_ids, log_likelihoods, average_within_cluster=True):
    """
    Returns list of log-masses per cluster (unnormalized across clusters).
    If average_within_cluster=True, divides each cluster's mass by its size.
    """
    masses = []
    for uid in sorted(set(cluster_ids)):
        idxs = [i for i, cid in enumerate(cluster_ids) if cid == uid]
        ll = np.array([log_likelihoods[i] for i in idxs])  # log p_i
        log_mass = logsumexp(ll)                           # log sum_i p_i in cluster
        if average_within_cluster and len(idxs) > 0:
            log_mass -= np.log(len(idxs))                  # average by cluster size
        masses.append(log_mass)
    return masses

# --- 3) Normalize across clusters (softmax in log space) ---
def normalize_log_probs(log_masses):
    """
    Convert unnormalized log masses to normalized log probabilities over clusters.
    """
    logZ = logsumexp(log_masses)                 # log sum_k exp(log_mass_k)
    return [lm - logZ for lm in log_masses]      # log p_k

# --- 4) Entropy from normalized log probabilities ---
def predictive_entropy_from_logprobs(norm_log_probs):
    """
    H(p) = - sum_k p_k * log p_k
    norm_log_probs must be normalized logs s.t. logsumexp = 0.
    """
    probs = np.exp(norm_log_probs)
    return float(-np.sum(probs * norm_log_probs))

