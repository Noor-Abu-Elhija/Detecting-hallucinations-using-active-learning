# src/semantic_entropy.py
# This module computes semantic entropy by clustering LLM completions
# into meaning-equivalent groups using NLI entailment checks, then
# measuring the distributional uncertainty over these semantic clusters.

import numpy as np
from src.entailment import EntailmentDeberta
from collections import defaultdict
from scipy.special import logsumexp


# --- 1) Semantic grouping of completions ---
def canonicalize(s: str) -> str:
    """Normalize string by trimming spaces and lowering case for exact-match grouping."""
    return " ".join(s.strip().lower().split())


def get_semantic_ids(completions, model, strict_entailment=False):
    """
    Cluster completions into semantic groups using a two-step process:
      (1) Exact-match collapse for identical strings.
      (2) NLI-based merging for semantically equivalent completions.
    Returns:
        cluster_ids: list[int] cluster ID per completion.
    """
    n = len(completions)
    cluster_ids = [-1] * n

    # Step 1: Fast collapse for identical strings
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

    # Helper: NLI entailment-based equivalence check
    def are_equivalent(a, b):
        imp1 = model.check_implication(a, b)  # entailment: 2, neutral: 1, contradiction: 0
        imp2 = model.check_implication(b, a)
        if strict_entailment:
            return (imp1 == 2) and (imp2 == 2)
        # Soft equivalence: entailment one way, not contradiction the other
        return (imp1 == 2 and imp2 != 0) or (imp2 == 2 and imp1 != 0)

    # Step 2: Merge near-duplicates via entailment
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


# --- 2) Compute per-cluster log-probability masses ---
def cluster_log_masses(cluster_ids, log_likelihoods, average_within_cluster=True):
    """
    Compute unnormalized log-probability mass for each semantic cluster.
    If `average_within_cluster` is True, divide each cluster’s mass by its size.
    """
    masses = []
    for uid in sorted(set(cluster_ids)):
        idxs = [i for i, cid in enumerate(cluster_ids) if cid == uid]
        ll = np.array([log_likelihoods[i] for i in idxs])
        log_mass = logsumexp(ll)  # log(sum_i p_i)
        if average_within_cluster and len(idxs) > 0:
            log_mass -= np.log(len(idxs))  # normalize by cluster size
        masses.append(log_mass)
    return masses


# --- 3) Normalize across clusters ---
def normalize_log_probs(log_masses):
    """Convert cluster log masses into normalized log probabilities."""
    logZ = logsumexp(log_masses)
    return [lm - logZ for lm in log_masses]


# --- 4) Compute predictive entropy ---
def predictive_entropy_from_logprobs(norm_log_probs):
    """Compute entropy H(p) = -Σ p_k log p_k from normalized log-probabilities."""
    probs = np.exp(norm_log_probs)
    return float(-np.sum(probs * norm_log_probs))
