# src/semantic_entropy.py

import numpy as np
from src.entailment import EntailmentDeberta


def get_semantic_ids(completions, model, strict_entailment=False):
    """
    Groups completions into semantic clusters based on pairwise entailment.
    Returns a list of cluster IDs per completion.
    """

    def are_equivalent(a, b):
        imp1 = model.check_implication(a, b)
        imp2 = model.check_implication(b, a)

        print(f"[{a} → {b}] = {imp1}, [{b} → {a}] = {imp2}")  # Debug

        if strict_entailment:
            return imp1 == 2 and imp2 == 2
        else:
            return imp1 == 2 and imp2 != 1  # <- use soft bi-entailment only

    cluster_ids = [-1] * len(completions)
    next_cluster_id = 0

    for i in range(len(completions)):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = next_cluster_id
        for j in range(i + 1, len(completions)):
            if cluster_ids[j] == -1 and are_equivalent(completions[i], completions[j]):
                cluster_ids[j] = next_cluster_id
        next_cluster_id += 1

    return cluster_ids


def logsumexp_by_id(cluster_ids, log_likelihoods, agg="sum_normalized"):
    """
    Groups log-likelihoods by semantic cluster and returns the aggregated log-probs.
    """
    unique_ids = sorted(set(cluster_ids))
    log_probs_per_cluster = []

    for uid in unique_ids:
        indices = [i for i, cid in enumerate(cluster_ids) if cid == uid]
        ll_values = [log_likelihoods[i] for i in indices]

        if agg == "sum_normalized":
            ll_array = np.array(ll_values)
            ll_array -= np.log(np.sum(np.exp(ll_array)))
            logsumexp_value = np.log(np.sum(np.exp(ll_array)))
        else:
            raise ValueError("Unknown aggregation method")

        log_probs_per_cluster.append(logsumexp_value)

    return log_probs_per_cluster


def predictive_entropy_rao(log_probs):
    """
    Rao's semantic entropy: -Σ p(x) log p(x)
    where log_probs are already normalized.
    """
    probs = np.exp(log_probs)
    entropy = -np.sum(probs * log_probs)
    return entropy
