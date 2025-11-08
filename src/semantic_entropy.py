# scripts/semantic_entropy.py
# This script calculates *semantic entropy* for model-generated answers by clustering
# semantically equivalent completions using entailment (DeBERTa) and measuring
# uncertainty over meaning-level clusters.

import os
import sys
import json
import datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TRANSFORMERS_NO_TF"] = "1"

from scripts.generate_answers import load_falcon_model, generate_with_probs, format_prompt
from src.clustring_with_entailment import (
    EntailmentDeberta,
    get_semantic_ids,
    normalize_log_probs,
    cluster_log_masses,
    predictive_entropy_from_logprobs,
)
from utils.arg_parser import get_args


def compute_semantic_entropy(
    completions,
    sequence_probs,
    strict_entailment: bool = False,
    average_within_cluster: bool = True,
):
    """
    Compute semantic entropy from generated completions.

    Steps:
        1. Use NLI (DeBERTa) to cluster semantically equivalent completions.
        2. Aggregate probabilities per semantic cluster in log-space.
        3. Normalize cluster masses to obtain a probability distribution.
        4. Compute predictive entropy (in nats) across meaning clusters.

    Args:
        completions (List[str]): Model-generated responses.
        sequence_probs (List[float]): Raw sequence-level probabilities.
        strict_entailment (bool): If True, require bidirectional entailment.
        average_within_cluster (bool): Whether to normalize per-cluster mass by size.

    Returns:
        tuple:
            semantic_ids (List[int]): Cluster assignment for each completion.
            entropy (float): Semantic-level predictive entropy.
    """
    entail_model = EntailmentDeberta()

    eps = 1e-45  # to prevent log(0)
    log_likelihoods = [float(np.log(max(prob, eps))) for prob in sequence_probs]

    # Step 1: Cluster by semantic equivalence
    semantic_ids = get_semantic_ids(
        completions, model=entail_model, strict_entailment=strict_entailment
    )

    # Step 2: Aggregate log masses per cluster
    cluster_logs = cluster_log_masses(
        semantic_ids, log_likelihoods, average_within_cluster=average_within_cluster
    )

    # Step 3: Normalize cluster probabilities
    norm_log_probs = normalize_log_probs(cluster_logs)

    # Step 4: Compute predictive entropy
    entropy = predictive_entropy_from_logprobs(norm_log_probs)

    return semantic_ids, entropy
