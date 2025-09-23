# scripts/semantic_entropy.py

import os
import sys
import json
import datetime
import scipy
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TRANSFORMERS_NO_TF"] = "1"

from scripts.generate_answers import load_falcon_model, generate_with_probs, format_prompt
from src.clustring_with_entailment import EntailmentDeberta, get_semantic_ids, normalize_log_probs, cluster_log_masses, predictive_entropy_from_logprobs
from utils.arg_parser import get_args


def compute_semantic_entropy(completions, sequence_probs, strict_entailment=False,
                             average_within_cluster=True):
    entail_model = EntailmentDeberta()

    # Robust log-likelihoods
    eps = 1e-45
    log_likelihoods = [float(np.log(max(prob, eps))) for prob in sequence_probs]

    # Cluster
    semantic_ids = get_semantic_ids(completions, model=entail_model,
                                    strict_entailment=strict_entailment)

    # Aggregate per cluster
    cluster_logs = cluster_log_masses(semantic_ids, log_likelihoods,
                                      average_within_cluster=average_within_cluster)

    # Normalize across clusters
    norm_log_probs = normalize_log_probs(cluster_logs)

    # Entropy
    entropy = predictive_entropy_from_logprobs(norm_log_probs)

    return semantic_ids, entropy


