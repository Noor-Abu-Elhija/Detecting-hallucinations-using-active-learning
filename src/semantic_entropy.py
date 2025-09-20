# scripts/semantic_entropy.py

import os
import sys
import json
import datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TRANSFORMERS_NO_TF"] = "1"

from scripts.generate_answers import load_falcon_model, generate_with_probs, format_prompt
from src.semantic_entropy import EntailmentDeberta, get_semantic_ids, predictive_entropy_rao, logsumexp_by_id
from utils.arg_parser import get_args


def compute_semantic_entropy(completions, sequence_probs):
    entail_model = EntailmentDeberta()
    semantic_ids = get_semantic_ids(completions, model=entail_model)

    log_likelihoods = [np.log(prob) if prob > 0 else float('-inf') for prob in sequence_probs]
    log_probs_per_cluster = logsumexp_by_id(semantic_ids, log_likelihoods, agg="sum_normalized")
    entropy = predictive_entropy_rao(log_probs_per_cluster)

    return semantic_ids, entropy

