# scripts/test_feature_extraction.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
from src.feature_extraction import compute_embedding_variance, compute_embedding_entropy
from utils.arg_parser import get_args


def test_feature_extraction(num_neighbors):
    # Simulated input
    embedding = np.array([0.1, 0.2, 0.3])

    # Create `num_neighbors` neighbor embeddings of dim 3
    rng = np.random.default_rng(seed=42)
    neighbors = rng.normal(loc=0.2, scale=0.05, size=(num_neighbors, 3))

    var = compute_embedding_variance(embedding, neighbors)
    ent = compute_embedding_entropy(embedding, neighbors)

    print(f"Embedding variance (uncertainty): {var:.5f}")
    print(f"Embedding entropy (uncertainty):  {ent:.5f}")


if __name__ == "__main__":
    args = get_args()
    test_feature_extraction(args.num_neighbors)
