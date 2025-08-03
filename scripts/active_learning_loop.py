# scripts/active_learning_loop.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sentence_transformers import SentenceTransformer
from src.feature_extraction import compute_embedding_variance, compute_embedding_entropy
from src.active_selector import select_by_uncertainty
from utils.arg_parser import get_args


def compute_uncertainties(embeddings, metric, num_neighbors):
    uncertainties = []

    for i in range(len(embeddings)):
        # Exclude self
        others = np.delete(embeddings, i, axis=0)
        neighbors = others[np.random.choice(len(others), size=num_neighbors, replace=False)]

        if metric == 'variance':
            score = compute_embedding_variance(embeddings[i], neighbors)
        elif metric == 'entropy':
            score = compute_embedding_entropy(embeddings[i], neighbors)
        else:
            raise ValueError(f"Unknown uncertainty metric: {metric}")

        uncertainties.append(score)

    return np.array(uncertainties)


def main():
    args = get_args()

    # Placeholder responses — replace this later with real LLM outputs
    responses = [
        "Paris is the capital of France.",
        "Einstein invented the light bulb.",
        "The Eiffel Tower is in Berlin.",
        "Water boils at 100 degrees Celsius.",
        "Napoleon was defeated at the Battle of Waterloo.",
        "Mount Everest is the tallest mountain.",
        "The moon is made of cheese.",
        "Shakespeare wrote The Great Gatsby.",
    ]

    print(f"Encoding {len(responses)} responses using {args.embedding_model}...")
    model = SentenceTransformer(args.embedding_model)
    embeddings = model.encode(responses, convert_to_numpy=True)

    print("Computing uncertainties...")
    uncertainties = compute_uncertainties(embeddings, args.uncertainty_metric, args.num_neighbors)

    print(f"Selecting top {args.top_k} uncertain samples...")
    top_indices = select_by_uncertainty(uncertainties, args.top_k)

    print("\nMost uncertain samples:")
    for i in top_indices:
        print(f"[{i}] Score={uncertainties[i]:.4f} → {responses[i]}")


if __name__ == "__main__":
    main()
