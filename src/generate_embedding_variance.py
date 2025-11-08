# scripts/generate_embedding_variance.py
# This script generates multiple completions for a question using Falcon,
# computes embedding-based variance metrics (including weighted and KMeans),
# and saves the results as a structured JSON report.

import os, sys, json, datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TRANSFORMERS_NO_TF"] = "1"

from sentence_transformers import SentenceTransformer
from scripts.generate_answers import load_falcon_model, generate_with_probs, format_prompt
from utils.arg_parser import get_args
from src.embedding_variance import (
    compute_embedding_variance,
    compute_embedding_variance_weighted,
    compute_kmeans_variance,
)

def save_output(path, payload):
    """Save computed results to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved embedding variance output to {path}")

def main():
    args = get_args()
    question = args.question or "What is the capital of France?"
    prompt = format_prompt(question)

    # Step 1: Generate model completions with Falcon
    tokenizer, model = load_falcon_model()
    completions, token_probs, sequence_probs = generate_with_probs(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        num_return_sequences=args.num_generations,
        temperature=args.temperature,
    )

    # Step 2: Compute sentence embeddings on CPU
    embed_model = getattr(args, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    st = SentenceTransformer(embed_model, device="cpu")
    emb = st.encode(completions, convert_to_numpy=True).astype("float32")

    # Step 3: Compute global embedding variance (weighted/unweighted)
    use_weighted = getattr(args, "weighted", False)
    if use_weighted:
        weights = np.asarray(sequence_probs, dtype=np.float64)
        weights = np.clip(weights, 1e-40, 1.0)
        weights /= weights.sum()
        centroid, per_var, overall = compute_embedding_variance_weighted(emb, weights)
    else:
        centroid, per_var, overall = compute_embedding_variance(emb)

    # Step 4: Optionally compute KMeans-based variance
    k = getattr(args, "k", 0)
    kmeans_result = None
    if k and k > 0:
        try:
            labels, cents, km_per_var, km_overall, cluster_vars = compute_kmeans_variance(emb, k)
            kmeans_result = {
                "k": int(k),
                "labels": labels,
                "centroids": [c.tolist() for c in cents],
                "per_answer_var": km_per_var.tolist(),
                "overall_var": float(km_overall),
                "cluster_variances": {
                    str(ci): {"count": int(v["count"]), "variance": float(v["variance"])}
                    for ci, v in cluster_vars.items()
                },
            }
        except Exception as e:
            print(f"[warn] KMeans skipped: {e}")

    # Step 5: Display results
    print("\n=== Embedding Variance Report ===")
    print(f"Question: {question}")
    print(f"Embedding model: {embed_model}  |  Weighted: {use_weighted}")
    for i, (c, v) in enumerate(zip(completions, per_var)):
        print(f"[{i}] var_to_centroid={float(v):.6f} -> {c}")
    print(f"\nGlobal centroid variance (mean squared distance): {float(overall):.6f}")

    # Step 6: Save structured JSON output
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("outputs", f"embedding_variance_{ts}.json")
    payload = {
        "question": question,
        "completions": completions,
        "sequence_probs": sequence_probs,
        "token_probs": token_probs,
        "embedding_model": embed_model,
        "embeddings_shape": [int(emb.shape[0]), int(emb.shape[1])],
        "global_centroid": centroid.tolist(),
        "per_answer_variance": per_var.tolist(),
        "overall_variance": float(overall),
        "kmeans": kmeans_result,
    }
    save_output(out_path, payload)

if __name__ == "__main__":
    main()
