# scripts/full_pipeline.py
# This script runs the full hallucination-detection pipeline:
# it encodes candidate responses, measures uncertainty, verifies them via ANN and NLI, and outputs factuality results.

import os, sys
os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sentence_transformers import SentenceTransformer
from src.feature_extraction import compute_embedding_entropy
from src.active_selector import select_by_uncertainty
from src.ann_verifier import ANNVerifier
from src.nli import NLI

ANN_THRESHOLD = 0.92  # stricter to reduce false positives

def main():
    """Run the end-to-end hallucination detection pipeline using entropy, ANN retrieval, and NLI verification."""
    responses = [
        "Paris is the capital of France.",
        "Einstein invented the light bulb.",
        "The Eiffel Tower is in Berlin.",
        "Water boils at 100 degrees Celsius.",
        "Shakespeare wrote The Great Gatsby.",
        "Mount Everest is the tallest mountain.",
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(responses, convert_to_numpy=True).astype("float32")

    # Compute embedding entropy for each response
    ent = []
    for i, e in enumerate(emb):
        neighbors = np.delete(emb, i, axis=0)
        ent.append(compute_embedding_entropy(e, neighbors))
    ent = np.array(ent, dtype=np.float32)

    # Select the most uncertain items based on entropy
    top_indices = select_by_uncertainty(ent, top_k=2)

    # Build a small trusted corpus for factual verification
    trusted = [
        "Paris is the capital of France.",
        "Water boils at 100 degrees Celsius.",
        "Mount Everest is the tallest mountain.",
        "Thomas Edison invented the light bulb.",
        "The Eiffel Tower is in Paris.",
        "F. Scott Fitzgerald wrote The Great Gatsby.",
    ]
    trusted_emb = model.encode(trusted, convert_to_numpy=True).astype("float32")
    verifier = ANNVerifier(trusted_emb)

    # Initialize the NLI model for entailment verification
    nli = NLI()

    # Combine uncertainty, ANN, and NLI to determine factual support
    results = []
    for i in top_indices:
        is_supported_ann, max_sim, idxs = verifier.verify(emb[i], k=5, threshold=ANN_THRESHOLD)
        nearest_idx = int(idxs[0]) if len(idxs) else -1
        nearest_txt = trusted[nearest_idx] if nearest_idx >= 0 else ""
        nli_label, nli_conf, _scores = nli.predict(premise=nearest_txt, hypothesis=responses[i])
        supported = bool(is_supported_ann and nli_label == "entailment")

        results.append({
            "idx": int(i),
            "response": responses[i],
            "entropy": float(ent[i]),
            "max_cosine": float(max_sim),
            "nearest_text": nearest_txt,
            "nli_label": nli_label,
            "nli_conf": float(nli_conf),
            "supported": supported,
        })

    # Display concise summary of verification results
    for r in results:
        print(f"[{r['idx']}] supported={r['supported']} | "
              f"ent={r['entropy']:.4f} | cos={r['max_cosine']:.3f} | "
              f"nli={r['nli_label']}({r['nli_conf']:.2f}) | "
              f"resp='{r['response']}' | near='{r['nearest_text']}'")

if __name__ == "__main__":
    main()
