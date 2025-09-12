# scripts/full_pipeline.py
import os, sys
os.environ["TRANSFORMERS_NO_TF"] = "1"  # silence TF in transformers/sentence-transformers
# ensure src/ is importable when running from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sentence_transformers import SentenceTransformer
from src.feature_extraction import compute_embedding_entropy
from src.active_selector import select_by_uncertainty
from src.ann_verifier import ANNVerifier


def main():
    print(">>> starting full_pipeline...", flush=True)

    # -------------------------------------------------------------------------
    # 1) Candidate responses (replace later with Falcon completions)
    # -------------------------------------------------------------------------
    responses = [
        "Paris is the capital of France.",
        "Einstein invented the light bulb.",
        "The Eiffel Tower is in Berlin.",
        "Water boils at 100 degrees Celsius.",
        "Shakespeare wrote The Great Gatsby.",
        "Mount Everest is the tallest mountain.",
    ]
    print(f">>> {len(responses)} responses loaded.", flush=True)

    # -------------------------------------------------------------------------
    # 2) Encode responses
    #    Tip: first run may take time to download the model; next runs are fast.
    # -------------------------------------------------------------------------
    print(">>> loading SentenceTransformer (first run may download the model)...", flush=True)
    # You can swap to a smaller model if downloads are slow: "paraphrase-MiniLM-L3-v2"
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    print(f">>> model '{model_name}' loaded. Encoding responses...", flush=True)

    embeddings = model.encode(responses, convert_to_numpy=True).astype("float32")
    print(">>> responses encoded.", flush=True)

    # -------------------------------------------------------------------------
    # 3) Compute per-item semantic entropy (neighbors = all other items)
    # -------------------------------------------------------------------------
    print(">>> computing semantic entropies...", flush=True)
    uncertainties = []
    for i, emb in enumerate(embeddings):
        neighbors = np.delete(embeddings, i, axis=0)
        ent = compute_embedding_entropy(emb, neighbors)
        uncertainties.append(ent)
    uncertainties = np.array(uncertainties, dtype=np.float32)

    # show top uncertainties
    top_k = 2
    print(f">>> selecting top-{top_k} most uncertain samples...", flush=True)
    top_indices = select_by_uncertainty(uncertainties, top_k=top_k)

    print("\nTop uncertain responses:")
    for i in top_indices:
        print(f"[{i}] entropy={uncertainties[i]:.4f}  →  {responses[i]}", flush=True)

    # -------------------------------------------------------------------------
    # 4) Build small trusted corpus + ANN verifier (cosine similarity)
    #    Later: replace with Wikipedia/SQuAD chunks + cached embeddings.
    # -------------------------------------------------------------------------
    trusted_corpus = [
        "Paris is the capital of France.",
        "Water boils at 100 degrees Celsius.",
        "Mount Everest is the tallest mountain.",
        "Thomas Edison invented the light bulb.",
        "The Eiffel Tower is in Paris.",
        "F. Scott Fitzgerald wrote The Great Gatsby.",
    ]
    print("\n>>> encoding trusted corpus and building ANN index...", flush=True)
    trusted_embeddings = model.encode(trusted_corpus, convert_to_numpy=True).astype("float32")
    verifier = ANNVerifier(trusted_embeddings)
    print(">>> ANN verifier ready (cosine-based).", flush=True)

    # -------------------------------------------------------------------------
    # 5) Verify the most-uncertain answers against trusted sources
    # -------------------------------------------------------------------------
    print("\nVerification results:")
    for i in top_indices:
        is_supported, max_sim, idxs = verifier.verify(embeddings[i], k=5, threshold=0.80)
        nearest_idx = int(idxs[0]) if len(idxs) else -1
        nearest_txt = trusted_corpus[nearest_idx] if nearest_idx >= 0 else "<none>"

        print(f"[{i}] {responses[i]}", flush=True)
        print(f"  Supported? {is_supported} | max_cosine={max_sim:.4f} | nearest='{nearest_txt}'", flush=True)


if __name__ == "__main__":
    main()
