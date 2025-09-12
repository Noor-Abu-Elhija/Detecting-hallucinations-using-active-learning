# scripts/full_pipeline.py
import os, sys
os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sentence_transformers import SentenceTransformer
from src.feature_extraction import compute_embedding_entropy
from src.active_selector import select_by_uncertainty
from src.ann_verifier import ANNVerifier
from src.nli import NLI  # <— NEW

ANN_THRESHOLD = 0.92  # stricter to reduce false positives

def main():
    # 1) Candidate responses (swap later with Falcon generations)
    responses = [
        "Paris is the capital of France.",
        "Einstein invented the light bulb.",
        "The Eiffel Tower is in Berlin.",
        "Water boils at 100 degrees Celsius.",
        "Shakespeare wrote The Great Gatsby.",
        "Mount Everest is the tallest mountain.",
    ]

    # 2) Encode
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(responses, convert_to_numpy=True).astype("float32")

    # 3) Entropy per item (neighbors = others)
    ent = []
    for i, e in enumerate(emb):
        neighbors = np.delete(emb, i, axis=0)
        ent.append(compute_embedding_entropy(e, neighbors))
    ent = np.array(ent, dtype=np.float32)

    # pick the most uncertain items
    top_indices = select_by_uncertainty(ent, top_k=2)

    # 4) Tiny trusted corpus + ANN
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

    # 5) NLI checker
    nli = NLI()  # facebook/bart-large-mnli

    # 6) Verify top uncertain answers with ANN + NLI
    results = []
    for i in top_indices:
        is_supported_ann, max_sim, idxs = verifier.verify(emb[i], k=5, threshold=ANN_THRESHOLD)
        nearest_idx = int(idxs[0]) if len(idxs) else -1
        nearest_txt = trusted[nearest_idx] if nearest_idx >= 0 else ""

        # NLI: does nearest_txt ENTAIL the candidate response?
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

    # concise output
    for r in results:
        print(f"[{r['idx']}] supported={r['supported']} | "
              f"ent={r['entropy']:.4f} | cos={r['max_cosine']:.3f} | "
              f"nli={r['nli_label']}({r['nli_conf']:.2f}) | "
              f"resp='{r['response']}' | near='{r['nearest_text']}'")

if __name__ == "__main__":
    main()