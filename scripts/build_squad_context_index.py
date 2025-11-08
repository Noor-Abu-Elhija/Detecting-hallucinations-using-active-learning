# scripts/build_squad_context_index.py
# This script builds a dense vector index of unique SQuAD contexts using a SentenceTransformer model.
# It encodes all unique paragraph contexts and saves them into a FAISS-based index for retrieval tasks.

import os, sys
from typing import List
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from src.corpus_index import CorpusIndex

os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_squad_unique_contexts(split: str = "train") -> List[str]:
    """Load all unique context passages from the specified SQuAD split."""
    ds = load_dataset("squad", split=split)
    unique_contexts = set(item["context"] for item in ds)
    return list(unique_contexts)

def main(index_dir: str, embed_model: str, split: str):
    """Embed unique SQuAD contexts and save them as a searchable FAISS index."""
    os.makedirs(index_dir, exist_ok=True)
    print(f"Loading SQuAD unique contexts ({split})…")
    texts = load_squad_unique_contexts(split=split)
    print(f"Got {len(texts)} unique contexts. Embedding with {embed_model}…")

    model = SentenceTransformer(embed_model, device='cpu')
    emb = model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=True).astype("float32")

    print("Building index…")
    ci = CorpusIndex(embeddings=emb, doc_texts=texts)
    ci.save(index_dir)
    print(f"Saved CONTEXT-BASED index to {index_dir}")

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "indexes/squad_context_v1"
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "all-mpnet-base-v2"
    split_arg = sys.argv[3] if len(sys.argv) > 3 else "train"
    main(out_dir, model_arg, split_arg)
