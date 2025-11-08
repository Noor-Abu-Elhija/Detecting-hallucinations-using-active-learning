# scripts/build_squad_index.py
# This script builds a dense FAISS index for SQuAD by chunking contexts, embedding them, and saving for retrieval tasks.
# It can also load SQuAD question–answer pairs for downstream hallucination or QA evaluation.

import os, sys, json
from typing import List, Dict
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from src.corpus_index import chunk_text, CorpusIndex

os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_squad_texts(split: str = "train") -> List[str]:
    """Load and chunk SQuAD contexts into overlapping text segments for indexing."""
    ds = load_dataset("squad", split=split)
    texts = []
    for item in ds:
        ctx = item["context"]
        for ch in chunk_text(ctx, max_tokens=180, overlap=30):
            texts.append(ch)
    return texts

def load_squad_qa(split: str = "train") -> List[Dict[str, str]]:
    """Load SQuAD questions and their answers (no context)."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")

    ds = load_dataset("squad_v2", split=split)
    qa_pairs = []
    for item in ds:
        question = item["question"].strip()
        answers = item["answers"]["text"] if item["answers"]["text"] else ["N/A"]
        qa_pairs.append({"question": question, "answers": answers})
    return qa_pairs

def main(index_dir: str = "indexes/squad", embed_model: str = "all-mpnet-base-v2", split="train"):
    """Embed chunked SQuAD contexts and save them into a FAISS-based index."""
    os.makedirs(index_dir, exist_ok=True)
    print(f"Loading SQuAD ({split})…")
    texts = load_squad_texts(split=split)
    print(f"Got {len(texts)} chunks. Embedding with {embed_model}…")

    model = SentenceTransformer(embed_model, device='cpu')
    emb = model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True).astype("float32")

    print("Building index…")
    ci = CorpusIndex(embeddings=emb, doc_texts=texts)
    ci.save(index_dir)
    print(f"Saved index to {index_dir}")

if __name__ == "__main__":
    # Example: python -u scripts/build_squad_index.py indexes/squad all-mpnet-base-v2 train
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "indexes/squad"
    model = sys.argv[2] if len(sys.argv) > 2 else "all-mpnet-base-v2"
    split = sys.argv[3] if len(sys.argv) > 3 else "train"
    main(out_dir, model, split)
