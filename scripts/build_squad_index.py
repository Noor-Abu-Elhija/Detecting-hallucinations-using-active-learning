# scripts/build_squad_index.py
import os, sys, json
from typing import List, Dict

os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
from datasets import load_dataset  # pip install datasets
import numpy as np
from sentence_transformers import SentenceTransformer
from src.corpus_index import chunk_text, CorpusIndex


def load_squad_texts(split: str = "train") -> List[str]:
    ds = load_dataset("squad", split=split)
    texts = []
    for item in ds:
        ctx = item["context"]
        for ch in chunk_text(ctx, max_tokens=180, overlap=30):
            texts.append(ch)
    return texts

def load_squad_qa(split: str = "train") -> List[Dict[str, str]]:
    """
    Load SQuAD questions with their answers (without contexts).
    Returns a list of dicts:
        {"question": "...", "answers": ["...", "..."]}
    """
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")

    ds = load_dataset("squad_v2", split=split)
    qa_pairs = []

    for item in ds:
        question = item["question"].strip()
        answers = item["answers"]["text"] if item["answers"]["text"] else ["N/A"]

        qa_pairs.append({
            "question": question,
            "answers": answers
        })

    return qa_pairs


def main(index_dir: str = "indexes/squad", embed_model: str = "all-mpnet-base-v2", split="train"):
    os.makedirs(index_dir, exist_ok=True)
    print(f"Loading SQuAD ({split})…")
    texts = load_squad_texts(split=split)
    print(f"Got {len(texts)} chunks. Embedding with {embed_model}…")

    # --- THIS IS THE MODIFIED SECTION ---
    # Create the model object first, explicitly telling it to use the CPU
    model = SentenceTransformer(embed_model, device='cpu')

    # Now, use the model to encode the texts
    emb = model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True).astype("float32")
    # --- END OF MODIFIED SECTION ---

    print("Building index…")
    ci = CorpusIndex(embeddings=emb, doc_texts=texts)
    ci.save(index_dir)
    print(f"Saved index to {index_dir}")


if __name__ == "__main__":
    # quick CLI: python -u scripts/build_squad_index.py
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "indexes/squad"
    model = sys.argv[2] if len(sys.argv) > 2 else "all-mpnet-base-v2"
    split = sys.argv[3] if len(sys.argv) > 3 else "train"
    main(out_dir, model, split)