# src/ann_verifier.py
import faiss
import numpy as np

class ANNVerifier:
    def __init__(self, embeddings: np.ndarray, dim: int, nlist: int = 100):
        self.index = faiss.IndexFlatL2(dim)  # exact search
        self.index.add(embeddings)

    def verify(self, answer_embedding: np.ndarray, k: int = 5, threshold: float = 0.8):
        answer_embedding = np.array([answer_embedding]).astype("float32")
        distances, indices = self.index.search(answer_embedding, k)
        min_distance = float(distances[0][0])
        return min_distance < (1 - threshold), min_distance, indices[0]
