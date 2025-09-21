# scripts/run_eval_metrics.py
import os, sys, argparse, json
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.generate_answers import load_falcon_model, generate_with_probs, format_prompt
from src.semantic_entropy import compute_semantic_entropy
from src.generate_with_regular_entropy import compute_regular_entropy
from src.embedding_variance import compute_embedding_variance
from src.embedding_variance import compute_embedding_variance_weighted
from src.embedding_variance import compute_kmeans_variance
from src.ann_verifier import ANNVerifier
from src.nli import NLI
from src.corpus_index import CorpusIndex
from utils.arg_parser import get_args


def run_generation(question: str, model, tokenizer, num_return_sequences: int, temperature: float, max_new_tokens: int):
    prompt = format_prompt(question)
    completions, sequence_probs = generate_with_probs(prompt, model, tokenizer, num_return_sequences, temperature,
                                                      max_new_tokens)
    return completions, sequence_probs


# REPLACE the old evaluate_metrics function in scripts/run_eval.py with this

def evaluate_metrics(
        question: str,
        completions: List[str],
        sequence_probs: List[float],
        embedder: SentenceTransformer,
        metric: str,
        corpus_index: Optional[CorpusIndex] = None,
        ann_threshold: float = 0.9,
        k: int = 5,
        original_question_text: str = ""  # Add this new parameter
) -> Dict[str, Any]:
    results = {
        "question": question,
        "completions": completions,
        "sequence_probs": sequence_probs
    }

    # The entropy/variance metrics remain the same
    if metric == "semantic_entropy":
        H, semantic_ids = compute_semantic_entropy(completions, sequence_probs)
        results["semantic_entropy"] = H
        results["semantic_ids"] = semantic_ids
    elif metric == "entropy":
        H = compute_regular_entropy(sequence_probs)
        results["entropy"] = H
    elif metric == "variance":
        # Note: Your script has multiple variance types, you can choose which one to use here
        # Using the simple one for now.
        var = compute_embedding_variance(embedder.encode(completions))
        results["variance"] = var
    elif metric == "weighted variance":
        var = compute_embedding_variance_weighted(sequence_probs)
        results["variance"] = var
    elif metric == "kmeans variance":
        var = compute_kmeans_variance(sequence_probs)
        results["variance"] = var
    # --- ANN METRIC WITH YOUR NLI FIX ---
    elif metric == "ann":
        if corpus_index is None:
            raise ValueError("CorpusIndex is required for ANN metric")

        comp_embs = embedder.encode(completions, convert_to_numpy=True).astype("float32")

        # Force NLI to CPU to prevent CUDA errors
        nli = NLI()
        nli.model.to('cpu')  # Ensure model is on CPU
        nli.device = 'cpu'

        ann_details = []
        supported_flags = []

        for i, ce in enumerate(comp_embs):
            sims, idxs = corpus_index.search(ce, k=k)
            max_sim = float(sims[0]) if len(sims) else 0.0
            nearest_txt = corpus_index.texts[int(idxs[0])] if len(idxs) else ""

            # --- YOUR BRILLIANT NLI FIX ---
            # Reconstruct a full, contextual hypothesis from the question and the short answer
            reconstructed_hypothesis = f"The answer to '{original_question_text}' is '{completions[i]}'."

            label_nli, conf_nli, _ = nli.predict(premise=nearest_txt, hypothesis=reconstructed_hypothesis)

            supported = max_sim >= ann_threshold and label_nli == "entailment"
            supported_flags.append(supported)

            ann_details.append({
                "completion": completions[i],
                "is_supported": supported,
                "max_similarity": max_sim,
                "nearest_sentence": nearest_txt,
                "nli_label": label_nli
            })

        results["ann_details"] = ann_details
        results["supported_ratio"] = float(sum(supported_flags)) / len(supported_flags) if supported_flags else 0.0

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return results


def load_questions(path: Optional[str]) -> List[Dict[str, Any]]:
    if path is None:
        return [
        {"question": "Who invented the light bulb?", "label": 1},
        {"question": "Where is the Eiffel Tower located?", "label": 1},
        {"question": "Is the Moon made of cheese?", "label": 0},
        ]
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def main():
    args = get_args()
    questions = load_questions(args.dataset)

    # --- SETUP: Load models ONCE outside the loop for efficiency ---
    print("Loading models (this may take a moment)...")
    # Load Falcon on the CPU (as defined in generate_answers.py)
    falcon_tokenizer, falcon_model = load_falcon_model()

    # Load the embedding model on the CPU to prevent CUDA errors
    embedder = SentenceTransformer(args.embed_model, device='cpu')

    # Load the SQuAD index (only if we're running the 'ann' metric)
    corpus_index = None
    if args.metric == 'ann':
        if not args.index_dir:
            raise ValueError("You must provide --index_dir when using the 'ann' metric.")
        print(f"Loading sentence-chunk index from {args.index_dir}...")
        corpus_index = CorpusIndex.load(args.index_dir)

    # --- PROCESSING LOOP ---
    all_results = []
    for i, q_item in enumerate(questions):
        question_text = q_item["question"]
        print(f"\nProcessing question {i + 1}/{len(questions)}: '{question_text}'")

        # 1. Generate new completions for each question using the formatted prompt
        completions, sequence_probs = run_generation(
            question=question_text,
            model=falcon_model,
            tokenizer=falcon_tokenizer,
            num_return_sequences=args.num_generations,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )

        # 2. Evaluate the specified metric for these completions
        results = evaluate_metrics(
            question=question_text,
            completions=completions,
            sequence_probs=sequence_probs,
            embedder=embedder,
            metric=args.metric,
            corpus_index=corpus_index,
            ann_threshold=args.ann_threshold,
            k=args.k,
            # We pass the original question down to use in our NLI fix
            original_question_text=question_text
        )
        all_results.append(results)

    # --- SAVE RESULTS ---
    if args.save_json:
        # Create parent directory if it doesn't exist
        output_dir = os.path.dirname(args.save_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved all {len(all_results)} results to {args.save_json}")


if __name__ == "__main__":
    main()