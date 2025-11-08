# scripts/run_eval_metrics.py
# This script runs full metric evaluation on generated answers using Falcon-7B,
# computing uncertainty (entropy, variance) and factual verification via ANN + NLI.

import os
import sys
import json
import random
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nli import NLI
from utils.arg_parser import get_args
from src.corpus_index import CorpusIndex
from src.ann_verifier import ANNVerifier
from scripts.build_squad_index import load_squad_qa
from src.semantic_entropy import compute_semantic_entropy
from src.embedding_variance import compute_kmeans_variance, compute_embedding_variance, compute_embedding_variance_weighted
from src.generate_with_regular_entropy import compute_regular_entropy
from scripts.generate_answers import load_falcon_model, generate_with_probs, format_prompt


def run_generation(question: str, model, tokenizer, num_return_sequences: int, temperature: float, max_new_tokens: int):
    """Generate multiple completions for a question using Falcon-7B."""
    prompt = format_prompt(question)
    completions, sequence_probs = generate_with_probs(prompt, model, tokenizer, num_return_sequences, temperature, max_new_tokens)
    return completions, sequence_probs


def evaluate_metrics(
    question: str,
    completions: List[str],
    sequence_probs: List[float],
    embedder: SentenceTransformer,
    metric: str,
    corpus_index: Optional[CorpusIndex] = None,
    ann_threshold: float = 0.9,
    k: int = 5,
    original_question_text: str = "",
    correct_answer: str = "N/A"
) -> Dict[str, Any]:
    """Compute uncertainty and factuality metrics for generated completions."""
    results = {
        "question": question,
        "completions": completions,
        "sequence_probs": sequence_probs,
        "correct_answer": correct_answer
    }

    if metric in {"semantic_entropy", "all"}:
        semantic_ids, H = compute_semantic_entropy(completions, sequence_probs)
        results["semantic_entropy"] = H
        results["semantic_ids"] = semantic_ids
    if metric in {"entropy", "all"}:
        H = compute_regular_entropy(sequence_probs)
        results["entropy"] = H
    if metric in {"variance", "all"}:
        _, _, overall = compute_embedding_variance(embedder.encode(completions))
        results["variance"] = overall
    if metric in {"weighted variance", "all"}:
        _, _, overall = compute_embedding_variance_weighted(embedder.encode(completions), np.array(sequence_probs))
        results["weighted_variance"] = overall
    if metric in {"kmeans variance", "all"}:
        _, _, _, overall, cluster_vars = compute_kmeans_variance(embedder.encode(completions), k=k)
        results["kmeans_variance"] = overall
        results["cluster_variances"] = cluster_vars
    if metric in {"ann", "all"}:
        if corpus_index is None:
            raise ValueError("CorpusIndex is required for ANN metric.")
        comp_embs = embedder.encode(completions, convert_to_numpy=True).astype("float32")

        nli = NLI()
        nli.model.to('cpu')
        nli.device = 'cpu'

        ann_details, supported_flags = [], []
        for i, ce in enumerate(comp_embs):
            sims, idxs = corpus_index.search(ce, k=k)
            max_sim = float(sims[0]) if len(sims) else 0.0
            nearest_txt = corpus_index.texts[int(idxs[0])] if len(idxs) else ""
            reconstructed_hypothesis = f"The answer to '{original_question_text}' is '{completions[i]}'."
            label_nli, conf_nli, _ = nli.predict(premise=nearest_txt, hypothesis=reconstructed_hypothesis)

            supported = max_sim >= ann_threshold and (label_nli in ["entailment", "neutral"])
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

    return results


def load_questions(path: str) -> List[Dict[str, Any]]:
    """Load labeled or evaluation questions from a JSONL file."""
    if not path:
        raise ValueError("A dataset file path must be provided via --dataset.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at path: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    """Run evaluation across multiple questions and metrics."""
    args = get_args()
    all_qa_pairs = load_squad_qa("train")
    qa_pairs = random.sample(all_qa_pairs, args.num_of_question)
    questions = [item["question"] for item in qa_pairs]

    print("Loading models (this may take a moment)...")
    falcon_tokenizer, falcon_model = load_falcon_model()
    embedder = SentenceTransformer(args.embed_model, device='cpu')

    corpus_index = None
    if args.metric in {'ann', 'all'}:
        if not args.index_dir:
            raise ValueError("You must provide --index_dir when using 'ann' or 'all'.")
        print(f"Loading sentence-chunk index from {args.index_dir}...")
        corpus_index = CorpusIndex.load(args.index_dir)

    all_results = []
    for i, question_text in enumerate(questions):
        print(f"\nProcessing question {i + 1}/{args.num_of_question}: '{question_text}'")
        completions, sequence_probs = run_generation(
            question=question_text,
            model=falcon_model,
            tokenizer=falcon_tokenizer,
            num_return_sequences=args.num_generations,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )

        results = evaluate_metrics(
            question=question_text,
            completions=completions,
            sequence_probs=sequence_probs,
            embedder=embedder,
            metric=args.metric,
            corpus_index=corpus_index,
            ann_threshold=args.ann_threshold,
            k=args.k,
            original_question_text=question_text,
            correct_answer=qa_pairs[i]["answers"]
        )
        all_results.append(results)

    if args.save_json:
        output_dir = os.path.dirname(args.save_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(all_results)} results to {args.save_json}")

if __name__ == "__main__":
    main()
