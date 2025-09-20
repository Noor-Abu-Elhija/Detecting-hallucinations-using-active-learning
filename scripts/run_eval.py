# scripts/run_eval_metrics.py
import os, sys, argparse, json
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.generate_answers import load_falcon_model, generate_with_probs, format_prompt
from src.semantic_entropy import compute_semantic_entropy
from src.generate_with_regular_entropy import compute_regular_entropy
from src.variance_utils import compute_variance
from src.ann_verifier import ANNVerifier
from src.nli import NLI
from src.corpus_index import CorpusIndex
from utils.arg_parser import get_args


def run_generation(question: str, model, tokenizer, num_return_sequences: int, temperature: float, max_new_tokens: int):
    prompt = format_prompt(question)
    completions, sequence_probs = generate_with_probs(prompt, model, tokenizer, num_return_sequences, temperature,
                                                      max_new_tokens)
    return completions, sequence_probs


def evaluate_metrics(
        question: str,
        completions: List[str],
        sequence_probs: List[float],
        embedder: SentenceTransformer,
        metric: str,
        corpus_index: Optional[CorpusIndex] = None,
        ann_threshold: float = 0.9,
        k: int = 5
) -> Dict[str, Any]:
    results = {
        "question": question,
        "completions": completions,
        "sequence_probs": sequence_probs
    }

    if metric == "semantic_entropy":
        H, semantic_ids = compute_semantic_entropy(completions, sequence_probs)
        results["semantic_entropy"] = H
        results["semantic_ids"] = semantic_ids

    elif metric == "entropy":
        H = compute_regular_entropy(sequence_probs)
        results["entropy"] = H

    elif metric == "variance":
        var = compute_variance(sequence_probs)
        results["variance"] = var

    elif metric == "ann":
        if corpus_index is None:
            raise ValueError("CorpusIndex is required for ANN metric")
        comp_embs = embedder.encode(completions, convert_to_numpy=True).astype("float32")
        nli = NLI()
        supported_flags = []

        for i, ce in enumerate(comp_embs):
            sims, idxs = corpus_index.search(ce, k=k)
            max_sim = float(sims[0]) if len(sims) else 0.0
            nearest_txt = corpus_index.texts[int(idxs[0])] if len(idxs) else ""
            label_nli, conf_nli, _ = nli.predict(premise=nearest_txt, hypothesis=completions[i])
            supported = max_sim >= ann_threshold and label_nli == "entailment"
            supported_flags.append(supported)

        results["supported"] = supported_flags
        results["supported_ratio"] = float(sum(supported_flags)) / len(supported_flags)

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


    results = evaluate(
    questions=questions,
    embed_model=args.embed_model,
    generations=args.generations,
    temperature=args.temperature,
    metric=args.metric
    )


    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to {args.save_json}")


if __name__ == "__main__":
    main()
