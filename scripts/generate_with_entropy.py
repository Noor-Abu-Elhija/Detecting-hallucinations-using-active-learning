# scripts/generate_with_entropy.py

import os
import sys
import json
import datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TRANSFORMERS_NO_TF"] = "1"

from scripts.generate_answers import load_falcon_model, generate_with_probs, format_prompt
from src.semantic_entropy import EntailmentDeberta, get_semantic_ids, predictive_entropy_rao, logsumexp_by_id
from utils.arg_parser import get_args


def compute_semantic_entropy(completions, sequence_probs):
    entail_model = EntailmentDeberta()
    semantic_ids = get_semantic_ids(completions, model=entail_model)

    log_likelihoods = [np.log(prob) if prob > 0 else float('-inf') for prob in sequence_probs]
    log_probs_per_cluster = logsumexp_by_id(semantic_ids, log_likelihoods, agg="sum_normalized")
    entropy = predictive_entropy_rao(log_probs_per_cluster)

    return semantic_ids, entropy


def save_output(question, completions, token_probs, sequence_probs, semantic_ids, entropy, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"semantic_entropy_{timestamp}.json")

    data = {
        "question": question,
        "completions": completions,
        "token_probs": token_probs,
        "sequence_probs": sequence_probs,
        "semantic_ids": semantic_ids,
        "semantic_entropy": entropy
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved semantic entropy output to {filename}")


def main():
    args = get_args()
    question = args.question or "What is the capital of France?"
    prompt = format_prompt(question)

    tokenizer, model = load_falcon_model()
    completions, token_probs, sequence_probs = generate_with_probs(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        num_return_sequences=args.num_generations,
        temperature=args.temperature
    )

    semantic_ids, entropy = compute_semantic_entropy(completions, sequence_probs)

    print(f"\nSemantic Entropy: {entropy:.4f}")
    save_output(question, completions, token_probs, sequence_probs, semantic_ids, entropy)


if __name__ == "__main__":
    main()
