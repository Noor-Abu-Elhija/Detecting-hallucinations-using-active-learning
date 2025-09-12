# scripts/generate_with_entropy.py

import os
import sys
import json
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["TRANSFORMERS_NO_TF"] = "1"

from src.falcon_generate_logprobs import load_falcon_model, generate_with_logprobs
from src.semantic_entropy import EntailmentDeberta, get_semantic_ids, logsumexp_by_id, predictive_entropy_rao
from utils.arg_parser import get_args


def compute_semantic_entropy(completions, log_likelihoods):
    entail_model = EntailmentDeberta()
    semantic_ids = get_semantic_ids(completions, model=entail_model)

    avg_log_likelihoods = [sum(ll) / len(ll) for ll in log_likelihoods]
    log_probs_per_cluster = logsumexp_by_id(semantic_ids, avg_log_likelihoods, agg="sum_normalized")
    entropy = predictive_entropy_rao(log_probs_per_cluster)

    return semantic_ids, entropy


def save_output(question, completions, log_likelihoods, semantic_ids, entropy, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"semantic_entropy_{timestamp}.json")

    data = {
        "question": question,
        "completions": completions,
        "log_likelihoods": log_likelihoods,
        "semantic_ids": semantic_ids,
        "semantic_entropy": entropy
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved full semantic entropy output to {filename}")


def main():
    args = get_args()

    question = "Who discovered America?"  # Can be dynamic later

    tokenizer, model = load_falcon_model()
    completions, log_likelihoods = generate_with_logprobs(
        prompt=question,
        model=model,
        tokenizer=tokenizer,
        num_return_sequences=args.num_generations,
        temperature=args.temperature
    )

    semantic_ids, entropy = compute_semantic_entropy(completions, log_likelihoods)

    print(f"Semantic Entropy: {entropy:.4f}")
    save_output(question, completions, log_likelihoods, semantic_ids, entropy)


if __name__ == "__main__":
    main()
