# scripts/generate_with_regular_entropy.py
# Generate multiple completions and compute plain token-level entropy per completion.
# Saves a small JSON with the question, completions, per-step and per-sequence entropies.

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TRANSFORMERS_NO_TF"] = "1"

from scripts.generate_answers import load_falcon_model, generate_with_probs, format_prompt
from utils.arg_parser import get_args
from src.regular_entropy import sequence_token_entropy, batch_regular_entropy

def save_output(question, completions, token_probs, per_seq_avg, per_step_entropies, mean_over_sequences, out_path=None):
    data = {
        "question": question,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "completions": completions,
        "per_sequence_avg_entropy": per_seq_avg.tolist(),
        "mean_over_sequences": float(mean_over_sequences),
        "per_step_entropies": [e.tolist() for e in per_step_entropies],
    }
    if out_path is None:
        slug = question.strip().replace(" ", "_")[:60]
        out_path = f"regular_entropy_{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")

def main():
    args = get_args()
    question = args.question or "Who wrote the novel 1984?"
    prompt = format_prompt(question)

    tokenizer, model = load_falcon_model()

    completions, token_probs, sequence_probs = generate_with_probs(
        prompt,
        model,
        tokenizer,
        num_return_sequences=args.num_generations,
        temperature=args.temperature
    )

    # token_probs is expected to be a list of (T_i, V) arrays (per completion)
    per_seq_avg = []
    per_step_list = []
    for mat in token_probs:
        avg, per_step = sequence_token_entropy(mat, normalize=getattr(args, 'normalize_entropy', False))
        per_seq_avg.append(avg)
        per_step_list.append(per_step)

    mean_over_sequences, _ = batch_regular_entropy(token_probs, normalize=getattr(args, 'normalize_entropy', False))

    print("\nCompletions & average entropies:")
    for i, (c, e) in enumerate(zip(completions, per_seq_avg), 1):
        print(f"[{i}] H_avg={e:.4f}  ::  {c}")

    print(f"\nMean entropy over sequences: {mean_over_sequences:.4f}")

    # Optional save path from args
    out_path = getattr(args, 'save_json', None)
    save_output(question, completions, token_probs, np.array(per_seq_avg), per_step_list, mean_over_sequences, out_path)

if __name__ == '__main__':
    main()
