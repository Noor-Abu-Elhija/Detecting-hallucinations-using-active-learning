# scripts/generate_with_regular_entropy.py
# Generate multiple completions and compute plain token-level entropy per completion.
# Self-contained: uses Hugging Face directly (no imports from other project files).
# Saves a JSON with: question, completions, per-sequence avg entropy, per-step entropies.

import os
import json
import argparse
from datetime import datetime
from typing import List, Tuple
import numpy as np

# Ensure we use PyTorch backend for transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"

from src.regular_entropy import sequence_token_entropy, batch_regular_entropy


def hf_generate_token_probs(
    prompt: str,
    model_name: str = "gpt2",
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    num_return_sequences: int = 3,
    device: str | None = None,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Returns:
      completions: list[str] of decoded generated tokens (only the new part)
      token_probs: list[np.ndarray], each (T_i, V) with next-token probabilities per step
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if device:
        model.to(device)

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True,  # crucial to get per-step logits
    )

    scores = gen.scores  # list of length T; each is (num_seqs, V) logits at step t
    T = len(scores)
    completions, probs_list = [], []

    # Build (T, V) probability matrix per sequence
    for i in range(num_return_sequences):
        # Stack logits across steps for sequence i -> (T, V)
        logits = torch.stack([scores[t][i] for t in range(T)], dim=0)
        probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
        probs_list.append(probs)

    # Decode only the generated part for readability
    for i in range(num_return_sequences):
        gen_ids = gen.sequences[i][-T:]  # last T tokens are newly generated
        completions.append(tok.decode(gen_ids, skip_special_tokens=True))

    return completions, probs_list


def save_output(
    question: str,
    completions: List[str],
    per_seq_avg: List[float],
    per_step_entropies: List[List[float]],
    mean_over_sequences: float,
    out_path: str | None = None,
):
    data = {
        "question": question,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "completions": completions,
        "per_sequence_avg_entropy": per_seq_avg,
        "mean_over_sequences": float(mean_over_sequences),
        "per_step_entropies": per_step_entropies,
    }
    if out_path is None:
        slug = question.strip().replace(" ", "_")[:60]
        out_path = f"regular_entropy_{slug}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


def main():
    ap = argparse.ArgumentParser("Generate and compute Regular (token-level) Entropy")
    ap.add_argument("--question", type=str, required=True, help="Prompt/question to answer")
    ap.add_argument("--model", type=str, default="gpt2", help="HF model name (e.g., gpt2)")
    ap.add_argument("--num_generations", type=int, default=3, help="Number of completions per question")
    ap.add_argument("--max_new_tokens", type=int, default=32, help="Max new tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--device", type=str, default=None, help='Set to "cuda" if you have a GPU')
    ap.add_argument("--normalize_entropy", action="store_true", help="Divide by log(V) to scale entropy to [0,1]")
    ap.add_argument("--save_json", type=str, default=None, help="Optional path to save JSON")
    args = ap.parse_args()

    completions, token_probs = hf_generate_token_probs(
        prompt=args.question,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_return_sequences=args.num_generations,
        device=args.device,
    )

    # Compute per-completion entropy (average across steps)
    per_seq_avg = []
    per_step_list = []
    for mat in token_probs:
        avg, per_step = sequence_token_entropy(mat, normalize=args.normalize_entropy)
        per_seq_avg.append(float(avg))
        per_step_list.append(per_step.tolist())

    # Mean over completions
    mean_over_sequences, _ = batch_regular_entropy(token_probs, normalize=args.normalize_entropy)

    print("\nCompletions & average entropies:")
    for i, (c, e) in enumerate(zip(completions, per_seq_avg), 1):
        print(f"[{i}] H_avg={e:.4f} :: {c}")
    print(f"\nMean entropy over sequences: {float(mean_over_sequences):.4f}")

    # Save if requested
    save_output(
        args.question,
        completions,
        per_seq_avg,
        per_step_list,
        float(mean_over_sequences),
        args.save_json,
    )


if __name__ == "__main__":
    main()
