# scripts/generate_regular_entropy.py

import os, sys, json, datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TRANSFORMERS_NO_TF"] = "1"

from scripts.generate_answers import load_falcon_model, generate_with_probs, format_prompt
from utils.arg_parser import get_args


def compute_regular_entropy(sequence_probs):
    """
    Regular (sequence) entropy:
      1) convert sequence probabilities to a normalized distribution
      2) H = - sum_i p_i * log(p_i)  (nats, because we use natural log)

    Uses a numerically-stable log-sum-exp pathway if you pass *log* probs.
    Here we receive raw probabilities from generate_with_probs, so we:
      - clip tiny values to avoid log(0)
      - renormalize, then compute Shannon entropy.
    """
    probs = np.asarray(sequence_probs, dtype=np.float64)

    # avoid zeros -> add tiny epsilon, then renormalize
    eps = 1e-40
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum()

    entropy = -np.sum(probs * np.log(probs))   # nats
    return float(entropy), probs.tolist()


def save_output(question, completions, token_probs, sequence_probs, norm_probs, entropy, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"regular_entropy_{ts}.json")

    data = {
        "question": question,
        "completions": completions,
        "token_probs": token_probs,             # list of per-token probabilities for each completion
        "sequence_probs": sequence_probs,       # product of token probs per completion
        "sequence_probs_normalized": norm_probs,
        "regular_entropy_nats": entropy
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved regular entropy output to {path}")


def main():
    args = get_args()
    question = args.question or "What is the capital of France?"
    prompt = format_prompt(question)

    # 1) Falcon loader (cpu, fp32) + sampling with scores
    tok, model = load_falcon_model()

    # 2) Generate completions + per-token probs + per-sequence product probs
    completions, token_probs, sequence_probs = generate_with_probs(
        prompt=prompt,
        model=model,
        tokenizer=tok,
        num_return_sequences=args.num_generations,
        temperature=args.temperature
    )

    # 3) Regular entropy over completions
    H, norm_probs = compute_regular_entropy(sequence_probs)

    print("\n=== Regular Entropy Report ===")
    for i, (c, sp) in enumerate(zip(completions, sequence_probs)):
        print(f"[{i}] p(seq)={sp:.3e}  ->  {c}")
    print(f"\nRegular (sequence) entropy over {len(completions)} samples: {H:.4f} nats\n")

    # 4) Save JSON
    save_output(question, completions, token_probs, sequence_probs, norm_probs, H)


if __name__ == "__main__":
    main()
