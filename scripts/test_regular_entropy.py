# scripts/test_regular_entropy.py
import os, sys
os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.arg_parser import get_args
from src.regular_entropy import load_llm, compute_regular_entropy_for_prompt

def main():
    args = get_args()
    tok, model, device = load_llm(args.model_id)
    texts, seq_logprobs, H = compute_regular_entropy_for_prompt(
        tok, model, device, args.prompt,
        num_samples=args.num_samples, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p
    )
    print(f"\nPrompt: {args.prompt!r}\nModel: {args.model_id}\nDevice: {device}\n")
    for i, (t, lp) in enumerate(zip(texts, seq_logprobs)):
        print(f"[{i}] logprob={lp: .3f}  completion: {t!r}")
    print(f"\nRegular (sequence) entropy over {len(texts)} samples: {H:.4f} nats\n")

if __name__ == "__main__":
    main()
