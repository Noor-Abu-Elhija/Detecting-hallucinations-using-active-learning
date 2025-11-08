# scripts/create_eval_set.py
# This script creates a small evaluation set from the SQuAD validation split.
# It saves questions and their assumed factual labels (1) into a JSONL file for model testing.

import json
from datasets import load_dataset
import argparse

def create_squad_eval_file(output_file: str, num_questions: int):
    """Generate a JSONL evaluation file from the SQuAD validation set with factual labels."""
    print("Loading SQuAD validation set...")
    dataset = load_dataset("squad", split="validation")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            if count >= num_questions:
                break

            data = {
                "question": item["question"],
                "label": 1  # All SQuAD questions are assumed factual
            }
            f.write(json.dumps(data) + "\n")
            count += 1

    print(f"Created evaluation file with {count} questions at '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an evaluation set from SQuAD.")
    parser.add_argument("--output_file", type=str, default="squad_eval_questions.jsonl",
                        help="Path to save the output .jsonl file.")
    parser.add_argument("--num_questions", type=int, default=100,
                        help="Number of questions to include.")
    args = parser.parse_args()
    create_squad_eval_file(args.output_file, args.num_questions)
