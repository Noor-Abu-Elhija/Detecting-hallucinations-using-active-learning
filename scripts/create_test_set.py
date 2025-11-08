# scripts/create_test_set.py
# This script creates a small, fixed test set (10 questions) from the SQuAD validation split.
# It saves each question with its correct answers and a factual label (1) into a JSONL file.

import json
from datasets import load_dataset

def main():
    """Generate a reproducible test set from the SQuAD validation split."""
    print("Loading SQuAD validation set...")
    dataset = load_dataset("squad", split="validation")

    output_filename = "squad_test_10.jsonl"
    num_questions_to_take = 10

    print(f"Creating '{output_filename}' with the first {num_questions_to_take} questions...")

    with open(output_filename, "w", encoding="utf-8") as f:
        for i in range(num_questions_to_take):
            item = dataset[i]
            correct_answers = item['answers']['text']
            data = {
                "question": item["question"],
                "label": 1,  # All SQuAD questions are factual
                "correct_answer": correct_answers
            }
            f.write(json.dumps(data) + "\n")

    print(f"Successfully created '{output_filename}'")

if __name__ == "__main__":
    main()
