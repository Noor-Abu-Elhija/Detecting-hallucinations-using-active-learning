# scripts/create_eval_set.py
import json
from datasets import load_dataset
import argparse


def create_squad_eval_file(output_file: str, num_questions: int):
    """
    Creates a .jsonl file for evaluation from the SQuAD validation set.
    Each question is assumed to have a factually supported answer, so label is 1.
    """
    print("Loading SQuAD validation set...")
    # We use the 'validation' split here so we aren't testing on our training data
    dataset = load_dataset("squad", split="validation")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            if count >= num_questions:
                break

            # Create a dictionary for each question
            data = {
                "question": item["question"],
                "label": 1  # We assume all SQuAD questions are fact-based
            }

            # Write the JSON object as a single line in the file
            f.write(json.dumps(data) + "\n")
            count += 1

    print(f"Successfully created evaluation file with {count} questions at '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an evaluation set from SQuAD.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="squad_eval_questions.jsonl",
        help="The path to save the output .jsonl file."
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=100,
        help="Number of questions to include in the set."
    )
    args = parser.parse_args()

    create_squad_eval_file(args.output_file, args.num_questions)