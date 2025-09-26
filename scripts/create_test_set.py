# scripts/create_test_set.py
import json
from datasets import load_dataset


def main():
    """
    Creates a fixed, non-randomized test set from the SQuAD validation split.
    """
    print("Loading SQuAD validation set...")
    # Use the 'validation' split for testing to keep it separate from the train set
    dataset = load_dataset("squad", split="validation")

    output_filename = "squad_test_10.jsonl"
    num_questions_to_take = 10

    print(f"Creating '{output_filename}' with the first {num_questions_to_take} questions...")

    with open(output_filename, "w", encoding="utf-8") as f:
        # Take the FIRST 10 questions (no shuffle!) for a reproducible test set
        for i in range(num_questions_to_take):
            item = dataset[i]

            # --- THIS IS THE FIX ---
            # The 'answers' field is now a dictionary, not a list of dictionaries.
            # We access the list of answer texts directly via the 'text' key.
            correct_answers = item['answers']['text']

            data = {
                "question": item["question"],
                "label": 1,  # All SQuAD questions are fact-based
                "correct_answer": correct_answers
            }
            f.write(json.dumps(data) + "\n")

    print(f"Successfully created '{output_filename}'")


if __name__ == "__main__":
    main()