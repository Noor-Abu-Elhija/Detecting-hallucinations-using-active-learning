# scripts/label_completions.py
# This script allows a human annotator to manually label generated answers as factually supported or not.
# It loads model outputs, displays them with context, and saves user-provided labels into a JSONL file.

import json
import argparse
import os

def main():
    """Run an interactive labeling session for human annotation of model completions."""
    parser = argparse.ArgumentParser(description="Manually label generated completions.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the model output JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the labeled JSONL file.")
    args = parser.parse_args()

    # Load input data
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{args.input_file}'")
        return
    except json.JSONDecodeError:
        print(f"ERROR: '{args.input_file}' is not a valid JSON file.")
        return

    print("--- Starting Human Labeling Session ---")
    print("Review each generated answer and decide if it is factually supported by the context.")
    print("Enter 'y' for YES, 'n' for NO, or 's' to SKIP.\n")

    # Create output directory if missing
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Open output file for labeled results
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        for i, item in enumerate(data):
            question = item.get('question', 'N/A')
            completions_details = item.get('ann_details') or item.get('completions', [])
            if not completions_details:
                continue

            print(f"\n--- Question {i+1}/{len(data)} ---")
            print(f"QUESTION: {question}")

            for j, detail in enumerate(completions_details):
                completion = detail.get('completion', 'N/A')
                nearest_sentence = detail.get('nearest_sentence', detail.get('nearest_text', 'N/A'))
                machine_label = detail.get('is_supported', detail.get('supported', False))

                print(f"\n  [Completion {j+1}/{len(completions_details)}]")
                print(f"  Generated Answer: '{completion}'")
                print(f"  Nearest Sentence: '{nearest_sentence}'")
                print(f"  Machine Guess: {'SUPPORTED' if machine_label else 'NOT SUPPORTED'}")

                while True:
                    user_input = input("  Is the answer factually supported? (y/n/s): ").lower()
                    if user_input in ['y', 'n', 's']:
                        break
                    print("  Invalid input. Please enter 'y', 'n', or 's'.")

                if user_input == 's':
                    print("  Skipped.")
                    continue

                human_label = (user_input == 'y')

                labeled_point = {
                    'question': question,
                    'completion': completion,
                    'nearest_sentence': nearest_sentence,
                    'human_label_is_supported': human_label,
                    'semantic_entropy_for_question': item.get('semantic_entropy'),
                    'variance_for_question': item.get('variance'),
                }

                out_f.write(json.dumps(labeled_point) + '\n')

    print(f"\n--- Labeling session complete! ---")
    print(f"Your annotations have been saved to '{args.output_file}'")

if __name__ == "__main__":
    main()
