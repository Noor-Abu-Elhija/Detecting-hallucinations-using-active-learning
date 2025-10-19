# scripts/label_completions.py
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Manually label generated completions.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSON file from your teammates' run.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save your new human-labeled JSONL file.")
    args = parser.parse_args()

    # --- Load the dataset ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: The input file was not found at '{args.input_file}'")
        print("Please make sure you are in the main project directory when running this script.")
        return
    except json.JSONDecodeError:
        print(f"FATAL ERROR: The input file '{args.input_file}' is not a valid JSON file.")
        return

    print("--- Starting Human Labeling Session ---")
    print("For each generated answer, you will see the context it was compared against.")
    print("Your task is to determine if the answer is factually supported by that context.")
    print("Enter 'y' for YES, 'n' for NO, or 's' to SKIP.\n")

    # --- Create the output directory if it doesn't exist ---
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # --- Open the output file to write labels ---
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        for i, item in enumerate(data):
            question = item.get('question', 'N/A')
            # Handle potential differences in the input JSON structure
            if 'ann_details' in item:
                completions_details = item['ann_details']
            elif 'completions' in item and isinstance(item['completions'], list):
                 completions_details = item['completions'] # Adapt if the structure is different
            else:
                completions_details = []


            if not completions_details:
                continue

            print(f"\n--- Question {i+1}/{len(data)} ---")
            print(f"QUESTION: {question}")

            for j, detail in enumerate(completions_details):
                completion = detail.get('completion', 'N/A')
                nearest_sentence = detail.get('nearest_sentence', detail.get('nearest_text', 'N/A')) # Check for both possible keys
                machine_label = detail.get('is_supported', detail.get('supported', False)) # Check for both possible keys

                print(f"\n  [Completion {j+1}/{len(completions_details)}]")
                print(f"  Generated Answer: '{completion}'")
                print(f"  Nearest Sentence (from knowledge base): '{nearest_sentence}'")
                print(f"  Machine's Guess: {'SUPPORTED' if machine_label else 'NOT SUPPORTED'}")

                # Loop until a valid input is received
                while True:
                    user_input = input("  Is the answer factually supported by the nearest sentence? (y/n/s): ").lower()
                    if user_input in ['y', 'n', 's']:
                        break
                    else:
                        print("  Invalid input. Please enter 'y', 'n', or 's'.")

                if user_input == 's':
                    print("  Skipped.")
                    continue

                human_label = (user_input == 'y')

                # Prepare the final labeled data point
                labeled_point = {
                    'question': question,
                    'completion': completion,
                    'nearest_sentence': nearest_sentence,
                    'human_label_is_supported': human_label,
                    'semantic_entropy_for_question': item.get('semantic_entropy'),
                    'variance_for_question': item.get('variance'),
                }

                # Write this labeled point as a new line in output file
                out_f.write(json.dumps(labeled_point) + '\n')

    print(f"\n--- Labeling session complete! ---")
    print(f"Your labels have been saved to '{args.output_file}'")


if __name__ == "__main__":
    main()
