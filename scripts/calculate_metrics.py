# scripts/calculate_metrics.py
import json
import argparse
from sklearn.metrics import classification_report


def main(results_path, questions_path, ratio_threshold):
    # Load the detailed results from your experiment
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    # Load the original questions to get the ground-truth labels
    with open(questions_path, 'r', encoding='utf-8') as f:
        # Create a dictionary mapping: question_string -> label
        questions_data = {item['question']: item['label'] for item in [json.loads(line) for line in f]}

    y_true = []  # The ground-truth labels (0 or 1)
    y_pred = []  # Your system's predictions (0 or 1)

    for result in results_data:
        question = result['question']
        true_label = questions_data.get(question)

        # Only process this result if we have a ground-truth label for it
        if true_label is not None:
            supported_ratio = result.get('supported_ratio', 0.0)
            prediction = 1 if supported_ratio >= ratio_threshold else 0

            y_true.append(true_label)
            y_pred.append(prediction)

    # --- THIS IS THE NEW, ROBUST FIX ---
    # Check if we actually found any matching labels before creating a report.
    if not y_true:
        print("\n--- ERROR ---")
        print("Could not generate a report because no matching ground-truth labels were found.")
        print("Please check the following:")
        print(
            f"1. Does the results file '{results_path}' contain the same questions as the labels file '{questions_path}'?")
        print("2. Are the question strings an exact match in both files?")
        print("-----------------\n")
        return  # Exit the script gracefully

    print(f"\n--- Classification Report ---")
    print(f"Results file: {results_path}")
    print(f"Decision rule: Predicted 'supported' if supported_ratio >= {ratio_threshold}")
    print(f"Found {len(y_true)} matching entries to evaluate.")
    print("-" * 30)

    report = classification_report(
        y_true,
        y_pred,
        target_names=['false/hallucination (0)', 'true/supported (1)'],
        zero_division=0
    )
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate final metrics from a results JSON file.")
    parser.add_argument('results_path', type=str, help='Path to the JSON results file')
    parser.add_argument('questions_path', type=str, help='Path to the original JSONL questions file with labels')
    parser.add_argument('--ratio_threshold', type=float, default=0.5,
                        help="Threshold for 'supported_ratio' to be considered a 'supported' prediction.")
    args = parser.parse_args()

    main(args.results_path, args.questions_path, args.ratio_threshold)