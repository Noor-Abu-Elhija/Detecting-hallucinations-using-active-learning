# scripts/calculate_metrics.py
# This script compares experiment results to ground-truth labels and prints a classification report.
# It evaluates how well the predicted supported_ratio values align with true labels using a threshold.

import json
import argparse
from sklearn.metrics import classification_report

def main(results_path, questions_path, ratio_threshold):
    """Load experiment results and labeled questions, then compute evaluation metrics."""
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    print(f"Building label lookup from: {questions_path}")
    label_lookup = {}
    with open(questions_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            label_lookup[item['question']] = item['label']
    print(f"Found {len(label_lookup)} ground-truth labels.")
    
    y_true, y_pred = [], []
    found_matches = 0

    # Match each result with its true label and predict based on threshold
    for result in results_data:
        question = result['question']
        if question in label_lookup:
            found_matches += 1
            true_label = label_lookup[question]
            supported_ratio = result.get('supported_ratio', 0.0)
            prediction = 1 if supported_ratio >= ratio_threshold else 0
            y_true.append(true_label)
            y_pred.append(prediction)

    if not y_true:
        print("\n--- ERROR ---")
        print("No matching questions found between results and labels file.")
        print("-----------------\n")
        return

    print(f"\n--- Classification Report ---")
    print(f"Found {found_matches} matching entries out of {len(results_data)} results.")
    print("-" * 30)
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=['false/hallucination (0)', 'true/supported (1)'],
        zero_division=0
    )
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate final classification metrics from a results JSON file.")
    parser.add_argument('results_path', type=str, help='Path to the JSON results file')
    parser.add_argument('questions_path', type=str, help='Path to the JSONL questions file with ground-truth labels')
    parser.add_argument('--ratio_threshold', type=float, default=0.5,
                        help="Threshold for supported_ratio to classify as supported (1).")
    args = parser.parse_args()
    main(args.results_path, args.questions_path, args.ratio_threshold)
