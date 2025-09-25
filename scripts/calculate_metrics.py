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
        questions_data = {item['question']: item['label'] for item in [json.loads(line) for line in f]}

    y_true = []  # The ground-truth labels (0 or 1)
    y_pred = []  # Your system's predictions (0 or 1)

    for result in results_data:
        question = result['question']

        # This is where we make the final decision for the whole question
        # If the ratio of supported answers is >= our threshold, we predict 1 (true/supported)
        # This is the equivalent of '--agg majority' if the threshold is 0.5
        supported_ratio = result.get('supported_ratio', 0.0)
        prediction = 1 if supported_ratio >= ratio_threshold else 0

        # Find the true label for this question
        true_label = questions_data.get(question)

        if true_label is not None:
            y_true.append(true_label)
            y_pred.append(prediction)

    print(f"\n--- Classification Report ---")
    print(f"Results file: {results_path}")
    print(f"Decision rule: Predicted 'supported' if supported_ratio >= {ratio_threshold}")
    print("-" * 30)

    # Use scikit-learn to generate a beautiful report
    report = classification_report(
        y_true,
        y_pred,
        target_names=['false/hallucination (0)', 'true/supported (1)'],
        zero_division=0
    )
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate final metrics from a results JSON file.")
    parser.add_argument('results_path', type=str, help='Path to the JSON results file from run_final_experiment.py')
    parser.add_argument('questions_path', type=str, help='Path to the original JSONL questions file with labels')
    parser.add_argument('--ratio_threshold', type=float, default=0.5,
                        help="Threshold for 'supported_ratio' to be considered a 'supported' prediction.")
    args = parser.parse_args()

    main(args.results_path, args.questions_path, args.ratio_threshold)