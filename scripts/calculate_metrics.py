# scripts/calculate_metrics.py
import json
import argparse
from sklearn.metrics import classification_report

def main(results_path, questions_path, ratio_threshold):
    # Load the detailed results from your experiment
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    # --- THIS IS THE NEW, SMARTER LOGIC ---
    # Load the original questions file and turn it into a lookup dictionary (an "answer key")
    # The key will be the question string, and the value will be the label (0 or 1).
    print(f"Building label lookup from: {questions_path}")
    label_lookup = {}
    with open(questions_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            label_lookup[item['question']] = item['label']
    print(f"Found {len(label_lookup)} ground-truth labels.")
    
    y_true = []  # The ground-truth labels
    y_pred = []  # Your system's predictions
    found_matches = 0

    # Loop through the results from your (potentially shuffled) experiment
    for result in results_data:
        question = result['question']
        
        # Check if this question exists in our answer key
        if question in label_lookup:
            found_matches += 1
            true_label = label_lookup[question]
            
            supported_ratio = result.get('supported_ratio', 0.0)
            prediction = 1 if supported_ratio >= ratio_threshold else 0
            
            y_true.append(true_label)
            y_pred.append(prediction)

    if not y_true:
        print("\n--- ERROR ---")
        print("Could not generate a report because NO matching questions were found between the results and the labels file.")
        print("-----------------\n")
        return

    print(f"\n--- Classification Report ---")
    print(f"Found {found_matches} matching entries to evaluate out of {len(results_data)} results.")
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
