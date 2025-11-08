# scripts/final_active_learning_loop.py
# This script runs a full Active Learning iteration: trains a model on labeled data,
# selects the most uncertain samples, and requests human (oracle) labels for them.

import json
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def get_human_label(question, completion, nearest_sentence):
    """Ask the human oracle to label whether the model's answer is factually supported."""
    print("\n" + "="*30)
    print("  ORACLE: Please label the following sample:")
    print("="*30)
    print(f"  QUESTION: {question}")
    print(f"  Generated Answer: '{completion}'")
    print(f"  Nearest Sentence: '{nearest_sentence}'")
    
    while True:
        user_input = input("  Is this answer factually supported? (y/n): ").lower()
        if user_input in ['y', 'n']:
            return user_input == 'y'
        else:
            print("  Invalid input. Please enter 'y' or 'n'.")

def main():
    """Run one iteration of an active learning loop for hallucination detection."""
    parser = argparse.ArgumentParser(description="Run one full iteration of the Active Learning loop.")
    parser.add_argument("--raw_results", type=str, required=True, help="Path to the raw JSON file.")
    parser.add_argument("--initial_labels", type=str, required=True, help="Path to the initial human-labeled JSONL file.")
    parser.add_argument("--budget", type=int, default=5, help="Number of new samples to label in this iteration.")
    args = parser.parse_args()

    print("--- Loading all data points from raw results ---")
    all_points = {}
    with open(args.raw_results, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    for question_data in raw_data:
        question = question_data['question']
        for detail in question_data.get('ann_details', []):
            completion = detail['completion']
            key = (question, completion)
            nli_label = detail.get('nli_label', 'neutral')
            features = [
                question_data.get('variance', 0),
                question_data.get('semantic_entropy', 0),
                detail.get('max_similarity', 0),
                1 if nli_label == 'contradiction' else 0,
                1 if nli_label == 'neutral' else 0
            ]
            all_points[key] = {
                "features": features,
                "nearest_sentence": detail.get('nearest_sentence', '')
            }

    print(f"\n--- Partitioning data into Labeled (L) and Unlabeled (U) sets ---")
    labeled_keys = set()
    with open(args.initial_labels, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            labeled_keys.add((record['question'], record['completion']))

    unlabeled_keys = [k for k in all_points.keys() if k not in labeled_keys]
    
    print(f"Found {len(labeled_keys)} initial labels. Training initial model 'M'...")
    training_points = [p for key, p in all_points.items() if key in labeled_keys]
    
    oracle_labels = {}
    with open(args.initial_labels, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            oracle_labels[(record['question'], record['completion'])] = record['human_label_is_supported']

    X_train = np.array([p['features'] for p in training_points])
    y_train = np.array([not oracle_labels[(list(labeled_keys)[i])] for i, p in enumerate(training_points)])

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    model = LogisticRegression(class_weight='balanced', ra_
