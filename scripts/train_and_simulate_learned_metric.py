# scripts/train_and_simulate_learned_metric_v3_optimized.py
import json
import argparse
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser(description="Train a 'judge' model and simulate active learning (Optimized Version).")
    parser.add_argument("--raw_results", type=str, required=True, help="Path to the raw JSON file with all features.")
    parser.add_argument("--human_labels", type=str, required=True, help="Path to the human-labeled JSONL file.")
    parser.add_argument("--top_k", type=int, default=5, help="The number of samples to select.")
    args = parser.parse_args()

    print("--- Loading all available data ---")
    oracle = {}
    all_points = []

    try:
        with open(args.human_labels, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                key = (record['question'], record['completion'])
                oracle[key] = record['human_label_is_supported']
    except FileNotFoundError:
        print(f"FATAL: Human labels file not found at '{args.human_labels}'")
        return

    try:
        with open(args.raw_results, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Raw results file not found at '{args.raw_results}'")
        return

    total_hallucinations_in_pool = sum(1 for supported in oracle.values() if not supported)
    print(f"Loaded {len(oracle)} human-labeled answers. Found {total_hallucinations_in_pool} total hallucinations.\n")

    for question_data in raw_data:
        for detail in question_data.get('ann_details', []):
            nli_label = detail.get('nli_label', 'neutral')
            features = [
                question_data.get('variance', 0),
                question_data.get('semantic_entropy', 0),
                detail.get('max_similarity', 0),
                1 if nli_label == 'contradiction' else 0,
                1 if nli_label == 'neutral' else 0
            ]
            
            point = {
                "question": question_data['question'],
                "completion": detail['completion'],
                "features": features
            }
            all_points.append(point)

    training_points = [p for p in all_points if (p['question'], p['completion']) in oracle]
    
    print(f"--- Training a SMARTER 'Judge' model on {len(training_points)} labels ---")
    X_train = np.array([d['features'] for d in training_points])
    y_train = np.array([not oracle[(d['question'], d['completion'])] for d in training_points])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    judge_model = LogisticRegression(class_weight='balanced', random_state=42)
    judge_model.fit(X_train_scaled, y_train)
    print("Training complete.\n")

    # --- OPTIMIZATION: Prepare all data for batch prediction ---
    all_features = np.array([item['features'] for item in all_points])
    
    # --- OPTIMIZATION: Scale and Predict ONCE on the entire batch ---
    all_features_scaled = scaler.transform(all_features)
    all_uncertainty_scores = judge_model.predict_proba(all_features_scaled)[:, 1]
    
    # --- OPTIMIZATION: Add the scores back to the data ---
    for i, item in enumerate(all_points):
        item['uncertainty'] = all_uncertainty_scores[i]

    print(f"--- Simulating Active Learning with the 'Learned Metric' (selecting top {args.top_k}) ---")
    all_points.sort(key=lambda x: x['uncertainty'], reverse=True)
    selected_by_learned = all_points[:args.top_k]
    
    learned_found_count = 0
    for item in selected_by_learned:
        key = (item['question'], item['completion'])
        if key in oracle and not oracle.get(key, True):
            learned_found_count += 1
    
    print(f"Learned Metric selection found {learned_found_count} hallucinations.")

    print(f"\n--- Simulating Random Sampling (selecting {args.top_k} at random) ---")
    random.seed(42)
    selected_by_random = random.sample(all_points, args.top_k)
    
    random_found_count = 0
    for item in selected_by_random:
        key = (item['question'], item['completion'])
        if key in oracle and not oracle.get(key, True):
            random_found_count += 1
            
    print(f"Random Sampling selection found {random_found_count} hallucinations.")

    print("\n" + "="*30)
    print("  FINAL ACTIVE LEARNING REPORT (V3 - Optimized)")
    print("="*30)
    print(f"METHOD 1: Active Learning (using a SMARTER learned metric)")
    print(f"  - Selected top {args.top_k} samples.")
    print(f"  - Found {learned_found_count} hallucinations.")
    print("-" * 30)
    print(f"METHOD 2: Random Sampling")
    print(f"  - Selected {args.top_k} random samples.")
    print(f"  - Found {random_found_count} hallucinations.")
    print("="*30)

    if learned_found_count > random_found_count:
        print("\nCONCLUSION: SUCCESS! The new model trained on your labels is MORE EFFICIENT than random sampling.")
    else:
        print("\nCONCLUSION: The new model still couldn't find a strong pattern. We may need more data or better features.")

if __name__ == "__main__":
    main()
