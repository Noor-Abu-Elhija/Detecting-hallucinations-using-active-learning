# scripts/train_and_simulate_learned_metric_v3_optimized.py
# This script trains a Logistic Regression model (a “judge”) to predict hallucination likelihood
# using features like variance, entropy, similarity, and NLI labels — then simulates active learning
# to compare learned selection vs random sampling in finding hallucinations efficiently.

import json
import argparse
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def main():
    """Train a learned uncertainty metric and simulate an Active Learning round."""
    parser = argparse.ArgumentParser(description="Train a 'judge' model and simulate active learning (Optimized Version).")
    parser.add_argument("--raw_results", type=str, required=True, help="Path to the JSON file with extracted features.")
    parser.add_argument("--human_labels", type=str, required=True, help="Path to the human-labeled JSONL file.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of samples to select per round.")
    args = parser.parse_args()

    print("--- Loading data and human labels ---")
    oracle = {}
    all_points = []

    # Load human labels (ground truth)
    try:
        with open(args.human_labels, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                oracle[(record['question'], record['completion'])] = record['human_label_is_supported']
    except FileNotFoundError:
        print(f"ERROR: Human labels file not found at '{args.human_labels}'")
        return

    # Load raw experimental data
    try:
        with open(args.raw_results, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Raw results file not found at '{args.raw_results}'")
        return

    total_hallucinations = sum(1 for supported in oracle.values() if not supported)
    print(f"Loaded {len(oracle)} labels ({total_hallucinations} hallucinations detected).\n")

    # Extract features
    for q in raw_data:
        for d in q.get('ann_details', []):
            nli_label = d.get('nli_label', 'neutral')
            features = [
                q.get('variance', 0),
                q.get('semantic_entropy', 0),
                d.get('max_similarity', 0),
                1 if nli_label == 'contradiction' else 0,
                1 if nli_label == 'neutral' else 0
            ]
            all_points.append({
                "question": q['question'],
                "completion": d['completion'],
                "features": features
            })

    # Train model on labeled subset
    labeled_data = [p for p in all_points if (p['question'], p['completion']) in oracle]
    print(f"--- Training learned 'Judge' model on {len(labeled_data)} samples ---")

    X_train = np.array([d['features'] for d in labeled_data])
    y_train = np.array([not oracle[(d['question'], d['completion'])] for d in labeled_data])

    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(class_weight='balanced', random_state=42).fit(scaler.transform(X_train), y_train)
    print("Model training complete.\n")

    # Predict uncertainty on all samples
    X_all = np.array([p['features'] for p in all_points])
    all_probs = model.predict_proba(scaler.transform(X_all))[:, 1]

    for i, p in enumerate(all_points):
        p['uncertainty'] = all_probs[i]

    # Active learning simulation
    print(f"--- Simulating Active Learning (top {args.top_k} by model uncertainty) ---")
    all_points.sort(key=lambda x: x['uncertainty'], reverse=True)
    top_uncertain = all_points[:args.top_k]
    learned_found = sum(
        not oracle.get((p['question'], p['completion']), True)
        for p in top_uncertain
    )
    print(f"Learned Metric found {learned_found} hallucinations.")

    # Random sampling simulation
    print(f"\n--- Simulating Random Sampling (selecting {args.top_k} at random) ---")
    random.seed(42)
    random_found = sum(
        not oracle.get((p['question'], p['completion']), True)
        for p in random.sample(all_points, args.top_k)
    )
    print(f"Random Sampling found {random_found} hallucinations.\n")

    # Summary
    print("=" * 30)
    print("  FINAL ACTIVE LEARNING REPORT (V3 - Optimized)")
    print("=" * 30)
    print(f"Active Learning (Learned Metric): {learned_found}/{args.top_k}")
    print(f"Random Sampling: {random_found}/{args.top_k}")
    print("=" * 30)

    if learned_found > random_found:
        print("\n✅ CONCLUSION: The learned model is MORE EFFICIENT at discovering hallucinations!")
    else:
        print("\n⚠️ CONCLUSION: The learned model was NOT superior — more data or features may be needed.")

if __name__ == "__main__":
    main()
