# scripts/simulate_active_learning.py
# This script simulates an Active Learning cycle by comparing uncertainty-based selection
# against random sampling in discovering hallucinated answers using human-labeled ground truth.

import json
import argparse
import random
import numpy as np

def main():
    """Simulate Active Learning vs Random Sampling using uncertainty scores and human labels."""
    parser = argparse.ArgumentParser(description="Simulate and compare Active Learning vs. Random Sampling.")
    parser.add_argument("--raw_results", type=str, required=True, help="Path to the raw JSON results file.")
    parser.add_argument("--human_labels", type=str, required=True, help="Path to the human-labeled JSONL file (ground truth).")
    parser.add_argument("--top_k", type=int, default=10, help="Number of samples to select per simulation.")
    parser.add_argument("--metric", type=str, default="variance", choices=["variance", "semantic_entropy"],
                        help="Uncertainty metric used for sorting.")
    args = parser.parse_args()

    # --- Load human labels (the Oracle) ---
    print(f"--- Loading Human-Labeled Data from '{args.human_labels}' ---")
    oracle = {}
    try:
        with open(args.human_labels, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                key = (record['question'], record['completion'])
                oracle[key] = record['human_label_is_supported']
    except FileNotFoundError:
        print(f"FATAL: Human labels file not found at '{args.human_labels}'")
        return

    total_hallucinations = sum(1 for supported in oracle.values() if not supported)
    print(f"Loaded {len(oracle)} labeled samples ({total_hallucinations} hallucinations).\n")

    # --- Load raw data containing uncertainty metrics ---
    print(f"--- Loading Raw Experiment Data from '{args.raw_results}' ---")
    unlabeled_pool = []
    try:
        with open(args.raw_results, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Raw results file not found at '{args.raw_results}'")
        return

    for q_data in raw_data:
        question = q_data['question']
        uncertainty_score = q_data.get(args.metric, 0.0)
        if q_data.get('completions'):
            completion = q_data['completions'][0]
            unlabeled_pool.append({
                "question": question,
                "completion": completion,
                "uncertainty": uncertainty_score
            })

    print(f"Created unlabeled pool of {len(unlabeled_pool)} items.\n")

    # --- Active Learning Simulation (most uncertain samples) ---
    print(f"--- Simulating Active Learning (top {min(args.top_k, len(unlabeled_pool))} by '{args.metric}') ---")
    unlabeled_pool.sort(key=lambda x: x['uncertainty'], reverse=True)
    select_k = min(args.top_k, len(unlabeled_pool))
    selected_by_uncertainty = unlabeled_pool[:select_k]

    uncertainty_found = sum(
        not oracle.get((item['question'], item['completion']), True)
        for item in selected_by_uncertainty
    )
    print(f"Active Learning found {uncertainty_found} hallucinations.")

    # --- Random Sampling Simulation ---
    print(f"\n--- Simulating Random Sampling (selecting {select_k} at random) ---")
    random.seed(42)
    selected_random = random.sample(unlabeled_pool, select_k)

    random_found = sum(
        not oracle.get((item['question'], item['completion']), True)
        for item in selected_random
    )
    print(f"Random Sampling found {random_found} hallucinations.")

    # --- Summary Report ---
    print("\n" + "="*30)
    print("  ACTIVE LEARNING SIMULATION REPORT")
    print("="*30)
    print(f"Total unique answers in pool: {len(unlabeled_pool)}")
    print(f"Total hallucinations in ground truth: {total_hallucinations}")
    print("-" * 30)
    print(f"Active Learning ({args.metric}): {uncertainty_found} / {select_k} found")
    print(f"Random Sampling: {random_found} / {select_k} found")
    print("="*30)

    if uncertainty_found > random_found:
        print("\nCONCLUSION: Active Learning was MORE EFFICIENT at finding hallucinations.")
    elif uncertainty_found == random_found:
        print("\nCONCLUSION: Active Learning was EQUAL to random sampling — metric not informative.")
    else:
        print("\nCONCLUSION: Active Learning performed WORSE than random sampling.")

if __name__ == "__main__":
    main()
