# scripts/simulate_active_learning.py
import json
import argparse
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Simulate and compare Active Learning vs. Random Sampling.")
    parser.add_argument("--raw_results", type=str, required=True, help="Path to the raw JSON file from the experiment (e.g., all_test_results_0_temp.json)")
    parser.add_argument("--human_labels", type=str, required=True, help="Path to the human-labeled JSONL file (your ground truth).")
    parser.add_argument("--top_k", type=int, default=10, help="The number of samples to select in each simulation.")
    parser.add_argument("--metric", type=str, default="variance", choices=["variance", "semantic_entropy"], help="Which uncertainty metric to use for sorting.")
    args = parser.parse_args()

    # --- Part 1: Load Human Labels into an "Oracle" ---
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
    
    total_hallucinations_in_pool = sum(1 for supported in oracle.values() if not supported)
    print(f"Loaded {len(oracle)} human-labeled answers. Found {total_hallucinations_in_pool} total hallucinations in the dataset.\n")

    # --- Part 2: Load the Raw Data with Uncertainty Scores ---
    print(f"--- Loading Raw Experiment Data from '{args.raw_results}' ---")
    unlabeled_pool = []
    try:
        with open(args.raw_results, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Raw results file not found at '{args.raw_results}'")
        return

    for question_data in raw_data:
        question = question_data['question']
        uncertainty_score = question_data.get(args.metric, 0.0)
        
        # Since all completions are the same, we only need to process one
        if question_data['completions']:
            completion = question_data['completions'][0]
            unlabeled_pool.append({
                "question": question,
                "completion": completion,
                "uncertainty": uncertainty_score
            })
    
    print(f"Created an 'unlabeled pool' of {len(unlabeled_pool)} items to select from (one per question due to duplicate answers).\n")

    # --- Part 3: Simulate Active Learning ---
    print(f"--- Simulating Active Learning (selecting top {min(args.top_k, len(unlabeled_pool))} most uncertain using '{args.metric}') ---")
    
    unlabeled_pool.sort(key=lambda x: x['uncertainty'], reverse=True)
    
    select_k = min(args.top_k, len(unlabeled_pool))
    selected_by_uncertainty = unlabeled_pool[:select_k]
    
    uncertainty_found_count = 0
    for item in selected_by_uncertainty:
        key = (item['question'], item['completion'])
        if not oracle.get(key, True):
            uncertainty_found_count += 1
            
    print(f"Active Learning selection found {uncertainty_found_count} hallucinations.")

    # --- Part 4: Simulate Random Sampling ---
    print(f"\n--- Simulating Random Sampling (selecting {select_k} at random) ---")
    
    random.seed(42)
    selected_by_random = random.sample(unlabeled_pool, select_k)
    
    random_found_count = 0
    for item in selected_by_random:
        key = (item['question'], item['completion'])
        if not oracle.get(key, True):
            random_found_count += 1
            
    print(f"Random Sampling selection found {random_found_count} hallucinations.")

    # --- Part 5: Final Report ---
    print("\n" + "="*30)
    print("  ACTIVE LEARNING SIMULATION REPORT")
    print("="*30)
    print(f"Total unique answers in pool: {len(unlabeled_pool)}")
    print(f"Total known hallucinations: {total_hallucinations_in_pool}")
    print("-" * 30)
    print(f"METHOD 1: Active Learning (using '{args.metric}')")
    print(f"  - Selected top {select_k} samples.")
    print(f"  - Found {uncertainty_found_count} hallucinations.")
    print("-" * 30)
    print(f"METHOD 2: Random Sampling")
    print(f"  - Selected {select_k} random samples.")
    print(f"  - Found {random_found_count} hallucinations.")
    print("="*30)

    if uncertainty_found_count > random_found_count:
        print("\nCONCLUSION: Active Learning was MORE EFFICIENT at finding hallucinations. Mission successful!")
    elif uncertainty_found_count == random_found_count:
        print("\nCONCLUSION: Active Learning was NOT more efficient than random sampling. This uncertainty metric may not be effective.")
    else:
        print("\nCONCLUSION: Active Learning performed WORSE than random sampling. This is a significant finding.")

if __name__ == "__main__":
    main()
