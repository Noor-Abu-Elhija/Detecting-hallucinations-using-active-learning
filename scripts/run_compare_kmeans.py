import json
import argparse
import numpy as np
from tabulate import tabulate  # pip install tabulate

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def summarize(results):
    overall_vars = [r.get("variance", 0.0) for r in results]
    avg = float(np.mean(overall_vars)) if overall_vars else 0.0
    std = float(np.std(overall_vars)) if overall_vars else 0.0
    return avg, std

def main():
    parser = argparse.ArgumentParser(description="Compare KMeans variance results between two runs")
    parser.add_argument("--k2_file", type=str, required=True, help="Path to results JSON with k=2")
    parser.add_argument("--k3_file", type=str, required=True, help="Path to results JSON with k=3")
    args = parser.parse_args()

    results_k2 = load_results(args.k2_file)
    results_k3 = load_results(args.k3_file)

    avg2, std2 = summarize(results_k2)
    avg3, std3 = summarize(results_k3)

    table = [
        ["k=2", len(results_k2), avg2, std2],
        ["k=3", len(results_k3), avg3, std3],
    ]
    print(tabulate(table, headers=["Run", "Num Questions", "Avg Variance", "Std Dev"], floatfmt=".4f"))

    # השוואה שורה-שורה (אופציונלי)
    print("\n--- Detailed comparison (first 5 questions) ---")
    for i, (r2, r3) in enumerate(zip(results_k2, results_k3)):
        if i >= 5:
            break
        print(f"\nQ{i+1}: {r2['question']}")
        print(f"  k=2 variance: {r2.get('variance')}")
        print(f"  k=3 variance: {r3.get('variance')}")

if __name__ == "__main__":
    main()
