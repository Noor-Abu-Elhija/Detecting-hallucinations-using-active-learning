import json
import numpy as np

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_results(results):
    variances = []
    trivial_clusters = 0
    total_clusters = 0

    for q in results:
        # Overall variance
        if "variance" in q:
            variances.append(q["variance"])

        # Cluster-level stats (if available)
        if "cluster_variances" in q:
            for cluster_id, stats in q["cluster_variances"].items():
                total_clusters += 1
                if stats["count"] == 1:  # trivial cluster
                    trivial_clusters += 1

    avg_var = np.mean(variances) if variances else 0
    return {
        "num_questions": len(results),
        "avg_variance": avg_var,
        "trivial_clusters": trivial_clusters,
        "total_clusters": total_clusters,
        "trivial_ratio": trivial_clusters / total_clusters if total_clusters > 0 else 0,
    }

def compare_files(file_k2, file_k3):
    data_k2 = load_results(file_k2)
    data_k3 = load_results(file_k3)

    stats_k2 = analyze_results(data_k2)
    stats_k3 = analyze_results(data_k3)

    print("\n=== KMeans Variance Comparison ===")
    print(f"File (K=2): {file_k2}")
    print(f"  Questions: {stats_k2['num_questions']}")
    print(f"  Avg variance: {stats_k2['avg_variance']:.4f}")
    print(f"  Trivial clusters: {stats_k2['trivial_clusters']} / {stats_k2['total_clusters']} "
          f"({stats_k2['trivial_ratio']:.2%})")

    print(f"\nFile (K=3): {file_k3}")
    print(f"  Questions: {stats_k3['num_questions']}")
    print(f"  Avg variance: {stats_k3['avg_variance']:.4f}")
    print(f"  Trivial clusters: {stats_k3['trivial_clusters']} / {stats_k3['total_clusters']} "
          f"({stats_k3['trivial_ratio']:.2%})")

    print("\n=== Recommendation ===")
    if stats_k3["avg_variance"] < stats_k2["avg_variance"] * 0.9 and stats_k3["trivial_ratio"] < 0.5:
        print("✅ K=3 seems better (lower variance, reasonable clustering).")
    else:
        print("✅ K=2 seems more stable (K=3 adds trivial clusters or not enough variance reduction).")

if __name__ == "__main__":
    # update paths to your actual output files
    compare_files("out/kmeans_test_k2.json", "out/kmeans_test_k3.json")
