
#  Detecting Hallucinations Using Active Learning

This project explores **hallucination detection in Large Language Models (LLMs)** using a hybrid pipeline that combines **uncertainty-based metrics** and **active learning**.  
It integrates **Semantic Entropy**, **Entropy**, **Variance-based metrics**, and **Approximate Nearest Neighbor (ANN–NLI)** similarity to iteratively identify and label hallucination-prone samples.

---

## Overview

Large Language Models often produce **confident but incorrect statements** (“hallucinations”).  
This project aims to detect such outputs efficiently by leveraging multiple uncertainty signals and refining a hallucination classifier through **active learning loops**.

We extend the work of **Farquhar et al. (2024, *Nature*)**, introducing a **multi-metric selection strategy** that guides annotation toward the most informative examples.

---

## Methodology

### 1. Metrics Implemented

| Metric | Description |
|--------|--------------|
| **Entropy** | Measures token-level uncertainty from output probabilities. |
| **Semantic Entropy** | Estimates uncertainty in *semantic space* by comparing multiple paraphrased generations (Farquhar et al., 2024). |
| **Variance** | Computes embedding variance across multiple generations. |
| **Weighted Variance** | Embedding variance weighted by model confidence. |
| **k-Means Variance** | Clustering-based semantic spread estimation. |
| **ANN–NLI** | Checks factual consistency using Approximate Nearest Neighbor retrieval and Natural Language Inference. |

### 2. Active Learning Loop

The system iteratively refines the hallucination classifier using an **acquisition metric** `M ∈ {Entropy, SemanticEntropy, Variance, WeightedVariance, k-MeansVariance, ANN–NLI}`:

```text
Require: 
    Unlabeled pool U of questions,
    small labeled seed set L,
    number of iterations N,
    batch size k,
    acquisition metric M
Ensure:
    Trained classifier C that predicts hallucination likelihood

Initialize: Train a classifier C₀ on L
for i in 1...N:
    Compute M for all u ∈ U
    Select top-k samples with highest M(u)
    Query oracle for true labels
    Add labeled samples to L
    Retrain classifier Cᵢ

```

##  How to Run

You can reproduce the full pipeline using the following commands step-by-step:

```bash
# Clone the repository
git clone https://github.com/Noor-Abu-Elhija/Detecting-hallucinations-using-active-learning.git
cd Detecting-hallucinations-using-active-learning
```
#  Create and activate the Conda environment
```bash
conda env create -f environment.yml
conda activate project
  --metric all
```
# Run uncertainty metric evaluation (Entropy, Semantic Entropy, etc.)

python -u scripts/run_eval.py \
  --index_dir indexes/squad_v2 \
  --save_json out/all_test_results_1_temp.json \
  --embed_model all-mpnet-base-v2 \
  --num_generations 5 \
  --num_of_question 800 \
  --temperature 1 \
  --max_new_tokens 50 \
  --ann_threshold 0.55 \
  --k 3 \
  --metric all
```

#  Run the Active Learning loop
```bash
python src/active_selector.py \
  --dataset datasets/squad_train.json \
  --initial_labels datasets/labeled_seed.json \
  --metric SemanticEntropy \
  --iterations 5 \
  --batch_size 20 \
  --output out/active_learning_results.json
  --metric all
```
# View outputs
# Results are saved under:
# out/



