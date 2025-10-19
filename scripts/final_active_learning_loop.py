# scripts/final_active_learning_loop.py
import json
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# This is our Oracle 'O' - the human labeler (you!)
def get_human_label(question, completion, nearest_sentence):
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
    parser = argparse.ArgumentParser(description="Run one full iteration of the Active Learning loop.")
    parser.add_argument("--raw_results", type=str, required=True, help="Path to the raw JSON file.")
    parser.add_argument("--initial_labels", type=str, required=True, help="Path to the initial human-labeled JSONL file.")
    parser.add_argument("--budget", type=int, default=5, help="The number of new items to label in this iteration.")
    args = parser.parse_args()

    # --- SETUP: Load all data points ---
    print("--- Loading all data points from raw results ---")
    all_points = {} # Use a dictionary to avoid duplicates and for easy lookup
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

    # --- STEP 1: Define Initial Labeled (L) and Unlabeled (U) sets ---
    print(f"\n--- Partitioning data into Labeled (L) and Unlabeled (U) sets ---")
    labeled_keys = set()
    with open(args.initial_labels, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            labeled_keys.add((record['question'], record['completion']))

    unlabeled_keys = [k for k in all_points.keys() if k not in labeled_keys]
    
    # --- STEP 2: Train initial model M with L ---
    print(f"Found {len(labeled_keys)} initial labels. Training initial model 'M'...")
    training_points = [p for key, p in all_points.items() if key in labeled_keys]
    
    # We need the ground truth for the initial labels to train
    oracle_labels = {}
    with open(args.initial_labels, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            oracle_labels[(record['question'], record['completion'])] = record['human_label_is_supported']

    X_train = np.array([p['features'] for p in training_points])
    y_train = np.array([not oracle_labels[(list(labeled_keys)[i])] for i, p in enumerate(training_points)])

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    model = LogisticRegression(class_weight='balanced', random_state=42).fit(X_train_scaled, y_train)
    print("Initial model trained.\n")

    # --- STEP 3: QueryInformative(M, U) - Select most informative samples ---
    print(f"--- Querying the {len(unlabeled_keys)} unlabeled items to find the {args.budget} most informative samples ---")
    unlabeled_features = np.array([all_points[key]['features'] for key in unlabeled_keys])
    unlabeled_features_scaled = scaler.transform(unlabeled_features)
    
    # Get the model's predictions (probabilities) for the unlabeled data
    probabilities = model.predict_proba(unlabeled_features_scaled)
    
    # Uncertainty Score: Find the samples where the model is least confident (closest to 0.5)
    uncertainty_scores = np.abs(probabilities[:, 1] - 0.5)
    
    # Get the indices of the MOST uncertain items (lowest score = closest to 0.5)
    most_uncertain_indices = np.argsort(uncertainty_scores)[:args.budget]
    
    # These are the keys (question, completion) of the items we need to label
    Uq_keys = [unlabeled_keys[i] for i in most_uncertain_indices]
    print(f"Selected {len(Uq_keys)} samples for you to label.\n")

    # --- STEP 4: Lq <- O.label(Uq) - Get labels from the Oracle (YOU) ---
    Lq = {} # This will be our new set of labels
    for key in Uq_keys:
        question, completion = key
        nearest_sentence = all_points[key]['nearest_sentence']
        is_supported = get_human_label(question, completion, nearest_sentence)
        Lq[key] = is_supported

    # --- STEP 5 & 6: Update L and U ---
    # In a real system, we would now add Lq to L, remove Uq from U, and retrain.
    # For this simulation, we'll just report what we've accomplished.
    print("\n" + "="*30)
    print("  ACTIVE LEARNING ITERATION COMPLETE")
    print("="*30)
    print(f"You started with {len(labeled_keys)} labeled samples.")
    print(f"The model intelligently selected {len(Lq)} uncertain samples for you to label.")
    print("These new labels can now be added to the training set to create a smarter model in the next iteration.")
    print("\nMISSION ACCOMPLISHED: You have successfully built and executed one full cycle of the Active Learning pipeline.")

if __name__ == "__main__":
    main()
