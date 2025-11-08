import json
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

METRIC_TO_COLUMN_MAP = {
    'ann': 'supported_ratio',
    'semantic_entropy': 'semantic_entropy',
    'entropy': 'entropy',
    'variance': 'variance',
    'kmeans_variance': 'kmeans_variance',
    'weighted_variance': 'weighted_variance'
}

def run_simulation(data_filepath, metric_name):
    """
    Loads data, runs the active learning loop, and saves the FULL, updated dataset.
    """
    print(f"Loading data from: {os.path.basename(data_filepath)}")
    df = pd.read_json(data_filepath)

    feature_column_name = METRIC_TO_COLUMN_MAP[metric_name]
    print(f"Using feature column: '{feature_column_name}' for metric '{metric_name}'")

    # --- Prepare the single 'feature' column for the model ---
    if metric_name == 'entropy' and isinstance(df[feature_column_name].iloc[0], list):
        df['feature'] = df[feature_column_name].apply(lambda x: x[0] if isinstance(x, list) and x else np.nan)
    else:
        df['feature'] = df[feature_column_name]

    # --- Separate data without deleting anything ---
    # This correctly handles boolean True/False and leaves other values (like "null") as NaN
    label_map = {True: 1, False: 0, 'true': 1, 'false': 0}
    df['is_hallucination'] = df['is_hallucination'].map(label_map)
    
    # Clean only rows that are missing the essential feature value
    df.dropna(subset=['feature'], inplace=True)

    labeled_pool = df[df['is_hallucination'].notna()].copy()
    unlabeled_pool = df[df['is_hallucination'].isna()].copy()
    
    labeled_pool['is_hallucination'] = labeled_pool['is_hallucination'].astype(int)

    print(f"\n--- Starting Active Learning for Metric: '{metric_name}' ---")
    print(f"Initial Labeled Pool (L) size: {len(labeled_pool)}")
    print(f"Unlabeled Pool (U) size: {len(unlabeled_pool)}")

    iteration = 0
    QUERY_SIZE = 5
    while len(unlabeled_pool) > 0:
        iteration += 1
        
        # 1. Train model on the current labeled pool
        X_train = labeled_pool[['feature']].values
        y_train = labeled_pool['is_hallucination'].values
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train)

        # 2. Select queries from unlabeled pool based on feature uncertainty
        query_size = min(QUERY_SIZE, len(unlabeled_pool))
        if metric_name == 'ann':
            uncertainty_scores = abs(unlabeled_pool['feature'])
            query_indices = uncertainty_scores.nsmallest(query_size).index
        else:
            uncertainty_scores = unlabeled_pool['feature']
            query_indices = uncertainty_scores.nlargest(query_size).index

        # 3. Get pseudo-labels for the queries
        X_to_predict = unlabeled_pool.loc[query_indices][['feature']].values
        X_to_predict_scaled = scaler.transform(X_to_predict)
        pseudo_labels = model.predict(X_to_predict_scaled)
        
        # 4. Update the pools
        newly_labeled_samples = unlabeled_pool.loc[query_indices].copy()
        newly_labeled_samples['is_hallucination'] = pseudo_labels
        
        labeled_pool = pd.concat([labeled_pool, newly_labeled_samples])
        unlabeled_pool = unlabeled_pool.drop(query_indices)
        
        print(f"Iter {iteration:3d}: Labeled Size = {len(labeled_pool):4d}, Unlabeled Remaining = {len(unlabeled_pool):4d}")

    # --- THE CRITICAL FIX: UPDATE THE ORIGINAL DATAFRAME ---
    print("\nLoop complete. Updating the original dataframe with final labels...")
    
    # Create a map from the question to its final predicted label
    final_label_map = labeled_pool.set_index('question')['is_hallucination']
    
    # Use the map to fill in the 'is_hallucination' column in the original full dataframe
    df['is_hallucination'] = df['question'].map(final_label_map)
    df.dropna(subset=['is_hallucination'], inplace=True) # Remove any rows that failed to map
    df['is_hallucination'] = df['is_hallucination'].astype(int)

    # Now, save the UPDATED ORIGINAL dataframe, which has all columns
    output_basename = os.path.basename(data_filepath).replace('.json', '')
    output_filename = f"{output_basename}_final_labels_by_{metric_name}.json"
    output_path = os.path.join(os.path.dirname(data_filepath), output_filename)
    
    # Drop the temporary 'feature' column before saving
    df.drop(columns=['feature'], inplace=True)
    df.to_json(output_path, orient='records', indent=2)
    
    print(f"\nSuccess! FULL labeled file saved to {output_path}")
    print(f"File size should now be similar to the original.")
    print(f"Final counts: {df['is_hallucination'].value_counts().to_dict()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the final, corrected Active Learning loop.")
    parser.add_argument("--data-file", required=True, help="Name of the main data file.")
    parser.add_argument("--metric", required=True, choices=METRIC_TO_COLUMN_MAP.keys())
    
    args = parser.parse_args()
    
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    data_filepath = os.path.join(base_path, "out", args.data_file)

    run_simulation(data_filepath, args.metric)
