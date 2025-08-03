# utils/arg_parser.py

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Active Learning for Hallucination Detection")

    # Add parameters here
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of most uncertain samples to select')
    parser.add_argument('--uncertainty_metric', type=str, choices=['variance', 'entropy'],
                        default='variance', help='Which uncertainty metric to use')
    parser.add_argument('--num_neighbors', type=int, default=5,
                        help='Number of neighbors to compute uncertainty from')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name')

    return parser.parse_args()
