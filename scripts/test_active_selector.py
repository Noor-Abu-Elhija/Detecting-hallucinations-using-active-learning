# scripts/test_active_selector.py

import numpy as np
from src.active_selector import select_by_uncertainty, select_random
from utils.arg_parser import get_args

def test_select_by_uncertainty(top_k):
    uncertainties = np.array([0.1, 0.3, 0.05, 0.9, 0.6])
    selected = select_by_uncertainty(uncertainties, top_k)
    print(f"Top-{top_k} selected by uncertainty: {selected} (scores: {uncertainties[selected]})")

def test_select_random(top_k):
    unlabeled_size = 10
    selected = select_random(unlabeled_size, top_k, seed=123)
    print(f"Random {top_k} selection (seed=123): {selected}")

if __name__ == "__main__":
    args = get_args()
    test_select_by_uncertainty(args.top_k)
    test_select_random(args.top_k)
