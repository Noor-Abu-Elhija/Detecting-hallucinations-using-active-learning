# scripts/test_regular_entropy.py
import numpy as np
from src.regular_entropy import sequence_token_entropy, batch_regular_entropy

# Create a fake (T,V) prob matrix for 2 completions
rng = np.random.default_rng(0)
probs1 = rng.random((5, 10)); probs1 /= probs1.sum(axis=1, keepdims=True)
probs2 = rng.random((7, 10)); probs2 /= probs2.sum(axis=1, keepdims=True)

avg1, steps1 = sequence_token_entropy(probs1, normalize=True)
avg2, steps2 = sequence_token_entropy(probs2, normalize=True)

mean_over, per_seq = batch_regular_entropy([probs1, probs2], normalize=True)

print('avg1:', round(avg1,4), 'avg2:', round(avg2,4))
print('mean_over:', round(mean_over,4), 'per_seq:', [round(x,4) for x in per_seq.tolist()])
