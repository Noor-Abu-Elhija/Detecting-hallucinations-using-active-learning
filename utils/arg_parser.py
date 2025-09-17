# utils/arg_parser.py
import argparse

# --------------------------------------------------------------------
# Back-compat: your original args used by active_learning_loop & tests
# --------------------------------------------------------------------
def get_args():
    """
    Legacy arguments for scripts that already import utils.arg_parser.get_args()
    (e.g., active_learning_loop.py, test_feature_extraction.py, etc.)
    """
    parser = argparse.ArgumentParser(description="Active Learning for Hallucination Detection (legacy)")

    # Active-learning toy loop
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of most uncertain samples to select')
    parser.add_argument('--uncertainty_metric', type=str, choices=['variance', 'entropy'],
                        default='variance', help='Which uncertainty metric to use')
    parser.add_argument('--num_neighbors', type=int, default=5,
                        help='Number of neighbors to compute uncertainty from')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name for embeddings')

    # Falcon7B parameters (used by your Falcon scripts)
    parser.add_argument('--num_generations', type=int, default=5,
                        help='How many answers to generate per question')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='Sampling temperature for generation (higher = more diverse)')
    parser.add_argument('--question', type=str,
                        help='Question to answer')

    # === Regular entropy demo args (for scripts/test_regular_entropy.py) ===
    # Note: we REUSE the existing --temperature above; no duplicate flag added.
    parser.add_argument("--model_id", type=str, default="tiiuae/falcon-7b-instruct",
                        help="HF causal LM to use; try 'gpt2' on CPU.")
    parser.add_argument("--prompt", type=str, default="The capital of France is",
                        help="Prompt to sample completions from.")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="How many completions to sample for entropy computation.")
    parser.add_argument("--max_new_tokens", type=int, default=24,
                        help="Maximum tokens to generate per completion.")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling parameter (keep top cumulative probability mass).")

    # Embedding variance options (for scripts/generate_embedding_variance.py)
    parser.add_argument('--k', type=int, default=0,
                        help='If >0, run KMeans with k clusters and report per-cluster variance')
    parser.add_argument('--weighted', action='store_true',
                        help='Weight variance by sequence probabilities')

    return parser.parse_args()


# --------------------------------------------------------------------
# New: Full-pipeline (demo) arguments
# --------------------------------------------------------------------
def get_full_pipeline_args():
    """
    Arguments tailored for scripts/full_pipeline.py
    """
    p = argparse.ArgumentParser(description="Full pipeline: entropy → ANN → NLI (demo)")

    p.add_argument('--top_k', type=int, default=2,
                   help='How many most-uncertain responses to verify')
    p.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2',
                   help='SentenceTransformer model for embeddings')
    p.add_argument('--trusted', type=str, default=None,
                   help='Path to TXT with trusted sentences (one per line). If None, use built-in small list')
    p.add_argument('--ann_threshold', type=float, default=0.92,
                   help='Cosine similarity threshold for ANN support (0..1)')
    p.add_argument('--nli_model', type=str, default='facebook/bart-large-mnli',
                   help='HF model for NLI (entailment/neutral/contradiction)')
    return p.parse_args()


# --------------------------------------------------------------------
# New: Evaluation harness arguments (run_eval.py)
# --------------------------------------------------------------------
def get_eval_args():
    """
    Arguments for scripts/run_eval.py (evaluation + metrics).
    """
    p = argparse.ArgumentParser(description="Evaluation: generation → semantic entropy → ANN+NLI → metrics")

    # Data
    p.add_argument('--dataset', type=str, default=None,
                   help="Path to JSONL with {'question':..., 'label':0/1 (optional)}")
    p.add_argument('--trusted', type=str, default=None,
                   help='Path to TXT with trusted sentences. If None, use built-in small list')

    # Embeddings / Retrieval
    p.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2',
                   help='SentenceTransformer model for embeddings (e.g., all-mpnet-base-v2)')
    p.add_argument('--ann_threshold', type=float, default=0.90,
                   help='Cosine similarity threshold for support (0..1)')
    p.add_argument('--k', type=int, default=5,
                   help='Top-k neighbors to retrieve from ANN')

    # Generation backend
    p.add_argument('--gen_backend', type=str, default='hf', choices=['hf', 'falcon'],
                   help='hf = small HF models on CPU; falcon = Falcon-7B-Instruct (GPU)')
    p.add_argument('--gen_model', type=str, default='gpt2',
                   help='HF model name when gen_backend=hf (e.g., gpt2, TinyLlama/TinyLlama-1.1B-Chat-v1.0)')
    p.add_argument('--generations', type=int, default=5,
                   help='Number of completions per question (M)')
    p.add_argument('--temperature', type=float, default=0.8,
                   help='Sampling temperature for generation')

    # Semantic entropy (simple cosine cluster)
    p.add_argument('--sim_threshold', type=float, default=0.85,
                   help='Cosine threshold for clustering completions when computing entropy')

    # Decision aggregation at question level
    p.add_argument('--agg', type=str, default='any', choices=['any', 'majority'],
                   help="How to aggregate completion-level support into a question prediction")

    # NLI
    p.add_argument('--nli_model', type=str, default='facebook/bart-large-mnli',
                   help='HF model for entailment/contradiction')

    # Output
    p.add_argument('--save_json', type=str, default=None,
                   help='Optional path to save full JSON results')



    return p.parse_args()
