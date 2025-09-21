# utils/arg_parser.py
import argparse

def get_args(argv=None):
    p = argparse.ArgumentParser(
        description="Active Learning + Evaluation: generation → entropy/variance → ANN+NLI"
    )

    # ---------- Data & retrieval ----------
    g_data = p.add_argument_group("Data & Retrieval")
    g_data.add_argument('--dataset', type=str, default=None,
                        help="Path to JSONL with {'question':..., 'label':0/1 (optional)}")
    g_data.add_argument('--index_dir', type=str, default=None,
                        help="Path to ANN/FAISS index dir (required for --metric ann)")
    g_data.add_argument('--trusted', type=str, default=None,
                        help="TXT with trusted sentences (one per line)")

    # Embedding model (keep both names for back-compat)
    g_data.add_argument('--embed_model', '--embedding_model', dest='embed_model',
                        type=str, default='all-MiniLM-L6-v2',
                        help='SentenceTransformer model for embeddings')

    # Shared 'k' (used by ANN top-k and by some variance scripts)
    g_data.add_argument('--k', type=int, default=5,
                        help='Top-k neighbors for ANN retrieval (some scripts reuse as generic k)')

    g_data.add_argument('--ann_threshold', type=float, default=0.90,
                        help='Cosine similarity threshold for ANN support (0..1)')

    # ---------- Metrics & decisions ----------
    g_metrics = p.add_argument_group("Metrics & Decisions")
    g_metrics.add_argument('--metric', type=str, default=None,  # make explicit for eval harness
                           choices=['semantic_entropy', 'entropy', 'variance',
                                    'weighted variance', 'kmeans variance', 'ann'],
                           help="Which metric to compute")
    g_metrics.add_argument('--sim_threshold', type=float, default=0.85,
                           help='Cosine threshold for clustering in semantic entropy')
    g_metrics.add_argument('--agg', type=str, default='any', choices=['any', 'majority'],
                           help="How to aggregate completion-level support into a question prediction")

    # ---------- Generation ----------
    g_gen = p.add_argument_group("Generation")
    g_gen.add_argument('--gen_backend', type=str, default='falcon', choices=['hf', 'falcon'],
                       help='hf = small HF models on CPU; falcon = Falcon-7B-Instruct')
    g_gen.add_argument('--gen_model', type=str, default='gpt2',
                       help='HF model to use when gen_backend=hf')
    # keep both flags as aliases to the same dest
    g_gen.add_argument('--num_generations', '--generations', dest='num_generations',
                       type=int, default=5, help='How many answers per question')
    g_gen.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    g_gen.add_argument('--max_new_tokens', type=int, default=64,
                       help='Max tokens to generate per completion')
    g_gen.add_argument('--top_p', type=float, default=0.95,
                       help='Nucleus sampling (used by some sampling demos)')
    g_gen.add_argument('--question', type=str,
                       help='Single-question entrypoint for certain demos')

    # ---------- Models ----------
    g_models = p.add_argument_group("Models")
    g_models.add_argument('--nli_model', type=str, default='facebook/bart-large-mnli',
                          help='HF model for NLI (entail/neutral/contradict)')

    # ---------- Embedding-variance specific (legacy) ----------
    g_var = p.add_argument_group("Embedding Variance (legacy scripts)")
    g_var.add_argument('--weighted', action='store_true',
                       help='Weight variance by sequence probabilities')
    g_var.add_argument('--kmeans_k', '--variance_k', dest='variance_k', type=int, default=0,
                       help='If >0, run KMeans with this k and report per-cluster variance')

    # ---------- Output ----------
    g_out = p.add_argument_group("Output")
    g_out.add_argument('--save_json', type=str, default=None,
                       help='Path to save full JSON results')

    # Parse
    args = p.parse_args(argv)

    # Optional guard that helps users:
    if args.metric == 'ann' and not args.index_dir:
        p.error("--index_dir is required when --metric ann")

    return args
