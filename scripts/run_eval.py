# scripts/run_eval.py
import os, sys, json, argparse
from typing import List, Dict, Any, Tuple
os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.ann_verifier import ANNVerifier
from src.nli import NLI

# Optional Falcon backend (GPU recommended)
try:
    from src.falcon_generate_logprobs import load_falcon_model, generate_with_logprobs
    _HAS_FALCON_FILE = True
except Exception:
    _HAS_FALCON_FILE = False


# -----------------------------
# Generation backends
# -----------------------------
def _factual_prompt(q: str) -> str:
    # nudges small models to answer cleanly
    return f"Answer briefly and factually: {q}\nAnswer:"

def sample_completions_hf(
    prompt: str,
    model_name: str = "gpt2",
    num_return_sequences: int = 5,
    temperature: float = 0.8,
    max_new_tokens: int = 48,
) -> List[str]:
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt2 = _factual_prompt(prompt)
    inputs = tok(prompt2, return_tensors="pt")
    outputs = mdl.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )
    texts = tok.batch_decode(outputs, skip_special_tokens=True)
    return [t[len(prompt2):].strip() if t.startswith(prompt2) else t.strip() for t in texts]

def sample_completions_falcon(
    prompt: str,
    num_return_sequences: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 64,
) -> List[str]:
    if not _HAS_FALCON_FILE:
        raise RuntimeError("Falcon backend requested but src/falcon_generate_logprobs.py not found/importable.")
    tok, mdl = load_falcon_model("tiiuae/falcon-7b-instruct")
    prompt2 = _factual_prompt(prompt)
    completions, _logprobs = generate_with_logprobs(
        prompt=prompt2,
        model=mdl,
        tokenizer=tok,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    # strip the prompt we prepended (if present)
    return [c.strip() for c in completions]

# ---------------------------------------
# Simple semantic clustering + entropy H
# ---------------------------------------
def semantic_entropy_simple(
    texts: List[str],
    embedder: SentenceTransformer,
    sim_threshold: float = 0.85,
) -> Tuple[float, List[int]]:
    if not texts:
        return 0.0, []
    embs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    cluster_centroids = []
    ids = []
    for v in embs:
        if not cluster_centroids:
            cluster_centroids.append([v.copy(), 1])
            ids.append(0)
            continue
        cents = np.stack([c[0] for c in cluster_centroids], axis=0)
        sims = cents @ v
        j = int(np.argmax(sims))
        if float(sims[j]) >= sim_threshold:
            vec, cnt = cluster_centroids[j]
            new_cnt = cnt + 1
            new_vec = (vec * cnt + v) / new_cnt
            new_vec = new_vec / (np.linalg.norm(new_vec) + 1e-12)
            cluster_centroids[j] = [new_vec, new_cnt]
            ids.append(j)
        else:
            cluster_centroids.append([v.copy(), 1])
            ids.append(len(cluster_centroids) - 1)

    counts = {}
    for cid in ids:
        counts[cid] = counts.get(cid, 0) + 1
    n = len(texts)
    ps = np.array([c / n for c in counts.values()], dtype=np.float32)
    H = float(-(ps * np.log(ps + 1e-12)).sum())
    return H, ids

# -----------------
# Eval core logic
# -----------------
def evaluate(
    questions: List[Dict[str, Any]],
    trusted_corpus: List[str],
    embed_model: str = "all-MiniLM-L6-v2",
    gen_backend: str = "hf",         # "hf" | "falcon"
    gen_model: str = "gpt2",
    generations: int = 5,
    temperature: float = 0.8,
    ann_threshold: float = 0.90,
    agg: str = "any",                # "any" | "majority"
) -> Dict[str, Any]:
    embedder = SentenceTransformer(embed_model)
    trusted_emb = embedder.encode(trusted_corpus, convert_to_numpy=True).astype("float32")
    ann = ANNVerifier(trusted_emb)
    nli = NLI()

    if gen_backend == "falcon":
        gen_fn = lambda q: sample_completions_falcon(q, num_return_sequences=generations,
                                                     temperature=temperature, max_new_tokens=64)
    else:
        gen_fn = lambda q: sample_completions_hf(q, model_name=gen_model, num_return_sequences=generations,
                                                 temperature=temperature, max_new_tokens=48)

    preds, labels, per_question = [], [], []

    for q in questions:
        qtext = q["question"]
        lbl = q.get("label")

        completions = gen_fn(qtext)
        H, cluster_ids = semantic_entropy_simple(completions, embedder)

        comp_embs = embedder.encode(completions, convert_to_numpy=True).astype("float32")
        supported_flags, details = [], []
        for i, ce in enumerate(comp_embs):
            is_sup, max_sim, idxs = ann.verify(ce, k=5, threshold=ann_threshold)
            nearest_idx = int(idxs[0]) if len(idxs) else -1
            nearest_txt = trusted_corpus[nearest_idx] if nearest_idx >= 0 else ""
            label_nli, conf_nli, _ = nli.predict(premise=nearest_txt, hypothesis=completions[i])
            supported = bool(is_sup and label_nli == "entailment")
            supported_flags.append(supported)
            details.append({
                "completion": completions[i],
                "cluster_id": int(cluster_ids[i]),
                "max_cosine": float(max_sim),
                "nearest_text": nearest_txt,
                "nli_label": label_nli,
                "nli_conf": float(conf_nli),
                "supported": supported,
            })

        if agg == "any":
            pred_supported = (sum(supported_flags) >= 1)
        else:  # majority
            pred_supported = (sum(supported_flags) > (len(supported_flags) / 2))

        preds.append(int(pred_supported))
        if lbl is not None:
            labels.append(int(lbl))

        per_question.append({
            "question": qtext,
            "semantic_entropy": float(H),
            "completions": details,
            "pred_supported": int(pred_supported),
            "label": int(lbl) if lbl is not None else None,
        })

    out = {"per_question": per_question}
    if labels:
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        out["metrics"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "report": classification_report(labels, preds, target_names=["false/hallucination", "true/supported"], zero_division=0)
        }
    return out

# -------------
# CLI
# -------------
def load_questions(path: str | None) -> List[Dict[str, Any]]:
    if path is None:
        return [
            {"question": "Who invented the light bulb?", "label": 1},
            {"question": "Where is the Eiffel Tower located?", "label": 1},
            {"question": "Is the Moon made of cheese?", "label": 0},
        ]
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def load_trusted(path: str | None) -> List[str]:
    if path is None:
        return [
            "Thomas Edison invented the light bulb.",
            "The Eiffel Tower is in Paris.",
            "Paris is the capital of France.",
            "Water boils at 100 degrees Celsius.",
            "Mount Everest is the tallest mountain.",
            "The Moon is a natural satellite of Earth.",
        ]
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--trusted", type=str, default=None)
    p.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--gen_backend", type=str, default="hf", choices=["hf", "falcon"])
    p.add_argument("--gen_model", type=str, default="gpt2")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--ann_threshold", type=float, default=0.90)
    p.add_argument("--agg", type=str, default="any", choices=["any", "majority"])
    p.add_argument("--save_json", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    questions = load_questions(args.dataset)
    trusted = load_trusted(args.trusted)

    results = evaluate(
        questions=questions,
        trusted_corpus=trusted,
        embed_model=args.embed_model,
        gen_backend=args.gen_backend,
        gen_model=args.gen_model,
        generations=args.generations,
        temperature=args.temperature,
        ann_threshold=args.ann_threshold,
        agg=args.agg,
    )

    if "metrics" in results:
        print("\n=== Metrics ===")
        for k, v in results["metrics"].items():
            if k == "report":
                print("\n" + v)
            else:
                print(f"{k}: {v:.3f}")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to {args.save_json}")

if __name__ == "__main__":
    main()
