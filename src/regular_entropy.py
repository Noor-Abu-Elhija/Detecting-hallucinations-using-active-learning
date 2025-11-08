# src/regular_entropy.py
# This module computes regular (sequence-level) entropy for text generation models,
# quantifying uncertainty across sampled completions from an LLM such as Falcon.

import math
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.inference_mode()
def load_llm(model_id: str = "tiiuae/falcon-7b-instruct", device: Optional[str] = None):
    """
    Load a causal language model (LLM) with automatic device and dtype handling.
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=dtype
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return tok, model, device


@torch.inference_mode()
def sample_completions(
    tok,
    model,
    device: str,
    prompt: str,
    num_samples: int = 8,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multiple completions for a given prompt, returning token IDs and scores.
    """
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_samples,
        return_dict_in_generate=True,
        output_scores=True,
    )

    prompt_len = enc["input_ids"].shape[1]
    full_ids = out.sequences
    gen_ids = full_ids[:, prompt_len:]
    scores = torch.stack(out.scores)  # [gen_len, K, vocab]

    texts = [
        tok.decode(full_ids[i, prompt_len:], skip_special_tokens=True)
        for i in range(full_ids.size(0))
    ]
    return texts, enc["input_ids"], gen_ids, scores


@torch.inference_mode()
def sequence_logprobs_from_scores(gen_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """
    Compute total log-probabilities of each sampled sequence from per-token scores.
    """
    gen_len, K, _ = scores.shape
    assert gen_ids.shape == (K, gen_len)
    log_probs = torch.log_softmax(scores, dim=-1)
    tok_lp = log_probs.gather(-1, gen_ids.T.unsqueeze(-1)).squeeze(-1)
    return tok_lp.sum(dim=0)  # [K]


def regular_entropy_from_logprobs(seq_logprobs: torch.Tensor, base: float = math.e) -> float:
    """
    Compute Shannon entropy across generated sequences, given their log-probabilities.
    """
    probs = torch.softmax(seq_logprobs, dim=0)
    ent = -(probs * (torch.log2(probs) if base == 2 else torch.log(probs))).sum()
    return float(ent)


def compute_regular_entropy_for_prompt(
    tok,
    model,
    device: str,
    prompt: str,
    num_samples: int = 8,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    """
    Main helper to compute the sequence-level entropy for a prompt using Falcon.
    """
    texts, _, gen_ids, scores = sample_completions(
        tok, model, device, prompt,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    seq_lp = sequence_logprobs_from_scores(gen_ids, scores)
    H = regular_entropy_from_logprobs(seq_lp, base=math.e)
    return texts, seq_lp.tolist(), H
