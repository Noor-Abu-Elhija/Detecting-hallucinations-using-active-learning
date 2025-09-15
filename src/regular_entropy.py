# src/regular_entropy.py
# Regular (sequence) entropy from an LLM:
# 1) sample K completions,
# 2) compute each completion's sequence log-likelihood from generation scores,
# 3) compute Shannon entropy across the K completions.

import math
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.inference_mode()
def load_llm(model_id: str = "tiiuae/falcon-7b-instruct",
             device: Optional[str] = None):
    """
    Load tokenizer + model for causal LM. Uses GPU if available; otherwise CPU.
    If VRAM is tight, test first with a tiny model like 'gpt2'.

    Returns:
        tok:   Pretrained tokenizer
        model: Pretrained causal LM
        device: 'cuda' or 'cpu'
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Choose dtype sensibly (bf16/fp16 on GPU when possible; fp32 on CPU)
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)

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
    Sample K completions from the model and return:
        texts:    list[str] decoded completions (without the prompt)
        input_ids: prompt token ids (1 x prompt_len)
        gen_ids:  generated token ids (K x gen_len)
        scores:   per-step logits from generate (gen_len x K x vocab)
    """
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        do_sample=True, temperature=temperature, top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_samples,
        return_dict_in_generate=True, output_scores=True,  # <-- critical
    )

    prompt_len = enc["input_ids"].shape[1]
    full_ids = out.sequences  # [K, prompt_len + gen_len]
    gen_ids = full_ids[:, prompt_len:]  # [K, gen_len]

    # Stack list[gen_len] -> tensor [gen_len, K, vocab]
    scores = torch.stack(out.scores)

    # Decode only completion part (skip prompt)
    texts = [
        tok.decode(full_ids[i, prompt_len:], skip_special_tokens=True)
        for i in range(full_ids.size(0))
    ]
    return texts, enc["input_ids"], gen_ids, scores


@torch.inference_mode()
def sequence_logprobs_from_scores(
    gen_ids: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    """
    Sum log P(next_token_t | history) over t for each generated sequence.
    Args:
        gen_ids: [K, gen_len]
        scores:  [gen_len, K, vocab] (pre-softmax logits from generate)
    Returns:
        seq_logprobs: [K] tensor of total log-likelihood per sequence
    """
    gen_len, K, vocab = scores.shape
    assert gen_ids.shape == (K, gen_len), \
        f"gen_ids shape {gen_ids.shape} must be (K, gen_len)=({K}, {gen_len})"

    # Convert logits -> log-probs
    log_probs = torch.log_softmax(scores, dim=-1)          # [gen_len, K, vocab]

    # Gather log-prob of each actually generated token at each step
    # gen_ids.T: [gen_len, K] -> unsqueeze for gather index
    tok_lp = log_probs.gather(-1, gen_ids.T.unsqueeze(-1)).squeeze(-1)  # [gen_len, K]

    # Sum over time steps -> per-sequence log-likelihood
    seq_lp = tok_lp.sum(dim=0)  # [K]
    return seq_lp


def regular_entropy_from_logprobs(
    seq_logprobs: torch.Tensor,
    base: float = math.e,
) -> float:
    """
    Convert sequence log-likelihoods into probabilities over K samples (softmax),
    then compute Shannon entropy: H = -sum_i p_i log(p_i)
    Returns:
        Entropy (float). Units: nats (base=e) or bits (base=2).
    """
    probs = torch.softmax(seq_logprobs, dim=0)  # [K]
    if base == 2:
        ent = -(probs * torch.log2(probs)).sum()
    else:
        ent = -(probs * torch.log(probs)).sum()  # nats
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
    Convenience wrapper:
      - samples K completions
      - computes per-sequence log-likelihoods
      - returns completions, their log-likelihoods, and regular entropy
    """
    texts, _, gen_ids, scores = sample_completions(
        tok, model, device, prompt,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    seq_lp = sequence_logprobs_from_scores(gen_ids, scores)
    H = regular_entropy_from_logprobs(seq_lp, base=math.e)  # nats
    return texts, seq_lp.tolist(), H
