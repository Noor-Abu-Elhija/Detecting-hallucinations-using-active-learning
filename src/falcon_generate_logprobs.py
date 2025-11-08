# src/falcon_generate_logprobs.py
# This module provides Falcon-7B generation with token-level log probabilities,
# used for entropy-based uncertainty estimation in language model outputs.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def load_falcon_model(model_name="tiiuae/falcon-7b-instruct"):
    """
    Load the Falcon-7B model and tokenizer on CPU with standard precision (float32).
    Ensures pad_token is defined for stable generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    return tokenizer, model


def generate_with_logprobs(prompt, model, tokenizer,
                           num_return_sequences=5, temperature=0.7, max_new_tokens=100):
    """
    Generate multiple completions from Falcon and compute per-token log probabilities.

    Args:
        prompt (str): Input text to generate from.
        model: Loaded Falcon model.
        tokenizer: Corresponding tokenizer.
        num_return_sequences (int): Number of completions to generate.
        temperature (float): Sampling temperature.
        max_new_tokens (int): Maximum tokens to generate.

    Returns:
        completions (List[str]): Generated text completions.
        all_logprobs (List[List[float]]): Log probabilities per token per sequence.
    """
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        num_return_sequences=num_return_sequences,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False
    )

    sequences = outputs.sequences
    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    completions = [decoded[i][len(prompt):].strip() for i in range(num_return_sequences)]

    all_logprobs = []
    for i in range(num_return_sequences):
        seq_len = sequences[i].shape[0]
        input_len = model_inputs['input_ids'].shape[1]
        gen_len = seq_len - input_len

        logits = torch.stack(outputs.scores[:gen_len], dim=1)[i]
        token_ids = sequences[i][input_len:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[range(gen_len), token_ids].tolist()
        all_logprobs.append(token_log_probs)

    return completions, all_logprobs
