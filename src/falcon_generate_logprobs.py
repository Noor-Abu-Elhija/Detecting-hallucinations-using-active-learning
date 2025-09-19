# src/falcon_generate_logprobs.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def load_falcon_model(model_name="tiiuae/falcon-7b-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # IMPORTANT: Use "auto" to leverage the GPU and bfloat16 for performance
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16  # Use bfloat16 for better performance on modern GPUs
    )
    return tokenizer, model


# RENAMED from generate_with_probs to generate_with_logprobs
def generate_with_logprobs(prompt, model, tokenizer, num_return_sequences=5, temperature=0.7, max_new_tokens=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        num_return_sequences=num_return_sequences,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id
    )

    sequences = outputs.sequences
    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    completions = [decoded[i][len(prompt):].strip() for i in range(num_return_sequences)]

    # NOTE: The run_eval script actually ignores this second return value,
    # but we provide it to match the function call signature.
    # We are calculating sequence log probabilities here.
    all_logprobs = []
    for i in range(num_return_sequences):
        seq_len = sequences[i].shape[0]
        input_len = input_ids.shape[1]
        gen_len = seq_len - input_len

        logits = torch.stack(outputs.scores[:gen_len], dim=1)[i]
        token_ids = sequences[i][input_len:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[range(gen_len), token_ids].tolist()
        all_logprobs.append(token_log_probs)

    # MODIFIED: Return only 2 items to match what run_eval.py expects
    return completions, all_logprobs