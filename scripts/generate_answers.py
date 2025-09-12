# src/falcon_generate_logprobs.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_falcon_model(model_name="tiiuae/falcon-7b-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return tokenizer, model


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
        return_dict_in_generate=True
    )

    # Decode sequences
    sequences = outputs.sequences
    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)

    # Compute token log-probs for the *new* tokens only
    all_logprobs = []
    for i in range(num_return_sequences):
        seq_len = sequences[i].shape[0]
        input_len = input_ids.shape[1]
        gen_len = seq_len - input_len

        # Gather logits and input_ids
        logits = torch.stack(outputs.scores[:gen_len], dim=1)[i]  # shape: (gen_len, vocab)
        token_ids = sequences[i][input_len:]                      # shape: (gen_len)

        # log softmax + gather true token scores
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[range(gen_len), token_ids].tolist()
        all_logprobs.append(token_log_probs)

    # Extract just the generated answers (new tokens)
    completions = [decoded[i][len(prompt):].strip() for i in range(num_return_sequences)]

    return completions, all_logprobs
