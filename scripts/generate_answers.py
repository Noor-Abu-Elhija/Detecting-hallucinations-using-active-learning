# scripts/generate_answers.py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_falcon_model(model_name="tiiuae/falcon-7b-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32
    )

    return tokenizer, model


def format_prompt(question: str) -> str:
    return f"Answer the following question in a single brief but complete sentence.\n\nQuestion: {question}\n\nAnswer:"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_with_probs(prompt, model, tokenizer, num_return_sequences=5, temperature=0.7, max_new_tokens=100):
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
        pad_token_id=tokenizer.eos_token_id  # prevent error due to missing pad token
    )

    sequences = outputs.sequences
    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    completions = [decoded[i][len(prompt):].strip() for i in range(num_return_sequences)]

    all_token_probs = []
    all_sequence_probs = []

    for i in range(num_return_sequences):
        input_len = input_ids.shape[1]
        gen_len = len(outputs.scores)  # total number of generated tokens

        token_ids = sequences[i][input_len:input_len + gen_len]

        logits = torch.stack(outputs.scores[:len(token_ids)], dim=1)[i]  # slice to actual token count
        probs = torch.softmax(logits, dim=-1)

        # Compute probabilities for each generated token
        token_probs = []
        for j in range(len(token_ids)):
            prob = probs[j, token_ids[j]].item()
            if prob > 0:  # Avoid any 0.0 entries
                token_probs.append(prob)

        sequence_prob = np.prod(token_probs)

        all_token_probs.append(token_probs)
        all_sequence_probs.append(sequence_prob)

    return completions, all_token_probs, all_sequence_probs





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_generations", type=int, default=5, help="Number of completions to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--question", type=str, default="What is the capital of France?", help="Question to ask")
    args = parser.parse_args()

    prompt = format_prompt(args.question)
    tokenizer, model = load_falcon_model()
    tokenizer, model = load_falcon_model()

    completions, token_probs, sequence_probs = generate_with_probs(
        prompt,
        model,
        tokenizer,
        num_return_sequences=args.num_generations,
        temperature=args.temperature
    )

    for i, (text, lp) in enumerate(zip(completions, sequence_probs)):
        print(f"\n--- Completion {i+1} ---")
        print(f"Answer: {text}")
        print(f"prob: {lp}")
