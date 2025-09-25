# scripts/create_test_set.py
import json
from datasets import load_dataset

# Load the SQuAD validation set
dataset = load_dataset("squad", split="validation")

# Take the FIRST 10 questions (no shuffle!)
with open("squad_test_10.jsonl", "w", encoding="utf-8") as f:
    for i in range(10):
        item = dataset[i]
        data = {"question": item["question"], "label": 1, "correct_answer": [ans['text'] for ans in item['answers']]}
        f.write(json.dumps(data) + "\n")

print("Successfully created 'squad_test_10.jsonl' with 10 fixed questions.")