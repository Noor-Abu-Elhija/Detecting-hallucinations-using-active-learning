# src/entailment.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EntailmentDeberta:
    def __init__(self):

        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-xlarge-mnli").to(self.device)
        self.model.eval()

    def check_implication(self, a: str, b: str) -> float:
        """Check entailment score from a → b"""
        inputs = self.tokenizer(
            a, b,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        entailment_score = probs[0][2].item()  # Index 2 = entailment

        return entailment_score
